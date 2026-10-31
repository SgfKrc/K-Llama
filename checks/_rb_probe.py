"""Rollback probe for the recurrent (gated-delta-net) snapshot mechanism.

Question: after `llama_memory_seq_rm(seq, P, -1)` the state must be exactly the
state of position P-1, so re-writing the same token at P must reproduce the
first hidden bit-for-bit.  This script measures how far off it is for three
write shapes:

  A) single-token writes   (n_tokens == 1, the spec-decode draft shape)
  B) two-token write       (n_tokens == 2)
  C) K-token write         (n_tokens == K == n_rs_seq + 1)

All three then roll back to P and re-write the same token at P.

Interpretation used in the report:
  * C exact   -> the in-ubatch snapshot slots are fine
  * A/B off   -> the slots do not survive across ubatches
  * B small   -> the conv-state slot scatter hits its max(0, ...) clamp
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

from _shimlib import head_step, load as _shim_load

SHIM = r"H:\slimarc-proto\exp-c\shim\qlh_keep_head.dll"
MODEL = (r"H:\qlh_models\qwen3-6-35b-a3b-opus47-distill-apex-mtp"
         r"\Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf")

os.environ["SA_BACKEND_DIR"] = os.path.dirname(SHIM)
os.environ["SA_N_CPU_MOE"] = "0"

lib = _shim_load(SHIM)


def load_head():
    _shim_load(SHIM, range="0:32", logits=0, mtp=0,
               gpu=int(os.environ.get("PROBE_GPU", "12")))
    ne, nl = ctypes.c_int32(0), ctypes.c_int32(0)
    err = ctypes.create_string_buffer(512)
    h = lib.qlh_kh_load(MODEL.encode(), 1024, 8, 64, 1, 0, 0,
                        ctypes.byref(ne), ctypes.byref(nl), err, len(err))
    if not h:
        raise SystemExit("load failed: %s" % err.value.decode())
    return h, int(ne.value)


def rel(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    d = np.abs(a - b)
    na = float(np.linalg.norm(a))
    return float(d.max()), float(np.linalg.norm(a - b) / max(na, 1e-12)), float(d.max() / max(na, 1e-12))


def main() -> int:
    head, W = load_head()
    print("n_embd=%d  n_rs_seq=%s" % (W, os.environ.get("PROBE_RS_SEQ", "4")))

    buf = (ctypes.c_int32 * 256)()
    raw = "中国的首都是北京，那里有很多名胜古迹。北京是中国的政治文化中心，也是一座历史悠久的城市。".encode("utf-8")
    n = lib.qlh_kh_tokenize(head, raw, len(raw), buf, 256)
    toks = [buf[i] for i in range(n)]
    print("tokens=%d" % n)

    def build(n_tok):
        """single-token chain over toks[0..n_tok-1], also returns hidden at n_tok-1"""
        lib.qlh_kh_reset(head)
        h = None
        for p in range(n_tok):
            h = head_step(head, [toks[p]], p, W)
        return h

    base = int(os.environ.get("PROBE_P", "10"))
    P = base
    if len(toks) < base + 6:
        raise SystemExit("need at least %d tokens, got %d" % (base + 6, len(toks)))

    # determinism control: two independent rebuilds must agree
    build(P - 1)
    hA = head_step(head, [toks[P]], P, W)
    build(P - 1)
    hA2 = head_step(head, [toks[P]], P, W)
    print("\n[control] rebuild determinism: %s" % (rel(hA, hA2),))

    # A: single-token write at P, roll back, rewrite
    print("\n  write shape        P   max|d|     L2rel   maxrel")
    for Pp in range(base, base + 3):
        hh = build(Pp - 1)
        h1 = head_step(head, [toks[Pp]], Pp, W)
        rc = lib.qlh_kh_seq_rm(head, 0, Pp)
        h2 = head_step(head, [toks[Pp]], Pp, W)
        mx, mn, rl = rel(h1, h2)
        print("  A single(1)      %3d  %.3e  %.3e  %.4f%%  rm=%d" % (Pp, mx, mn, 100 * rl, rc))

    # B: two-token write [P-1, P], roll back to P, rewrite P
    for Pp in range(base, base + 3):
        build(Pp - 2)
        hb = head_step(head, [toks[Pp - 1], toks[Pp]], Pp - 1, W)
        rc = lib.qlh_kh_seq_rm(head, 0, Pp)
        h1 = head_step(head, [toks[Pp]], Pp, W)
        mx, mn, rl = rel(hb[-1], h1)
        print("  B two(2)         %3d  %.3e  %.3e  %.4f%%  rm=%d" % (Pp, mx, mn, 100 * rl, rc))

    # C: K-token write [P-4..P] (K = n_rs_seq+1 = 5), roll back to P, rewrite P
    K = int(os.environ.get("PROBE_RS_SEQ", "4")) + 1
    for Pp in range(base, base + 3):
        build(Pp - K)
        hc = head_step(head, toks[Pp - K + 1:Pp + 1], Pp - K + 1, W)
        rc = lib.qlh_kh_seq_rm(head, 0, Pp)
        h1 = head_step(head, [toks[Pp]], Pp, W)
        mx, mn, rl = rel(hc[-1], h1)
        print("  C K-token(%d)     %3d  %.3e  %.3e  %.4f%%  rm=%d" % (K, Pp, mx, mn, 100 * rl, rc))

    # D: K-token write, roll back 3 steps (to P-2), rewrite [P-2..P]
    for Pp in range(base, base + 2):
        build(Pp - K)
        hc = head_step(head, toks[Pp - K + 1:Pp + 1], Pp - K + 1, W)
        rc = lib.qlh_kh_seq_rm(head, 0, Pp - 2)
        hback = head_step(head, toks[Pp - 2:Pp + 1], Pp - 2, W)
        mx, mn, rl = rel(hc[-3:], hback)
        print("  D K-token rm3    %3d  %.3e  %.3e  %.4f%%  rm=%d" % (Pp, mx, mn, 100 * rl, rc))

    # E: single-token chain, roll back one position (rm P-1) and rewrite token P-1
    for Pp in range(base, base + 2):
        build(Pp - 1)
        h1 = head_step(head, [toks[Pp - 1]], Pp - 1, W)
        rc = lib.qlh_kh_seq_rm(head, 0, Pp - 1)
        h2 = head_step(head, [toks[Pp - 1]], Pp - 1, W)
        mx, mn, rl = rel(h1, h2)
        print("  E single rm(P-1) %3d  %.3e  %.3e  %.4f%%  rm=%d" % (Pp, mx, mn, 100 * rl, rc))

    # F: single-token write, roll back 1, then continue two decodes: the exact shape
    #    a speculative loop uses (reject one draft, keep going).
    for Pp in range(base, base + 2):
        build(Pp - 1)
        h1 = head_step(head, [toks[Pp]], Pp, W)
        h2 = head_step(head, [toks[Pp + 1]], Pp + 1, W)
        h3 = head_step(head, [toks[Pp + 2]], Pp + 2, W)
        lib.qlh_kh_seq_rm(head, 0, Pp)
        r1 = head_step(head, [toks[Pp]], Pp, W)
        r2 = head_step(head, [toks[Pp + 1]], Pp + 1, W)
        r3 = head_step(head, [toks[Pp + 2]], Pp + 2, W)
        print("  F single +2 after rm %3d  first=%.3e  second=%.3e  third=%.3e"
              % (Pp, rel(h1, r1)[0], rel(h2, r2)[0], rel(h3, r3)[0]))

    # G: K-token write, then rm by 1..K-1 steps, rewrite the tail
    Kb = K
    for Pp in range(base, base + 1):
        for back in range(1, Kb):
            build(Pp - Kb)
            hc = head_step(head, toks[Pp - Kb + 1:Pp + 1], Pp - Kb + 1, W)
            rc = lib.qlh_kh_seq_rm(head, 0, Pp - back + 1)
            if rc != 0:
                print("  G K-token rm%-2d   %3d  rm refused" % (back, Pp))
                continue
            hb = head_step(head, toks[Pp - back + 1:Pp + 1], Pp - back + 1, W)
            mx, mn, rl = rel(hc[-back:], hb)
            print("  G K-token rm%-2d   %3d  %.3e  %.3e  %.4f%%" % (back, Pp, mx, mn, 100 * rl))

    # H: write 2 tokens, rm by 2, rewrite the SAME 2-token batch.  Same shape means the same
    #    GDN kernel (batched), so what is left here is arithmetic, not rollback bookkeeping.
    for Pp in range(base, base + 2):
        build(Pp - 2)
        hc = head_step(head, [toks[Pp - 1], toks[Pp]], Pp - 1, W)
        rc = lib.qlh_kh_seq_rm(head, 0, Pp - 1)
        hb = head_step(head, [toks[Pp - 1], toks[Pp]], Pp - 1, W)
        print("  H 2tok rm2 same  %3d  max=%.3e  (row0 %.3e row1 %.3e)  rm=%d"
              % (Pp, rel(hc, hb)[0], rel(hc[0:1], hb[0:1])[0], rel(hc[1:2], hb[1:2])[0], rc))

    # I: cost of the snapshot shift, measured on steady-state single-token decodes
    import time
    n_perf = min(8, len(toks) - base)
    build(base)
    t0 = time.perf_counter()
    for q in range(n_perf):
        head_step(head, [toks[base + q]], base + q, W)
    t1 = time.perf_counter()
    print("\n[cost] single-token decode: %.2f ms/step over %d steps" % (1000 * (t1 - t0) / n_perf, n_perf))

    return 0


if __name__ == "__main__":
    sys.exit(main())
