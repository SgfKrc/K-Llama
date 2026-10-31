"""Is the reject-branch retract numerically clean?

Clean path:  head writes token_{P+1} at P+1, then nxt at P+2  -> hidden_{P+2}
Spec path:   head writes [token_{P+1}, d] at P+1, P+2, then seq_rm(0, P+2), then nxt at P+2
             -> hidden_{P+2}

Both must give the SAME hidden for the token sequence to stay identical. Repeated for several
P, each time from a freshly rebuilt history (so this measures cleanliness, not accumulation).

Also compares the tail's token at P+2 between a batched 2-row write and two 1-row writes after
the same history, and the batched-then-rolled-back variant.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

from _shimlib import head_step, load as _shim_load, tail_batch, tail_one

SHIM = r"H:\slimarc-proto\exp-c\shim\qlh_keep_head.dll"
MODEL = (r"H:\qlh_models\qwen3-6-35b-a3b-opus47-distill-apex-mtp"
         r"\Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf")

os.environ["SA_BACKEND_DIR"] = os.path.dirname(SHIM)
os.environ["SA_N_CPU_MOE"] = "0"

lib = _shim_load(SHIM)


def load(rng, need_logits, mtp, gpu):
    _shim_load(SHIM, range=rng, logits=need_logits, mtp=mtp, gpu=gpu)
    ne, nl = ctypes.c_int32(0), ctypes.c_int32(0)
    err = ctypes.create_string_buffer(512)
    h = lib.qlh_kh_load(MODEL.encode(), 1024, 8, 64, 1, 0, 0,
                        ctypes.byref(ne), ctypes.byref(nl), err, len(err))
    if not h:
        raise SystemExit("load %s failed: %s" % (rng, err.value.decode()))
    return h


def main() -> int:
    gpu = int(os.environ.get("PROBE_GPU", "12"))
    steps = int(os.environ.get("PROBE_STEPS", "8"))
    head = load("0:32", 0, 0, gpu)
    tail = load("32:40", 1, 1, 0)
    W = lib.qlh_kh_n_embd(head)

    buf = (ctypes.c_int32 * 256)()
    raw = "中国的首都是".encode("utf-8")
    n = lib.qlh_kh_tokenize(tail, raw, len(raw), buf, 256, 0)
    toks = [buf[i] for i in range(n)]

    # plain chain: tokens at n.. and hidden at n..
    lib.qlh_kh_reset(head)
    lib.qlh_kh_reset(tail)
    hp = head_step(head, toks, 0, W)
    tok = {n: tail_one(tail, hp[-1:], n - 1, W)}
    hid = {}
    p = n
    hid[p] = head_step(head, [tok[p]], p, W)
    while p < n + steps + 2:
        tok[p + 1] = tail_one(tail, hid[p], p, W)
        hid[p + 1] = head_step(head, [tok[p + 1]], p + 1, W)
        p += 1
    print("plain tokens: %r" % [tok[i] for i in range(n, n + steps + 1)])

    def build(P):
        """head holds 0..P, tail holds n-1..P-1 (the round invariant)."""
        lib.qlh_kh_reset(head)
        lib.qlh_kh_reset(tail)
        hp_ = head_step(head, toks, 0, W)
        tail_one(tail, hp_[-1:], n - 1, W)
        hh = head_step(head, [tok[n]], n, W)
        for q in range(n, P):
            tail_one(tail, hh, q, W)
            hh = head_step(head, [tok[q + 1]], q + 1, W)
        # head holds P, tail holds P-1
        tail_one(tail, hh, P, W)                     # tail holds P
        return hh

    print("")
    print("  P   clean grid   spec path    max|diff|   tokens (tail@P+2)")
    worst = 0.0
    for P in range(n, n + steps):
        nxt = tok[P + 2]        # what the reject branch would write at P+2
        fake = tok[P + 2]       # stand-in for a wrong draft: same position, different token
        wrong = tok[P + 3]      # a genuinely different token to stand in for a bad draft

        # clean: head writes token_{P+1} at P+1, then nxt at P+2
        hP = build(P)
        hA = head_step(head, [tok[P + 1]], P + 1, W)
        hclean = head_step(head, [nxt], P + 2, W)
        tclean = tail_one(tail, hclean, P + 2, W)

        # spec A: two SINGLE writes at P+1,P+2, rm to P+2, then nxt at P+2 (HEAD_BATCH=0 path)
        hP = build(P)
        head_step(head, [tok[P + 1]], P + 1, W)
        head_step(head, [wrong], P + 2, W)
        rcA = lib.qlh_kh_seq_rm(head, 0, P + 2)
        hspecA = head_step(head, [nxt], P + 2, W)
        tsA = tail_one(tail, hspecA, P + 2, W)

        # spec B: one BATCHED write at P+1,P+2, rm to P+2, then nxt at P+2 (HEAD_BATCH=1 path)
        hP = build(P)
        head_step(head, [tok[P + 1], wrong], P + 1, W)
        rcB = lib.qlh_kh_seq_rm(head, 0, P + 2)
        hspecB = head_step(head, [nxt], P + 2, W)
        tsB = tail_one(tail, hspecB, P + 2, W)

        dA = float(np.max(np.abs(hclean.astype(np.float64) - hspecA.astype(np.float64))))
        dB = float(np.max(np.abs(hclean.astype(np.float64) - hspecB.astype(np.float64))))
        worst = max(worst, dA)
        print("  %-3d clean=%d  single+rm: %.2e tok %d %s   batch+rm: %.2e tok %d %s"
              % (P, tclean, dA, tsA, "SAME" if tclean == tsA else "DIFF",
                 dB, tsB, "SAME" if tclean == tsB else "DIFF"))

    print("")
    print("worst clean-vs-retract hidden diff: %.3e" % worst)

    # tail: batched 2-row write vs two single writes, at a mid history
    P = n + 3
    build(P)
    hP = head_step(head, [tok[P + 1]], P + 1, W)
    a0 = tail_one(tail, hP, P + 1, W)
    rows = head_step(head, [tok[P + 1], tok[P + 2]], P + 1, W)
    # same hidden rows into the tail, batched, but on a rebuilt state
    build(P)
    hb = tail_batch(tail, rows, P + 1, W)
    build(P)
    x0 = tail_one(tail, rows[0:1], P + 1, W)
    x1 = tail_one(tail, rows[1:2], P + 2, W)
    print("")
    print("tail batched [%d,%d] vs single [%d,%d] -> %s"
          % (hb[0], hb[1], x0, x1, "SAME" if hb == [x0, x1] else "DIFF"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
