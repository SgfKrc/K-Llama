"""End-to-end rollback check over the head/tail split.

head 0:32 takes tokens and returns the pre-norm hidden; tail 32:40 takes that hidden and
returns a token.  Both have linear-attention layers with recurrent state, so a rollback has
to be exact in both.

  1. build the chain up to P-1 (head and tail advance together)
  2. write token P on the head, feed the hidden to the tail        -> hidden1 / token1
  3. seq_rm(P) on both, write token P again                        -> hidden2 / token2
  4. hidden1 == hidden2 (bitwise) and token1 == token2

Also reports the same for a two-token write (the verify-batch shape) so the kernel-path
difference can be told apart from a rollback error.
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

from _shimlib import head_step, load as _shim_load, tail_batch

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
    head = load("0:32", 0, 0, gpu)
    W = int(lib.qlh_kh_n_embd(head))
    tail = load("32:40", 1, 1, 0)

    buf = (ctypes.c_int32 * 256)()
    raw = "中国的首都是北京，那里有很多名胜古迹。".encode("utf-8")
    n = lib.qlh_kh_tokenize(tail, raw, len(raw), buf, 256)
    toks = [buf[i] for i in range(n)]
    P = int(os.environ.get("PROBE_P", "8"))

    def build(upto):
        """advance head+tail together over toks[0..upto-1]"""
        lib.qlh_kh_reset(head)
        lib.qlh_kh_reset(tail)
        hp_ = head_step(head, toks[:upto], 0, W)
        tail_batch(tail, hp_, 0, W)

    def step(p, tokens):
        h = head_step(head, tokens, p, W)
        # the tail consumes every row the head produced (positions p .. p+n-1)
        t = tail_batch(tail, h, p, W)
        return h, [t[-1]]

    print("n_embd=%d tokens=%d P=%d n_rs_seq=%s" % (W, n, P, os.environ.get("PROBE_RS_SEQ", "4")))

    # single-token write, roll back, rewrite
    build(P)
    h1, t1 = step(P, [toks[P]])
    rc_h = lib.qlh_kh_seq_rm(head, 0, P)
    rc_t = lib.qlh_kh_seq_rm(tail, 0, P)
    h2, t2 = step(P, [toks[P]])
    d = float(np.max(np.abs(h1.astype(np.float64) - h2.astype(np.float64))))
    print("single: hidden max|d|=%.3e  token %d/%d %s  rm=%d/%d"
          % (d, t1[0], t2[0], "SAME" if t1 == t2 else "DIFF", rc_h, rc_t))

    # two-token write (verify shape), roll back the last one, rewrite single
    build(P - 1)
    hb, tb = step(P - 1, [toks[P - 1], toks[P]])
    lib.qlh_kh_seq_rm(head, 0, P)
    lib.qlh_kh_seq_rm(tail, 0, P)
    hc, tc = step(P, [toks[P]])
    d = float(np.max(np.abs(hb[-1].astype(np.float64) - hc[-1].astype(np.float64))))
    print("verify-shape: hidden max|d|=%.3e  token %d/%d %s"
          % (d, tb[-1], tc[-1], "SAME" if tb[-1:] == tc else "DIFF"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
