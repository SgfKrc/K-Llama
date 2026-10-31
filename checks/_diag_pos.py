"""Position-accounting diagnostics for the speculative loop (throwaway probe).

Three independent questions, all on the real 35B two-segment split:

  A. Does a 2-token batched head call produce the same hiddens as two 1-token calls?
     (the speculative loop reads hb[0]/hb[1] and trusts both)
  B. Does a 2-row batched tail call produce the same tokens as two 1-token calls,
     when the tail already holds history at earlier positions?
  C. After qlh_kh_seq_rm(h, 0, p0), does rewriting position p0 give bit-identical
     hiddens/tokens to the first write? (hybrid recurrent rollback cleanliness)
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

from _shimlib import head_step, load as _shim_load, tail_batch, tail_one

SHIM = r"H:\slimarc-proto\exp-c\shim\qlh_keep_head.dll"
MODEL = (r"H:\qlh_models\qwen3-6-35b-a3b-opus47-distill-apex-mtp"
         r"\Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf")

os.environ["SA_BACKEND_DIR"] = r"H:\slimarc-proto\exp-c\shim"
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


def diff(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def main() -> int:
    gpu = int(os.environ.get("PROBE_GPU", "12"))
    head = load("0:32", 0, 0, gpu)
    tail = load("32:40", 1, 1, 0)
    W = lib.qlh_kh_n_embd(head)
    nv = lib.qlh_kh_n_vocab(tail)

    buf = (ctypes.c_int32 * 256)()
    raw = "中国的首都是".encode("utf-8")
    n = lib.qlh_kh_tokenize(tail, raw, len(raw), buf, 256)
    toks = [buf[i] for i in range(n)]
    print("n_embd=%d n_vocab=%d prompt_tokens=%d %r" % (W, nv, n, toks))

    # --- reference: plain single-step path, head hidden at n-1, n, n+1 ---
    lib.qlh_kh_reset(head)
    hp = head_step(head, toks, 0, W)          # 0..n-1
    lib.qlh_kh_reset(tail)
    t_n = tail_one(tail, hp[-1:], n - 1, W)   # token@n
    h_n = head_step(head, [t_n], n, W)        # hidden@n
    lib.qlh_kh_reset(tail)
    t_n_again = tail_one(tail, hp[-1:], n - 1, W)
    print("reset determinism: token@n %d vs %d -> %s"
          % (t_n, t_n_again, "SAME" if t_n == t_n_again else "DIFF"))
    # NOTE: hp/tail were reset above; rebuild head state for the rest
    lib.qlh_kh_reset(head)
    hp = head_step(head, toks, 0, W)
    lib.qlh_kh_reset(tail)
    t_n = tail_one(tail, hp[-1:], n - 1, W)   # token@n  (tail pos n-1)
    h_n = head_step(head, [t_n], n, W)        # hidden@n  (head pos n)
    t_n1 = tail_one(tail, h_n, n, W)          # token@n+1 (tail pos n)
    h_n1 = head_step(head, [t_n1], n + 1, W)  # hidden@n+1 (head pos n+1)
    t_n2 = tail_one(tail, h_n1, n + 1, W)     # token@n+2 (tail pos n+1)
    print("single-step chain: t@n=%d t@n+1=%d t@n+2=%d" % (t_n, t_n1, t_n2))

    # --- A: batched head over [t_n1, t_n2] vs two single steps ---
    lib.qlh_kh_reset(head)
    hp_a = head_step(head, toks, 0, W)
    h_a0 = head_step(head, [t_n1], n, W)
    h_a1 = head_step(head, [t_n2], n + 1, W)
    lib.qlh_kh_reset(head)
    head_step(head, toks, 0, W)
    hb = head_step(head, [t_n1, t_n2], n, W)
    print("")
    print("A) head batch(2) vs single:")
    print("   hb[0] max|diff| = %.3e   rc=%s" % (diff(hb[0], h_a0), "ok"))
    print("   hb[1] max|diff| = %.3e" % diff(hb[1], h_a1))

    # --- B: batched tail over [h_n, h_n1] vs two single steps, with history ---
    lib.qlh_kh_reset(tail)
    tail_one(tail, hp[-1:], n - 1, W)          # pos n-1
    b0 = tail_one(tail, h_n, n, W)             # pos n   -> token@n+1
    b1 = tail_one(tail, h_n1, n + 1, W)        # pos n+1 -> token@n+2
    lib.qlh_kh_reset(tail)
    tail_one(tail, hp[-1:], n - 1, W)
    bb = tail_batch(tail, np.stack([h_n, h_n1]), n, W)
    print("")
    print("B) tail batch(2) vs single (history at n-1):")
    print("   single -> [%d, %d]" % (b0, b1))
    print("   batch  -> %r" % bb)
    print("   MATCH: %s" % (bb == [b0, b1]))

    # --- C: rollback cleanliness ---
    # head: write pos n+1 (h_a1), rm to n+1, rewrite pos n+1 -> same hidden?
    lib.qlh_kh_reset(head)
    head_step(head, toks, 0, W)
    head_step(head, [t_n1], n, W)
    h_c_first = head_step(head, [t_n2], n + 1, W)
    rc_rm = lib.qlh_kh_seq_rm(head, 0, n + 1)
    h_c_again = head_step(head, [t_n2], n + 1, W)
    print("")
    print("C) head rollback: seq_rm rc=%d  rewrite max|diff| = %.3e"
          % (rc_rm, diff(h_c_first, h_c_again)))

    # tail: write pos n+2 then roll back and rewrite
    lib.qlh_kh_reset(tail)
    tail_one(tail, hp[-1:], n - 1, W)
    tail_one(tail, h_n, n, W)
    tail_one(tail, h_n1, n + 1, W)
    t_c_first = tail_one(tail, h_n2 := h_n1, n + 2, W)
    rc_rm2 = lib.qlh_kh_seq_rm(tail, 0, n + 2)
    t_c_again = tail_one(tail, h_n1, n + 2, W)
    print("C) tail rollback: seq_rm rc=%d  rewrite %d vs %d -> %s"
          % (rc_rm2, t_c_first, t_c_again, "SAME" if t_c_first == t_c_again else "DIFF"))

    # --- D: what does the batched tail return for row 1 when row 0's hidden is wrong? ---
    # (verifies the "free token" trick: tb[1] must equal the token the trunk emits after d)
    lib.qlh_kh_reset(tail)
    tail_one(tail, hp[-1:], n - 1, W)
    tail_one(tail, h_n, n, W)                  # pos n -> token@n+1
    # feed [h_n1 (true hidden of n+1), garbage] at n+1, n+2 and see row 0
    garbage = np.zeros(W, dtype=np.float32)
    dd = tail_batch(tail, np.stack([h_n1, garbage]), n + 1, W)
    print("")
    print("D) tail row0 with garbage row1: %d (expect %d) -> %s"
          % (dd[0], t_n2, "SAME" if dd[0] == t_n2 else "DIFF"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
