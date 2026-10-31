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

SHIM = r"H:\slimarc-proto\exp-c\shim\qlh_keep_head.dll"
MODEL = (r"H:\qlh_models\qwen3-6-35b-a3b-opus47-distill-apex-mtp"
         r"\Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf")

os.environ["SA_BACKEND_DIR"] = os.path.dirname(SHIM)
os.environ["SA_N_CPU_MOE"] = "0"
os.add_dll_directory(os.path.dirname(SHIM))

lib = ctypes.CDLL(SHIM)
P32 = ctypes.POINTER(ctypes.c_int32)
PF = ctypes.POINTER(ctypes.c_float)

for name, argt, rest in [
    ("qlh_kh_load", [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                     ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, P32, P32,
                     ctypes.c_char_p, ctypes.c_size_t], ctypes.c_void_p),
    ("qlh_kh_forward", [ctypes.c_void_p, P32, ctypes.c_int32, ctypes.c_int32, PF], ctypes.c_int32),
    ("qlh_kh_forward_embd_token", [ctypes.c_void_p, PF, ctypes.c_int32, ctypes.c_int32,
                                   P32, P32, P32, P32], ctypes.c_int32),
    ("qlh_kh_forward_embd_tokens", [ctypes.c_void_p, PF, ctypes.c_int32, ctypes.c_int32,
                                    P32, P32, P32, P32, ctypes.c_int32], ctypes.c_int32),
    ("qlh_kh_tokenize", [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, P32,
                         ctypes.c_int32, ctypes.c_int32], ctypes.c_int32),
    ("qlh_kh_n_embd", [ctypes.c_void_p], ctypes.c_int32),
    ("qlh_kh_seq_rm", [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32], ctypes.c_int32),
]:
    fn = getattr(lib, name)
    fn.argtypes = argt
    fn.restype = rest
lib.qlh_kh_reset.argtypes = [ctypes.c_void_p]
lib.qlh_kh_reset.restype = None


def load(rng, need_logits, mtp, gpu):
    os.environ["LLAMA_LAYER_RANGE"] = rng
    os.environ["SA_NEEDS_LOGITS"] = str(need_logits)
    os.environ["SA_MTP"] = str(mtp)
    os.environ["SA_N_GPU_LAYERS"] = str(gpu)
    os.environ["SA_N_RS_SEQ"] = os.environ.get("PROBE_RS_SEQ", "4")
    ne, nl = ctypes.c_int32(0), ctypes.c_int32(0)
    err = ctypes.create_string_buffer(512)
    h = lib.qlh_kh_load(MODEL.encode(), 1024, 8, 64, 1, 0, 0,
                        ctypes.byref(ne), ctypes.byref(nl), err, len(err))
    if not h:
        raise SystemExit("load %s failed: %s" % (rng, err.value.decode()))
    return h


def hw(head, tokens, at, W):
    out = np.zeros((len(tokens), W), dtype=np.float32)
    rc = lib.qlh_kh_forward(head, (ctypes.c_int32 * len(tokens))(*tokens), len(tokens), at,
                            out.ctypes.data_as(PF))
    if rc != 0:
        raise RuntimeError("head rc=%d at %d" % (rc, at))
    return out


def t1(tail, hidden, at, W):
    t = ctypes.c_int32(-1)
    rc = lib.qlh_kh_forward_embd_token(tail, hidden.ctypes.data_as(PF), 1, at,
                                       None, None, None, ctypes.byref(t))
    if rc != 0:
        raise RuntimeError("tail rc=%d at %d" % (rc, at))
    return t.value


def tn(tail, rows, at, W):
    k = rows.shape[0]
    out = (ctypes.c_int32 * k)()
    rc = lib.qlh_kh_forward_embd_tokens(tail, rows.ctypes.data_as(PF), k, at,
                                        None, None, None, out, k)
    if rc != 0:
        raise RuntimeError("tail rows rc=%d at %d" % (rc, at))
    return list(out)


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
    hp = hw(head, toks, 0, W)
    tok = {n: t1(tail, hp[-1:], n - 1, W)}
    hid = {}
    p = n
    hid[p] = hw(head, [tok[p]], p, W)
    while p < n + steps + 2:
        tok[p + 1] = t1(tail, hid[p], p, W)
        hid[p + 1] = hw(head, [tok[p + 1]], p + 1, W)
        p += 1
    print("plain tokens: %r" % [tok[i] for i in range(n, n + steps + 1)])

    def build(P):
        """head holds 0..P, tail holds n-1..P-1 (the round invariant)."""
        lib.qlh_kh_reset(head)
        lib.qlh_kh_reset(tail)
        hp_ = hw(head, toks, 0, W)
        t1(tail, hp_[-1:], n - 1, W)
        hh = hw(head, [tok[n]], n, W)
        for q in range(n, P):
            t1(tail, hh, q, W)
            hh = hw(head, [tok[q + 1]], q + 1, W)
        # head holds P, tail holds P-1
        t1(tail, hh, P, W)                     # tail holds P
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
        hA = hw(head, [tok[P + 1]], P + 1, W)
        hclean = hw(head, [nxt], P + 2, W)
        tclean = t1(tail, hclean, P + 2, W)

        # spec A: two SINGLE writes at P+1,P+2, rm to P+2, then nxt at P+2 (HEAD_BATCH=0 path)
        hP = build(P)
        hw(head, [tok[P + 1]], P + 1, W)
        hw(head, [wrong], P + 2, W)
        rcA = lib.qlh_kh_seq_rm(head, 0, P + 2)
        hspecA = hw(head, [nxt], P + 2, W)
        tsA = t1(tail, hspecA, P + 2, W)

        # spec B: one BATCHED write at P+1,P+2, rm to P+2, then nxt at P+2 (HEAD_BATCH=1 path)
        hP = build(P)
        hw(head, [tok[P + 1], wrong], P + 1, W)
        rcB = lib.qlh_kh_seq_rm(head, 0, P + 2)
        hspecB = hw(head, [nxt], P + 2, W)
        tsB = t1(tail, hspecB, P + 2, W)

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
    hP = hw(head, [tok[P + 1]], P + 1, W)
    a0 = t1(tail, hP, P + 1, W)
    rows = hw(head, [tok[P + 1], tok[P + 2]], P + 1, W)
    # same hidden rows into the tail, batched, but on a rebuilt state
    build(P)
    hb = tn(tail, rows, P + 1, W)
    build(P)
    x0 = t1(tail, rows[0:1], P + 1, W)
    x1 = t1(tail, rows[1:2], P + 2, W)
    print("")
    print("tail batched [%d,%d] vs single [%d,%d] -> %s"
          % (hb[0], hb[1], x0, x1, "SAME" if hb == [x0, x1] else "DIFF"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
