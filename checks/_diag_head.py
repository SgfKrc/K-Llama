"""Isolate which stage breaks when the block is processed as a batch: head or tail?

State is rebuilt exactly as the real loop has it at p=3 (head holds 0..3, tail holds 2..3).
Then token@6 is produced four ways:
    single head + single tail   (the plain path, ground truth)
    batch  head + batch  tail   (what the spec loop does)
    batch  head + single tail
    single head + batch  tail
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

SHIM = r"H:\slimarc-proto\exp-c\shim\qlh_keep_head.dll"
MODEL = (r"H:\qlh_models\qwen3-6-35b-a3b-opus47-distill-apex-mtp"
         r"\Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf")

os.environ["SA_BACKEND_DIR"] = r"H:\slimarc-proto\exp-c\shim"
os.environ["SA_N_CPU_MOE"] = "0"
os.add_dll_directory(r"H:\slimarc-proto\exp-c\shim")

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


def head_step(head, tokens, n_past, W):
    out = np.zeros((len(tokens), W), dtype=np.float32)
    rc = lib.qlh_kh_forward(head, (ctypes.c_int32 * len(tokens))(*tokens), len(tokens), n_past,
                            out.ctypes.data_as(PF))
    if rc != 0:
        raise RuntimeError("head rc=%d" % rc)
    return out


def tail_one(tail, hidden, n_past, W):
    t = ctypes.c_int32(-1)
    rc = lib.qlh_kh_forward_embd_token(tail, hidden.ctypes.data_as(PF), 1, n_past,
                                       None, None, None, ctypes.byref(t))
    if rc != 0:
        raise RuntimeError("tail rc=%d" % rc)
    return t.value


def tail_batch(tail, hidden, n_past, W):
    k = hidden.shape[0]
    out = (ctypes.c_int32 * k)()
    rc = lib.qlh_kh_forward_embd_tokens(tail, hidden.ctypes.data_as(PF), k, n_past,
                                        None, None, None, out, k)
    if rc != 0:
        raise RuntimeError("tail batch rc=%d" % rc)
    return list(out)


def main() -> int:
    gpu = int(os.environ.get("PROBE_GPU", "12"))
    head = load("0:32", 0, 0, gpu)
    tail = load("32:40", 1, 1, 0)
    W = lib.qlh_kh_n_embd(head)

    buf = (ctypes.c_int32 * 256)()
    raw = "中国的首都是".encode("utf-8")
    n = lib.qlh_kh_tokenize(tail, raw, len(raw), buf, 256, 0)
    toks = [buf[i] for i in range(n)]

    # plain chain to token@6 with hiddens 3..5
    lib.qlh_kh_reset(head)
    lib.qlh_kh_reset(tail)
    hp = head_step(head, toks, 0, W)
    t = {n: tail_one(tail, hp[-1:], n - 1, W)}
    hid = {}
    p = n
    hid[p] = head_step(head, [t[p]], p, W)
    while p < n + 3:
        t[p + 1] = tail_one(tail, hid[p], p, W)
        hid[p + 1] = head_step(head, [t[p + 1]], p + 1, W)
        p += 1
    print("plain: t@3=%d t@4=%d t@5=%d t@6=%d" % (t[3], t[4], t[5], t[6]))

    def rebuild(p_end, head_mode, tail_mode):
        """head holds 0..p_end, tail holds n-1..p_end; returns hidden@p_end."""
        lib.qlh_kh_reset(head)
        lib.qlh_kh_reset(tail)
        hpre = head_step(head, toks, 0, W)
        t0 = tail_one(tail, hpre[-1:], n - 1, W)      # tail wrote n-1
        if p_end == n:
            hh = head_step(head, [t[n]], n, W)
            return hh                                   # head wrote n, tail wrote n-1
        # write tokens n+1..p_end into head one at a time (head only, for state purposes)
        tokpos = {n: t[n]}
        hh = head_step(head, [t[n]], n, W)
        q = n
        while q < p_end:
            tt = tail_one(tail, hh, q, W)               # tail wrote q -> token@q+1
            tokpos[q + 1] = tt
            hh = head_step(head, [tt], q + 1, W)        # head wrote q+1
            q += 1
        return hh                                        # head wrote p_end, tail wrote p_end-1

    # ---- ground truth: single head at 4, single head at 5, single tail at 4 and 5 ----
    rebuild(3, "s", "s")                     # head 0..3, tail n-1..2  (tail wrote 2)
    h4 = head_step(head, [t[4]], 4, W)       # head 0..4
    # tail currently wrote 2; write 3 and 4 in two single steps
    tail_one(tail, hid[3], 3, W)             # tail wrote 3
    tail_one(tail, h4, 4, W)                 # tail wrote 4
    h5_single = head_step(head, [t[5]], 5, W)
    gt = tail_one(tail, h5_single, 5, W)     # token@6
    print("")
    print("GT (single/single): token@6 = %d  (plain says %d)" % (gt, t[6]))

    # ---- batch head + batch tail ----
    rebuild(3, "b", "b")
    hs = head_step(head, [t[4], t[5]], 4, W)
    tail_one(tail, hid[3], 3, W)
    bb = tail_batch(tail, hs, 4, W)
    print("batch/batch  : tb = %r   token@6 = %d  %s"
          % (bb, bb[1], "OK" if bb[1] == gt else "BAD"))

    # ---- batch head + single tail ----
    rebuild(3, "b", "b")
    hs2 = head_step(head, [t[4], t[5]], 4, W)
    tail_one(tail, hid[3], 3, W)
    s0 = tail_one(tail, hs2[0], 4, W)
    s1 = tail_one(tail, hs2[1], 5, W)
    print("batch/single : tokens = [%d, %d]   token@6 = %d  %s   head-row1 vs single %.3e"
          % (s0, s1, s1, "OK" if s1 == gt else "BAD",
             float(np.max(np.abs(hs2[1].astype(np.float64) - h5_single[0].astype(np.float64))))))

    # ---- single head + batch tail ----
    rebuild(3, "s", "s")
    h4b = head_step(head, [t[4]], 4, W)
    h5b = head_step(head, [t[5]], 5, W)
    # rebuild tail to have written 3, then batch [h4b, h5b] at 4,5
    lib.qlh_kh_reset(tail)
    tail_one(tail, hp[-1:], n - 1, W)
    tail_one(tail, hid[3], 3, W)
    bs = tail_batch(tail, np.stack([h4b[0], h5b[0]]), 4, W)
    print("single/batch : tb = %r   token@6 = %d  %s" % (bs, bs[1], "OK" if bs[1] == gt else "BAD"))

    # magnitudes, to judge whether 2.6e-2 is small
    print("")
    print("|h@p|: " + "  ".join("%d=%.2f" % (k, float(np.linalg.norm(v))) for k, v in sorted(hid.items())))
    d = np.abs(hs2[1].astype(np.float64) - h5_single[0].astype(np.float64))
    print("batch row1 vs single: max %.4e  mean %.4e  rel-to-norm %.2e"
          % (d.max(), d.mean(), d.max() / float(np.linalg.norm(h5_single))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
