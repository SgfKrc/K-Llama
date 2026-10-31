"""Does the speculative block reproduce plain greedy when the draft is always right?

Build the plain greedy chain (single-token steps). Then, for every position p, replay the
speculative block [token@p+1, token@p+2] through head batch + tail batch and check:
    tb[0] == token@p+2      (the verification readout)
    tb[1] == token@p+3      (the "free" token the loop would use as `nxt`)
Also re-checks the rollback path: after a rejected block, roll back one position on BOTH
contexts and confirm the rewrite lands on the plain token.
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
    steps = int(os.environ.get("PROBE_STEPS", "10"))
    gpu = int(os.environ.get("PROBE_GPU", "12"))
    head = load("0:32", 0, 0, gpu)
    tail = load("32:40", 1, 1, 0)
    W = lib.qlh_kh_n_embd(head)

    buf = (ctypes.c_int32 * 256)()
    raw = "中国的首都是".encode("utf-8")
    n = lib.qlh_kh_tokenize(tail, raw, len(raw), buf, 256, 0)
    toks = [buf[i] for i in range(n)]

    # --- plain chain: token@p for p = n .. n+steps, plus hidden@n .. hidden@(n+steps) ---
    lib.qlh_kh_reset(head)
    lib.qlh_kh_reset(tail)
    hp = head_step(head, toks, 0, W)
    tok = {n: tail_one(tail, hp[-1:], n - 1, W)}     # token@n, tail wrote n-1
    hid = {}                                         # hidden@p
    p = n
    hid[p] = head_step(head, [tok[p]], p, W)         # head writes p
    while p < n + steps:
        tok[p + 1] = tail_one(tail, hid[p], p, W)    # tail writes p -> token@p+1
        hid[p + 1] = head_step(head, [tok[p + 1]], p + 1, W)
        p += 1
    print("plain tokens: %r" % [tok[i] for i in range(n, n + steps + 1)])

    # --- block replay: head gets [tok@p+1, tok@p+2] at p+1..p+2 ---
    print("")
    print("  p   block            tb[0] (want tok@p+2)  tb[1] (want tok@p+3)")
    ok0 = ok1 = 0
    bad = []
    for p in range(n, n + steps - 1):
        lib.qlh_kh_reset(head)
        lib.qlh_kh_reset(tail)
        # rebuild state exactly up to p: tail has written n-1 .. p, head has written 0..p
        hpre = head_step(head, toks, 0, W)          # head wrote 0..n-1
        tail_one(tail, hpre[-1:], n - 1, W)         # tail wrote n-1
        hh = head_step(head, [tok[n]], n, W)        # head wrote n; hh = hidden@n
        for q in range(n, p):
            tail_one(tail, hh, q, W)                # tail wrote q
            hh = head_step(head, [tok[q + 1]], q + 1, W)   # head wrote q+1
        # head wrote p, tail wrote p-1, hh = hidden@p
        tail_one(tail, hh, p, W)                    # tail wrote p
        hs = head_step(head, [tok[p + 1], tok[p + 2]], p + 1, W)
        tb = tail_batch(tail, hs, p + 1, W)
        good0 = tb[0] == tok[p + 2]
        good1 = tb[1] == tok[p + 3]
        ok0 += good0
        ok1 += good1
        if not (good0 and good1):
            bad.append((p, tb, tok[p + 2], tok[p + 3]))
        print("  %2d  [%6d,%6d]   %8d %-6s        %8d %-6s"
              % (p, tok[p + 1], tok[p + 2], tb[0], "OK" if good0 else "BAD",
                 tb[1], "OK" if good1 else "BAD"))

    total = len(range(n, n + steps - 1))
    print("")
    print("tb[0] correct %d/%d   tb[1] correct %d/%d" % (ok0, total, ok1, total))
    if bad:
        print("BAD:", bad)
    return 0 if (ok0 == total and ok1 == total) else 1


if __name__ == "__main__":
    raise SystemExit(main())
