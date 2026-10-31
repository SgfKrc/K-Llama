"""With a *perfect* draft, does the batched path still match plain greedy?

The earlier divergence test fed a wrong filler token into the KV cache, which corrupts every
later position -- that measured my bug, not batching. Here the second row of each block is the
token plain greedy actually produces, so the KV is never polluted. Any remaining difference
would then be genuine batch-vs-single numeric drift.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

SHIM = r"H:\slimarc-proto\exp-c\shim\qlh_keep_head.dll"
MODEL = r"H:\qlh_models\qwen3-5-2b-gguf\qwen35-2b-Q4_K_M.gguf"

os.environ["SA_BACKEND_DIR"] = r"H:\slimarc-proto\exp-c\shim"
os.environ["SA_N_GPU_LAYERS"] = "0"
os.environ["SA_N_CPU_MOE"] = "0"
os.add_dll_directory(r"H:\slimarc-proto\exp-c\shim")

lib = ctypes.CDLL(SHIM)
P32 = ctypes.POINTER(ctypes.c_int32)
PF = ctypes.POINTER(ctypes.c_float)

lib.qlh_kh_load.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            P32, P32, ctypes.c_char_p, ctypes.c_size_t]
lib.qlh_kh_load.restype = ctypes.c_void_p
lib.qlh_kh_forward.argtypes = [ctypes.c_void_p, P32, ctypes.c_int32, ctypes.c_int32, PF]
lib.qlh_kh_forward_embd_token.argtypes = [ctypes.c_void_p, PF, ctypes.c_int32, ctypes.c_int32,
                                          P32, P32, P32, P32]
lib.qlh_kh_forward_embd_tokens.argtypes = [ctypes.c_void_p, PF, ctypes.c_int32, ctypes.c_int32,
                                           P32, P32, P32, P32, ctypes.c_int32]
lib.qlh_kh_reset.argtypes = [ctypes.c_void_p]
lib.qlh_kh_n_embd.argtypes = [ctypes.c_void_p]
lib.qlh_kh_n_embd.restype = ctypes.c_int32
lib.qlh_kh_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, P32,
                                ctypes.c_int32, ctypes.c_int32]
lib.qlh_kh_tokenize.restype = ctypes.c_int32
lib.qlh_kh_token_to_piece.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p,
                                      ctypes.c_int32]


def load(range_env: str, need_logits: int):
    os.environ["LLAMA_LAYER_RANGE"] = range_env
    os.environ["SA_NEEDS_LOGITS"] = str(need_logits)
    ne, nl = ctypes.c_int32(0), ctypes.c_int32(0)
    err = ctypes.create_string_buffer(512)
    h = lib.qlh_kh_load(MODEL.encode(), 1024, 6, 64, 1, 0, 0,
                        ctypes.byref(ne), ctypes.byref(nl), err, len(err))
    if not h:
        raise SystemExit("load %s: %s" % (range_env, err.value.decode()))
    return h


def piece(h, tok):
    buf = ctypes.create_string_buffer(64)
    n = lib.qlh_kh_token_to_piece(h, tok, buf, len(buf))
    return buf.raw[:n].decode("utf-8", "replace") if n > 0 else ""


def main() -> int:
    code = None
    try:
        head = load("0:20", 0)
        tail = load("20:24", 1)
        W = lib.qlh_kh_n_embd(tail)

        buf = (ctypes.c_int32 * 256)()
        raw = "中国的首都是".encode()
        n = lib.qlh_kh_tokenize(tail, raw, len(raw), buf, 256, 0)
        toks = [buf[i] for i in range(n)]

        def head_run(ts, pos):
            o = np.zeros((len(ts), W), dtype=np.float32)
            rc = lib.qlh_kh_forward(head, (ctypes.c_int32 * len(ts))(*ts), len(ts), pos,
                                    o.ctypes.data_as(PF))
            if rc != 0:
                raise RuntimeError("head rc=%d" % rc)
            return o

        def tail_one(hid, pos):
            t = ctypes.c_int32(-1)
            rc = lib.qlh_kh_forward_embd_token(tail, hid.ctypes.data_as(PF), 1, pos,
                                               None, None, None, ctypes.byref(t))
            if rc != 0:
                raise RuntimeError("tail rc=%d" % rc)
            return t.value

        def tail_k(hid, pos, k):
            out = (ctypes.c_int32 * k)()
            rc = lib.qlh_kh_forward_embd_tokens(tail, hid.ctypes.data_as(PF), k, pos,
                                                None, None, None, out, k)
            if rc != 0:
                raise RuntimeError("tail batch rc=%d" % rc)
            return list(out)

        want = 30
        # 1) baseline: pure single step
        lib.qlh_kh_reset(head); lib.qlh_kh_reset(tail)
        hid = head_run(toks, 0)
        base = [tail_one(hid[-1:], n - 1)]
        p = n
        while len(base) < want:
            hid = head_run([base[-1]], p)
            base.append(tail_one(hid, p))
            p += 1

        # 2) batched, with the *correct* token as the second row
        lib.qlh_kh_reset(head); lib.qlh_kh_reset(tail)
        hid = head_run(toks, 0)
        got = [tail_one(hid[-1:], n - 1)]
        p = n
        i = 1
        mism = []
        mism2 = []
        while len(got) < want:
            perfect = base[i]                      # what greedy would emit next
            hb = head_run([got[-1], perfect], p)
            tb = tail_k(hb, p, 2)
            if tb[0] != perfect:
                mism.append((p, tb[0], perfect))
            # tb[1] 是"免费拿到的下一个 token"。它是块的**第 2 行**，与单步的结果是否相同
            # 是另一个问题——上一版只查了 tb[0]。
            if i + 1 < len(base) and tb[1] != base[i + 1]:
                mism2.append((p + 1, tb[1], base[i + 1]))
            got.append(tb[0]); got.append(tb[1])
            p += 2; i += 2
        got = got[:want]

        same = got == base
        print("baseline: %r" % "".join(piece(tail, t) for t in base[:24]))
        print("batched : %r" % "".join(piece(tail, t) for t in got[:24]))
        print("")
        print("完美草稿下批量 vs 单步: %s" % ("逐位一致" if same else "分叉"))
        if mism:
            print("  第 1 行（tb[0]，用于验货）不符 %d 处:" % len(mism))
            for pos, a, b in mism[:6]:
                print("    位置 %d: 批量 %d(%r)  单步 %d(%r)"
                      % (pos, a, piece(tail, a), b, piece(tail, b)))
        else:
            print("  第 1 行（tb[0]，验货用）: 全部一致  ← 投机接受判据可靠")
        if mism2:
            print("  第 2 行（tb[1]，想免费拿的下一 token）不符 %d/%d 处:"
                  % (len(mism2), len(got) // 2))
            for pos, a, b in mism2[:6]:
                print("    位置 %d: 批量 %d(%r)  单步 %d(%r)  ← 免费 token 不可信"
                      % (pos, a, piece(tail, a), b, piece(tail, b)))
        else:
            print("  第 2 行（tb[1]，免费 token）: 全部一致  ← 可以白拿")
        if not same:
            for j in range(min(len(base), len(got))):
                if base[j] != got[j]:
                    print("  首个分叉 @%d: batch=%d single=%d" % (j, got[j], base[j]))
                    break
        return 0 if same else 1
    finally:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
