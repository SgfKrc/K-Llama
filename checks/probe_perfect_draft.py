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

from _shimlib import head_step, load as _shim_load, tail_batch, tail_one

SHIM = r"H:\slimarc-proto\exp-c\shim\qlh_keep_head.dll"
MODEL = r"H:\qlh_models\qwen3-5-2b-gguf\qwen35-2b-Q4_K_M.gguf"

os.environ["SA_BACKEND_DIR"] = r"H:\slimarc-proto\exp-c\shim"
os.environ["SA_N_CPU_MOE"] = "0"

lib = _shim_load(SHIM, rs_seq=0)


def load(range_env: str, need_logits: int):
    _shim_load(SHIM, range=range_env, logits=need_logits, gpu=0, rs_seq=0)
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
        n = lib.qlh_kh_tokenize(tail, raw, len(raw), buf, 256)
        toks = [buf[i] for i in range(n)]

        want = 30
        # 1) baseline: pure single step
        lib.qlh_kh_reset(head); lib.qlh_kh_reset(tail)
        hid = head_step(head, toks, 0, W)
        base = [tail_one(tail, hid[-1:], n - 1, W)]
        p = n
        while len(base) < want:
            hid = head_step(head, [base[-1]], p, W)
            base.append(tail_one(tail, hid, p, W))
            p += 1

        # 2) batched, with the *correct* token as the second row
        lib.qlh_kh_reset(head); lib.qlh_kh_reset(tail)
        hid = head_step(head, toks, 0, W)
        got = [tail_one(tail, hid[-1:], n - 1, W)]
        p = n
        i = 1
        mism = []
        mism2 = []
        while len(got) < want:
            perfect = base[i]                      # what greedy would emit next
            hb = head_step(head, [got[-1], perfect], p, W)
            tb = tail_batch(tail, hb, p, W)
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
