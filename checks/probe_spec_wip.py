"""Speculative decode with the MTP head: does it actually produce the same tokens faster?

Semantics being verified. At position p the trunk emits t1 and leaves hidden h_p. The MTP head
predicts x_{p+2} from (h_p, t1) -- call that d. To verify, submit the block [t1, d] for positions
p+1 and p+2 in ONE batched call; the trunk's own token at p+1 is then its prediction of x_{p+2}.
If that equals d, both t1 and d are accepted (2 tokens for one batched step); otherwise only t1
survives and d is discarded.

Correctness bar: the token sequence must be identical to plain greedy decode. MTP only changes how
many tokens one step covers, never which tokens are correct.
"""
from __future__ import annotations

import ctypes
import os
import statistics
import sys
import time

import numpy as np

SHIM = r"H:\slimarc-proto\exp-c\shim\qlh_keep_head.dll"
MODEL = (r"H:\qlh_models\qwen3-6-35b-a3b-opus47-distill-apex-mtp"
         r"\Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf")

os.environ["SA_BACKEND_DIR"] = r"H:\slimarc-proto\exp-c\shim"
os.environ["SA_N_CPU_MOE"] = "0"
DEBUG = os.environ.get("PROBE_DEBUG") == "1"
os.add_dll_directory(r"H:\slimarc-proto\exp-c\shim")

lib = ctypes.CDLL(SHIM)
P32 = ctypes.POINTER(ctypes.c_int32)
PF = ctypes.POINTER(ctypes.c_float)

lib.qlh_kh_load.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            P32, P32, ctypes.c_char_p, ctypes.c_size_t]
lib.qlh_kh_load.restype = ctypes.c_void_p
lib.qlh_kh_forward.argtypes = [ctypes.c_void_p, P32, ctypes.c_int32, ctypes.c_int32, PF]
lib.qlh_kh_forward.restype = ctypes.c_int32
lib.qlh_kh_forward_embd_token.argtypes = [ctypes.c_void_p, PF, ctypes.c_int32, ctypes.c_int32,
                                          P32, P32, P32, P32]
lib.qlh_kh_forward_embd_token.restype = ctypes.c_int32
lib.qlh_kh_forward_embd_tokens.argtypes = [ctypes.c_void_p, PF, ctypes.c_int32, ctypes.c_int32,
                                           P32, P32, P32, P32, ctypes.c_int32]
lib.qlh_kh_forward_embd_tokens.restype = ctypes.c_int32
lib.qlh_kh_last_hidden.argtypes = [ctypes.c_void_p, PF, ctypes.c_int32]
lib.qlh_kh_last_hidden.restype = ctypes.c_int32
lib.qlh_kh_mtp_draft.argtypes = [ctypes.c_void_p, PF, ctypes.c_int32, ctypes.c_int32,
                                 ctypes.c_int32, P32]
lib.qlh_kh_mtp_draft.restype = ctypes.c_int32
lib.qlh_kh_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, P32,
                                ctypes.c_int32, ctypes.c_int32]
lib.qlh_kh_tokenize.restype = ctypes.c_int32
lib.qlh_kh_n_embd.argtypes = [ctypes.c_void_p]
lib.qlh_kh_n_embd.restype = ctypes.c_int32
lib.qlh_kh_reset.argtypes = [ctypes.c_void_p]
lib.qlh_kh_seq_rm.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
lib.qlh_kh_seq_rm.restype = ctypes.c_int32
lib.qlh_kh_token_to_piece.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p,
                                      ctypes.c_int32]
lib.qlh_kh_token_to_piece.restype = ctypes.c_int32


def load(rng: str, need_logits: int, mtp: int, gpu: int):
    os.environ["LLAMA_LAYER_RANGE"] = rng
    os.environ["SA_NEEDS_LOGITS"] = str(need_logits)
    os.environ["SA_MTP"] = str(mtp)
    os.environ["SA_N_GPU_LAYERS"] = str(gpu)
    # Recurrent-state snapshots: without them llama_memory_seq_rm refuses to roll back a hybrid
    # model's linear-layer state, and a rejected draft cannot be undone.
    os.environ["SA_N_RS_SEQ"] = os.environ.get("PROBE_RS_SEQ", "4")
    ne, nl = ctypes.c_int32(0), ctypes.c_int32(0)
    err = ctypes.create_string_buffer(512)
    h = lib.qlh_kh_load(MODEL.encode(), 1024, 8, 64, 1, 0, 0,
                        ctypes.byref(ne), ctypes.byref(nl), err, len(err))
    if not h:
        raise SystemExit("load %s failed: %s" % (rng, err.value.decode()))
    return h


def piece(h, tok: int) -> str:
    buf = ctypes.create_string_buffer(64)
    n = lib.qlh_kh_token_to_piece(h, tok, buf, len(buf))
    return buf.raw[:n].decode("utf-8", "replace") if n > 0 else "?"


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
    prompt = sys.argv[1] if len(sys.argv) > 1 else "中国的首都是"
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    gpu = int(os.environ.get("PROBE_GPU", "12"))

    head = load("0:32", 0, 0, gpu)
    tail = load("32:40", 1, 1, 0)
    W = lib.qlh_kh_n_embd(head)

    buf = (ctypes.c_int32 * 256)()
    raw = prompt.encode("utf-8")
    n = lib.qlh_kh_tokenize(tail, raw, len(raw), buf, 256, 0)
    toks = [buf[i] for i in range(n)]
    print("n_embd=%d  prompt=%d tokens  %r" % (W, n, prompt))

    hidden = head_step(head, toks, 0, W)

    # NOTE: tail is NOT prefilled over the whole prompt. probe_mtp_accept.py works this way and
    # produces sane text, so keep the same shape here; pre-filling the tail was what pushed the
    # head into an invalid batch at position 4.
    lib.qlh_kh_reset(tail)
    first = [tail_one(tail, hidden[-1:], n - 1, W)]

    # ---- plain greedy, timed ----

    plain = [first[-1]]          # the prompt's last position already yielded a token
    t0 = time.perf_counter()
    p = n                        # next position to fill
    h = head_step(head, [first[-1]], p, W)
    for _ in range(want - 1):
        t = tail_one(tail, h, p, W)
        plain.append(t)
        h = head_step(head, [t], p + 1, W)
        p += 1
    plain_ms = (time.perf_counter() - t0) * 1000

    # ---- speculative, timed ----
    # Reset BOTH contexts and re-prefill the head: its KV still holds the plain run's positions,
    # and writing position p+1 again there is an invalid batch, not a slow path.
    lib.qlh_kh_reset(head)
    hidden = head_step(head, toks, 0, W)
    lib.qlh_kh_reset(tail)
    first2 = [tail_one(tail, hidden[-1:], n - 1, W)]
    spec = [first2[-1]]
    skipped = 0
    t0 = time.perf_counter()
    p = n
    h = head_step(head, [first2[-1]], p, W)
    while len(spec) < want:
        # t1 is the trunk's token at position p; its hidden is what the draft head needs.
        t1 = tail_one(tail, h, p, W)
        lh = np.zeros(W, dtype=np.float32)
        rc = lib.qlh_kh_last_hidden(tail, lh.ctypes.data_as(PF), W)
        d = ctypes.c_int32(-1)
        drafted = (rc == 0 and
                   lib.qlh_kh_mtp_draft(tail, lh.ctypes.data_as(PF), W, t1, p,
                                        ctypes.byref(d)) == 0)

        # One batched pass over [t1, d] fills positions p+1..p+2 and yields the trunk's own
        # token at each of them. tb[0] is the trunk's prediction for p+2: if it equals d, the
        # draft was right. Either way tb[0] is a real token, so nothing computed is wasted.
        block = [t1, d.value] if drafted else [t1]
        if DEBUG:
            print("    p=%d tail_one@p, block=%d -> head/tail @%d..%d"
                  % (p, len(block), p + 1, p + len(block)))
        hb = head_step(head, block, p + 1, W)
        tb = tail_batch(tail, hb, p + 1, W)

        if drafted and tb[0] == d.value:
            spec.append(t1)
            spec.append(d.value)
            p += 2                       # t1 at p+1, d at p+2
            nxt = tb[1] if len(tb) > 1 else None
            if nxt is None:
                break
        else:
            spec.append(t1)
            p += 1                       # t1 at p+1
            nxt = tb[0]
            skipped += 1
            # The batched pass already wrote position p+1 for the rejected draft. Drop it from
            # both stages, otherwise the next step writes that position again and llama_decode
            # rejects the batch outright.
            for stage in (head, tail):
                rr = lib.qlh_kh_seq_rm(stage, 0, p + 1)
                if rr != 0:
                    print("      seq_rm rc=%d (position %d)" % (rr, p + 1))

        # nxt is the trunk's token for position p (free: already paid for above).
        if DEBUG:
            print("      tail_one already wrote p=%d; block wrote %d..%d; now head writes %d"
                  % (p - (2 if drafted and tb[0] == d.value else 1),
                     p - (2 if drafted and tb[0] == d.value else 1) + 1, p, p + 1))
        h = head_step(head, [nxt], p + 1, W)
        p += 1
    spec_ms = (time.perf_counter() - t0) * 1000

    spec = spec[:want]
    same = spec == plain[:len(spec)]
    print("")
    print("plain : %r" % "".join(piece(tail, t) for t in plain))
    print("spec  : %r" % "".join(piece(tail, t) for t in spec))
    print("token 序列一致: %s" % same)
    print("")
    print("plain %.0f ms (%.1f ms/token)" % (plain_ms, plain_ms / len(plain)))
    print("spec  %.0f ms (%.1f ms/token)   接受率 %.1f%%  拒稿 %d 次"
          % (spec_ms, spec_ms / max(len(spec), 1),
             100.0 * (len(spec) - skipped - (len(spec) - skipped) / 2) / max(len(spec), 1),
             100.0 * skipped / max(len(spec), 1)))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
