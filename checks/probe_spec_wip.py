"""Speculative decode with the MTP head: does it produce the same tokens as plain greedy?

Semantics being verified. At position p the trunk emits t1 and leaves hidden h_p. The MTP head
predicts x_{p+2} from (h_p, t1) -- call that d. To verify, submit the block [t1, d] for positions
p+1 and p+2 in ONE batched call; the trunk's own token at p+2 is then its prediction of x_{p+2}.
If that equals d, both t1 and d are accepted; otherwise only t1 survives and d is discarded.

Correctness bar: the token sequence must be identical to plain greedy decode. MTP only changes how
many tokens one step covers, never which tokens are correct.

================================================================================================
POSITION LEDGER

Two contexts: `head` (layers 0:32, token -> hidden) and `tail` (32:40, hidden -> token). Every
position k is written exactly once in EACH context:

    head writes token_k  at k      qlh_kh_forward            (returns hidden_k)
    tail writes hidden_k at k      qlh_kh_forward_embd_token / _tokens
    the tail's decode at k yields token_{k+1}

Counters `hn` / `tn` = positions written so far, checked after every action. Round invariant:

    hn == P+1 and tn == P          head holds hidden_P; the tail has written 0..P-1

One round of the speculative loop:

    a. t_next = tail_one(hidden_P, P)      tail writes P            tn -> P+1   = token_{P+1}
       emit t_next                                     (position P+1)
    b. lh = last_hidden()                  the row just decoded     = hidden_P
    c. d  = mtp_draft(lh, t_next, P)       (h_P, token_{P+1}) -> predicts token_{P+2}
    d. block [t_next, d] at P+1, P+2       head writes 2 rows       hn -> P+3
                                           tail writes 2 rows       tn -> P+3
       tb[0] = token_{P+2}  (the checkout that decides tb[0] == d)
       tb[1] = token_{P+3}  (a real trunk token: it must be EMITTED, see below)
    e. accept (tb[0] == d): emit d (P+2); nxt = tb[1];
                            head writes nxt at P+3                   hn -> P+4;
                            emit nxt (P+3);  P -> P+3
       reject            : seq_rm(head/tail, 0, P+2)  -- retract P+2 in BOTH contexts
                            hn = tn = P+2; nxt = tb[0] = token_{P+2};
                            head writes nxt at P+2                   hn -> P+3;
                            emit nxt (P+2);  P -> P+2

Where the previous WIP went wrong (three separate defects, all in the assembly):

  1. It emitted t1 and d but never `tb[1]`. tb[1] is a genuine trunk token at P+3; the next round
     recomputes token_{P+4} at the top and appends that, so one token per accepted round was
     dropped. The emitted list then maps to positions n, n+1, n+2, n+4, n+5, n+7 ... -- i.e. the
     token sequence can never equal plain greedy no matter what the positions do.
  2. `nxt` was written for the next step but the round then restarted its `tail_one` at a position
     inferred from `p += 2; p += 1` instead of from the ledger, so the retract point in the reject
     branch was one position off from the position the batch had actually written. Here both are
     derived from hn/tn and asserted.
  3. On reject it retracted [p+2, ..) after already bumping p, which makes the retract point and
     the next head write the same slot; fine in principle, but with no ledger there is nothing to
     assert it against. Here seq_rm target == the last position the batch wrote + 1, always.

Cost accounting (so the timings make sense): one accepted round writes 3 head positions (2 in the
block + the trailing nxt) and 3 tail positions for 3 emitted tokens; a rejected round writes 3 +
3 positions for 2 emitted tokens. Expected positions per token is 3/(2+r), which is ~1.09 at
r = 0.75 -- slightly worse than plain (1.0). k=1 with a single MTP layer cannot do better; the win
has to come from the batched decode being cheaper than two single ones, which is what the timings
measure.

================================================================================================
WHY THERE ARE THREE STRATEGIES

`block (head batch)` submits [t_next, d] to the head as one 2-token decode -- what a real
implementation wants. Measured on this machine that call is NOT bit-identical to two 1-token calls:
every row of the batch differs by ~4.6e-3 relative even with identical inputs and positions. The
model is MoE, so that shift moves an argmax (measured: token@6 108820 -> 95761) and the sequence
diverges. It is not a position error -- with `n_ubatch = 1` and no recurrent snapshots the same
block reproduces plain greedy bit for bit.

`block (head 1-tok)` keeps the head on 1-token decodes and only batches the tail, whose multi-row
path IS bit-identical to per-row calls (verified). It still diverges on long runs: the reject branch
must retract position P+2, and `qlh_kh_seq_rm`'s partial rollback moves the recurrent snapshot index
without restoring the exact state. Measured clean-vs-retracted hidden after one 1-token write, rm and
rewrite: 0.13..0.62 absolute on vectors of norm ~6.5 (2..10% relative).

`vtw (no retract)` verifies before writing, so nothing is ever retracted: token@(P+2) comes from a
1-token tail decode at P+1 and is only then written at P+2. It is plain decode plus a draft
comparison, so it reproduces plain greedy exactly -- the right harness for the ledger. It also
reports the draft hit rate, the number that says whether the (hidden@p, token@p+1) pairing is right.

Env:
    PROBE_DEBUG=1        print the position ledger every round
    PROBE_ONLY=vtw | block-batch | block-single
                         which strategy to run (default: vtw). One per process on purpose.
    PROBE_SELFTEST=1     only measure head batch-vs-single drift, then exit
    PROBE_RS_SEQ=n       recurrent snapshots (default 4); rollback needs >= 1
    PROBE_GPU=n          GPU layers for the head (default 12)
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

import numpy as np

SHIM = r"H:\slimarc-proto\exp-c\shim\qlh_keep_head.dll"
MODEL = (r"H:\qlh_models\qwen3-6-35b-a3b-opus47-distill-apex-mtp"
         r"\Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf")

os.environ["SA_BACKEND_DIR"] = os.path.dirname(SHIM)
os.environ["SA_N_CPU_MOE"] = "0"
DEBUG = os.environ.get("PROBE_DEBUG") == "1"
SELFTEST = os.environ.get("PROBE_SELFTEST") == "1"
os.add_dll_directory(os.path.dirname(SHIM))

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
lib.qlh_kh_reset.restype = None
lib.qlh_kh_seq_rm.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
lib.qlh_kh_seq_rm.restype = ctypes.c_int32
lib.qlh_kh_token_to_piece.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p,
                                      ctypes.c_int32]
lib.qlh_kh_token_to_piece.restype = ctypes.c_int32


# ---------------------------------------------------------------------------------- plumbing

def load(rng: str, need_logits: int, mtp: int, gpu: int):
    os.environ["LLAMA_LAYER_RANGE"] = rng
    os.environ["SA_NEEDS_LOGITS"] = str(need_logits)
    os.environ["SA_MTP"] = str(mtp)
    os.environ["SA_N_GPU_LAYERS"] = str(gpu)
    # Recurrent-state snapshots. Without them llama_memory_seq_rm refuses to roll back a hybrid
    # model's linear-layer state and a rejected draft cannot be undone. llama.cpp also requires
    # n_ubatch > n_rs_seq + 1, so this value bounds the micro-batch from below.
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


def head_write(head, tokens, at, W):
    out = np.zeros((len(tokens), W), dtype=np.float32)
    rc = lib.qlh_kh_forward(head, (ctypes.c_int32 * len(tokens))(*tokens), len(tokens), at,
                            out.ctypes.data_as(PF))
    if rc != 0:
        raise RuntimeError("head forward rc=%d at %d" % (rc, at))
    return out


def tail_one(tail, hidden, at, W):
    t = ctypes.c_int32(-1)
    rc = lib.qlh_kh_forward_embd_token(tail, hidden.ctypes.data_as(PF), 1, at,
                                       None, None, None, ctypes.byref(t))
    if rc != 0:
        raise RuntimeError("tail token rc=%d at %d" % (rc, at))
    return t.value


def tail_rows(tail, rows, at, W):
    k = rows.shape[0]
    out = (ctypes.c_int32 * k)()
    rc = lib.qlh_kh_forward_embd_tokens(tail, rows.ctypes.data_as(PF), k, at,
                                        None, None, None, out, k)
    if rc != 0:
        raise RuntimeError("tail rows rc=%d at %d" % (rc, at))
    return list(out)


def last_hidden(tail, W):
    lh = np.zeros(W, dtype=np.float32)
    rc = lib.qlh_kh_last_hidden(tail, lh.ctypes.data_as(PF), W)
    if rc != 0:
        raise RuntimeError("last_hidden rc=%d" % rc)
    return lh


def mtp_draft(tail, hidden, token, at, W):
    d = ctypes.c_int32(-1)
    if lib.qlh_kh_mtp_draft(tail, hidden.ctypes.data_as(PF), W, token, at,
                            ctypes.byref(d)) != 0:
        return None
    return d.value


def seq_rm(head, tail, p0):
    for name, h in (("head", head), ("tail", tail)):
        rc = lib.qlh_kh_seq_rm(h, 0, p0)
        if rc != 0:
            raise RuntimeError("seq_rm(%s, 0, %d) rc=%d" % (name, p0, rc))


# ---------------------------------------------------------------------------------- the loop

def generate(head, tail, prompt_tokens, want, W, spec, head_batch, no_rollback=False):
    """Generate `want` tokens. Returns (tokens, seconds, accepted_drafts, drafted_rounds)."""
    n = len(prompt_tokens)
    lib.qlh_kh_reset(head)
    lib.qlh_kh_reset(tail)

    hp = head_write(head, prompt_tokens, 0, W)       # head: 0 .. n-1
    if hp.shape[0] != n:
        raise RuntimeError("head prefill returned %d rows for %d tokens" % (hp.shape[0], n))

    tok = tail_one(tail, hp[-1:], n - 1, W)          # tail: n-1  ->  token@n
    hidden = head_write(head, [tok], n, W)           # head: n    ->  hidden@n

    hn, tn, P = n + 1, n, n
    tokens = [tok]                                   # token@n
    accepted = drafted = rejected = 0
    t0 = time.perf_counter()

    while len(tokens) < want:
        if not (hn == P + 1 and tn == P):
            raise AssertionError("round start ledger: hn=%d tn=%d P=%d" % (hn, tn, P))

        # a. tail writes P -> token@(P+1); hidden@P is what the head produced last round
        t_next = tail_one(tail, hidden, P, W)
        tn = P + 1
        tokens.append(t_next)                        # position P+1

        # b. the row the tail just decoded is hidden@P, exactly what MTP needs
        lh = last_hidden(tail, W)

        # c. draft token@(P+2)
        d = mtp_draft(tail, lh, t_next, P, W) if spec else None

        if no_rollback and d is not None:
            # Verify-then-write: the head never consumes the draft, so nothing is ever retracted.
            # token@(P+2) must be known before anything is written at P+2, so the block is
            # [t_next] then [t2]: two 1-token head writes, two 1-token tail writes, 2 tokens out.
            drafted += 1
            h1 = head_write(head, [t_next], P + 1, W)
            hn = P + 2
            t2 = tail_one(tail, h1, P + 1, W)
            tn = P + 2
            tokens.append(t2)                        # position P+2
            if t2 == d:
                accepted += 1
            hidden = head_write(head, [t2], P + 2, W)
            hn = P + 3
            P = P + 2
            if DEBUG:
                print("  P=%-3d verify-first: tail@%d head@%d,%d tail@%d,%d  d=%d t2=%d %s"
                      % (P - 2, P - 2, P - 1, P, P - 1, P, d, t2,
                         "ACCEPT" if t2 == d else "REJECT"))
            if not (hn == P + 1 and tn == P):
                raise AssertionError("ledger: hn=%d tn=%d P=%d" % (hn, tn, P))
            continue

        if d is None:
            # No draft (or not asked): one position, one token.
            hidden = head_write(head, [t_next], P + 1, W)
            hn = P + 2
            P = P + 1
            if DEBUG:
                print("  P=%-3d tail@P, head@%d (no draft)" % (P - 1, P))
            if not (hn == P + 1 and tn == P):
                raise AssertionError("ledger after plain step: hn=%d tn=%d P=%d" % (hn, tn, P))
            continue

        drafted += 1
        # d. one verification block over [t_next, d] at P+1, P+2
        if head_batch:
            rows = head_write(head, [t_next, d], P + 1, W)
            hn = P + 3
            drift = None
        else:
            r0 = head_write(head, [t_next], P + 1, W)
            r1 = head_write(head, [d], P + 2, W)
            rows = np.stack([r0[0], r1[0]])
            hn = P + 3
            drift = None

        tb = tail_rows(tail, rows, P + 1, W)
        tn = P + 3
        if not (hn == P + 3 and tn == P + 3):
            raise AssertionError("ledger after block: hn=%d tn=%d" % (hn, tn))

        if tb[0] == d:
            # e. accept: d is token@(P+2); tb[1] is token@(P+3) and IS emitted
            accepted += 1
            tokens.append(d)                         # position P+2
            nxt = tb[1]
            hidden = head_write(head, [nxt], P + 3, W)
            hn = P + 4
            tokens.append(nxt)                       # position P+3
            P = P + 3
            if DEBUG:
                print("  P=%-3d block@[%d,%d] tb=%s ACCEPT d=%d nxt=%d (head@%d)"
                      % (P - 3, P - 2, P - 1, tb, d, nxt, P))
        else:
            # e'. reject: the block already wrote P+2 in BOTH contexts; retract it there
            rejected += 1
            seq_rm(head, tail, P + 2)
            hn = tn = P + 2
            nxt = tb[0]                              # token@(P+2), the trunk's own answer
            hidden = head_write(head, [nxt], P + 2, W)
            hn = P + 3
            tokens.append(nxt)                       # position P+2
            P = P + 2
            if DEBUG:
                print("  P=%-3d block@[%d,%d] tb=%s REJECT d=%d -> nxt=%d (rm to %d, head@%d)"
                      % (P - 2, P - 1, P, tb, d, nxt, P, P))

        if not (hn == P + 1 and tn == P):
            raise AssertionError("round end ledger: hn=%d tn=%d P=%d" % (hn, tn, P))

    dt = time.perf_counter() - t0
    return tokens[:want], dt, accepted, drafted


def selftest(head, tail, prompt_tokens, W):
    """Show the one number that decides whether a batched head can ever match plain."""
    n = len(prompt_tokens)
    lib.qlh_kh_reset(head)
    lib.qlh_kh_reset(tail)
    hp = head_write(head, prompt_tokens, 0, W)
    t = {n: tail_one(tail, hp[-1:], n - 1, W)}
    hid = {}
    p = n
    hid[p] = head_write(head, [t[p]], p, W)
    while p < n + 2:
        t[p + 1] = tail_one(tail, hid[p], p, W)
        hid[p + 1] = head_write(head, [t[p + 1]], p + 1, W)
        p += 1
    print("plain chain: token@%d..%d = %r" % (n, n + 2, [t[i] for i in range(n, n + 3)]))

    words = [t[n + 1], t[n + 2]]
    lib.qlh_kh_reset(head)
    head_write(head, prompt_tokens, 0, W)
    singles = np.stack([head_write(head, [words[0]], n, W)[0],
                        head_write(head, [words[1]], n + 1, W)[0]])
    lib.qlh_kh_reset(head)
    head_write(head, prompt_tokens, 0, W)
    batch = head_write(head, words, n, W)
    for i in range(2):
        d = float(np.max(np.abs(batch[i].astype(np.float64) - singles[i].astype(np.float64))))
        print("  head row %d: max|batch-single| = %.3e   rel = %.2e"
              % (i, d, d / float(np.linalg.norm(singles[i]))))
    print("  n_rs_seq=%s (llama.cpp requires n_ubatch > n_rs_seq+1, so a 2-token batch cannot be"
          " split into 1-token ubatches)" % os.environ.get("SA_N_RS_SEQ"))
    return 0


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
    if n <= 0:
        raise SystemExit("tokenize failed")
    toks = [buf[i] for i in range(n)]
    print("n_embd=%d  prompt=%d tokens  %r" % (W, n, prompt))
    print("rs_seq=%s   gpu_layers=%d" % (os.environ.get("SA_N_RS_SEQ", "4"), gpu))

    if SELFTEST:
        return selftest(head, tail, toks, W)

    only = os.environ.get("PROBE_ONLY", "").strip()

    plain, plain_s, _, _ = generate(head, tail, toks, want, W, spec=False, head_batch=False)
    plain_ms = plain_s * 1000 / len(plain)
    print("")
    print("plain            : %r" % "".join(piece(tail, t) for t in plain))
    print("                   %.1f ms/token" % plain_ms)

    # One strategy per process: the MTP context has no reset entry point, so a second generate()
    # in the same process keeps drafting into a context whose positions already ran past -- the
    # drafts then fail outright (measured: 0/0 accepted). Re-run with PROBE_ONLY= to switch.
    runs = []
    if only == "":
        runs.append(("vtw (no retract)", True, True))
    elif only == "vtw":
        runs.append(("vtw (no retract)", True, True))
    elif only == "block-batch":
        runs.append(("block (head batch)", False, True))
    elif only == "block-single":
        runs.append(("block (head 1-tok)", False, False))
    else:
        raise SystemExit("PROBE_ONLY must be vtw | block-batch | block-single")

    all_same = True
    for label, nb, hb in runs:
        out, secs, acc, drafted = generate(head, tail, toks, want, W, spec=True,
                                           head_batch=hb, no_rollback=nb)
        same = out == plain
        all_same = all_same and same
        ms = secs * 1000 / len(out)
        bad = next((i for i in range(min(len(plain), len(out))) if plain[i] != out[i]), None)
        print("")
        print("%-17s: %r" % (label, "".join(piece(tail, t) for t in out)))
        print("                   一致 %-5s  %s接受率 %.1f%% (%d/%d)  %.1f ms/token (%+.0f%%)"
              % (same,
                 "" if same else "首个分歧@%d %d!=%d  " % (bad, plain[bad], out[bad]),
                 100.0 * acc / max(drafted, 1), acc, drafted,
                 ms, 100.0 * (ms - plain_ms) / plain_ms))
    return 0 if all_same else 1


if __name__ == "__main__":
    raise SystemExit(main())
