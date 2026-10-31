"""Independent check: same two-stage topology, but the DFlash call sequence is aligned
with common/speculative.cpp + examples/speculative-simple.cpp.

It does NOT import or modify check_p4_rebuild.py. What changes vs. that driver:

  * the draft KV is injected for positions 0..n_past-1 only -- the anchor token is the
    *first* slot of the noise block (its own position n_past), exactly like
    examples/speculative-simple.cpp:191 (`.pos0 = n_past, .id_last = id_last`);
  * the target verifies [anchor, drafts...] in ONE batch, so ver[i] is the argmax of
    position n_past+i and therefore ver[i] must equal drafts[i] (no +-1 shift);
  * the accepted rows are replayed from the verification batch's layer_inp instead of
    being dropped;
  * no second forward of the prefill's last position (that extra forward pushes the
    tail's recurrent layers one step ahead);
  * SA_SPEC_K=16 gives the tail enough recurrent snapshots to roll back a 16-token
    verification block (llama-memory-recurrent.cpp:211 refuses rollback > n_rs_seq).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, r"G:\C\PYT\qlh")
os.environ.setdefault("SA_BACKEND_DIR", r"H:\slimarc-proto\exp-c\shim")
os.environ["SA_MEM_SAFETY"] = "0.88"
os.environ["SA_CTX"] = "512"
os.environ["SA_SPEC_K"] = "16"

import numpy as np

import chain_client as cc
from topology import load

SHIM = Path(r"H:\slimarc-proto\exp-c\shim\qlh_keep_head.dll")
DRAFT = r"H:\qlh_models\dflash\dflash-merged-Q4_K_M.gguf"
HEAD_LIDS = [2, 7, 12, 17, 23, 28]
TAIL_LIDS = [1, 6]
N_BLOCK = 16


def eight_layers(hc, tc):
    """[n_tokens, 8*2048] in the GGUF target_layers order, like speculative.cpp:1239-1248."""
    h = hc.layer_inp(HEAD_LIDS)
    t = tc.layer_inp(TAIL_LIDS)
    if h is None or t is None:
        raise RuntimeError("layer_inp unavailable")
    n = h.shape[1]
    return np.concatenate([h.transpose(1, 0, 2), t.transpose(1, 0, 2)], axis=1).reshape(n, -1)


def main() -> int:
    prompt = "def quicksort(arr):"
    n_predict = 16
    max_rounds = 4

    stages = load("topology-local-32-8.toml")
    head, tail = stages[0], stages[1]
    hproc = tproc = None
    hc = tc = None
    try:
        hproc, _ = cc.spawn_worker(head, SHIM, 512)
        hc = cc.StageClient(head)
        hc.connect(proc=hproc)
        toks = hc.tokenize(prompt)

        tproc, _ = cc.spawn_worker(tail, SHIM, 512)
        tc = cc.StageClient(tail)
        tc.connect(proc=tproc)
        tc.tokenize("x")

        for lid in HEAD_LIDS:
            hc.add_layer_inp(lid)
        for lid in TAIL_LIDS:
            tc.add_layer_inp(lid)

        # n_batch must exceed the whole re-injected prefix: draft_inject builds a batch of
        # n_rows tokens and llama_decode refuses batch.n_tokens > n_batch (the original
        # driver passes 32, so once the prefix grows past 32 rows every inject fails).
        rc = hc.draft_open(DRAFT, 512, 512, 0)
        if rc != 0:
            print("draft_open rc=%d" % rc)
            return 1
        print("enc_w = %d  prompt tokens = %d" % (hc.draft_n_embd_enc(), len(toks)))

        # prefill: the target consumes all prompt tokens; the last argmax is the anchor,
        # which has NOT entered the target KV yet.
        t0 = time.perf_counter()
        hid = hc.forward_tokens(toks, 0)
        ver0 = tc.forward_tokens_out(hid, 0)
        n_past = len(toks)
        anchor = int(ver0[-1])
        rows = [eight_layers(hc, tc)]
        print("prefill %.1fs  n_past=%d  rows=%d  anchor=%d"
              % (time.perf_counter() - t0,
                 n_past, sum(r.shape[0] for r in rows), anchor))

        stats = {"rounds": 0, "drafted": 0, "accepted": 0}
        gen: list[int] = []
        t0 = time.perf_counter()

        while len(gen) < n_predict and stats["rounds"] < max_rounds:
            rc = hc.draft_reset()
            if rc != 0:
                print("draft_reset rc=%d" % rc)
                break
            rc = hc.draft_inject(np.concatenate(rows, axis=0))
            if rc != 0:
                print("draft_inject rc=%d" % rc)
                break

            drafts = hc.draft_block(anchor, N_BLOCK)
            if not drafts:
                print("no drafts")
                break

            # verify [anchor, d0..d14] at positions n_past .. n_past+15
            bh = hc.forward_tokens([anchor] + drafts, n_past)
            ver = tc.forward_tokens_out(bh, n_past)

            accept = 0
            for i in range(len(drafts)):
                if int(ver[i]) == int(drafts[i]):
                    accept += 1
                else:
                    break

            emitted = [anchor] + [int(d) for d in drafts[:accept]]
            gen.extend(emitted)
            stats["rounds"] += 1
            stats["drafted"] += len(drafts)
            stats["accepted"] += accept
            print("  round %d: accept %2d/%d  pos=%d  first draft/ver: %d/%d"
                  % (stats["rounds"], accept, len(drafts), n_past,
                     int(drafts[0]), int(ver[0])))

            # rows for the next round: everything the target has now consumed
            keep = n_past + 1 + accept
            fresh = eight_layers(hc, tc)          # [16, 16384]: anchor + 15 drafts
            rows.append(fresh[: 1 + accept])
            if keep < n_past + 1 + len(drafts):
                tc.seq_rm(keep)
                hc.seq_rm(keep)

            if accept == len(drafts):
                nxt = int(ver[len(drafts)])        # bonus token
            else:
                nxt = int(ver[accept])             # corrected token at n_past+1+accept
            n_past = keep
            anchor = nxt

        dt = time.perf_counter() - t0
        text = "".join(_piece(hc, t) for t in gen[:n_predict])
        print("")
        print("  rounds   = %d" % stats["rounds"])
        print("  drafted  = %d" % stats["drafted"])
        print("  accepted = %d (%.1f%%)"
              % (stats["accepted"],
                 100.0 * stats["accepted"] / max(1, stats["drafted"])))
        print("  wall     = %.1fs" % dt)
        print("  text     = %r" % text[:160])
        return 0
    finally:
        for c in (hc, tc):
            if c is not None:
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
        for p in (hproc, tproc):
            if p is not None:
                try:
                    p.terminate()
                except Exception:  # noqa: BLE001
                    pass


def _piece(client, token: int) -> str:
    try:
        return client.piece(int(token)).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
