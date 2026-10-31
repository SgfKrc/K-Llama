#!/usr/bin/env python3
"""Cross-machine two-stage DFlash heuristic-acceptance harness.

This is the vendor-neutral form of the lab script. Nothing here knows a machine name, an
address, a drive letter or a filesystem layout: everything about *which* stages exist and
*how* to reach them comes from the topology file, and the two optional per-stage fields
``ssh_user`` / ``start_cmd`` describe how to bring a remote stage up.

    [stage]  host = "..."     port = 9103
             ssh_user = "me"  start_cmd = "C:\\path\\to\\tail-worker.cmd"

Leave ``ssh_user`` empty when the remote worker is already listening by other means; the
harness then only connects.

    python scripts/check_p4_cross.py topology-cross-mtp.toml

Environment (all optional):
    SA_P4_BLOCK      drafts per round, default 6 (=> k=5, the measured optimum)
    SA_N_RS_SEQ      recurrent snapshot depth for the head, default SA_P4_BLOCK
    SA_HEAD_GPU      head GPU layers override
    SA_HEAD_CMOE     head layers whose experts stay on CPU, default 24
    SA_DRAFT_GPU     GPU layers for the draft model, default 99
    SA_CTX           context size, default 1024

Why the default is k=5 and not k=3: measured 2026-10-31 on the 32:40 chain, one prompt
(`def quicksort`), decode phase only, 64-token runs: k=3 gives 3.94 tokens/round at
17.0 tok/s and 98.0% acceptance, k=5 gives 5.82 tokens/round at 20.1 tok/s and 96.4%.
Longer runs (128 tokens, where the text leaves the predictable region) keep the same
ordering: k=3 12.2, k=5 13.4. Past that the acceptance collapses -- k=8 lands 31.1% and
k=15 11.1%, both slower than k=3. The extra drafts stop paying for themselves at k=6.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import chain_client as cc  # noqa: E402
from topology import Stage, load  # noqa: E402

SHIM = Path(os.environ.get("SA_SHIM", REPO / "shim" / "qlh_keep_head.dll"))
DRAFT = os.environ.get("SA_DRAFT", "")
# The draft model wants the INPUT of these eight source layers. Which stage owns each one
# depends on where the chain is cut, so they are split at runtime instead of hard-coded:
# asking a 4-layer tail segment for layer 6 would address a layer it never loaded.
TARGET_LAYERS = [2, 7, 12, 17, 23, 28, 33, 38]


def split_target_layers(head_range, tail_range):
    """Source layers -> (head-local ids, tail-local ids) for a given cut point."""
    head_lo, head_hi = head_range
    tail_lo, tail_hi = tail_range
    head_ids = [l - head_lo for l in TARGET_LAYERS if head_lo <= l < head_hi]
    tail_ids = [l - tail_lo for l in TARGET_LAYERS if tail_lo <= l < tail_hi]
    return head_ids, tail_ids
N_BLOCK = int(os.environ.get("SA_P4_BLOCK", "6"))
CTX = int(os.environ.get("SA_CTX", "1024"))
# The head needs one recurrent snapshot per position a reject-all round retracts, which is
# N_BLOCK-1. chain_client's spawn_worker derives it from SA_SPEC_K instead (that knob drives
# the MTP path), so without this the default k=5 block would try to roll back 5 positions on
# a 2-deep snapshot ring and silently lose the state.
os.environ.setdefault("SA_N_RS_SEQ", str(N_BLOCK))


def eight_layers(hc, tc, enc_w: int, head_ids, tail_ids) -> np.ndarray:
    """The eight target-layer inputs for the most recent forward, in GGUF order."""
    h = hc.layer_inp(head_ids)
    t = tc.layer_inp(tail_ids)
    if h is None or t is None:
        raise RuntimeError("layer_inp unavailable (head=%s tail=%s)"
                           % (None if h is None else h.shape, None if t is None else t.shape))
    if h.shape[1] != t.shape[1]:
        raise RuntimeError("row mismatch: head %d tail %d" % (h.shape[1], t.shape[1]))
    n = h.shape[1]
    rows = np.concatenate([h.transpose(1, 0, 2), t.transpose(1, 0, 2)],
                          axis=1).reshape(n, -1)
    if rows.shape[1] != enc_w:
        raise RuntimeError("enc width %d != %d" % (rows.shape[1], enc_w))
    return rows


def start_remote(stage: Stage):
    """Bring a remote stage up over ssh, keeping the connection open.

    `start /b` looks like the obvious way to detach, but its child dies with the ssh session,
    so the worker never appears. Holding the ssh process open is what keeps the remote python
    alive; the returned handle stops it again on the way out.
    """
    ssh = subprocess.Popen(
        ["cmd", "/c", f'ssh -o BatchMode=yes {stage.ssh_target} "{stage.start_cmd}"'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  %s: launched over ssh (pid %d), waiting for %s:%d ..."
          % (stage.kind, ssh.pid, stage.host, stage.port))
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if ssh.poll() is not None:
            raise RuntimeError("ssh to %s exited with %s before the stage accepted"
                               % (stage.ssh_target, ssh.returncode))
        try:
            socket.create_connection((stage.host, stage.port), timeout=2).close()
            print("  %s: accepted" % stage.kind)
            return ssh
        except OSError:
            time.sleep(2)
    raise RuntimeError("%s:%d did not accept within 240s" % (stage.host, stage.port))


def main() -> int:
    topo = sys.argv[1] if len(sys.argv) > 1 else "topology-cross-mtp.toml"
    prompt = ("def quicksort(arr):\n"
              "    if len(arr) <= 1:\n"
              "        return arr\n")
    n_predict = int(os.environ.get("SA_N_PREDICT", "64"))

    stages = load(topo)
    head, tail = stages[0], stages[1]
    head = Stage(**{**head.__dict__,
                    "gpu_layers": int(os.environ.get("SA_HEAD_GPU", head.layers)),
                    "cpu_moe": int(os.environ.get("SA_HEAD_CMOE", "24"))})
    print("  head %s:%d layers=%d range=%s"
          % (head.host, head.port, head.layers, head.layer_range))
    print("  tail %s:%d layers=%d range=%s%s"
          % (tail.host, tail.port, tail.layers, tail.layer_range,
             "" if not tail.ssh_target else " via " + tail.ssh_target))

    hproc = remote = None
    hc = tc = None
    try:
        if tail.ssh_target and tail.start_cmd:
            remote = start_remote(tail)
        elif not tail.is_local:
            print("  %s: connecting to an externally started worker" % tail.kind)

        hproc, _ = cc.spawn_worker(head, SHIM, CTX)
        hc = cc.StageClient(head)
        hc.connect(proc=hproc)
        toks = hc.tokenize(prompt)
        tc = cc.StageClient(tail)
        tc.connect()
        tc.tokenize("x")
        # An externally started worker serves repeatedly: the previous run leaves its KV
        # populated, and the next run writes position 0..n on top of it, which llama_decode
        # rejects (rc=-1). Clearing costs one round trip and makes repeated runs possible.
        tc.seq_rm(0)
        print("  prompt tokens = %d" % len(toks))

        head_ids, tail_ids = split_target_layers(head.layer_range, tail.layer_range)
        print("  target layers: head-local %s, tail-local %s" % (head_ids, tail_ids))
        if len(head_ids) + len(tail_ids) != len(TARGET_LAYERS):
            raise RuntimeError("target layers split %d+%d != %d"
                               % (len(head_ids), len(tail_ids), len(TARGET_LAYERS)))
        for lid in head_ids:
            hc.add_layer_inp(lid)
        for lid in tail_ids:
            tc.add_layer_inp(lid)
        if not DRAFT:
            print("  SA_DRAFT is not set; pass the draft GGUF path in it")
            return 2
        if hc.draft_open(DRAFT, CTX, 512, int(os.environ.get("SA_DRAFT_GPU", "99"))) != 0:
            print("  draft_open failed")
            return 1
        enc_w = hc.draft_n_embd_enc()

        hid = hc.forward_tokens(toks, 0)
        ver0 = tc.forward_tokens_out(hid, 0)
        n_past = len(toks)
        anchor = int(ver0[-1])

        rows: list[np.ndarray] = [eight_layers(hc, tc, enc_w, head_ids, tail_ids)]
        pending: list[np.ndarray] = []
        primed = False
        stats = {"rounds": 0, "drafted": 0, "accepted": 0, "tokens": 0}
        gen: list[int] = []
        t_gen = time.perf_counter()

        while len(gen) < n_predict:
            if not primed:
                hc.draft_reset()
                hc.draft_inject(np.concatenate(rows, axis=0))
                primed = True
            else:
                hc.draft_rollback(n_past)
                hc.draft_inject(np.concatenate(pending, axis=0))
                pending = []

            drafts = hc.draft_block(anchor, N_BLOCK)
            if not drafts:
                gen.append(anchor)
                stats["rounds"] += 1
                stats["tokens"] += 1
                break

            bh = hc.forward_tokens([anchor] + drafts, n_past)
            ver = tc.forward_tokens_out(bh, n_past)

            accept = 0
            for i, d in enumerate(drafts):
                if i < len(ver) and int(ver[i]) == int(d):
                    accept += 1
                else:
                    break

            emitted = [anchor] + drafts[:accept]
            gen.extend(emitted)
            stats["rounds"] += 1
            stats["drafted"] += len(drafts)
            stats["accepted"] += accept
            stats["tokens"] += len(emitted)

            fresh = eight_layers(hc, tc, enc_w, head_ids, tail_ids)
            take = fresh[: 1 + accept]
            rows.append(take)
            pending.append(take)

            n_past += 1 + accept
            if len(drafts) > accept:
                tc.seq_rm(n_past)
                hc.seq_rm(n_past)
            anchor = int(ver[accept]) if accept < len(ver) else int(ver[-1])

        dt = time.perf_counter() - t_gen
        text = "".join(_piece(hc, t) for t in gen[:n_predict])
        print("")
        print("  rounds    = %d" % stats["rounds"])
        print("  drafted   = %d" % stats["drafted"])
        print("  accepted  = %d (%.1f%%)"
              % (stats["accepted"], 100.0 * stats["accepted"] / max(1, stats["drafted"])))
        print("  tokens    = %d (%.2f/round)"
              % (stats["tokens"], stats["tokens"] / max(1, stats["rounds"])))
        print("  wall      = %.2fs -> %.1f tok/s" % (dt, stats["tokens"] / dt))
        print("")
        print("  sample: %r" % text[:200])
        ok = stats["accepted"] > 0
        print("")
        print("  PASS" if ok else "  FAIL: zero acceptance")
        return 0 if ok else 1
    finally:
        for c in (hc, tc):
            if c is not None:
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
        for p in (hproc, remote):
            if p is not None:
                try:
                    p.terminate()
                except Exception:  # noqa: BLE001
                    pass


def _piece(client, token: int) -> str:
    try:
        v = client.piece(int(token))
        return v.decode("latin-1") if isinstance(v, (bytes, bytearray)) else str(v)
    except Exception:  # noqa: BLE001
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
