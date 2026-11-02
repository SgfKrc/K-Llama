"""Verify the multi-sequence batch fix at the C boundary.

The bug: `llama_batch_init(n_tokens, embd, 1)` allocated ONE seq_id slot per token, while
`qlh_fill_explicit` accepted `n_seq_id[i] > 1` and wrote `batch->n_seq_id[i] = 2`. llama.cpp
then read `seq_id[i][1]` -- uninitialised memory -- and `llama_batch_free` walked it. The
crash landed in free, which is why it looked unrelated to the model.

Fixed by: recording n_seq_max in the handle, allocating the batch with it, filling every
slot from a flat seq_ids array, and refusing a token that asks for more sequences than the
batch can hold.

This drives the C API directly because the Python layer does not expose n_seq_max.

2026-11-02: off the cut artifact, onto the whole checkpoint
-----------------------------------------------------------
The check used to borrow `archive-q35-2b\q35-2b-head20.gguf` -- a head segment from the
retired cut scheme -- because all it needs is one loadable model. seq_id assignment happens
in `llama_batch_init` / `qlh_fill_explicit`; nothing about it depends on the layer count or
on how the weights were split. So it now loads the 2B checkpoint whole and selects layers
at load time:

* `qwen35-2b-Q4_K_M.gguf` has `block_count = 25` and `nextn_predict_layers = 1`, i.e. a
  24-layer trunk;
* `LLAMA_LAYER_RANGE=0:20` keeps the first 20, the same layer set the old head20 artifact
  held, so `n_layer` still reports 20 and the output stays comparable;
* `layer_hi = 20 < 24`, which stays below the nextn boundary: only a range whose last layer
  is the model's last one keeps the MTP head (`llama-model-loader.cpp:761`). This script
  does not touch MTP, and a head range does not load `output.weight` either
  (`needs_output()` is true for the tail only), so the per-row logits flag is switched off
  after load.

The old location was `H:\\slimarc-proto\\shim\\bin`: a 2026-10-30 build exporting 19 of the
34 symbols in `shim/qlh_keep_head.def` (no `last_hidden`, no `logits`, no `mtp_*`, no
`draft_*`, no `seq_rm`). It now goes through `config.find_shim()`, the one rule all three
entry points use (`--shim` > `SA_SHIM` > this checkout's build).

What this script does and does not exercise
-------------------------------------------
`kv_unified` is hardcoded 0 in the shim (`shim/qlh_keep_head_gpu.c:677`), so a token that
asks for more than one sequence is refused by `qlh_fill_explicit`'s first check
(`per_token > 1 && !kv_unified`, same file :1176) before any slot is written. The `-5` in
case 1 is therefore the shim's own refusal code, not a llama.cpp KV assertion -- the latter
would only surface deeper in `llama_decode` as "failed to find a memory slot". The
over-capacity case is refused before the write for the same reason.

So, honestly:

* covered: a multi-sequence request is refused cleanly (not a corrupted batch that crashes
  in free), an over-capacity request is refused, a single-sequence forward still works;
* not covered: the `llama_batch_init(..., n_seq_max)` slot allocation itself. Both refusal
  paths return before `seq_id[i][j]` is written, and nothing on the Python side can see the
  allocation. Covering it needs a build with `kv_unified = 1`, which this repo has no
  switch for.
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from _shimlib import load as shim_load  # noqa: E402

MODEL = Path(r"H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf")
#: Whole checkpoint, first 20 of its 24 trunk layers -- the old head20 artifact's layer set.
LAYER_RANGE = "0:20"
N_EMBD = 2048
#: n_seq_max = 4 so a token may belong to several sequences.
N_SEQ_MAX = 4

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        ok = False


def main() -> int:
    if not MODEL.is_file():
        print(f"  [SKIP] whole checkpoint missing: {MODEL}")
        return 0
    shim = config.find_shim()
    print(f"  shim   {shim}")
    print(f"  model  {MODEL}")
    print(f"  range  LLAMA_LAYER_RANGE={LAYER_RANGE}  n_seq_max={N_SEQ_MAX}")

    # Sets LLAMA_LAYER_RANGE before the DLL loads (the shim links the C runtime
    # statically, so getenv sees the snapshot from DLL-load time) and declares the ABI.
    lib = shim_load(shim, range=LAYER_RANGE)

    i32, f32 = ctypes.c_int32, ctypes.c_float

    err = ctypes.create_string_buffer(512)
    n_embd, n_layer = i32(), i32()
    h = lib.qlh_kh_load(str(MODEL).encode("utf-8"), 512, 4, 512, N_SEQ_MAX,
                        0, 0,   # mode 0 = nextn (no cut_layer)
                        ctypes.byref(n_embd), ctypes.byref(n_layer), err, 512)
    if not h:
        print(f"  load failed: {err.value.decode(errors='replace')}")
        return 1
    lo, hi = (int(v) for v in LAYER_RANGE.split(":"))
    check("loaded with n_seq_max=4", n_embd.value == N_EMBD and n_layer.value == hi - lo,
          f"n_embd={n_embd.value} n_layer={n_layer.value} (want {N_EMBD}/{hi - lo})")

    # Hidden only: without this every row is marked as an output and llama.cpp runs
    # output.weight, which a head range does not load (needs_output() is tail-only).
    lib.qlh_kh_set_skip_hidden_logits(h, 1)

    embd_dim = n_embd.value
    out = (f32 * (3 * embd_dim))()
    buf = (f32 * (3 * embd_dim))()
    for i in range(3 * embd_dim):
        buf[i] = 0.01 * (i % 7)

    try:
        # Case 1: the crash scenario -- each token claims TWO sequences, flat ids.
        # Both tokens go to sequences {0,1} at increasing positions. Sharing a sequence is
        # the point; putting two tokens of one sequence at the SAME position is not, and
        # would be refused by llama.cpp's KV cache for an unrelated reason.
        #
        # rc=-5 comes from qlh_fill_explicit (kv_unified is 0), not from llama.cpp: the
        # request is refused before any slot is written. That is the property under test --
        # a clean refusal instead of an out-of-bounds write that crashes in llama_batch_free.
        n_seq_id = (i32 * 2)(2, 2)
        seq_ids = (i32 * 4)(0, 1, 0, 1)         # 2 tokens x 2 sequences, flattened
        pos = (i32 * 2)(0, 1)
        rc = lib.qlh_kh_forward_embd_seq(h, buf, 2, 0, n_seq_id, seq_ids, pos, out)
        check("token in 2 sequences refused cleanly", rc == -5,
              f"rc={rc} (want -5; crashing or a KV error would both be wrong)")

        # Case 2: single sequence, the path every existing caller uses.
        n1 = (i32 * 3)(1, 1, 1)
        s1 = (i32 * 3)(0, 0, 0)
        p1 = (i32 * 3)(3, 4, 5)
        rc = lib.qlh_kh_forward_embd_seq(h, buf, 3, 2, n1, s1, p1, out)
        check("single sequence still works", rc == 0, f"rc={rc}")

        # Case 3: ask for more sequences than the batch was built for -> refuse, not corrupt.
        n_big = (i32 * 3)(9, 1, 1)
        s_big = (i32 * 3)(0, 0, 0)
        rc = lib.qlh_kh_forward_embd_seq(h, buf, 3, 5, n_big, s_big, p1, out)
        check("over-capacity request refused", rc == -5, f"rc={rc} (want -5)")
    finally:
        lib.qlh_kh_close(h)

    print()
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
