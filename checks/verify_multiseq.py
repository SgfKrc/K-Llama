"""Verify the multi-sequence batch fix at the C boundary.

The bug: `llama_batch_init(n_tokens, embd, 1)` allocated ONE seq_id slot per token, while
`qlh_fill_explicit` accepted `n_seq_id[i] > 1` and wrote `batch->n_seq_id[i] = 2`. llama.cpp
then read `seq_id[i][1]` -- uninitialised memory -- and `llama_batch_free` walked it. The
crash landed in free, which is why it looked unrelated to the model.

Fixed by: recording n_seq_max in the handle, allocating the batch with it, filling every
slot from a flat seq_ids array, and refusing a token that asks for more sequences than the
batch can hold.

This drives the C API directly because the Python layer does not expose n_seq_max.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

sys.path.insert(0, r"G:\C\PYT\qlh")

BIN = r"H:\slimarc-proto\shim\bin"
MODEL = r"F:\qlh-models\archive-q35-2b\q35-2b-head20.gguf"
N_EMBD = 2048

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        ok = False


def main() -> int:
    os.add_dll_directory(BIN)
    os.environ["PATH"] = BIN + os.pathsep + os.environ.get("PATH", "")
    lib = ctypes.CDLL(os.path.join(BIN, "qlh_keep_head.dll"))

    i32, f32 = ctypes.c_int32, ctypes.c_float
    lib.qlh_kh_load.restype = ctypes.c_void_p
    lib.qlh_kh_load.argtypes = [ctypes.c_char_p, i32, i32, i32, i32, i32, i32,
                                ctypes.POINTER(i32), ctypes.POINTER(i32),
                                ctypes.c_char_p, ctypes.c_size_t]
    lib.qlh_kh_forward_embd_seq.restype = i32
    lib.qlh_kh_forward_embd_seq.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(f32), i32, i32,
        ctypes.POINTER(i32), ctypes.POINTER(i32), ctypes.POINTER(i32),
        ctypes.POINTER(f32)]
    lib.qlh_kh_close.restype = None
    lib.qlh_kh_close.argtypes = [ctypes.c_void_p]

    err = ctypes.create_string_buffer(512)
    n_embd, n_layer = i32(), i32()

    # n_seq_max = 4 so a token may belong to several sequences.
    h = lib.qlh_kh_load(MODEL.encode(), 512, 4, 512, 4, 0, 0,   # mode 0 = nextn (no cut_layer)
                        ctypes.byref(n_embd), ctypes.byref(n_layer), err, 512)
    if not h:
        print(f"  load failed: {err.value.decode(errors='replace')}")
        return 1
    check("loaded with n_seq_max=4", n_embd.value > 0,
          f"n_embd={n_embd.value} n_layer={n_layer.value}")

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
        n_seq_id = (i32 * 2)(2, 2)
        seq_ids = (i32 * 4)(0, 1, 0, 1)         # 2 tokens x 2 sequences, flattened
        pos = (i32 * 2)(0, 1)
        rc = lib.qlh_kh_forward_embd_seq(h, buf, 2, 0, n_seq_id, seq_ids, pos, out)
        # Refused, not crashed. kv_unified is off (llama.cpp's default), and in that mode
        # llama-kv-cache.cpp asserts n_seq_id == 1 for every token; allowing it through
        # would fail later with "failed to find a memory slot", which points at nothing.
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
