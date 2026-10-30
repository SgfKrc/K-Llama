"""The dtype the hidden state travels in between stages: one knob, one place.

Everything inside a stage stays float32 -- the shim's ABI, the keep_head_worker pipe,
the kernels. The only lossy point in the numeric path is the socket hop, so the wire
dtype is a switch rather than a constant:

    SA_WIRE_DTYPE=float32   (default) byte-for-byte the pre-float16 behaviour
    SA_WIRE_DTYPE=float16   half the bytes; quantizes hidden to fp16 on send
    SA_WIRE_DTYPE=int16     half the bytes plus a 4-byte scale; fixed point

Every FORWARD frame carries the dtype in its meta (``"dtype": "float16"``), so the two
ends do not need to be configured alike: a worker decodes what the client declared and
answers in the same dtype. A frame without ``dtype`` is float32, which is what makes
the switch safe to roll out one side at a time.

Why both quantized dtypes are opt-in rather than the default: the hidden is the
layer-segment relay's numeric path. Measured on the 2B chain (checks/verify_wire.py,
5 greedy prompts x 48 tokens): float16 (max|diff| 3.9e-3) flipped a token on 1 of 5,
and int16 fixed point (max|diff| 2.2e-4, 20x finer) also flipped one on 1 of 5 -- a
different prompt. Both flips landed where the next token was near a tie, so "most
prompts agree" is not the test. Numbers and method: docs/MEASUREMENTS.md §9. Keep
float32 unless a different output is acceptable.
"""
from __future__ import annotations

import os
import struct

import numpy as np

ENV = "SA_WIRE_DTYPE"
DEFAULT = "float32"
DTYPES = {"float32": np.dtype(np.float32), "float16": np.dtype(np.float16),
          "int16": np.dtype(np.int16)}
_ALIASES = {"f32": "float32", "f16": "float16", "i16": "int16"}
# int16 is fixed point, so it needs the scale it was divided by. It rides in the first
# 4 bytes of the payload rather than in the frame meta: the blob stays "the bytes the
# receiver turns back into numbers", and no caller has to thread a second field through.
_PREFIX = struct.Struct("<f")


def resolve(name: str | None = None) -> str:
    """The canonical dtype name, from an explicit value or the environment."""
    raw = (name if name else os.environ.get(ENV) or DEFAULT).strip().lower()
    raw = _ALIASES.get(raw, raw)
    if raw not in DTYPES:
        raise ValueError(f"transport dtype must be one of {sorted(DTYPES)} "
                         f"(or f32/f16/i16), got {raw!r}")
    return raw


def current_dtype() -> str:
    """What this process would put on the wire. Read once per forward, not cached:
    the environment is the documented switch, so it must take effect without a restart."""
    return resolve()


def dtype_of(name: str | None) -> np.dtype:
    """numpy dtype for a wire name, with float32 as the missing-value default."""
    return DTYPES[resolve(name)]


def item_size(name: str | None) -> int:
    return dtype_of(name).itemsize


def payload_size(shape: tuple[int, int], name: str | None) -> int:
    """Bytes one hidden occupies on the wire, including the int16 scale prefix."""
    n = int(shape[0]) * int(shape[1])
    return n * item_size(name) + (_PREFIX.size if resolve(name) == "int16" else 0)


def encode(array, name: str | None = None) -> memoryview:
    """A contiguous buffer in the wire dtype, without a copy when already so.

    The result is a byte-shaped memoryview, not bytes: ``len()`` must be a byte count
    (an (n, 2048) array's memoryview reports n), and protocol.encode joins it into the
    frame instead of materializing a second intermediate. cast("B") is free.

    The explicit dtype/contiguity check exists so the float32 fast path never enters
    numpy: the array already has the right type and layout, and calling
    ascontiguousarray anyway would cost more Python than the 8 KB memcpy it replaces.

    int16 is per-tensor symmetric fixed point: scale = max|x| / 32767, carried in the
    payload's first four bytes. Its step is 20-30x finer than float16's ULP on a hidden
    with these norms (measured: max|diff| 2.2e-4 vs 3.9e-3) -- and that was still
    enough to flip one prompt's tokens, so it exists as a measured alternative, not as
    a recommendation.
    """
    dtype = DTYPES[name] if name in DTYPES else dtype_of(name)
    if resolve(name) == "int16":
        arr = np.ascontiguousarray(array, dtype=np.float32).ravel()
        peak = float(np.abs(arr).max()) if arr.size else 0.0
        scale = peak / 32767.0 if peak else 1.0
        q = np.clip(np.rint(arr / scale), -32767, 32767).astype(np.int16)
        buf = bytearray(_PREFIX.size + q.nbytes)
        _PREFIX.pack_into(buf, 0, scale)
        memoryview(buf)[_PREFIX.size:] = memoryview(q).cast("B")
        return memoryview(buf)
    if not isinstance(array, np.ndarray) or array.dtype != dtype or not array.flags.c_contiguous:
        array = np.ascontiguousarray(array, dtype=dtype)
    return memoryview(array).cast("B")


def decode(buf, name: str | None, shape: tuple[int, int]) -> np.ndarray:
    """View the payload as the declared dtype. No copy; converts later, once."""
    count = int(shape[0]) * int(shape[1])
    if resolve(name) == "int16":
        (scale,) = _PREFIX.unpack_from(buf, 0)
        q = np.frombuffer(buf, dtype=np.int16, count=count, offset=_PREFIX.size)
        return q.reshape(shape).astype(np.float32) * np.float32(scale)
    return np.frombuffer(buf, dtype=dtype_of(name), count=count).reshape(shape)
