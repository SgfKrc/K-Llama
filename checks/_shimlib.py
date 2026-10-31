"""Shared ctypes ABI for the one-off probes in checks/.

Every probe used to copy its own 30-95 lines of: add_dll_directory, CDLL, one argtypes
block per symbol, then head/tail wrappers. This is the single copy.

Authority for every declaration below is ``keep_head_worker.py:78-159`` -- the binding
the real worker uses. Three symbols the probes need are not declared there; their C
signatures are read from ``shim/qlh_keep_head_gpu.c``:

    qlh_kh_n_vocab   c:1615   int32_t f(void *handle)
    qlh_kh_n_embd    c:1619   int32_t f(void *handle)
    qlh_kh_n_layer   c:1623   int32_t f(void *handle)

Ownership of the process-wide setenv stays with the caller's script: the shim links the
C runtime statically, so ``load()`` sets the ``SA_*`` variables *before* the first
``qlh_kh_load`` and a second call in the same process reuses the first values. Probes that
load two stages in one process therefore pass a different ``range``/``gpu`` per call, just
like they did with their own copies.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

P32 = ctypes.POINTER(ctypes.c_int32)
PF = ctypes.POINTER(ctypes.c_float)
I32 = ctypes.c_int32
VP = ctypes.c_void_p

#: Symbols without which a probe cannot work at all (same list as keep_head_worker).
_REQUIRED = ("qlh_kh_load", "qlh_kh_forward", "qlh_kh_forward_embd_seq",
             "qlh_kh_forward_embd_token", "qlh_kh_reset", "qlh_kh_close")

#: name -> (argtypes, restype). Copied from keep_head_worker.py:78-159.
_DECL = {
    "qlh_kh_load": (
        [ctypes.c_char_p, I32, I32, I32, I32, I32, I32, P32, P32,
         ctypes.c_char_p, ctypes.c_size_t], VP),
    "qlh_kh_forward": ([VP, P32, I32, I32, PF], I32),
    "qlh_kh_forward_embd_seq": ([VP, PF, I32, I32, P32, P32, P32, PF], I32),
    "qlh_kh_forward_embd_token": ([VP, PF, I32, I32, P32, P32, P32, P32], I32),
    "qlh_kh_forward_embd_tokens": ([VP, PF, I32, I32, P32, P32, P32, P32, I32], I32),
    "qlh_kh_last_hidden": ([VP, PF, I32], I32),
    "qlh_kh_add_layer_inp": ([VP, ctypes.c_uint32], I32),
    "qlh_kh_get_layer_inp": ([VP, ctypes.c_uint32, PF, I32], I32),
    "qlh_kh_last_n_tokens": ([VP], I32),
    "qlh_kh_draft_open": (
        [VP, ctypes.c_char_p, I32, I32, I32, ctypes.c_char_p, ctypes.c_size_t], I32),
    "qlh_kh_draft_inject": ([VP, PF, I32, P32], I32),
    "qlh_kh_draft_block": ([VP, I32, I32, P32, P32, I32], I32),
    "qlh_kh_draft_rollback": ([VP, I32], I32),
    "qlh_kh_draft_reset": ([VP], I32),
    "qlh_kh_draft_close": ([VP], I32),
    "qlh_kh_draft_n_embd_enc": ([VP], I32),
    "qlh_kh_mtp_draft": ([VP, PF, I32, I32, I32, P32], I32),
    "qlh_kh_mtp_reset": ([VP], I32),
    "qlh_kh_reset": ([VP], None),
    "qlh_kh_seq_rm": ([VP, I32, I32], I32),
    "qlh_kh_close": ([VP], None),
    # Probe-only readers (C signatures cited in the module docstring).
    "qlh_kh_n_embd": ([VP], I32),
    "qlh_kh_n_layer": ([VP], I32),
    "qlh_kh_n_vocab": ([VP], I32),
}

#: Optional symbols: an older shim still serves the numeric path without them.
_OPTIONAL = {
    "qlh_kh_set_skip_hidden_logits": ([VP, I32], I32),
    "qlh_kh_last_error": ([VP], I32),
    "qlh_kh_token_to_piece": ([VP, I32, ctypes.c_char_p, I32], I32),
    "qlh_kh_tokenize": ([VP, ctypes.c_char_p, I32, P32, I32], I32),
    "qlh_kh_is_eog": ([VP, I32], I32),
}

_LIB = None


def load(shim_path, *, range=False, logits=False, mtp=False, gpu=0, rs_seq="4"):
    """Set the per-stage ``SA_*`` variables, load the shim, declare its ABI.

    ``shim_path`` may be a Latin-free shortcut: the value is only used to find and open
    the DLL. Returns the ``ctypes.CDLL``; repeated calls with the same path reuse it.

    ``rs_seq`` keeps each probe's original ``SA_N_RS_SEQ``: most set it to "4" (or
    ``PROBE_RS_SEQ``); probe_perfect_draft never set it, and an unset value is 0 in the
    shim, so it passes ``rs_seq=0``.
    """
    if range:
        os.environ["LLAMA_LAYER_RANGE"] = str(range)
    os.environ["SA_NEEDS_LOGITS"] = "1" if logits else "0"
    os.environ["SA_MTP"] = "1" if mtp else "0"
    os.environ["SA_N_GPU_LAYERS"] = str(int(gpu))
    os.environ["SA_N_RS_SEQ"] = str(int(os.environ.get("PROBE_RS_SEQ", rs_seq)))

    global _LIB
    if _LIB is None:
        path = Path(shim_path)
        try:
            os.add_dll_directory(str(path.resolve().parent))
        except (AttributeError, OSError):
            pass
        lib = ctypes.CDLL(str(path))
        missing = [name for name in _REQUIRED if not hasattr(lib, name)]
        if missing:
            raise RuntimeError("%s is missing symbols %s" % (path, missing))
        for name, (argt, rest) in _DECL.items():
            fn = getattr(lib, name)
            fn.argtypes = argt
            fn.restype = rest
        for name, (argt, rest) in _OPTIONAL.items():
            if hasattr(lib, name):
                fn = getattr(lib, name)
                fn.argtypes = argt
                fn.restype = rest
        _LIB = lib
    return _LIB


def _lib():
    if _LIB is None:
        raise RuntimeError("checks._shimlib.load() must be called before the wrappers")
    return _LIB


def hw(handle, lib=None):
    """n_embd / n_layer of a loaded handle."""
    lib = lib or _lib()
    return int(lib.qlh_kh_n_embd(handle)), int(lib.qlh_kh_n_layer(handle))


def head_step(head, tokens, n_past, W, lib=None):
    """Write tokens at n_past on the head stage; return (len(tokens), W) float32."""
    lib = lib or _lib()
    out = np.zeros((len(tokens), W), dtype=np.float32)
    rc = lib.qlh_kh_forward(head, (ctypes.c_int32 * len(tokens))(*tokens), len(tokens),
                            n_past, out.ctypes.data_as(PF))
    if rc != 0:
        raise RuntimeError("head rc=%d at %d" % (rc, n_past))
    return out


def tail_one(tail, hidden, n_past, W, lib=None):
    """Feed one hidden row to the tail stage; return the produced token id."""
    lib = lib or _lib()
    t = ctypes.c_int32(-1)
    rc = lib.qlh_kh_forward_embd_token(tail, hidden.ctypes.data_as(PF), 1, n_past,
                                       None, None, None, ctypes.byref(t))
    if rc != 0:
        raise RuntimeError("tail rc=%d at %d" % (rc, n_past))
    return t.value


def tail_batch(tail, rows, n_past, W, lib=None):
    """Feed k hidden rows to the tail stage; return k token ids."""
    lib = lib or _lib()
    k = rows.shape[0]
    out = (ctypes.c_int32 * k)()
    rc = lib.qlh_kh_forward_embd_tokens(tail, rows.ctypes.data_as(PF), k, n_past,
                                        None, None, None, out, k)
    if rc != 0:
        raise RuntimeError("tail batch rc=%d at %d" % (rc, n_past))
    return list(out)
