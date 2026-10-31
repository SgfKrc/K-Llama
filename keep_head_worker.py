"""Worker process for keep_head: owns the shim ABI and answers one request per line.

A separate process is forced by the shim, not chosen for tidiness — see the
module docstring of keep_head.py. Two facts measured on this machine:

* the shim links the C runtime statically, so ``getenv`` sees the environment
  snapshot from DLL-load time and never refreshes; a second ``qlh_kh_load`` in the
  same process reuses the first ``SA_N_GPU_LAYERS``;
* it registers ggml backends once per process behind a ``static`` flag, reading
  ``SA_BACKEND_DIR`` at that moment.

Both values are per-stage, and the GPU stage and CPU stage run side by side. Only a
fresh process (with ``Popen(env=...)``) can differ.

Requests are JSON per line on stdin; responses are JSON per line on stdout. Hidden
states travel base64-encoded float32 so the pipe stays text.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import json
import os
from pathlib import Path
import sys

import numpy as np

MODE_CODES = {"nextn": 0, "layer_inp": 1}

#: Native return codes -> what actually went wrong (see qlh_keep_head.c).
FORWARD_ERRORS = {
    -1: "invalid arguments (handle / tokens / output buffer)",
    -2: "llama_decode failed",
    -3: "llama_get_embeddings_layer_inp returned nothing (layer slot not exported)",
    -5: "invalid arguments (handle / embd / output buffer)",
    -6: "tail logits or vocabulary unavailable",
    -7: "tail logits contain NaN or Inf",
}

_REQUIRED = ("qlh_kh_load", "qlh_kh_forward", "qlh_kh_forward_embd_seq",
             "qlh_kh_forward_embd_token", "qlh_kh_reset", "qlh_kh_close")


class KeepHeadUnavailable(RuntimeError):
    pass


def _f32(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class Binding:
    """Direct ctypes binding to the shim. Never leaves this process.

    Every ``argtypes`` below is explicit: ctypes defaults to ``c_int``, which
    truncates the 64-bit model handle (observed as ``OverflowError``).
    """

    def __init__(self, shim: Path, model: Path, mode: str, cut_layer: int | None,
                 n_ctx: int, n_threads: int, n_batch: int) -> None:
        try:
            self._dll_dir = os.add_dll_directory(str(shim.resolve().parent))
        except (AttributeError, OSError):
            self._dll_dir = None
        # SA_BACKEND_DIR is consumed while the DLL initialises, so it had to arrive
        # through Popen(env=...); setting it here would be too late.
        try:
            lib = ctypes.CDLL(str(shim))
        except OSError as exc:
            raise KeepHeadUnavailable(
                f"loading {shim} failed: {exc}; missing runtime DLLs?") from exc
        missing = [name for name in _REQUIRED if not hasattr(lib, name)]
        if missing:
            raise KeepHeadUnavailable(f"{shim} is missing symbols {missing}")
        self._lib = lib
        lib.qlh_kh_load.argtypes = [
            ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.c_char_p, ctypes.c_size_t]
        lib.qlh_kh_load.restype = ctypes.c_void_p
        lib.qlh_kh_forward.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        lib.qlh_kh_forward.restype = ctypes.c_int32
        lib.qlh_kh_forward_embd_seq.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float)]
        lib.qlh_kh_forward_embd_seq.restype = ctypes.c_int32
        lib.qlh_kh_forward_embd_token.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32)]
        lib.qlh_kh_forward_embd_token.restype = ctypes.c_int32
        lib.qlh_kh_forward_embd_tokens.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
        lib.qlh_kh_forward_embd_tokens.restype = ctypes.c_int32
        lib.qlh_kh_last_hidden.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
        lib.qlh_kh_last_hidden.restype = ctypes.c_int32
        # DFLASH-01: multi-slot intermediate hidden. Declaring these matters -- an undeclared
        # function makes ctypes default to 32-bit ints and the 64-bit handle overflows, which
        # killed the worker silently the first time it happened (qlh_kh_seq_rm).
        lib.qlh_kh_add_layer_inp.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.qlh_kh_add_layer_inp.restype = ctypes.c_int32
        lib.qlh_kh_get_layer_inp.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
        lib.qlh_kh_get_layer_inp.restype = ctypes.c_int32
        lib.qlh_kh_last_n_tokens.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_last_n_tokens.restype = ctypes.c_int32
        # DFLASH-01: the draft trio. Same reason as above -- an undeclared function makes
        # ctypes truncate the 64-bit handle to int and the worker dies without a traceback.
        lib.qlh_kh_draft_open.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_char_p, ctypes.c_size_t]
        lib.qlh_kh_draft_open.restype = ctypes.c_int32
        lib.qlh_kh_draft_inject.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32)]
        lib.qlh_kh_draft_inject.restype = ctypes.c_int32
        lib.qlh_kh_draft_block.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
        lib.qlh_kh_draft_block.restype = ctypes.c_int32
        lib.qlh_kh_draft_rollback.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.qlh_kh_draft_rollback.restype = ctypes.c_int32
        lib.qlh_kh_draft_close.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_draft_close.restype = ctypes.c_int32
        lib.qlh_kh_draft_n_embd_enc.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_draft_n_embd_enc.restype = ctypes.c_int32
        lib.qlh_kh_mtp_draft.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
            ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
        lib.qlh_kh_mtp_draft.restype = ctypes.c_int32
        lib.qlh_kh_mtp_reset.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_mtp_reset.restype = ctypes.c_int32
        lib.qlh_kh_reset.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_reset.restype = None
        # Rollback. Declaring this is not optional: without argtypes ctypes assumes every
        # parameter is a 32-bit C int, so the 64-bit model handle is converted through
        # c_int and raises "ArgumentError: argument 1: OverflowError: int too long to
        # convert" whenever the pointer does not fit in 31 bits -- which is the normal
        # case. That exception escaped the request loop, killed the worker, and surfaced
        # higher up as "keep-head worker exited before responding" on the *next* forward,
        # which is why the crash looked like a bad forward call rather than a bad seq_rm.
        lib.qlh_kh_seq_rm.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
        lib.qlh_kh_seq_rm.restype = ctypes.c_int32
        lib.qlh_kh_close.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_close.restype = None
        if hasattr(lib, "qlh_kh_set_skip_hidden_logits"):
            lib.qlh_kh_set_skip_hidden_logits.argtypes = [ctypes.c_void_p, ctypes.c_int32]
            lib.qlh_kh_set_skip_hidden_logits.restype = ctypes.c_int32
        if hasattr(lib, "qlh_kh_last_error"):
            lib.qlh_kh_last_error.argtypes = [ctypes.c_void_p]
            lib.qlh_kh_last_error.restype = ctypes.c_int32
        # Text I/O. Declared defensively: an older shim without these still works for
        # the numeric path, and the methods below raise a clear error if used.
        if hasattr(lib, "qlh_kh_token_to_piece"):
            lib.qlh_kh_token_to_piece.argtypes = [
                ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
            lib.qlh_kh_token_to_piece.restype = ctypes.c_int32
        if hasattr(lib, "qlh_kh_tokenize"):
            lib.qlh_kh_tokenize.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
            lib.qlh_kh_tokenize.restype = ctypes.c_int32
        if hasattr(lib, "qlh_kh_is_eog"):
            lib.qlh_kh_is_eog.argtypes = [ctypes.c_void_p, ctypes.c_int32]
            lib.qlh_kh_is_eog.restype = ctypes.c_int32
        # Sampling. Optional like the text helpers: a shim rebuilt from an older
        # source still serves the argmax path, and Binding.set_sampling raises a
        # clear error instead of silently leaving the caller on argmax.
        if hasattr(lib, "qlh_kh_set_sampling"):
            lib.qlh_kh_set_sampling.argtypes = [
                ctypes.c_void_p,
                ctypes.c_float,                                  # temperature
                ctypes.c_float,                                  # top_p
                ctypes.c_int32,                                  # top_k
                ctypes.c_float,                                  # repeat_penalty
                ctypes.c_int32,                                  # repeat_last_n
                ctypes.c_float,                                  # penalty_freq
                ctypes.c_float,                                  # penalty_present
                ctypes.c_uint32]                                 # seed
            lib.qlh_kh_set_sampling.restype = ctypes.c_int32
        if hasattr(lib, "qlh_kh_get_sampling"):
            lib.qlh_kh_get_sampling.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32)]
            lib.qlh_kh_get_sampling.restype = ctypes.c_int32

        n_embd, n_layer = ctypes.c_int32(0), ctypes.c_int32(0)
        err = ctypes.create_string_buffer(512)
        self._handle = lib.qlh_kh_load(
            str(model).encode("utf-8"), int(n_ctx), int(n_threads), int(n_batch),
            1, MODE_CODES[mode], int(cut_layer or 0),
            ctypes.byref(n_embd), ctypes.byref(n_layer), err, len(err))
        if not self._handle:
            raise KeepHeadUnavailable(
                f"keep-head init failed: {err.value.decode('utf-8', 'replace') or 'unknown'}")
        self.n_embd, self.n_layer = int(n_embd.value), int(n_layer.value)
        if self.n_embd <= 0:
            self.close()
            raise KeepHeadUnavailable("shim reported n_embd <= 0")
        # Stages whose job is to hand back hidden states have no use for logits: the
        # shim would otherwise run output.weight for all 248,320 vocabulary rows on
        # every single decode step and throw the result away. Measured on stage0
        # (8 layers, all on GPU): 32.0 ms/step with logits, 6.4 ms without.
        # Only the tail stage needs them, because that is where a token is chosen.
        #
        # SA_NEEDS_LOGITS arrives through the environment because that is how engine.py
        # already conveys per-stage load settings (the shim reads its own copy at load
        # time, so anything set later would be invisible to it).
        if os.environ.get("SA_NEEDS_LOGITS", "0") != "1":
            self.set_skip_hidden_logits(True)

    def set_skip_hidden_logits(self, skip: bool) -> None:
        """Tell the shim whether this stage needs logits."""
        fn = getattr(self._lib, "qlh_kh_set_skip_hidden_logits", None)
        if fn is None:
            return          # older shim: keep the default, just slower
        rc = fn(self._handle, 1 if skip else 0)
        if rc:
            raise KeepHeadUnavailable(f"set_skip_hidden_logits({skip}) failed: {rc}")

    def _checked(self, rc: int, result, what: str):
        """Raise on a native error code, else hand back ``result``."""
        if not rc:
            return result
        detail = FORWARD_ERRORS.get(rc, "unknown error code")
        if hasattr(self._lib, "qlh_kh_last_error"):
            # The raw llama_decode code is the only thing that separates a KV/ctx
            # problem from a malformed batch; -2 alone cannot.
            code = int(self._lib.qlh_kh_last_error(self._handle))
            detail += f" (llama_decode rc={code})" if code else ""
        raise KeepHeadUnavailable(f"{what} failed rc={rc}: {detail}")

    def tokens(self, tokens: list[int], n_past: int) -> np.ndarray:
        out = np.zeros((len(tokens), self.n_embd), dtype=np.float32)
        rc = self._lib.qlh_kh_forward(
            self._handle, (ctypes.c_int32 * len(tokens))(*tokens), len(tokens),
            int(n_past), _f32(out))
        return self._checked(rc, out, "forward")

    def hidden(self, arr: np.ndarray, n_past: int) -> np.ndarray:
        out = np.zeros((arr.shape[0], self.n_embd), dtype=np.float32)
        rc = self._lib.qlh_kh_forward_embd_seq(
            self._handle, _f32(arr), int(arr.shape[0]), int(n_past), None, None, None,
            _f32(out))
        return self._checked(rc, out, "embd forward")

    def token(self, arr: np.ndarray, n_past: int) -> int:
        token = ctypes.c_int32(-1)
        rc = self._lib.qlh_kh_forward_embd_token(
            self._handle, _f32(arr), int(arr.shape[0]), int(n_past), None, None, None,
            ctypes.byref(token))
        return self._checked(rc, int(token.value), "tail forward")

    def tokens_of(self, arr: np.ndarray, n_past: int) -> list[int]:
        """k rows of hidden -> k argmax token ids in one backend call.

        This is the verification primitive for speculative decoding: the candidate tokens are
        submitted as one batch and the backend returns one argmax per position, so verifying k
        candidates costs one round trip rather than k.
        """
        k = int(arr.shape[0])
        if k <= 0:
            return []
        out = (ctypes.c_int32 * k)()
        rc = self._lib.qlh_kh_forward_embd_tokens(
            self._handle, _f32(arr), k, int(n_past), None, None, None, out, k)
        self._checked(rc, None, "tail forward (batch)")
        return list(out)

    def add_layer_inp(self, lid: int) -> int:
        """Open one extra intermediate-hidden slot. 0 = ok, negative = shim error."""
        return int(self._lib.qlh_kh_add_layer_inp(self._handle, int(lid)))

    def layer_inp(self, lids: list[int]) -> np.ndarray | None:
        """All requested slots from the most recent forward, as [n_layers, n_tokens, n_embd].

        Returns None if any slot is unavailable -- a partial result would be worse than none,
        because the caller would silently feed zeros into the draft model.
        """
        if not lids:
            return None
        n_tok = int(self._lib.qlh_kh_last_n_tokens(self._handle))
        if n_tok <= 0 or self.n_embd <= 0:
            print(f"[kh.layer_inp] n_tokens={n_tok} n_embd={self.n_embd}",
                  file=sys.stderr, flush=True)
            return None
        out = np.zeros((len(lids), n_tok, self.n_embd), dtype=np.float32)
        for i, lid in enumerate(lids):
            rc = self._lib.qlh_kh_get_layer_inp(
                self._handle, int(lid), _f32(out[i]), int(out[i].size))
            if rc != 0:
                print(f"[kh.layer_inp] lid={lid} rc={rc}", file=sys.stderr, flush=True)
                return None
        return out

    def draft_open(self, path: str, n_ctx: int, n_batch: int, n_gpu_layers: int) -> int:
        err = ctypes.create_string_buffer(512)
        rc = int(self._lib.qlh_kh_draft_open(
            self._handle, path.encode("utf-8"), int(n_ctx), int(n_batch),
            int(n_gpu_layers), err, len(err)))
        if rc != 0:
            print(f"[kh.draft_open] rc={rc} {err.value.decode(errors='replace')}",
                  file=sys.stderr, flush=True)
        return rc

    def draft_n_embd_enc(self) -> int:
        return int(self._lib.qlh_kh_draft_n_embd_enc(self._handle))

    def draft_inject(self, rows: np.ndarray) -> int:
        arr = np.ascontiguousarray(rows, dtype=np.float32)
        return int(self._lib.qlh_kh_draft_inject(
            self._handle, _f32(arr), int(arr.shape[0]), None))

    def draft_block(self, anchor: int, n_block: int) -> list[int]:
        out = (ctypes.c_int32 * max(1, n_block))()
        n = int(self._lib.qlh_kh_draft_block(
            self._handle, int(anchor), int(n_block), None, out, n_block))
        if n < 0:
            print(f"[kh.draft_block] rc={n}", file=sys.stderr, flush=True)
            return []
        return [int(out[i]) for i in range(n)]

    def draft_rollback(self, n_keep: int) -> int:
        return int(self._lib.qlh_kh_draft_rollback(self._handle, int(n_keep)))

    def draft_close(self) -> int:
        return int(self._lib.qlh_kh_draft_close(self._handle))

    def last_hidden(self) -> np.ndarray | None:
        """The trunk's final hidden of the most recent decode (needs SA_MTP=1).

        This is what the MTP head was trained against -- the tail's t_h_nextn, not the 32-layer
        head output. Returns None when the model carries no NextN head.
        """
        buf = np.zeros(self.n_embd, dtype=np.float32)
        rc = self._lib.qlh_kh_last_hidden(self._handle, _f32(buf), self.n_embd)
        if rc != 0:
            # -5 bad handle/ctx/capacity, -6 llama_get_embeddings_nextn() == NULL.
            # Returning a bare None erased that distinction and cost a long detour.
            print(f"[kh.last_hidden] rc={rc} n_embd={self.n_embd}", file=sys.stderr, flush=True)
            return None
        return buf

    def mtp_draft(self, hidden: np.ndarray, token: int, n_past: int) -> int:
        """One draft token from the MTP head, given (hidden, token)."""
        out = ctypes.c_int32(-1)
        rc = self._lib.qlh_kh_mtp_draft(self._handle, _f32(hidden), int(hidden.shape[-1]),
                                        int(token), int(n_past), ctypes.byref(out))
        return self._checked(rc, int(out.value), "mtp draft")

    def mtp_reset(self) -> None:
        """Clear the draft context's memory. Positions accumulate in mtp_ctx, so recursive
        drafting has to start each candidate chain from a clean slate."""
        self._lib.qlh_kh_mtp_reset(self._handle)

    def seq_rm(self, n_past: int) -> None:
        """Drop positions >= n_past for sequence 0. Undoes a rejected draft block."""
        rc = self._lib.qlh_kh_seq_rm(self._handle, 0, int(n_past))
        if os.environ.get("SA_DEBUG_SEQRM"):
            print(f"[seq_rm] p0={n_past} rc={rc}", file=sys.stderr, flush=True)
        # -6 means llama_memory_seq_rm found nothing to remove in that range: the block never
        # reached that position, or an earlier rollback already covered it. That is a no-op,
        # not a failure -- counting it made a healthy run report rollback errors.
        if rc not in (0, -6):
            raise RuntimeError(f"seq_rm({n_past}) rc={rc} (SA_N_RS_SEQ>0 required)")

    def tokenize(self, text: bytes) -> list[int]:
        """Text -> token ids.

        llama_tokenize signals "buffer too small" by returning the negated required
        size, not by failing, so the probe call always comes back negative and the
        absolute value is the length to allocate.
        """
        if not hasattr(self._lib, "qlh_kh_tokenize"):
            raise RuntimeError("shim has no qlh_kh_tokenize (rebuild it from shim/)")
        probe = self._lib.qlh_kh_tokenize(self._handle, text, len(text), None, 0)
        need = -probe if probe < 0 else probe
        if need == 0:
            return []
        buf = (ctypes.c_int32 * need)()
        got = self._lib.qlh_kh_tokenize(self._handle, text, len(text), buf, need)
        if got < 0:
            # Second call should fit now; a negative result means the tokenizer is
            # non-deterministic in size, which would be a real bug.
            raise RuntimeError(f"tokenize still short after sizing: {got}")
        return list(buf[:got])

    def piece(self, token: int) -> bytes:
        """Token id -> UTF-8 bytes. May be a partial character; the caller has to
        reassemble across tokens (see the SSE/TUI consumers)."""
        if not hasattr(self._lib, "qlh_kh_token_to_piece"):
            raise RuntimeError("shim has no qlh_kh_token_to_piece (rebuild it)")
        need = self._lib.qlh_kh_token_to_piece(self._handle, int(token), None, 0)
        if need <= 0:
            return b""
        buf = ctypes.create_string_buffer(need)
        got = self._lib.qlh_kh_token_to_piece(self._handle, int(token), buf, need)
        return buf.raw[:got] if got > 0 else b""

    def is_eog(self, token: int) -> bool:
        if not hasattr(self._lib, "qlh_kh_is_eog"):
            return False
        return bool(self._lib.qlh_kh_is_eog(self._handle, int(token)))

    def set_sampling(self, *, temperature: float, top_p: float, top_k: int,
                     repeat_penalty: float, repeat_last_n: int,
                     penalty_freq: float, penalty_present: float,
                     seed: int) -> dict:
        """Install (or replace) this handle's sampler chain.

        ``temperature <= 0`` selects greedy, which is bit-for-bit the argmax this
        shim used before sampling existed -- it is NOT "leave the sampler unset".
        The distinction matters: an unset sampler is the untouched default path that
        every existing throughput number was measured on, while greedy routes the
        same choice through llama.cpp's sampler, so a caller asking for temperature 0
        also exercises the sampler code. Both agree exactly (measured: 0 mismatches
        over 2000 randomised vectors, ties included).
        """
        if not hasattr(self._lib, "qlh_kh_set_sampling"):
            raise RuntimeError(
                "shim has no qlh_kh_set_sampling (rebuild it from shim/)")
        rc = self._lib.qlh_kh_set_sampling(
            self._handle, float(temperature), float(top_p), int(top_k),
            float(repeat_penalty), int(repeat_last_n),
            float(penalty_freq), float(penalty_present), int(seed) & 0xFFFFFFFF)
        if rc != 0:
            # -1 bad arguments (NULL handle / NaN / Inf), -2 sampler allocation failed.
            detail = {0: "ok", -1: "invalid arguments",
                      -2: "sampler allocation failed"}.get(rc, "unknown error code")
            raise KeepHeadUnavailable(f"set_sampling failed rc={rc}: {detail}")
        return self.sampling()

    def clear_sampling(self) -> dict:
        """Drop the sampler chain and go back to the untouched argmax default.

        Distinct from ``set_sampling(temperature=0)`` on purpose: that installs a
        greedy chain, which produces the same tokens but is still a sampler (and would
        be reported as ``mode=greedy``). A long-lived worker must be able to return to
        the pristine state, otherwise a request that asks for sampling leaves the
        handle configured for every request after it.
        """
        if hasattr(self._lib, "qlh_kh_set_sampling"):
            # -1.0 is the shim's documented "disable" sentinel for temperature.
            rc = self._lib.qlh_kh_set_sampling(
                self._handle, ctypes.c_float(-1.0), ctypes.c_float(1.0), 0,
                ctypes.c_float(1.0), 0, ctypes.c_float(0.0), ctypes.c_float(0.0), 0)
            if rc != 0:
                raise KeepHeadUnavailable(f"clear_sampling failed rc={rc}")
        return self.sampling()

    def sampling(self) -> dict:
        """The parameters actually in effect, as the native side reports them.

        Read back instead of echoing the request: the clamping and the
        "value means off" rules (``top_k >= n_vocab``, ``top_p >= 1``, a
        ``repeat_last_n`` filled in with a default) all live in the shim, and
        recomputing them here would be a second copy of the same rules to drift.
        """
        if not hasattr(self._lib, "qlh_kh_get_sampling"):
            return {}
        temperature, top_p = ctypes.c_float(0), ctypes.c_float(1)
        top_k, repeat_last_n = ctypes.c_int32(0), ctypes.c_int32(0)
        repeat_penalty, penalty_freq = ctypes.c_float(1), ctypes.c_float(0)
        penalty_present, seed = ctypes.c_float(0), ctypes.c_uint32(0)
        active = self._lib.qlh_kh_get_sampling(
            self._handle,
            ctypes.byref(temperature), ctypes.byref(top_p), ctypes.byref(top_k),
            ctypes.byref(repeat_penalty), ctypes.byref(repeat_last_n),
            ctypes.byref(penalty_freq), ctypes.byref(penalty_present),
            ctypes.byref(seed))
        return {
            "mode": "greedy" if (active and temperature.value <= 0) else
                    ("sampling" if active else "argmax"),
            "temperature": round(temperature.value, 6),
            "top_p": round(top_p.value, 6),
            "top_k": int(top_k.value),
            "repeat_penalty": round(repeat_penalty.value, 6),
            "repeat_last_n": int(repeat_last_n.value),
            "frequency_penalty": round(penalty_freq.value, 6),
            "presence_penalty": round(penalty_present.value, 6),
            "seed": int(seed.value),
        }

    def reset(self) -> None:
        self._lib.qlh_kh_reset(self._handle)

    def close(self) -> None:
        handle, self._handle = getattr(self, "_handle", None), None
        if handle is not None and getattr(self, "_lib", None) is not None:
            self._lib.qlh_kh_close(handle)
        if getattr(self, "_dll_dir", None) is not None:
            try:
                self._dll_dir.close()
            except OSError:
                pass
            self._dll_dir = None


def _trace(message: str) -> None:
    """Diagnostic breadcrumb on stderr, off unless SA_KH_TRACE is set.

    stderr is inherited by the stage worker, so these lines land in the stage log next
    to llama.cpp's own output and survive a worker that dies without a response. That is
    the only way to tell "the parent closed my stdin" apart from "a native call killed
    the process": both leave the parent reading EOF.
    """
    if os.environ.get("SA_KH_TRACE"):
        print(f"[kh {os.getpid()}] {message}", file=sys.stderr, flush=True)


def _response(**payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=True, separators=(",", ":")), flush=True)


def _array_response(array: np.ndarray) -> None:
    # b64encode reads the array's buffer directly; the old .tobytes() built an 8 KB
    # intermediate per step that base64 then copied again.
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    _response(ok=True, shape=list(contiguous.shape),
              data=base64.b64encode(contiguous).decode("ascii"))


def _decode_hidden(request: dict) -> np.ndarray:
    shape = tuple(int(value) for value in request["shape"])
    return np.frombuffer(
        base64.b64decode(str(request["data"])), dtype=np.float32).reshape(shape)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shim", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("nextn", "layer_inp"), default="nextn")
    parser.add_argument("--cut-layer", type=int, default=None)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-batch", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = _args()
    try:
        binding = Binding(Path(args.shim), Path(args.model), args.mode, args.cut_layer,
                          args.n_ctx, args.n_threads, args.n_batch)
    except Exception as exc:  # noqa: BLE001 - the parent needs the text, not a traceback
        _response(ok=False, error=f"{type(exc).__name__}: {exc}")
        return 1

    _response(ok=True, n_embd=binding.n_embd, n_layer=binding.n_layer)
    _trace(f"ready n_embd={binding.n_embd} n_layer={binding.n_layer}")
    try:
        for raw in sys.stdin:
            if not raw.strip():
                continue
            request = json.loads(raw)
            operation = request.get("op")
            n_past = int(request.get("n_past", 0))
            _trace(f"req op={operation} n_past={n_past} "
                   f"n_tokens={len(request.get('tokens') or []) or request.get('shape')}")
            if operation == "close":
                _response(ok=True)
                _trace("close requested by parent")
                return 0
            if operation == "reset":
                binding.reset()
                _response(ok=True)
            elif operation == "tokens":
                _array_response(binding.tokens(request.get("tokens", []), n_past))
            elif operation == "hidden_trace":
                print(f"[kh.hidden_trace] shape={request.get('shape')} "
                      f"blob={len(binding._pending_blob) if hasattr(binding,'_pending_blob') else '?'}",
                      file=sys.stderr, flush=True)
                _response(ok=False, error="trace only")
            elif operation == "hidden":
                # 这一段的 dense hidden 可能拿不到（tail 段为了出 token 必须开 logits，
                # 而 cparams.embeddings = !needs_logits）。这不是致命错误：前向已经发生，
                # 调用方往往只是想要它顺手产生的 layer_inp。早先让它抛出去，会把整个
                # shim 子进程打死（handler 约定是"异常即退出"），后续 layer_inp 全部
                # 变成 "worker exited"。所以这里降级成 no_hidden，进程继续。
                try:
                    _in = _decode_hidden(request)
                    print(f"[kh.hidden] in.shape={getattr(_in,'shape',None)} n_past={n_past}",
                          file=sys.stderr, flush=True)
                    arr = binding.hidden(_in, n_past)
                except KeepHeadUnavailable:
                    _response(ok=True, no_hidden=True)
                else:
                    _array_response(arr)
            elif operation == "hidden_token":
                _response(ok=True, token=binding.token(_decode_hidden(request), n_past))
            elif operation == "hidden_tokens":
                _response(ok=True, tokens=binding.tokens_of(_decode_hidden(request), n_past))
            elif operation == "last_hidden":
                arr = binding.last_hidden()
                if arr is None:
                    _response(ok=False, error="no nextn hidden (needs SA_MTP=1 and a NextN head)")
                else:
                    _array_response(arr.reshape(1, -1))
            elif operation == "mtp_draft":
                _response(ok=True, token=binding.mtp_draft(
                    _decode_hidden(request), int(request["token"]), n_past))
            elif operation == "mtp_reset":
                binding.mtp_reset()
                _response(ok=True)
            elif operation == "layer_inp":
                arr = binding.layer_inp([int(v) for v in request.get("lids", [])])
                if arr is None:
                    _response(ok=False, error="layer_inp unavailable (see stage log)")
                else:
                    _array_response(arr.reshape(-1, arr.shape[-1]))
            elif operation == "draft_open":
                rc = binding.draft_open(str(request["path"]), int(request.get("n_ctx", 256)),
                                        int(request.get("n_batch", 32)),
                                        int(request.get("n_gpu_layers", 0)))
                _response(ok=(rc == 0), draft_rc=rc)
            elif operation == "draft_n_embd_enc":
                _response(ok=True, n_embd_enc=binding.draft_n_embd_enc())
            elif operation == "draft_inject":
                rc = binding.draft_inject(_decode_hidden(request))
                _response(ok=(rc == 0), draft_rc=rc)
            elif operation == "draft_block":
                ids = binding.draft_block(int(request["anchor"]), int(request.get("n_block", 16)))
                _response(ok=True, drafts=ids)
            elif operation == "draft_rollback":
                rc = binding.draft_rollback(int(request["n_keep"]))
                _response(ok=(rc == 0), draft_rc=rc)
            elif operation == "draft_close":
                binding.draft_close()
                _response(ok=True)
            elif operation == "add_layer_inp":
                rc = binding.add_layer_inp(int(request["lid"]))
                if rc != 0:
                    _response(ok=False, error=f"add_layer_inp({request['lid']}) rc={rc}")
                else:
                    _response(ok=True)
            elif operation == "seq_rm":
                binding.seq_rm(n_past)
                _response(ok=True)
            elif operation == "tokenize":
                # Text crosses the worker boundary as UTF-8 bytes, not str: the shim
                # wants raw bytes plus an explicit length so it never has to guess.
                _response(ok=True, tokens=binding.tokenize(request.get("text", "").encode("utf-8")))
            elif operation == "piece":
                _response(ok=True, piece=binding.piece(int(request["token"])).decode("latin-1"))
            elif operation == "is_eog":
                _response(ok=True, eog=binding.is_eog(int(request["token"])))
            elif operation == "set_sampling":
                # Every field has a default that means "off", so a request carrying
                # only `temperature` is well formed.
                _response(ok=True, sampling=binding.set_sampling(
                    temperature=float(request.get("temperature", 0.0)),
                    top_p=float(request.get("top_p", 1.0)),
                    top_k=int(request.get("top_k", 0)),
                    repeat_penalty=float(request.get("repeat_penalty", 1.0)),
                    repeat_last_n=int(request.get("repeat_last_n", 0)),
                    penalty_freq=float(request.get("frequency_penalty", 0.0)),
                    penalty_present=float(request.get("presence_penalty", 0.0)),
                    seed=int(request.get("seed", 0xFFFFFFFF))))
            elif operation == "get_sampling":
                _response(ok=True, sampling=binding.sampling())
            elif operation == "clear_sampling":
                _response(ok=True, sampling=binding.clear_sampling())
            else:
                raise ValueError(f"unknown operation: {operation!r}")
    except Exception as exc:  # noqa: BLE001 - serialize the failure instead of dying
        _response(ok=False, error=f"{type(exc).__name__}: {exc}")
        _trace(f"handler error, exiting: {type(exc).__name__}: {exc}")
        return 1
    except BaseException as exc:  # noqa: BLE001 - KeyboardInterrupt / SystemExit reach here
        _trace(f"non-Exception escape: {type(exc).__name__}: {exc}")
        raise
    else:
        _trace("stdin EOF, exiting")
    finally:
        binding.close()
        _trace("binding closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
