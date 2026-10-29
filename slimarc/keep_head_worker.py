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
        lib.qlh_kh_reset.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_reset.restype = None
        lib.qlh_kh_close.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_close.restype = None
        if hasattr(lib, "qlh_kh_last_error"):
            lib.qlh_kh_last_error.argtypes = [ctypes.c_void_p]
            lib.qlh_kh_last_error.restype = ctypes.c_int32

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


def _response(**payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=True, separators=(",", ":")), flush=True)


def _array_response(array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    _response(ok=True, shape=list(contiguous.shape),
              data=base64.b64encode(contiguous.tobytes()).decode("ascii"))


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
    try:
        for raw in sys.stdin:
            if not raw.strip():
                continue
            request = json.loads(raw)
            operation = request.get("op")
            n_past = int(request.get("n_past", 0))
            if operation == "close":
                _response(ok=True)
                return 0
            if operation == "reset":
                binding.reset()
                _response(ok=True)
            elif operation == "tokens":
                _array_response(binding.tokens(request.get("tokens", []), n_past))
            elif operation == "hidden":
                _array_response(binding.hidden(_decode_hidden(request), n_past))
            elif operation == "hidden_token":
                _response(ok=True, token=binding.token(_decode_hidden(request), n_past))
            else:
                raise ValueError(f"unknown operation: {operation!r}")
    except Exception as exc:  # noqa: BLE001 - serialize the failure instead of dying
        _response(ok=False, error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        binding.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
