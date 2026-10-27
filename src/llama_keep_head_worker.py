"""Private worker that isolates keep-head's llama.cpp ABI from the host process."""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
from pathlib import Path
import sys

import numpy as np

from llama_keep_head import KeepHeadUpstream, _add_dll_dirs


def _response(**payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=True, separators=(",", ":")), flush=True)


def _array_response(array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    _response(ok=True, shape=list(contiguous.shape),
              data=base64.b64encode(contiguous.tobytes()).decode("ascii"))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shim", required=True)
    parser.add_argument("--model")
    parser.add_argument("--probe-shim", action="store_true")
    parser.add_argument("--mode", choices=("nextn", "layer_inp"), default="nextn")
    parser.add_argument("--cut-layer", type=int, default=None)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--n-seq-max", type=int, default=1,
                        help="P3 多序列：context 的并行序列上限（>=batch 才允许 seq_id>0）")
    parser.add_argument("--dll-dir", action="append", default=[])
    args = parser.parse_args()
    if not args.probe_shim and args.model is None:
        parser.error("--model is required unless --probe-shim is specified")
    return args


def main() -> int:
    args = _args()
    if args.probe_shim:
        # Only this short-lived process loads the DLL; no model handle is created.
        handles = []
        try:
            _, handles = _add_dll_dirs(Path(args.shim).parent, args.dll_dir)
            lib = ctypes.CDLL(str(Path(args.shim).resolve()))
            symbols = ("qlh_kh_forward_embd_tokens", "qlh_kh_n_vocab", "qlh_kh_finite_logits_guard")
            supported = all(hasattr(lib, name) for name in symbols)
            if supported:
                lib.qlh_kh_finite_logits_guard.argtypes = []
                lib.qlh_kh_finite_logits_guard.restype = ctypes.c_int32
                supported = lib.qlh_kh_finite_logits_guard() == 1
            _response(ok=True, supports_multi_position_argmax=supported)
            return 0
        except Exception as exc:  # noqa: BLE001 - serialized probe boundary
            _response(ok=False, error=f"{type(exc).__name__}: {exc}")
            return 1
        finally:
            for handle in reversed(handles):
                handle.close()
    try:
        upstream = KeepHeadUpstream(
            args.shim, args.model, mode=args.mode, cut_layer=args.cut_layer,
            n_ctx=args.n_ctx, n_threads=args.n_threads, n_batch=args.n_batch,
            n_seq_max=args.n_seq_max,
            extra_dll_dirs=args.dll_dir, _worker_process=True)
    except Exception as exc:  # noqa: BLE001 - serialized worker boundary
        _response(ok=False, error=f"{type(exc).__name__}: {exc}")
        return 1

    _response(ok=True, n_embd=upstream.n_embd, n_layer=upstream.n_layer,
              n_vocab=upstream.n_vocab,
              supports_multi_position_argmax=upstream.supports_multi_position_argmax,
              # ★ PERF-SKIP-LOGITS-01：native handle 只在本进程里，父进程必须靠这个回执
              #   确认 `QLH_LAYER_SKIP_LOGITS` 真的落实了（而不是"我设了环境变量"）。
              skip_hidden_logits=upstream.skip_hidden_logits_applied)
    try:
        for raw in sys.stdin:
            if not raw.strip():
                continue
            request = json.loads(raw)
            operation = request.get("op")
            if operation == "close":
                _response(ok=True)
                return 0
            if operation == "load_observation":
                _response(ok=True, observation=upstream.materialization_observation())
                continue
            if operation == "reset":
                upstream.reset()
                _response(ok=True)
                continue
            if operation == "tokens":
                _array_response(upstream.forward_tokens_to_hidden(
                    request.get("tokens", []), n_past=int(request.get("n_past", 0))))
                continue
            if operation == "hidden":
                shape = tuple(int(value) for value in request["shape"])
                hidden = np.frombuffer(
                    base64.b64decode(str(request["data"])), dtype=np.float32).reshape(shape)
                _array_response(upstream.forward_hidden_to_hidden(
                    hidden, n_past=int(request.get("n_past", 0)),
                    seq_ids=request.get("seq_ids"), positions=request.get("positions")))
                continue
            if operation == "hidden_tokens":
                shape = tuple(int(value) for value in request["shape"])
                hidden = np.frombuffer(
                    base64.b64decode(str(request["data"])), dtype=np.float32).reshape(shape)
                tokens = upstream.forward_hidden_to_tokens(
                    hidden, n_past=request.get("n_past", 0),
                    seq_ids=request.get("seq_ids"), positions=request.get("positions"))
                _response(ok=True, tokens=tokens)
                continue
            if operation == "hidden_token":
                shape = tuple(int(value) for value in request["shape"])
                hidden = np.frombuffer(
                    base64.b64decode(str(request["data"])), dtype=np.float32).reshape(shape)
                token = upstream.forward_hidden_to_token(
                    hidden, n_past=int(request.get("n_past", 0)),
                    seq_ids=request.get("seq_ids"), positions=request.get("positions"))
                _response(ok=True, token=int(token))
                continue
            raise ValueError(f"unknown operation: {operation!r}")
    except Exception as exc:  # noqa: BLE001 - serialized worker boundary
        _response(ok=False, error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        upstream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
