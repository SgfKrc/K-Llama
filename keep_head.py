"""Self-contained keep-head client: cut GGUF in, hidden or token out.

Trimmed from QLH's ``src/llama_keep_head.py`` (~870 lines) to the six calls this
prototype makes. The native binding lives in ``keep_head_worker.py``; this module is
the parent-side handle.

**The worker subprocess stays, and that is measured, not assumed.** Two independent
reasons force it:

1. *Static CRT.* The shim links the C runtime statically, so its ``getenv`` reads a
   snapshot of the process environment taken when the DLL loads and never refreshes
   it. The stages need different ``SA_N_GPU_LAYERS`` (the GPU stage offloads every
   layer, the CPU stage must keep every layer on the host) and both live at the same
   time. Measured: ``_putenv_s("SA_N_GPU_LAYERS", "0")`` followed by a second
   ``qlh_kh_load`` still offloaded 9/9 layers to CUDA. Only a new process differs.
2. *Backend registry.* ``qlh_ensure_backends_loaded`` registers ggml backends once
   per process behind a ``static`` flag, reading ``SA_BACKEND_DIR`` at that moment.
   Set it after ``CDLL()`` and a later load reports ``backends loaded: count=0`` and
   fails with ``llama_model_load_from_file failed``.

A single stage *can* load in-process (verified), but the two-stage chain cannot, so
one path is cheaper than two. Upstream has a third reason this prototype lacks: pip's
``llama_cpp`` ships same-named ``ggml-base.dll`` / ``ggml.dll`` and Windows binds a
DLL basename once per process (WinError 127). Here nothing imports ``llama_cpp``.

M-RoPE needs nothing on this side: the shim derives ``n_pos_per_embd`` from the
model's rope type and expands positions itself, so qwen35moe works unmodified.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

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

WORKER_SCRIPT = "keep_head_worker.py"


class KeepHeadUnavailable(RuntimeError):
    """Shim or symbol missing, load failed, or a forward returned an error."""


class KeepHeadUpstream:
    """One cut GGUF behind one shim handle, driven in a dedicated worker process.

    ``mode="nextn"`` pairs with a head-cut artifact (``blk.0..K-1``) and exports the
    file's own last-layer output, so a K-layer file runs K layers.
    ``mode="layer_inp"`` reads layer ``cut_layer``'s input from a whole model, which
    runs every layer over the whole file and is only good for numerical reference.

    ``n_past`` belongs to the caller: it is passed on every call and never tracked
    here. ``reset()`` clears KV / recurrent memory so a new sequence may start at 0.
    """

    def __init__(
        self,
        shim_path: str | Path,
        model_path: str | Path,
        *,
        mode: str = "nextn",
        cut_layer: int | None = None,
        n_ctx: int = 4096,
        n_threads: int = 8,
        n_batch: int = 512,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if mode not in MODE_CODES:
            raise KeepHeadUnavailable(f"mode must be one of {sorted(MODE_CODES)}, got {mode!r}")
        if mode == "layer_inp" and cut_layer is None:
            raise KeepHeadUnavailable("layer_inp mode needs cut_layer")
        self.mode = mode
        self.cut_layer = None if cut_layer is None else int(cut_layer)
        self.shim_path = Path(shim_path)
        self.model_path = Path(model_path)
        if not self.shim_path.is_file():
            raise KeepHeadUnavailable(f"keep-head shim not found: {self.shim_path}")
        if not self.model_path.is_file():
            raise KeepHeadUnavailable(f"model not found: {self.model_path}")
        self.environment = dict(os.environ if environment is None else environment)
        self.n_ctx = int(n_ctx)
        self.n_threads = int(n_threads)
        self.n_batch = max(1, int(n_batch))
        self._worker: subprocess.Popen | None = None
        self._start_worker()

    # ------------------------------------------------------------------ worker
    def _start_worker(self) -> None:
        script = Path(__file__).with_name(WORKER_SCRIPT)
        if not script.is_file():
            raise KeepHeadUnavailable(f"keep-head worker script missing: {script}")
        command = [
            sys.executable, str(script),
            "--shim", str(self.shim_path), "--model", str(self.model_path),
            "--mode", self.mode, "--n-ctx", str(self.n_ctx),
            "--n-threads", str(self.n_threads), "--n-batch", str(self.n_batch)]
        if self.cut_layer is not None:
            command.extend(["--cut-layer", str(self.cut_layer)])
        kwargs: dict[str, object] = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        # stderr goes to the caller's stderr, not a pipe (llama.cpp's loader would fill
        # it and deadlock) nor DEVNULL (the only failure explanation would vanish).
        try:
            self._worker = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, text=True, encoding="utf-8",
                bufsize=1, env=self.environment, **kwargs)
            response = self._reply()
        except (OSError, KeepHeadUnavailable) as exc:
            self.close()
            raise KeepHeadUnavailable(f"keep-head worker init failed: {exc}") from exc
        self.n_embd, self.n_layer = int(response["n_embd"]), int(response["n_layer"])
        if self.n_embd <= 0:
            self.close()
            raise KeepHeadUnavailable("worker reported n_embd <= 0")

    def _reply(self) -> dict:
        worker = self._worker
        if worker is None or worker.poll() is not None or worker.stdout is None:
            raise KeepHeadUnavailable("keep-head worker is not running")
        line = worker.stdout.readline()
        if not line:
            raise KeepHeadUnavailable("keep-head worker exited before responding")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise KeepHeadUnavailable("keep-head worker sent an unparseable line") from exc
        if not response.get("ok"):
            raise KeepHeadUnavailable(str(response.get("error") or "worker request failed"))
        return response

    def _request(self, payload: dict) -> dict:
        worker = self._worker
        if worker is None or worker.poll() is not None or worker.stdin is None:
            raise KeepHeadUnavailable("keep-head worker is not running")
        try:
            worker.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            worker.stdin.flush()
            return self._reply()
        except (OSError, BrokenPipeError) as exc:
            raise KeepHeadUnavailable(f"keep-head worker communication failed: {exc}") from exc

    @staticmethod
    def _hidden_payload(op: str, arr: np.ndarray, n_past: int, **extra) -> dict:
        # base64 keeps the pipe line-oriented JSON instead of raw binary, and it can read
        # the array's buffer directly: .tobytes() first would add an 8 KB copy per step.
        payload = {"op": op, "shape": list(arr.shape), "n_past": int(n_past),
                   "data": base64.b64encode(arr).decode("ascii")}
        payload.update(extra)
        return payload

    @staticmethod
    def _array(response: dict) -> np.ndarray:
        try:
            shape = tuple(int(v) for v in response["shape"])
            raw = base64.b64decode(str(response["data"]))
            # No .copy(): b64decode already produced a fresh bytes object owned by this
            # view, so the copy was pure overhead on an 8 KB payload. The array is
            # read-only, which the numeric path only ever reads.
            return np.frombuffer(raw, dtype=np.float32).reshape(shape)
        except (KeyError, TypeError, ValueError) as exc:
            raise KeepHeadUnavailable("keep-head worker returned malformed hidden") from exc

    # -------------------------------------------------------------- validation
    @property
    def is_available(self) -> bool:
        return self._worker is not None and self._worker.poll() is None

    def _check_context(self, n_tokens: int, n_past: int) -> None:
        if n_tokens > self.n_batch:
            raise KeepHeadUnavailable(f"batch too large: tokens={n_tokens} limit={self.n_batch}")
        if n_past < 0 or n_past + n_tokens > self.n_ctx:
            raise KeepHeadUnavailable(
                f"context limit exceeded: past={n_past} tokens={n_tokens} n_ctx={self.n_ctx}")

    def _checked_hidden(self, hidden, n_past: int) -> np.ndarray:
        arr = np.ascontiguousarray(np.asarray(hidden, dtype=np.float32))
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.n_embd:
            raise ValueError(f"hidden shape must be [n_tokens, {self.n_embd}], got {arr.shape}")
        if arr.shape[0] == 0:
            raise ValueError("hidden must contain at least one token")
        self._check_context(int(arr.shape[0]), n_past)
        return arr

    # ----------------------------------------------------------------- forward
    def forward_tokens_to_hidden(self, tokens: Sequence[int], *, n_past: int = 0) -> np.ndarray:
        """Run ``tokens`` from ``n_past``; return ``[n_tokens, n_embd]`` float32 hidden."""
        toks = [int(token) for token in tokens]
        if not toks:
            raise ValueError("tokens must not be empty")
        self._check_context(len(toks), n_past)
        return self._array(self._request(
            {"op": "tokens", "tokens": toks, "n_past": int(n_past)}))

    def forward_hidden_to_hidden(self, hidden, *, n_past: int = 0) -> np.ndarray:
        """Feed ``hidden`` back in as embeddings; return this stage's hidden."""
        arr = self._checked_hidden(hidden, n_past)
        return self._array(self._request(self._hidden_payload("hidden", arr, n_past)))

    def forward_hidden_to_token(self, hidden, *, n_past: int = 0) -> int:
        """Feed ``hidden`` in and return the last row's argmax token id."""
        arr = self._checked_hidden(hidden, n_past)
        token = int(self._request(self._hidden_payload("hidden_token", arr, n_past))["token"])
        if not 0 <= token < 2**31:
            raise KeepHeadUnavailable(f"keep-head worker returned an invalid token id: {token}")
        return token

    def forward_hidden_to_tokens(self, hidden, *, n_past: int = 0) -> list[int]:
        """Feed k rows in, get k argmax token ids back, in one call.

        Each row is verified independently at its own position (n_past, n_past+1, ...), which is
        what speculative decoding needs: submit the whole candidate block, read one argmax per
        slot, then keep the agreeing prefix.
        """
        arr = self._checked_hidden(hidden, n_past)
        reply = self._request(self._hidden_payload("hidden_tokens", arr, n_past))
        toks = [int(t) for t in reply.get("tokens", [])]
        if len(toks) != int(arr.shape[0]):
            raise KeepHeadUnavailable(
                f"keep-head worker returned {len(toks)} tokens for {arr.shape[0]} rows")
        return toks

    def last_hidden(self) -> np.ndarray:
        """The trunk's final hidden of the most recent decode; raises when unavailable."""
        reply = self._request({"op": "last_hidden"})
        return self._array(reply)

    def mtp_draft(self, hidden, token: int, *, n_past: int) -> int:
        """One draft token from the MTP head."""
        arr = self._checked_hidden(hidden, n_past)
        return int(self._request(
            self._hidden_payload("mtp_draft", arr, n_past, token=int(token)))["token"])

    def mtp_reset(self) -> None:
        self._request({"op": "mtp_reset"})

    def seq_rm(self, n_past: int) -> None:
        self._request({"op": "seq_rm", "n_past": int(n_past)})

    def tokenize(self, text: str) -> list[int]:
        """Text -> token ids, for callers that start from a prompt instead of ids."""
        reply = self._request({"op": "tokenize", "text": text})
        return [int(t) for t in reply.get("tokens", [])]

    def piece(self, token: int) -> bytes:
        """Token id -> UTF-8 bytes.

        A piece can be a partial UTF-8 character: byte-level BPE splits multi-byte
        characters across tokens. Callers must buffer until the bytes decode, which
        is what ``stream_text`` does.
        """
        reply = self._request({"op": "piece", "token": int(token)})
        # latin-1 is a byte-preserving round trip for the JSON hop; the real encoding
        # is recovered here.
        return reply.get("piece", "").encode("latin-1")

    def is_eog(self, token: int) -> bool:
        return bool(self._request({"op": "is_eog", "token": int(token)}).get("eog", False))

    # ----------------------------------------------------------------- sampling
    def set_sampling(
        self,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repeat_penalty: float = 1.0,
        repeat_last_n: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: int = 0xFFFFFFFF,
    ) -> dict:
        """Install this stage's sampler chain; return the values actually in effect.

        Only meaningful on a stage that emits token ids -- the tail. A hidden-out
        stage has no logits to sample from, so nothing there reads this.

        Defaults are the "off" end of every knob (temperature 0 = greedy,
        top_p 1 = disabled, top_k 0 = disabled, penalties disabled), which is why a
        caller that says nothing keeps the argmax-equivalent deterministic path.
        """
        reply = self._request({
            "op": "set_sampling",
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repeat_penalty": float(repeat_penalty),
            "repeat_last_n": int(repeat_last_n),
            "frequency_penalty": float(frequency_penalty),
            "presence_penalty": float(presence_penalty),
            "seed": int(seed) & 0xFFFFFFFF,
        })
        return dict(reply.get("sampling") or {})

    def sampling(self) -> dict:
        """What the native side says is in effect (no request parameters involved)."""
        return dict(self._request({"op": "get_sampling"}).get("sampling") or {})

    def clear_sampling(self) -> dict:
        """Return this stage to the pristine argmax default.

        Call this when a request carries no sampling parameters: ``set_sampling`` with
        the defaults would install a greedy chain, which scales the same tokens but
        reports a different mode and would let one client's configuration leak into
        the next client's request.
        """
        return dict(self._request({"op": "clear_sampling"}).get("sampling") or {})

    def reset(self) -> None:
        """Clear KV / recurrent memory so the next sequence may start at position 0."""
        if self.is_available:
            self._request({"op": "reset"})

    def close(self) -> None:
        worker, self._worker = getattr(self, "_worker", None), None
        if worker is None:
            return
        try:
            if worker.poll() is None and worker.stdin is not None and not worker.stdin.closed:
                worker.stdin.write('{"op":"close"}\n')
                worker.stdin.flush()
        except (OSError, BrokenPipeError):
            pass
        finally:
            try:
                if worker.stdin is not None:
                    worker.stdin.close()
            except OSError:
                pass
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=10)
            if worker.stdout is not None:
                worker.stdout.close()

    def __enter__(self) -> "KeepHeadUpstream":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - last resort at interpreter exit
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
