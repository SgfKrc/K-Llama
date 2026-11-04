"""Thin wrapper over the keep-head shim.

The shim (qlh_keep_head.c) provides the hidden-in/hidden-out and
hidden-in/token-out primitives; keep_head.py binds it. This module only adds the
per-stage policy: which layers live on which device, and role-specific naming.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

import config
from keep_head import KeepHeadUpstream


ROLES = ("gpu", "cpu", "tail")


class EngineError(RuntimeError):
    pass


class StageEngine:
    """One pipeline stage backed by a cut GGUF and the keep-head shim.

    role='gpu'  consumes token ids and emits hidden (first stage).
    role='cpu'  consumes hidden and emits hidden (middle stage).
    role='tail' consumes hidden and emits the next token id (last stage).

    'tail' runs exactly like 'cpu' -- same embd-injection decode path, same hidden
    width -- only the export the worker asks for differs. The role is validated
    here rather than silently coerced so a typo in topology.toml fails at load.
    """

    def __init__(
        self,
        *,
        shim_path: Path,
        model_path: Path,
        role: str,
        ctx: int,
        n_threads: int,
        n_gpu_layers: int,
        n_cpu_moe: int = 0,
        n_batch: int = 512,
    ) -> None:
        if role not in ROLES:
            raise EngineError(f"role must be one of {ROLES}, got {role!r}")
        self.role = role
        # Per-stage device policy rides on an explicit environment mapping, because
        # each stage needs a different value (the CPU stage keeps every layer on the
        # host) and the shim reads these with getenv() at load time.
        env = dict(os.environ)
        env["SA_N_GPU_LAYERS"] = str(int(n_gpu_layers))
        env["SA_N_CPU_MOE"] = str(int(n_cpu_moe))
        # The shim loads ggml backends by directory; without this it only sees the
        # executable and CWD search paths, which do not contain our DLLs.
        # `setdefault`, not assignment: the shim and llama.dll are usually siblings, but
        # running a stage straight out of the repo (docs/DELIVERY.md §2.3) puts the shim in
        # `.qlh_tmp/shimout/` while llama.dll sits in the llama.cpp build dir. Forcing the
        # shim's own directory made that fail with "Could not find module ... (or one of its
        # dependencies)" naming a file that does exist. One word buys back a documented
        # command that was otherwise unusable.
        env.setdefault("SA_BACKEND_DIR", str(Path(shim_path).resolve().parent))
        # Only the tail stage chooses a token, so only it needs the output projection.
        # The rule lives in config.needs_logits() because the subprocess path needs it too.
        env["SA_NEEDS_LOGITS"] = config.needs_logits(role)
        # mode 0 (nextn) returns the model's own last-layer output, which is what a head
        # stage wants when its LLAMA_LAYER_RANGE already stops at the seam. mode 1
        # (layer_inp) reads a chosen layer's input but runs every layer in the file, so it
        # is only correct on a whole checkpoint.
        self._up = KeepHeadUpstream(
            shim_path=str(shim_path),
            model_path=str(model_path),
            mode="nextn",
            n_ctx=ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            environment=env,
        )

    @property
    def n_embd(self) -> int:
        return int(self._up.n_embd)

    @property
    def n_layer(self) -> int:
        """Block count the shim found in this artifact, i.e. the layers it runs."""
        return int(self._up.n_layer)

    @property
    def healthy(self) -> bool:
        """这一段的模型还在不在：keep-head 子进程活着才算。

        判据只有 keep_head.KeepHeadUpstream.is_available 一处，这里只转发——持有权重的
        是那个子进程，本进程活着（还在 accept、还在回包）不代表这一段还能算。
        """
        return self._up.is_available

    def tokenize(self, text: str) -> list[int]:
        """Text -> ids. Any stage can do this; all of them load the same vocab."""
        return self._up.tokenize(text)

    def piece(self, token: int) -> bytes:
        return self._up.piece(token)

    def is_eog(self, token: int) -> bool:
        return self._up.is_eog(token)

    def forward_tokens(self, tokens: list[int], *, n_past: int) -> np.ndarray:
        """token in, hidden out. Used by the first stage."""
        out = self._up.forward_tokens_to_hidden(tokens, n_past=n_past)
        return np.asarray(out, dtype=np.float32)

    def forward_hidden(self, hidden: np.ndarray, *, n_past: int) -> np.ndarray:
        """hidden in, hidden out. Used by middle (cpu) stages.

        The input may arrive in the wire dtype (float16 -- see wire.py), but the shim's
        ABI is float32 and stays float32: this is where the one conversion happens, and
        it disappears entirely on the float32 path because ascontiguousarray returns the
        same array when it is already contiguous float32.
        """
        out = self._up.forward_hidden_to_hidden_or_none(hidden, n_past=n_past)
        if out is None:
            return None            # 前向发生了，只是这一段不提供 dense hidden
        return np.asarray(out, dtype=np.float32)

    def forward_token(self, hidden: np.ndarray, *, n_past: int) -> int:
        """hidden in, next token out. Used by the tail stage."""
        return int(self._up.forward_hidden_to_token(hidden, n_past=n_past))

    def forward_tokens_out(self, hidden: np.ndarray, *, n_past: int) -> list[int]:
        """k rows of hidden in, k token ids out. The speculative verification primitive."""
        return self._up.forward_hidden_to_tokens(hidden, n_past=n_past)

    def draft_open(self, path: str, n_ctx: int = 256, n_batch: int = 32,
                   n_gpu_layers: int = 0) -> int:
        return int(self._up.draft_open(path, n_ctx, n_batch, n_gpu_layers))

    def draft_n_embd_enc(self) -> int:
        return int(self._up.draft_n_embd_enc())

    def draft_inject(self, rows) -> int:
        return int(self._up.draft_inject(rows))

    def draft_block(self, anchor: int, n_block: int = 16) -> list[int]:
        return list(self._up.draft_block(anchor, n_block))

    def draft_reset(self) -> int:
        return int(self._up.draft_reset())

    def draft_rollback(self, n_keep: int) -> int:
        return int(self._up.draft_rollback(n_keep))

    def draft_close(self) -> None:
        self._up.draft_close()

    def add_layer_inp(self, lid: int) -> None:
        """Open an intermediate-hidden slot; needed only by DFlash-style drafting."""
        self._up.add_layer_inp(int(lid))

    def layer_inp(self, lids: list[int]) -> np.ndarray | None:
        """Multi-slot intermediate hidden of the most recent forward.

        None when unavailable, matching last_hidden(): the caller treats "no hidden" as
        "this model does not support it" rather than as a fatal error.
        """
        try:
            return np.asarray(self._up.layer_inp(lids), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            print(f"[engine.layer_inp] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return None

    def last_hidden(self, row: int = -1) -> np.ndarray | None:
        """Row `row` of the final hidden of the most recent decode (MTP input).

        `row=-1` is the last token of that forward. A verification pass of n rows produces n
        NextN rows, so `row` addresses the batch row whose position the caller wants -- the
        cascaded drafter asks for the last accepted one.
        """
        try:
            return np.asarray(self._up.last_hidden(int(row)), dtype=np.float32).reshape(-1)
        except Exception as exc:  # noqa: BLE001 - the caller treats "no draft head" as normal
            # Silent by design (a model without a head is normal), but it also hid a real
            # failure: the chain reported no_nextn_head while the shim answered rc=0 when
            # called directly. Leave the reason in the stage log.
            print(f"[engine.last_hidden] {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            return None

    def set_skip_hidden_logits(self, skip: bool = True) -> None:
        """Relay stages should say this: their logits are computed and discarded."""
        self._up.set_skip_hidden_logits(bool(skip))

    def logits(self, row: int = -1) -> np.ndarray:
        """Logits of one row from the most recent decode, flat with length n_vocab."""
        return np.asarray(self._up.logits(int(row)), dtype=np.float32).reshape(-1)

    def mtp_draft(self, hidden: np.ndarray, token: int, *, n_past: int) -> int:
        return int(self._up.mtp_draft(hidden, int(token), n_past=n_past))

    def mtp_draft_logits(self, hidden: np.ndarray, token: int, *,
                         n_past: int) -> tuple[int, np.ndarray]:
        """Draft token plus its logits row; the distribution rejection sampling needs."""
        token_out, values = self._up.mtp_draft_logits(hidden, int(token), n_past=n_past)
        return int(token_out), np.asarray(values, dtype=np.float32).reshape(-1)

    def mtp_reset(self) -> None:
        self._up.mtp_reset()

    def seq_rm(self, n_past: int) -> None:
        """Roll back positions >= n_past (needs SA_N_RS_SEQ > 0 on hybrid models)."""
        self._up.seq_rm(int(n_past))

    def set_sampling(self, **params) -> dict:
        """Configure the tail stage's sampler; returns the effective parameters.

        Wraps nothing: the shim decides what "off" means for every field, and the
        reply it sends back is the authority on what the caller actually got.
        A 'gpu'/'cpu' stage accepts the call and ignores it in every way that
        matters -- it never produces tokens, so `set_sampling` there is a no-op that
        still succeeds, which keeps the caller from having to special-case roles.
        """
        if self.role != "tail":
            return {}
        return self._up.set_sampling(**params)

    def sampling(self) -> dict:
        """Effective sampler configuration (empty for stages that never sample)."""
        if self.role != "tail":
            return {}
        return self._up.sampling()

    def clear_sampling(self) -> dict:
        """Back to the argmax default (see KeepHeadUpstream.clear_sampling)."""
        if self.role != "tail":
            return {}
        return self._up.clear_sampling()

    def reset(self) -> None:
        self._up.reset()

    def close(self) -> None:
        try:
            self._up.close()
        except Exception:  # noqa: BLE001 - closing must not mask the original failure
            pass


