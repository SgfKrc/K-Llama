"""N-stage client: read topology.toml, spawn the local stages, drive greedy decode.

One topology file describes the whole chain, so this file no longer knows how many
stages there are. It spawns every stage that runs on this host, connects to all of
them (local and remote alike), and walks the chain left to right.

Payload shape follows the stage kind, not its position:

  gpu   tokens  (JSON list)   -> hidden (raw bytes, wire dtype)
  cpu   hidden  (wire dtype)  -> hidden (raw bytes, wire dtype)
  tail  hidden  (wire dtype)  -> token  (JSON int)

The wire dtype is float32 unless SA_WIRE_DTYPE asks for float16 (see wire.py); the
client declares it on every FORWARD and reads back whatever the worker declares.

Only the first stage is spawned without an input; it is the one that owns the
token embedding. Only the last one owns the output projection.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import protocol  # noqa: E402
import topology  # noqa: E402
import wire  # noqa: E402
from topology import Stage  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_SHIM = Path("H:/slimarc-proto/exp-c/shim/qlh_keep_head.dll")
LOG_DIR = HERE / "build" / "relay-records"

# Weights are not the only cost of a loaded stage: each one also builds a KV cache
# and compute buffers, and the CUDA stage takes VRAM rather than RAM. Staying well
# under available RAM is the point — Windows does not refuse an overshoot, it pages
# and the machine stops responding.
#
# What the budget actually counts was calibrated by measuring, because the naive
# "sum of file sizes" reading is off by a factor of two on this workload:
#
#   stage 0-8   4.53 GB file, all layers on GPU   -> 4.84 GB RAM gone
#   stage 8-24  8.36 GB file, pure CPU, after warm -> 1.80 GB RAM gone
#   both loaded together                          -> 6.67 GB total (sums to 6.64)
#
# Pure-CPU stages mmap their weights and only the resident pages that the graph
# actually touches stay committed -- 8.36 GB of file became 1.80 GB of RAM. The GPU
# stage is different: its weights go to VRAM, but `token_embd` is forced onto the CPU
# ("cannot be used with preferred buffer type CUDA_Host"), and the CUDA context plus
# compute buffer add up, so it costs roughly its file size in RAM.
#
# Hence two coefficients rather than one, both deliberately generous:
# Fraction of *currently available* RAM a local set of stages may claim. 0.75 leaves a margin for
# page cache that the OS will not hand back on demand. Override with SA_MEM_SAFETY when a run
# genuinely needs to sit closer to the limit (e.g. a 35B two-segment split on one box, which needs
# 17.96 GiB against a 17.71 GiB budget at 0.75).
MEM_SAFETY = float(os.environ.get("SA_MEM_SAFETY", "0.75"))
# Fraction of a CPU stage's file size that stays resident. Measured 0.215; 0.30 leaves
# a margin for a longer context than the 4096 the measurement used.
CPU_RESIDENT_RATIO = 0.30
# Fraction of a file charged to RAM for the layers that are copied to VRAM. Reading them
# for upload walks the whole tensor through the page cache, and those pages stay counted
# until something evicts them, so a fully-offloaded artifact costs about its file size.
GPU_RAM_RATIO = 1.00

# A stage whose `gpu_layers` is between 0 and its layer count is a MIX, and charging such
# a stage its full file size is what made 35B look like it could not fit on one machine.
# Measured on the 40-layer 35B (20.82 GiB file) with 14 layers on the GPU:
#
#   estimate under the old rule   20.82 GiB   -> refused, since budget was ~11-17 GiB
#   actual RAM used                9.10 GiB   -> 0.44x, and it ran at 12.64 t/s
#
# The two populations behave differently: uploaded layers cost their pages once and those
# pages are reclaimable, while CPU layers keep re-reading the ~21.5% they touch. So the
# ratio is a weighted average over the split, not a step function of "any GPU layers at
# all". 14/40 predicts 0.35*1.00 + 0.65*0.30 = 0.545 against a measured 0.44 -- still
# generous, which is the right side to err on, but no longer 2.3x pessimistic.

# VRAM is a separate budget from RAM and it does not fail loudly. On Windows the display
# driver (WDDM) lets an allocation exceed dedicated video memory and backs the excess with
# system RAM, so llama.cpp reports "offloaded N/N layers" and everything looks fine while
# every step copies weights over PCIe. Measured on a 8188 MiB RTX 4060 Laptop:
#
#   stage 0-8    4640 MiB file -> 5237 MiB VRAM delta   1.13x   5.6 ms/step
#   stage 24-32  4542 MiB file -> 5139 MiB VRAM delta   1.13x   6.9 ms/step
#   stage 8-24   8559 MiB file -> 7660 MiB VRAM delta   0.89x  42.1 ms/step  <- spilled
#
# The two that fit both cost exactly 597 MiB on top of the file size, so the model is
# linear: file size times a fixed factor plus a fixed per-stage overhead. The one that did
# not fit is the tell -- its delta is *smaller* than its file, and it ran 7.5x slower than
# the same artifact on CPU. That is the silent spill.
#
# Both constants are rounded up from the measurements; a stage that trips this check has
# never turned out to fit.
VRAM_PER_BYTE = 1.13
GPU_VRAM_OVERHEAD = 640 << 20   # measured 597 MiB of context + compute buffer, rounded up


def mem_budget_bytes() -> int:
    import psutil

    return int(psutil.virtual_memory().available * MEM_SAFETY)


def vram_info() -> tuple[int, int] | None:
    """(total, used) video memory in bytes, or None when no NVIDIA card is queryable.

    nvidia-smi is the only source that reflects *dedicated* usage the driver has already
    committed, which is exactly what decides whether the next allocation spills. Returning
    None (rather than raising) keeps CPU-only machines working.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    line = (out.stdout or "").strip().splitlines()
    if not line:
        return None
    try:
        total, used = (int(x) for x in line[0].split(",")[:2])
    except ValueError:
        return None
    mib = 1 << 20
    return total * mib, used * mib


def estimate_vram_bytes(stages) -> int:
    """Video memory the GPU-offloaded stages need, per the measured model above.

    Only local GPU stages count: a remote one runs on its own card.
    """
    source_total = sum(s.layers for s in stages)
    total = 0.0
    for stage in stages:
        if not stage.is_local or not stage.gpu_layers:
            continue
        # Only `gpu_layers` of the stage's layers reach the card. Charging the whole stage
        # made a 32-layer segment with 12 on the GPU look like it needed 19.45 GiB of VRAM
        # and get refused, when it actually uses 6.7 GiB.
        on_gpu = min(stage.gpu_layers, stage.layers)
        per_layer = stage_bytes(stage, source_total) / max(stage.layers, 1)
        total += per_layer * on_gpu * VRAM_PER_BYTE + GPU_VRAM_OVERHEAD
    return int(total)


def stage_bytes(stage, source_total: int) -> float:
    """Bytes of the artifact this stage actually touches.

    For a pre-cut segment that is the file size. For a C-scheme stage the file is the full
    checkpoint, so charging its size would claim 20.82 GiB of RAM for a stage that loads
    8 of 40 layers -- which is exactly the pessimism that made 35B look unable to fit on
    one machine. Scale by the layer fraction instead.
    """
    size = stage.model.stat().st_size
    if stage.layer_range and source_total > 0:
        return size * (stage.layers / source_total)
    return float(size)


def ram_ratio(stage) -> float:
    """What fraction of one artifact's size to charge to system RAM.

    A stage that offloads *some* layers is a mix, and it must be charged as one. The old
    rule was a step: any `gpu_layers` at all meant the full file size in RAM. For 35B that
    turned a 9.10 GiB load into a 20.82 GiB estimate, so the budget refused a configuration
    that actually runs at 12.64 t/s -- and the workaround became splitting the model across
    machines, which is slower (3.40 t/s) than not splitting it at all.

    Layers go to VRAM but their pages still pass through the page cache on the way, and
    they are reclaimable afterwards; CPU layers keep re-reading the ~21.5% they touch.
    Weighting the two by the split predicts 0.545 where 0.44 was measured: still on the
    cautious side, which is the direction that keeps the machine alive.
    """
    layers = max(stage.layers, 1)
    on_gpu = min(max(stage.gpu_layers, 0), layers)
    frac_gpu = on_gpu / layers
    return frac_gpu * GPU_RAM_RATIO + (1.0 - frac_gpu) * CPU_RESIDENT_RATIO


def estimate_ram_bytes(stages) -> int:
    """What loading these stages is expected to cost in system RAM.

    Replaces summing file sizes, which assumed every byte of every artifact becomes
    resident. Only local stages count: a remote one is loaded on its own machine.
    """
    source_total = sum(s.layers for s in stages)
    total = 0.0
    for stage in stages:
        if not stage.is_local:
            continue
        total += stage_bytes(stage, source_total) * ram_ratio(stage)
    return int(total)

# A worker finishes loading its artifact tens of seconds after spawn, and serve()
# only starts listening afterwards, so connect has to retry rather than fail fast.
CONNECT_DEADLINE = 120.0
# create_connection's own timeout; the per-attempt probe, not the overall deadline.
CONNECT_ATTEMPT = 5.0
# Prefill can run for far longer than any sane connect timeout, so the socket is
# widened once the session is established.
SESSION_TIMEOUT = 600.0


class StageClient:
    """One TCP session to one stage. Holds no engine state of its own."""

    def __init__(self, stage: Stage) -> None:
        self.stage = stage
        self._addr = (stage.host, stage.port)
        self._sock: socket.socket | None = None
        self._n_embd = 0

    def connect(self, deadline_s: float = CONNECT_DEADLINE,
                proc: subprocess.Popen | None = None) -> float:
        """Retry until the stage accepts, but not past a worker that already died.

        Without the ``proc`` check a stage whose shim failed to load would be waited
        on for the full 120 s deadline -- the connect error would read "did not
        accept" when the real answer is already in the stage log.
        """
        deadline = time.monotonic() + deadline_s
        started = time.monotonic()
        while True:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"stage {self.stage.index} ({self.stage.kind}) worker exited with "
                    f"code {proc.returncode} before accepting on "
                    f"{self.stage.host}:{self.stage.port}")
            try:
                sock = socket.create_connection(self._addr, timeout=CONNECT_ATTEMPT)
            except OSError as exc:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"stage {self.stage.index} ({self.stage.kind}) did not accept on "
                        f"{self.stage.host}:{self.stage.port} within {deadline_s:.0f}s: {exc}"
                    ) from exc
                time.sleep(0.5)
                continue
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(SESSION_TIMEOUT)
            self._sock = sock
            return time.monotonic() - started

    @property
    def sock(self) -> socket.socket:
        if self._sock is None:
            raise RuntimeError(f"stage {self.stage.index} is not connected")
        return self._sock

    @property
    def n_embd(self) -> int:
        """Hidden width, known after the first forward.

        A single-stage topology never runs `forward_tokens`, so ask the worker for it
        via a tokenize round trip, which is cheap and always available.
        """
        if not self._n_embd:
            self._forward({"op": "tokenize", "text": ""})
        return self._n_embd

    def _forward(self, meta: dict, blob: bytes = b"") -> protocol.Message:
        protocol.send(self.sock, protocol.FORWARD, meta, blob)
        msg = protocol.read(self.sock)
        if msg.kind == protocol.ERROR:
            raise RuntimeError(f"stage {self.stage.index} ({self.stage.kind}): {msg.meta}")
        return msg

    def forward_tokens(self, tokens: list[int], n_past: int) -> np.ndarray:
        msg = self._forward({"tokens": tokens, "n_tokens": len(tokens),
                             "n_past": n_past, "dtype": wire.current_dtype()})
        return self._decode_hidden(msg)

    def forward_hidden(self, hidden: np.ndarray, n_past: int) -> np.ndarray:
        dtype = wire.current_dtype()
        msg = self._forward({"n_tokens": int(hidden.shape[0]), "n_past": n_past,
                             "dtype": dtype}, wire.encode(hidden, dtype))
        return self._decode_hidden(msg)

    def forward_token(self, hidden: np.ndarray, n_past: int) -> int:
        dtype = wire.current_dtype()
        msg = self._forward({"n_tokens": int(hidden.shape[0]), "n_past": n_past,
                             "dtype": dtype}, wire.encode(hidden, dtype))
        return int(msg.meta["token"])

    def forward_tokens_out(self, hidden: np.ndarray, n_past: int) -> list[int]:
        """k rows of hidden -> k token ids, in one round trip. Speculative verification."""
        dtype = wire.current_dtype()
        msg = self._forward({"n_tokens": int(hidden.shape[0]), "n_past": n_past,
                             "dtype": dtype}, wire.encode(hidden, dtype))
        toks = [int(t) for t in msg.meta["tokens"]]
        if len(toks) != int(hidden.shape[0]):
            raise RuntimeError(
                f"stage {self.stage.index}: {len(toks)} tokens for {hidden.shape[0]} rows")
        return toks

    def last_hidden(self) -> np.ndarray | None:
        """Final hidden of the most recent decode (MTP input). None when unavailable."""
        try:
            msg = self._forward({"op": "last_hidden", "n_tokens": 1, "n_past": 0})
        except Exception:  # noqa: BLE001 - "no draft head" is a normal state
            return None
        return self._decode_hidden(msg).reshape(-1)

    def mtp_draft(self, hidden: np.ndarray, token: int, n_past: int) -> int:
        """One draft token from the MTP head held by this stage."""
        # last_hidden() hands back a flat (n_embd,) row; the wire wants one row of that width.
        # Reading shape[0] without this gave a 2048-row request and the worker rejected it.
        arr = hidden.reshape(1, -1) if hidden.ndim == 1 else hidden
        dtype = wire.current_dtype()
        # No "op": this rides on FORWARD like a normal hidden-in call, with `token` marking it
        # as a draft request. An op would be intercepted by the text-helper dispatch before the
        # blob is decoded, and mtp_draft needs the hidden.
        msg = self._forward({"n_tokens": int(arr.shape[0]),
                             "n_past": n_past, "dtype": dtype, "token": int(token)},
                            wire.encode(arr, dtype))
        return int(msg.meta["token"])

    def mtp_reset(self) -> None:
        self._forward({"op": "mtp_reset", "n_tokens": 0, "n_past": 0})

    def seq_rm(self, n_past: int) -> None:
        """Drop everything at and after ``n_past``. Undoes a rejected draft."""
        self._forward({"op": "seq_rm", "n_tokens": 0, "n_past": n_past})

    def _decode_hidden(self, msg: protocol.Message) -> np.ndarray:
        """Decode the reply's hidden in whatever dtype the worker declared.

        The answer's dtype is not assumed to match this side's setting: a worker that
        ignored `dtype` (an older build) still answers float32, and the declared value
        is what decides how the bytes are read. Reading it wrong would be silent garbage,
        so the byte count is checked against it.
        """
        n_embd = int(msg.meta["n_embd"])
        n_tokens = int(msg.meta["n_tokens"])
        dtype = msg.meta.get("dtype", wire.DEFAULT)
        self._n_embd = n_embd
        expected = wire.payload_size((n_tokens, n_embd), dtype)
        if len(msg.blob) != expected:
            raise RuntimeError(
                f"stage {self.stage.index} ({self.stage.kind}) sent {len(msg.blob)} bytes "
                f"for {n_tokens}x{n_embd} {dtype} hidden (expected {expected})")
        return wire.decode(msg.blob, dtype, (n_tokens, n_embd))

    def tokenize(self, text: str) -> list[int]:
        """Text -> ids via the stage's own vocabulary (every stage loads the same one)."""
        msg = self._forward({"op": "tokenize", "text": text})
        if "n_embd" in msg.meta:
            self._n_embd = int(msg.meta["n_embd"])
        return [int(t) for t in msg.meta.get("tokens", [])]

    def piece(self, token: int) -> bytes:
        """Token id -> UTF-8 bytes, possibly a partial character."""
        msg = self._forward({"op": "piece", "token": int(token)})
        return str(msg.meta.get("piece", "")).encode("latin-1")

    def is_eog(self, token: int) -> bool:
        return bool(self._forward({"op": "is_eog", "token": int(token)}).meta.get("eog"))

    def set_sampling(self, **params) -> dict:
        """Install the tail stage's sampler; returns the parameters in effect.

        Rides on FORWARD's meta like the other ops, so the six-message protocol is
        untouched. The tail stage is the only one that produces tokens, so this is
        a no-op for every other kind.
        """
        if self.stage.kind != "tail":
            return {}
        return dict(self._forward({"op": "set_sampling", **params}).meta.get("sampling") or {})

    def sampling(self) -> dict:
        """Effective sampler configuration as the tail stage reports it."""
        if self.stage.kind != "tail":
            return {}
        return dict(self._forward({"op": "get_sampling"}).meta.get("sampling") or {})

    def clear_sampling(self) -> dict:
        """Return the tail stage to the untouched argmax default."""
        if self.stage.kind != "tail":
            return {}
        return dict(self._forward({"op": "clear_sampling"}).meta.get("sampling") or {})

    def reset(self) -> None:
        protocol.send(self.sock, protocol.RESET)
        msg = protocol.read(self.sock)
        if msg.kind == protocol.ERROR:
            raise RuntimeError(f"stage {self.stage.index} reset failed: {msg.meta}")

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            protocol.send(self._sock, protocol.BYE)
        except OSError:
            pass
        self._sock.close()
        self._sock = None


def spawn_worker(stage: Stage, shim: Path, ctx: int) -> tuple[subprocess.Popen, Path]:
    """Start one stage process and return (process, log file).

    Worker output goes to a file, not to this terminal: llama.cpp's loader writes
    hundreds of lines per stage and would bury the timings. The file is still the
    first thing to read when a stage dies before accepting.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"stage-{stage.index}-{stage.kind}.log"
    handle = open(log, "w", encoding="utf-8", errors="replace")
    cmd = [
        sys.executable, str(HERE / "stage_worker.py"),
        "--role", stage.kind,
        "--model", str(stage.model),
        "--shim", str(shim),
        "--host", stage.host,
        "--port", str(stage.port),
        "--ctx", str(ctx),
        "--threads", str(stage.threads),
        "--n-gpu-layers", str(stage.gpu_layers),
        "--n-cpu-moe", str(stage.cpu_moe),
        "--layers", str(stage.layers),
    ]
    if stage.cpus:
        cmd.extend(["--cpus", stage.cpus])
    # Only the tail stage needs the output projection. Without this the shim computes
    # logits for all 248,320 vocabulary rows on every step of every stage and discards
    # all but the tail's -- measured 32.0 vs 3.6 ms/step on stage0.
    env = dict(os.environ)
    env["SA_NEEDS_LOGITS"] = "1" if stage.kind == "tail" else "0"
    # Speculative decode needs two things the plain path does not: the draft head (only the
    # tail holds it) and recurrent snapshots to roll a rejected draft back. Both are opt-in
    # because they cost memory and graph nodes that a non-speculative run should not pay.
    if stage.kind == "tail":
        env.setdefault("SA_MTP", "1" if os.environ.get("SA_SPEC", "1") != "0" else "0")
    env.setdefault("SA_N_RS_SEQ", os.environ.get("SA_SPEC_K", "2"))
    # Layer selection at load time: with a range, `model` is the full checkpoint and
    # llama.cpp (with the C-scheme patch) exposes only these layers. Unset for the
    # pre-cut-artifact path, which needs no help.
    if stage.layer_range:
        env["LLAMA_LAYER_RANGE"] = f"{stage.layer_range[0]}:{stage.layer_range[1]}"
    try:
        proc = subprocess.Popen(cmd, cwd=str(HERE), stdout=handle,
                                stderr=subprocess.STDOUT, env=env)
    finally:
        handle.close()
    return proc, log


def forward(clients: list[StageClient], tokens: list[int], n_past: int,
            stage_ms: list[list[float]]) -> int:
    """Walk the chain once and return the next token.

    The payload is chosen by each stage's kind, not by its position: a gpu stage
    takes tokens, any number of cpu stages pass hidden through, and the tail returns
    one token. Position is asserted once up front because a mis-ordered topology
    would otherwise feed tokens into a hidden-in endpoint and fail inside the shim.
    """
    if not clients or clients[0].stage.kind != "gpu" or clients[-1].stage.kind != "tail":
        raise RuntimeError("chain must start with a gpu stage and end with a tail stage")

    hidden: np.ndarray | None = None
    token = -1
    for i, client in enumerate(clients):
        kind = client.stage.kind
        t0 = time.perf_counter()
        if kind == "gpu":
            hidden = client.forward_tokens(tokens, n_past)
        elif kind == "cpu":
            if hidden is None:
                raise RuntimeError(f"stage {i} (cpu) has no hidden to consume")
            hidden = client.forward_hidden(hidden, n_past)
        else:  # tail
            if hidden is None:
                raise RuntimeError(f"stage {i} (tail) has no hidden to consume")
            if hidden.shape[0] > 1:
                # A prefill pass hands the tail several rows at once. The worker answers with
                # one argmax per position, and the token this step should return is the last
                # one -- the same token a single-row call on the final position would give.
                token = client.forward_tokens_out(hidden, n_past)[-1]
            else:
                token = client.forward_token(hidden, n_past)
        stage_ms[i].append((time.perf_counter() - t0) * 1000.0)
    return token


def forward_batch(clients: list[StageClient], tokens: list[int], n_past: int,
                  stage_ms: list[list[float]]) -> list[int]:
    """Walk the chain once with a *block* of tokens and return one token per position.

    This is the verification primitive: the whole candidate block goes through the chain in one
    round trip, and the tail answers with an argmax per position. Verified bit-identical to
    running the same positions one at a time (see checks/probe_perfect_draft.py).
    """
    if not clients or clients[0].stage.kind != "gpu" or clients[-1].stage.kind != "tail":
        raise RuntimeError("chain must start with a gpu stage and end with a tail stage")

    hidden: np.ndarray | None = None
    for i, client in enumerate(clients):
        kind = client.stage.kind
        t0 = time.perf_counter()
        if kind == "gpu":
            hidden = client.forward_tokens(tokens, n_past)
        elif kind == "cpu":
            if hidden is None:
                raise RuntimeError(f"stage {i} (cpu) has no hidden to consume")
            hidden = client.forward_hidden(hidden, n_past)
        else:  # tail
            if hidden is None:
                raise RuntimeError(f"stage {i} (tail) has no hidden to consume")
            out = client.forward_tokens_out(hidden, n_past)
            stage_ms[i].append((time.perf_counter() - t0) * 1000.0)
            return out
        stage_ms[i].append((time.perf_counter() - t0) * 1000.0)
    raise RuntimeError("chain has no tail stage")


def speculate(clients: list[StageClient], token: int, n_past: int,
              stage_ms: list[list[float]], k: int,
              stats: dict) -> list[int]:
    """One speculative step: trunk token + up to k recursive MTP drafts, verified in one block.

    Returns the accepted tokens for this step (1 + accepted drafts), always starting with the
    token the trunk itself produced. The trailing token of the verified block is free -- the
    block already computed it -- so it is returned as well and the caller must not recompute it.

    Correctness contract: the returned list is what plain greedy would have produced. The
    verification compares the drafts against the trunk's own argmax at each position, and any
    position after the first mismatch is rolled back on *every* stage.
    """
    tail = clients[-1]

    # 1. trunk step: one token, and the final hidden the draft head needs
    t1 = forward(clients, [token], n_past, stage_ms)
    h = tail.last_hidden()
    if h is None:                      # no NextN head (or SA_MTP off): plain decode
        stats["no_draft"] = stats.get("no_draft", 0) + 1
        return [t1]

    # 2. recursive drafts. Each one feeds the previous guess back in with the same hidden.
    drafts: list[int] = []
    tail.mtp_reset()
    cur = t1
    try:
        for j in range(k):
            d = tail.mtp_draft(h, cur, n_past + 1 + j)
            drafts.append(d)
            cur = d
    except Exception:                  # noqa: BLE001 - a broken draft head must not break decode
        stats["draft_error"] = stats.get("draft_error", 0) + 1
        return [t1]

    # 3. verify the whole block [t1, d0, d1, ...] in one pass
    block = [t1] + drafts
    try:
        tb = forward_batch(clients, block, n_past + 1, stage_ms)
    except Exception:                  # noqa: BLE001
        stats["verify_error"] = stats.get("verify_error", 0) + 1
        return [t1]

    # 4. accept the longest prefix where the draft equals the trunk
    accepted = 0
    for j in range(len(drafts)):
        if drafts[j] == tb[j]:
            accepted += 1
        else:
            break

    keep = 1 + accepted                # positions the block legitimately wrote
    if keep < len(block):
        # roll back everything past the accepted prefix, on both ends of the chain
        for c in clients:
            try:
                c.seq_rm(n_past + 1 + keep)
            except Exception:          # noqa: BLE001 - keep decoding even if rollback fails
                stats["rollback_error"] = stats.get("rollback_error", 0) + 1

    stats["drafts"] = stats.get("drafts", 0) + len(drafts)
    stats["accepted"] = stats.get("accepted", 0) + accepted
    out = block[:keep]
    # tb[accepted] is the trunk's token at the next position -- already paid for
    out.append(tb[accepted])
    return out


def _children(pid: int) -> list:
    """Snapshot a worker's descendants, taken while its parent is still alive."""
    try:
        import psutil

        return psutil.Process(pid).children(recursive=True)
    except Exception:  # noqa: BLE001 - a process that already left has no children
        return []


def _report_death(proc: subprocess.Popen, log: Path, stage: Stage) -> None:
    # A stage that died on its own is the interesting case, and its log tail is the
    # only diagnostic that survives the process.
    if proc.returncode in (0, None) or not log.is_file():
        return
    tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-15:]
    print(f"  stage {stage.index} ({stage.kind}) exited {proc.returncode}, "
          f"log tail: {log}", file=sys.stderr)
    for line in tail:
        print(f"    {line}", file=sys.stderr)


def _terminate(procs: list[tuple[subprocess.Popen, Path, Stage]]) -> None:
    """Stop every stage and every process a stage started.

    Each stage_worker keeps a keep_head_worker child, and that child is the one that
    actually holds the 8-16 GiB of weights. It only learns its parent is gone through
    a closed stdin pipe, so it can outlive a terminated parent for as long as it takes
    to notice -- or forever if it is stuck in a native call. A leak here is not tidiness:
    the memory stays committed and the next run's pre-check then fails on a machine
    that has nothing loaded. So reap the children explicitly, and never let one failing
    stage skip the cleanup of the others.
    """
    for proc, log, stage in procs:
        kids = _children(proc.pid)
        try:
            if proc.poll() is not None:
                _report_death(proc, log, stage)
                continue
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(f"  stage {stage.index} ({stage.kind}) ignored kill",
                          file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - cleanup must reach the next stage
            print(f"  stage {stage.index} ({stage.kind}) shutdown failed: {exc}",
                  file=sys.stderr)
        finally:
            for kid in kids:
                try:
                    if kid.is_running():
                        kid.terminate()
                        kid.wait(timeout=10)
                except Exception:  # noqa: BLE001 - already gone or gone after EOF
                    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shim", default=str(DEFAULT_SHIM))
    ap.add_argument("--topology", default="", help="override SA_TOPOLOGY for this run")
    ap.add_argument("--prompt-tokens", type=int, default=32,
                    help="synthetic prompt length; ignored with --prompt-text")
    # The synthetic default is a sequence of unrelated token ids. It is fine for timing
    # (fixed shape, no tokenizer cost) but it makes every model look degenerate, so it is
    # not evidence about output quality either way.
    ap.add_argument("--prompt-text", default=None,
                    help="natural-language prompt instead of synthetic token ids")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--spec-k", type=int, default=0,
                    help="MTP 递归草稿数；0 = 不投机。代码场景 k=2 实测 1.19x，k>2 只会更慢")
    # The checkpoint recommends temp 1.00 / top_k 20 / top_p 0.95. Greedy (the shim default)
    # degenerates into repetition on this model, so sampling is the default here and
    # --greedy opts back into it for reproducible comparisons.
    ap.add_argument("--greedy", action="store_true",
                    help="temperature 0; deterministic but prone to repetition on 35B")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--chat", action="store_true",
                    help="render --prompt-text as ChatML and parse special tokens")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the topology and the memory budget, load no model")
    args = ap.parse_args()

    if args.topology:
        os.environ[config.TOPOLOGY_ENV] = args.topology
    stages = topology.load(config.topology_path())
    ctx = config.ctx()
    shim = Path(args.shim)
    if not shim.is_file():
        raise SystemExit(f"keep-head shim not found: {shim}")

    if args.prompt_tokens + args.gen > ctx:
        raise SystemExit(f"prompt+gen={args.prompt_tokens + args.gen} exceeds ctx={ctx}")

    print(f"topology: {config.topology_path()}  stages={len(stages)}  ctx={ctx}")
    for s in stages:
        where = "local" if s.is_local else s.host
        print(f"  stage {s.index}: {s.kind:<4} {s.layers:>2} layers  "
              f"{where}:{s.port}  gpu_layers={s.gpu_layers} cpu_moe={s.cpu_moe} "
              f"threads={s.threads}  {s.model.name}")

    # Loading every local stage at once is the whole memory cost of the run, and
    # Windows answers an overshoot by thrashing swap rather than failing — on this
    # machine that hard-locked the desktop twice. Refuse up front instead.
    #
    # The estimate is not the sum of file sizes: see CPU_RESIDENT_RATIO for the
    # measurement showing a pure-CPU stage keeps only ~21% of its artifact resident.
    local_bytes = sum(s.model.stat().st_size for s in stages if s.is_local)
    local_count = sum(1 for s in stages if s.is_local)
    estimate = estimate_ram_bytes(stages)
    budget = mem_budget_bytes()
    print(f"\nlocal stages: {local_count}  weights {local_bytes / 2**30:.2f} GiB  "
          f"expected resident {estimate / 2**30:.2f} GiB  "
          f"budget {budget / 2**30:.2f} GiB")
    if estimate > budget:
        raise SystemExit(
            f"refusing to start: local stages need an estimated "
            f"{estimate / 2**30:.2f} GiB resident but only "
            f"{budget / 2**30:.2f} GiB is safe (available x {MEM_SAFETY:.0%}). "
            f"Move a stage to another host or lower the count.")

    # VRAM gets its own check because an overshoot there is silent: WDDM backs the excess
    # with system RAM, llama.cpp still reports every layer offloaded, and the run simply
    # crawls. See VRAM_PER_BYTE for the measurement.
    if any(s.is_local and s.gpu_layers for s in stages):
        vram = vram_info()
        want = estimate_vram_bytes(stages)
        if vram is None:
            print("\nVRAM: nvidia-smi unavailable, skipping the check. An overflow would "
                  "be silent -- verify with nvidia-smi while it runs.")
        else:
            total, used = vram
            free = total - used
            print(f"VRAM: need ~{want / 2**30:.2f} GiB  free {free / 2**30:.2f} GiB  "
                  f"of {total / 2**30:.2f} GiB")
            if want > free:
                # Per-stage VRAM cost, scaled to the layers that actually go to the card.
                # Using the file size here charged a whole 20.82 GiB checkpoint to a stage
                # that offloads 12 of its 32 layers, so it was refused as a 24 GiB stage.
                src_total = sum(x.layers for x in stages)
                gpu = [(s, stage_bytes(s, src_total) / max(s.layers, 1)
                           * min(s.gpu_layers, s.layers) * VRAM_PER_BYTE
                           + GPU_VRAM_OVERHEAD)
                       for s in stages if s.is_local and s.gpu_layers]
                gpu.sort(key=lambda pair: pair[1], reverse=True)
                worst, cost = gpu[0]
                raise SystemExit(
                    f"refusing to start: GPU stages need ~{want / 2**30:.2f} GiB of video "
                    f"memory but only {free / 2**30:.2f} GiB is free "
                    f"(short by {(want - free) / 2**30:.2f} GiB).\n"
                    f"  Largest: stage {worst.index} ({worst.model.name}) needs "
                    f"~{cost / 2**30:.2f} GiB.\n"
                    f"  Lower its gpu_layers, or set gpu_layers = 0 to run it on CPU. "
                    f"Windows does not refuse the allocation -- it spills to system RAM "
                    f"and runs slower than pure CPU.")

    if args.dry_run:
        print("\ndry run: topology and budget look fine, no model loaded")
        return 0

    procs: list[tuple[subprocess.Popen, Path, Stage]] = []
    clients: list[StageClient] = []
    # Sample VRAM before any worker starts, so the delta after loading is attributable to
    # this run rather than to whatever else shares the card.
    vram_before = vram_info() if any(s.is_local and s.gpu_layers for s in stages) else None
    try:
        # Spawn everything first, then connect: the stages load in parallel, so the
        # chain is ready in the time of the slowest artifact, not the sum of them.
        # Must be set BEFORE the workers spawn: they inherit the environment at start-up,
        # and a later assignment would not reach the shim that tokenizes.
        if args.chat:
            os.environ["SA_PARSE_SPECIAL"] = "1"
        spawn_start = time.perf_counter()
        for s in stages:
            if s.is_local:
                procs.append((*spawn_worker(s, shim, ctx), s))
        clients = [StageClient(s) for s in stages]
        local_procs = {stage.index: proc for proc, _log, stage in procs}
        for c in clients:
            c.connect(proc=local_procs.get(c.stage.index))
        print(f"workers ready in {time.perf_counter() - spawn_start:.1f}s")

        # Verify the up-front VRAM estimate against what actually landed. The estimate is
        # deliberately generous, so it can only catch a gross miss; this catches the case
        # the estimate got wrong. The tell is simple and does not depend on the constants:
        # a stage that fits always occupies MORE video memory than its file size, because
        # the compute buffer and context come on top. A stage that spills occupies LESS --
        # the driver quietly backed the remainder with system RAM, and llama.cpp still
        # printed "offloaded N/N layers".
        if vram_before is not None:
            vram_after = vram_info()
            if vram_after is not None:
                landed = vram_after[1] - vram_before[1]
                # Same scaling as the estimate: with a layer range the file is the whole
                # checkpoint, and only this stage's share of it is copied to the card.
                source_total = sum(x.layers for x in stages)
                # Same scaling as the estimate: only `gpu_layers` of each stage reach the
                # card, so a 32-layer segment with 12 offloaded must be compared against
                # 12 layers' worth of bytes, or every such load looks like a spill.
                want_disk = sum(
                    stage_bytes(s, source_total) / max(s.layers, 1)
                    * min(s.gpu_layers, s.layers)
                    for s in stages if s.is_local and s.gpu_layers)
                if want_disk and landed < want_disk:
                    gpu_stages = [s for s in stages if s.is_local and s.gpu_layers]
                    print(f"\n!! VRAM spill detected after load: {landed / 2**30:.2f} GiB "
                          f"resident for {want_disk / 2**30:.2f} GiB of GPU weights.\n"
                          f"   Stage(s) {[s.index for s in gpu_stages]} did not fully fit. "
                          f"Windows spilled the rest to system RAM; decode will be slower "
                          f"than pure CPU.\n"
                          f"   Set gpu_layers = 0 for the largest stage, or lower it.")
                else:
                    print(f"VRAM verified: {landed / 2**30:.2f} GiB resident for "
                          f"{want_disk / 2**30:.2f} GiB of GPU weights")

        if args.chat and args.prompt_text:
            from chat import apply_chatml
            args.prompt_text = apply_chatml([{"role": "user", "content": args.prompt_text}])
        _sampling = (dict(temperature=0.0) if args.greedy else
                     dict(temperature=args.temperature, top_k=args.top_k,
                          top_p=args.top_p, seed=args.seed))
        # Every stage holds its own sampler; all of them must agree or the tail samples with
        # a different distribution than the chain was tuned for.
        for c in clients:
            c.set_sampling(**_sampling)
        print(f"  sampling: {clients[-1].sampling()}")

        if args.prompt_text:
            prompt = clients[0].tokenize(args.prompt_text)
            if not prompt:
                raise SystemExit("--prompt-text tokenized to nothing")
        else:
            # N unrelated ids. Keeps the shape fixed for timing; says nothing about
            # whether the model writes sensible text.
            prompt = list(range(100, 100 + args.prompt_tokens * 100, 100))[: args.prompt_tokens]
        origin = repr(args.prompt_text) if args.prompt_text else "(synthetic token ids)"
        print(f"  prompt: {len(prompt)} tokens  {origin}")
        for run in range(1, args.repeat + 1):
            # Fresh KV per run: a timing number only means something from a fixed start.
            for c in clients:
                c.reset()

            stage_ms: list[list[float]] = [[] for _ in stages]
            t0 = time.perf_counter()
            token = forward(clients, prompt, 0, stage_ms)
            prefill_s = time.perf_counter() - t0
            # Per-stage numbers below are decode steps only: a prefill pass is an order
            # of magnitude larger and would dominate the median of a short --gen run.
            stage_ms = [[] for _ in stages]

            step_ms: list[float] = []
            out_tokens = [token]
            n_past = len(prompt)
            spec_stats: dict = {}
            k = args.spec_k
            while len(out_tokens) < args.gen:
                start = time.perf_counter()
                if k > 0:
                    produced = speculate(clients, token, n_past, stage_ms, k, spec_stats)
                else:
                    produced = [forward(clients, [token], n_past, stage_ms)]
                step_ms.append((time.perf_counter() - start) * 1000.0)
                # produced[0] is the token this step committed first; the rest are the extra
                # tokens the block settled. A step that yields m tokens advances n_past by m.
                token = produced[0]
                out_tokens.append(token)
                n_past += 1
                for extra in produced[1:]:
                    if len(out_tokens) >= args.gen:
                        break
                    token = extra
                    out_tokens.append(token)
                    n_past += 1

            arr = np.asarray(step_ms)
            pp = len(prompt) / prefill_s if prefill_s > 0 else 0.0
            # --gen 1 leaves no decode step to time; an empty median is nan, not a number.
            tg = f"{1000.0 / np.median(arr):.1f} t/s" if arr.size else "n/a (--gen 1)"
            print(f"  run {run}: pp{len(prompt)}={pp:.1f} t/s  "
                  f"tg{args.gen}={tg}" +
                  (f"  (median {np.median(arr):.1f} ms, p90 {np.percentile(arr, 90):.1f} ms)"
                   if arr.size else ""))
            for i, s in enumerate(stages):
                ms = float(np.median(stage_ms[i])) if stage_ms[i] else 0.0
                where = "local" if s.is_local else s.host
                print(f"    stage {i} ({s.kind + ',':<5}{s.layers:>3} layers, {where}): "
                      f"{ms:.1f} ms")
            if spec_stats.get("drafts"):
                acc = spec_stats.get("accepted", 0)
                drf = spec_stats["drafts"]
                steps = max(spec_stats.get("steps", 1), 1)
                print(f"    spec:   k={k} 接受 {acc}/{drf} = {100.0*acc/max(drf,1):.1f}%"
                      f"  平均每轮 {len(out_tokens)/max(len(step_ms),1):.2f} token"
                      + (f"  回滚失败 {spec_stats['rollback_error']}"
                         if spec_stats.get("rollback_error") else ""))
            print(f"    tokens: {out_tokens[:8]}  unique={len(set(out_tokens))}"
                  f"{' (stuck on one token)' if len(set(out_tokens)) == 1 else ''}")
            # Decode the generated ids, otherwise a degenerate run and a healthy one
            # look the same except for a "unique" count.
            try:
                _text = b"".join(clients[-1].piece(t) for t in out_tokens)
                print(f"    text:   {_text.decode('utf-8', 'replace')!r}")
            except Exception:  # noqa: BLE001 - a decode failure must not kill a timing run
                pass
    finally:
        # One bad session must not leave the rest of the stages running: that is how
        # the next run ends up loading models on top of still-resident ones.
        for c in clients:
            try:
                c.close()
            except Exception as exc:  # noqa: BLE001 - keep closing the others
                print(f"  stage {c.stage.index} close failed: {exc}", file=sys.stderr)
        _terminate(procs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
