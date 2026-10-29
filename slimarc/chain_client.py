"""N-stage client: read topology.toml, spawn the local stages, drive greedy decode.

One topology file describes the whole chain, so this file no longer knows how many
stages there are. It spawns every stage that runs on this host, connects to all of
them (local and remote alike), and walks the chain left to right.

Payload shape follows the stage kind, not its position:

  gpu   tokens  (JSON list)   -> hidden (raw float32)
  cpu   hidden  (raw float32) -> hidden (raw float32)
  tail  hidden  (raw float32) -> token  (JSON int)

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
from topology import Stage  # noqa: E402

HERE = Path(__file__).resolve().parent
# Resolve the shim relative to the repo so a checkout runs anywhere; SA_SHIM
# overrides it for a machine whose DLLs live outside the tree.
DEFAULT_SHIM = Path(os.environ.get("SA_SHIM") or (HERE / "shim" / "bin" / "qlh_keep_head.dll"))
LOG_DIR = HERE / "build" / "relay-records"

# Weights are not the only cost of a loaded stage: each one also builds a KV cache
# and compute buffers, and the CUDA stage takes VRAM rather than RAM. Staying well
# under available RAM is the point — Windows does not refuse an overshoot, it pages
# and the machine stops responding. 75% of available leaves room for that overhead.
MEM_SAFETY = 0.75


def mem_budget_bytes() -> int:
    import psutil

    return int(psutil.virtual_memory().available * MEM_SAFETY)

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

    def _forward(self, meta: dict, blob: bytes = b"") -> protocol.Message:
        protocol.send(self.sock, protocol.FORWARD, meta, blob)
        msg = protocol.read(self.sock)
        if msg.kind == protocol.ERROR:
            raise RuntimeError(f"stage {self.stage.index} ({self.stage.kind}): {msg.meta}")
        return msg

    def forward_tokens(self, tokens: list[int], n_past: int) -> np.ndarray:
        msg = self._forward({"tokens": tokens, "n_tokens": len(tokens), "n_past": n_past})
        self._n_embd = int(msg.meta["n_embd"])
        return np.frombuffer(msg.blob, dtype=np.float32).reshape(
            int(msg.meta["n_tokens"]), self._n_embd)

    def forward_hidden(self, hidden: np.ndarray, n_past: int) -> np.ndarray:
        blob = np.ascontiguousarray(hidden, dtype=np.float32).tobytes()
        msg = self._forward({"n_tokens": int(hidden.shape[0]), "n_past": n_past}, blob)
        n_embd = int(msg.meta["n_embd"])
        return np.frombuffer(msg.blob, dtype=np.float32).reshape(
            int(msg.meta["n_tokens"]), n_embd)

    def forward_token(self, hidden: np.ndarray, n_past: int) -> int:
        blob = np.ascontiguousarray(hidden, dtype=np.float32).tobytes()
        msg = self._forward({"n_tokens": int(hidden.shape[0]), "n_past": n_past}, blob)
        return int(msg.meta["token"])

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
    try:
        proc = subprocess.Popen(cmd, cwd=str(HERE), stdout=handle, stderr=subprocess.STDOUT)
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
            token = client.forward_token(hidden, n_past)
        stage_ms[i].append((time.perf_counter() - t0) * 1000.0)
    return token


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
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--gen", type=int, default=64)
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
    local_bytes = sum(s.model.stat().st_size for s in stages if s.is_local)
    local_count = sum(1 for s in stages if s.is_local)
    budget = mem_budget_bytes()
    print(f"\nlocal stages: {local_count}  weights {local_bytes / 2**30:.2f} GiB  "
          f"budget {budget / 2**30:.2f} GiB")
    if local_bytes > budget:
        raise SystemExit(
            f"refusing to start: local artifacts total {local_bytes / 2**30:.2f} GiB "
            f"but only {budget / 2**30:.2f} GiB is safe (available x {MEM_SAFETY:.0%}). "
            f"Move a stage to another host or lower the count.")

    if args.dry_run:
        print("\ndry run: topology and budget look fine, no model loaded")
        return 0

    procs: list[tuple[subprocess.Popen, Path, Stage]] = []
    clients: list[StageClient] = []
    try:
        # Spawn everything first, then connect: the stages load in parallel, so the
        # chain is ready in the time of the slowest artifact, not the sum of them.
        spawn_start = time.perf_counter()
        for s in stages:
            if s.is_local:
                procs.append((*spawn_worker(s, shim, ctx), s))
        clients = [StageClient(s) for s in stages]
        local_procs = {stage.index: proc for proc, _log, stage in procs}
        for c in clients:
            c.connect(proc=local_procs.get(c.stage.index))
        print(f"workers ready in {time.perf_counter() - spawn_start:.1f}s")

        prompt = list(range(100, 100 + args.prompt_tokens * 100, 100))[: args.prompt_tokens]
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
            for _ in range(args.gen - 1):
                start = time.perf_counter()
                token = forward(clients, [token], n_past, stage_ms)
                step_ms.append((time.perf_counter() - start) * 1000.0)
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
            print(f"    tokens: {out_tokens[:8]}  unique={len(set(out_tokens))}"
                  f"{' (stuck on one token)' if len(set(out_tokens)) == 1 else ''}")
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
