"""Stage worker: one cut artifact, one TCP endpoint.

Runs as a separate process so every stage owns its llama context, KV cache and
(for GPU-offloaded stages) VRAM. Protocol is the six-message set in protocol.py:
hello / forward / result / error / reset / bye.

Three roles, three payload shapes:

  gpu   tokens  (JSON list)  -> hidden (raw float32)
  cpu   hidden  (raw float32)-> hidden (raw float32)
  tail  hidden  (raw float32)-> token  (JSON int)
"""
from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import protocol  # noqa: E402
from engine import ROLES, StageEngine  # noqa: E402


class StageWorker:
    def __init__(self, engine: StageEngine, role: str) -> None:
        self._engine = engine
        self._role = role

    def serve(self, host: str, port: int) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"[{self._role}] listening on {host}:{port}", flush=True)

        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[{self._role}] client {addr}", flush=True)
        try:
            self._session(conn)
        finally:
            conn.close()
            server.close()

    def _session(self, conn: socket.socket) -> None:
        while True:
            try:
                msg = protocol.read(conn)
            except protocol.ProtocolError:
                return

            if msg.kind == protocol.BYE:
                return
            if msg.kind == protocol.RESET:
                self._engine.reset()
                protocol.send(conn, protocol.RESULT, {"ok": True})
                continue
            if msg.kind != protocol.FORWARD:
                protocol.send(conn, protocol.ERROR, {"code": "unexpected_message", "name": msg.name})
                continue

            n_tokens = int(msg.meta.get("n_tokens", 0))
            n_past = int(msg.meta.get("n_past", 0))
            try:
                if self._role == "gpu":
                    tokens = list(msg.meta["tokens"])
                    hidden = self._engine.forward_tokens(tokens, n_past=n_past)
                    protocol.send(
                        conn, protocol.RESULT,
                        {"n_tokens": hidden.shape[0], "n_embd": hidden.shape[1], "n_past": n_past},
                        hidden.astype(np.float32).tobytes(),
                    )
                else:
                    n_embd = self._engine.n_embd
                    if len(msg.blob) != n_tokens * n_embd * 4:
                        protocol.send(conn, protocol.ERROR, {
                            "code": "hidden_size_mismatch",
                            "expected": n_tokens * n_embd * 4,
                            "got": len(msg.blob),
                        })
                        continue
                    hidden = np.frombuffer(msg.blob, dtype=np.float32).reshape(n_tokens, n_embd)
                    if self._role == "tail":
                        token = self._engine.forward_token(hidden, n_past=n_past)
                        protocol.send(conn, protocol.RESULT, {"token": token, "n_past": n_past})
                    else:
                        out = self._engine.forward_hidden(hidden, n_past=n_past)
                        protocol.send(
                            conn, protocol.RESULT,
                            {"n_tokens": out.shape[0], "n_embd": out.shape[1], "n_past": n_past},
                            out.astype(np.float32).tobytes(),
                        )
            except Exception as exc:  # noqa: BLE001 - the peer must see the failure
                protocol.send(conn, protocol.ERROR, {"code": "forward_failed", "detail": str(exc)[:400]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=ROLES)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shim", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--n-cpu-moe", type=int, default=0)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--cpus", default="", help="comma-separated logical CPU ids to pin to")
    args = ap.parse_args()

    if args.cpus:
        try:
            import psutil

            allowed = [int(x) for x in args.cpus.split(",") if x.strip()]
            psutil.Process().cpu_affinity(allowed)
            print(f"[{args.role}] pinned to CPUs {allowed}", flush=True)
        except Exception as exc:  # noqa: BLE001 - pinning is best effort
            print(f"[{args.role}] cpu pin failed: {exc}", flush=True)

    engine = StageEngine(
        shim_path=Path(args.shim),
        model_path=Path(args.model),
        role=args.role,
        ctx=args.ctx,
        n_threads=args.threads,
        cut_layer=None,
        n_gpu_layers=args.n_gpu_layers,
        n_cpu_moe=args.n_cpu_moe,
    )
    try:
        # --layers is what the topology declares; the artifact is what actually runs.
        # A mismatch means the topology points at a cut for a different range, which
        # still loads and still produces plausible-but-wrong hidden states.
        if engine.n_layer != args.layers:
            print(f"[{args.role}] WARNING: declared {args.layers} layers but "
                  f"{Path(args.model).name} holds {engine.n_layer}",
                  file=sys.stderr, flush=True)
        print(f"[{args.role}] {engine.n_layer} layers, n_embd={engine.n_embd}",
              flush=True)
        StageWorker(engine, args.role).serve(args.host, args.port)
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
