"""Measure what the per-step wire path costs, with the model out of the way.

Three modes, each isolating one layer of the stack:

* ``protocol`` -- a loopback echo server that only speaks protocol.py. No model, no
  numpy conversion: the number is encode + sendall + recv + decode. Run it before and
  after a transport change and the delta is the copy work that was removed.
* ``worker``   -- a real tail stage behind a worker (2B topology), compared against the
  same stage driven through StageEngine in-process. The difference is the worker
  protocol's own cost -- the 11 ms that loopback showed on ROG.
* ``proxy``    -- the same worker, but the client connects through a TCP relay that
  forwards at most one MTU-sized chunk per 5 ms. That is the measured Tailscale
  behaviour (1280 B MTU, ~5 ms per packet) reproduced locally, so the payload-size
  effect of float16 is measurable without touching the ROG machine. It is a model of
  the link, not the link: the relay counts the chunks it forwards, and that count is
  printed so the assumption is visible.

``SA_WIRE_DTYPE`` selects the transport dtype for the worker/proxy modes; each mode
prints what it used.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import protocol  # noqa: E402
from chain_client import StageClient, spawn_worker  # noqa: E402
from topology import load  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
DEFAULT_SHIM = Path(r"H:\slimarc-proto\exp-c\shim\qlh_keep_head.dll")
GIB = 1 << 30
#: Every wire dtype the transport supports: float32 is the baseline, float16 the
#: requested experiment (rejected on output grounds, see docs/MEASUREMENTS.md §9), int16
#: the finer fixed-point alternative measured against it.
WIRE_MODES = ("float32", "float16", "int16")


def wire_dtype() -> str:
    try:
        import wire  # noqa: WPS433

        return wire.current_dtype()
    except Exception:  # noqa: BLE001
        return "float32"


def stats(samples: list[float]) -> str:
    a = np.asarray(samples, dtype=np.float64)
    return (f"median {np.median(a):7.3f}  mean {a.mean():7.3f}  "
            f"p10 {np.percentile(a, 10):7.3f}  p90 {np.percentile(a, 90):7.3f}")


# --------------------------------------------------------------------- protocol
class EchoServer(threading.Thread):
    """Answer FORWARD with RESULT, echoing the blob untouched."""

    def __init__(self, port: int) -> None:
        super().__init__(daemon=True)
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", port))
        self._sock.listen(1)
        self._sock.settimeout(10)
        self.stop_flag = threading.Event()

    def run(self) -> None:
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                msg = protocol.read(conn)
                if msg.kind == protocol.BYE:
                    return
                protocol.send(conn, protocol.RESULT, msg.meta, msg.blob)
        except protocol.ProtocolError:
            pass
        finally:
            conn.close()
            self._sock.close()


def bench_protocol(args: argparse.Namespace) -> int:
    server = EchoServer(args.proto_port)
    server.start()
    time.sleep(0.2)
    sock = socket.create_connection(("127.0.0.1", args.proto_port), timeout=5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    n_embd = 2048
    try:
        for dtype in WIRE_MODES:
            arr = np.random.default_rng(7).standard_normal((1, n_embd)).astype(
                {"float32": np.float32, "float16": np.float16}[dtype])
            blob = arr.tobytes()
            # Warm up the path once: the first round trip allocates the socket buffers.
            for _ in range(20):
                protocol.send(sock, protocol.FORWARD, {"n_tokens": 1}, blob)
                protocol.read(sock)
            samples = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                protocol.send(sock, protocol.FORWARD, {"n_tokens": 1}, blob)
                msg = protocol.read(sock)
                samples.append((time.perf_counter() - t0) * 1e6)
            got = len(msg.blob)
            print(f"  protocol echo {dtype:<8} blob={len(blob):5d} B -> {got:5d} B  "
                  f"{stats(samples)} us/step")
        protocol.send(sock, protocol.BYE)
    finally:
        sock.close()
        server.join(timeout=5)
    print("  (what this measures: frame encode, sendall, recv, meta+blob decode on a "
          "loopback socket; no model, no numpy->wire conversion)")
    _encode_cost(args)
    return 0


def _encode_cost(args: argparse.Namespace) -> None:
    """The copy work alone, isolated from the socket.

    "legacy" is the old three-step assembly (tobytes, then +blob, then +header); "new" is
    one join over a memoryview of the array. Timing both with the same payload says what
    the removed copies are worth in microseconds -- which is the honest size of this
    optimisation, since an 8 KB memcpy is far below one network RTT.
    """
    import json
    import struct
    import timeit

    import wire

    arr = np.random.default_rng(3).standard_normal((1, 2048)).astype(np.float32)
    meta = {"n_tokens": 1, "n_past": 0}

    def legacy() -> bytes:
        blob = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        head = json.dumps(meta, separators=(",", ":")).encode()
        payload = struct.pack(">I", len(head)) + head + blob
        return struct.pack(">IB", len(payload), 2) + payload

    def new() -> bytes:
        return protocol.encode(protocol.FORWARD, meta, wire.encode(arr, "float32"))

    n = 200_000
    for label, fn in (("legacy", legacy), ("new   ", new)):
        us = timeit.timeit(fn, number=n) / n * 1e6
        print(f"  encode {label} frame {len(fn()):5d} B  {us:6.3f} us/frame")
    print("  (legacy copies the 8 KB payload twice inside encode; new copies it once. "
          "Both are sub-microsecond, i.e. invisible against a network RTT.)")


# ----------------------------------------------------------------------- worker
def _tail_stage(topology_path: str):
    stages = load(topology_path)
    tail = stages[-1]
    if tail.kind != "tail":
        raise SystemExit(f"last stage of {topology_path} is {tail.kind}, not tail")
    return stages, tail


def bench_worker(args: argparse.Namespace) -> int:
    """Real tail worker vs the same stage in-process: the protocol's own cost.

    Runs both transport dtypes against the same loaded worker in one process. Switching
    SA_WIRE_DTYPE between loops is enough because wire.current_dtype() reads it per
    forward -- and it keeps the two measurements under the same CPU state, which a
    second process would not.
    """
    stages, tail = _tail_stage(args.topology)
    shim = Path(args.shim)
    # Real hidden from the head stage, not noise: MoE routing depends on the values, so
    # a benchmark on zeros would time a different graph than a decode step does.
    head = StageClient(stages[0])
    tailc = StageClient(tail)
    engine = None
    procs = []
    try:
        for s in stages:
            procs.append(spawn_worker(s, shim, args.ctx))
        for c in (head, tailc):
            c.connect()
            c.set_sampling(temperature=0.0)
        hidden = head.forward_tokens(head.tokenize("中国的首都是"), 0)
        hidden = np.ascontiguousarray(hidden[-1:], dtype=np.float32)
        print(f"  hidden {hidden.shape} {hidden.dtype}  norm={np.linalg.norm(hidden):.2f}")

        # In-process first. A second copy of the same artifact; the 2B tail is ~0.6 GB,
        # so this is the only way to subtract compute from the worker round trip.
        from engine import StageEngine

        engine = StageEngine(shim_path=shim, model_path=tail.model, role="tail",
                             ctx=args.ctx, n_threads=tail.threads, cut_layer=None,
                             n_gpu_layers=0)
        engine.reset()
        n_past = 4
        direct_samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            engine.forward_token(hidden, n_past=n_past)
            direct_samples.append((time.perf_counter() - t0) * 1000.0)
            n_past += 1
        direct_ms = float(np.median(direct_samples))
        print(f"  inproc  (StageEngine.forward_token, same artifact)  "
              f"{stats(direct_samples)} ms")

        for dtype in WIRE_MODES:
            os.environ["SA_WIRE_DTYPE"] = dtype
            tailc.reset()
            n_past = 4
            samples = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                tailc.forward_token(hidden, n_past)
                samples.append((time.perf_counter() - t0) * 1000.0)
                n_past += 1
            worker_ms = float(np.median(samples))
            print(f"  worker {dtype:<8} (StageClient.forward_token)  {stats(samples)} ms")
            print(f"      -> protocol cost {worker_ms - direct_ms:6.3f} ms/step "
                  f"({worker_ms:.3f} vs {direct_ms:.3f})")
    finally:
        os.environ.pop("SA_WIRE_DTYPE", None)
        for c in (head, tailc):
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        if engine is not None:
            engine.close()
        for proc, _log in procs:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                proc.kill()
    return 0


# ------------------------------------------------------------------------ proxy
class DelayProxy(threading.Thread):
    """TCP relay: forward at most `chunk` bytes per `delay` seconds, per direction.

    Recv-side chunking is what makes the packet count visible: a 8 KiB payload arrives
    as ceil(bytes/chunk) chunks, and the relay sleeps after each one, which is the
    per-packet RTT Tailscale showed (1280 B MTU, ~5 ms).
    """

    def __init__(self, upstream_port: int, listen_port: int, chunk: int, delay: float) -> None:
        super().__init__(daemon=True)
        self.upstream_port = upstream_port
        self.listen_port = listen_port
        self.chunk = chunk
        self.delay = delay
        self.chunks = {"up": 0, "down": 0}
        self.bytes = {"up": 0, "down": 0}
        self.ready = threading.Event()
        self._ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._ls.bind(("127.0.0.1", listen_port))
        self._ls.listen(1)
        self._ls.settimeout(15)

    def run(self) -> None:
        self.ready.set()
        try:
            conn, _ = self._ls.accept()
        except OSError:
            return
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 16)
        up = socket.create_connection(("127.0.0.1", self.upstream_port), timeout=10)
        up.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        t1 = threading.Thread(target=self._pump, args=(conn, up, "up"), daemon=True)
        t2 = threading.Thread(target=self._pump, args=(up, conn, "down"), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=120)
        t2.join(timeout=120)
        for s in (conn, up, self._ls):
            try:
                s.close()
            except OSError:
                pass

    def _pump(self, src: socket.socket, dst: socket.socket, tag: str) -> None:
        try:
            while True:
                data = src.recv(self.chunk)
                if not data:
                    break
                dst.sendall(data)
                self.chunks[tag] += 1
                self.bytes[tag] += len(data)
                if self.delay:
                    time.sleep(self.delay)
        except OSError:
            pass
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass


def bench_proxy(args: argparse.Namespace) -> int:
    """Same worker, same hidden, two transport dtypes, through the MTU/5ms relay.

    Loopback is measured once (it does not depend on the wire dtype) and each dtype gets
    its own relay instance so the chunk counters attribute only that dtype's bytes.
    """
    stages, tail = _tail_stage(args.topology)
    shim = Path(args.shim)
    head = StageClient(stages[0])
    procs = []
    try:
        for s in stages:
            procs.append(spawn_worker(s, shim, args.ctx))
        head.connect()
        head.set_sampling(temperature=0.0)
        prompt = head.tokenize("请把下面这句话翻译成英文：层段之间的hidden state每一步都要传一次。")
        hidden = head.forward_tokens(prompt, 0)
        hidden = np.ascontiguousarray(hidden[-1:], dtype=np.float32)
        print(f"  hidden {hidden.shape} {hidden.dtype}  norm={np.linalg.norm(hidden):.2f}")

        # Direct loopback first: the reference the relay is added to.
        direct = StageClient(tail)
        direct.connect()
        direct.set_sampling(temperature=0.0)
        direct.reset()
        n_past = len(prompt)
        samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            direct.forward_token(hidden, n_past)
            samples.append((time.perf_counter() - t0) * 1000.0)
            n_past += 1
        direct_ms = float(np.median(samples))
        print(f"  loopback           {stats(samples)} ms")
        direct.close()

        results = {}
        for i, dtype in enumerate(WIRE_MODES):
            os.environ["SA_WIRE_DTYPE"] = dtype
            proxy = DelayProxy(tail.port, args.proxy_port + i, args.chunk,
                               args.delay / 1000.0)
            proxy.start()
            proxy.ready.wait(5)
            relayed = StageClient(tail)
            relayed._addr = ("127.0.0.1", args.proxy_port + i)  # noqa: SLF001 - the point
            relayed.connect()
            relayed.set_sampling(temperature=0.0)
            relayed.reset()
            n_past = len(prompt)
            samples = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                relayed.forward_token(hidden, n_past)
                samples.append((time.perf_counter() - t0) * 1000.0)
                n_past += 1
            relayed.close()
            proxied_ms = float(np.median(samples))
            results[dtype] = proxied_ms
            print(f"  1280B/5ms {dtype:<8} {stats(samples)} ms   "
                  f"{proxy.chunks['up']}/{proxy.chunks['down']} chunks "
                  f"({proxy.chunks['up'] / max(args.iters, 1):.1f} up/step, "
                  f"{proxy.bytes['up'] / max(args.iters, 1):.0f} B up/step)")
        effect = ", ".join(f"{d} +{v - direct_ms:.2f} ms" for d, v in results.items())
        print(f"  -> payload effect over loopback: {effect}")
        if "float32" in results:
            for d, v in results.items():
                if d != "float32":
                    print(f"     {d} vs float32: {results['float32'] - v:+.2f} ms/step")
        print(f"  (the relay forwards <= {args.chunk} B per direction per "
              f"{args.delay:.0f} ms; packet count, not bandwidth, is the model)")
    finally:
        os.environ.pop("SA_WIRE_DTYPE", None)
        for c in (head,):
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        for proc, _log in procs:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                proc.kill()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("protocol", "worker", "proxy"))
    ap.add_argument("--topology", default="topology-2b.toml")
    ap.add_argument("--shim", default=str(DEFAULT_SHIM))
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--proto-port", type=int, default=9471)
    ap.add_argument("--proxy-port", type=int, default=9472)
    ap.add_argument("--chunk", type=int, default=1240,
                    help="bytes forwarded per direction per delay tick (1280 MTU - 40)")
    ap.add_argument("--delay", type=float, default=5.0, help="ms per chunk")
    args = ap.parse_args()
    print(f"mode={args.mode} wire={wire_dtype()}")
    if args.mode == "protocol":
        return bench_protocol(args)
    if args.mode == "worker":
        return bench_worker(args)
    return bench_proxy(args)


if __name__ == "__main__":
    raise SystemExit(main())
