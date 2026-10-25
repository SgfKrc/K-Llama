#!/usr/bin/env python
"""Reproduce COMM-WIRE-01 framing costs and optional 32-step native A/B.

Uses production task-worker validation, negotiation and TCP framing. The wire
microbenchmark uses loopback TCP, not a cluster or end-to-end generation TPS.
Native mode reuses two verified proper GGUF segments without copying models.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import time
import base64

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from task_worker_protocol import (build_message, decode_message, split_inline_hidden,
                                  restore_inline_hidden, stage_input_sha256,
                                  stage_output_sha256, WorkerProtocolError)
from task_worker_adapter import TaskWorkerControlPlane
from tcp_comm import TCPClient, MessageType, build_message as build_frame

SCHEMA = "qlh.comm_wire_bench.v1"


class BenchError(ValueError):
    pass


@contextmanager
def _master_switch(enabled):
    key = "QLH_TASK_WORKER_BINARY_HIDDEN"
    previous = os.environ.get(key)
    os.environ[key] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


class LoopbackLink:
    """Real TCP sockets, with receive on another thread to fit large frames."""
    def __init__(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(30)
            self.sender = socket.create_connection(listener.getsockname(), timeout=30)
            try:
                self.receiver, _ = listener.accept()
            except BaseException:
                self.sender.close()
                raise
        for stream in (self.sender, self.receiver):
            stream.settimeout(30)
            stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reader = TCPClient(client_id="comm-wire-reader")
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.closed = False

    def exchange(self, packet, *, reverse=False):
        sender, receiver = ((self.receiver, self.sender) if reverse
                            else (self.sender, self.receiver))
        future = self.executor.submit(self.reader.recv_data, receiver)
        try:
            sender.sendall(packet)
            frame = future.result(timeout=35)
        except BaseException:
            self.close()
            raise
        if frame is None:
            raise BenchError("TCP closed before a complete frame")
        return frame

    def close(self):
        if self.closed:
            return
        self.closed = True
        for stream in (self.sender, self.receiver):
            try:
                stream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            stream.close()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class Peer:
    """Negotiate the real control plane over TCP; no transport mocks."""
    def __init__(self, coordinator, name, advertised, *, master_enabled=True):
        self.coordinator, self.name = coordinator, name
        self.advertised, self.master_enabled = advertised, master_enabled
        self.worker = TaskWorkerControlPlane()
        self.generation = 0
        self.link = None
        self.connect()

    def connect(self):
        self.generation += 1
        self.link = LoopbackLink()
        try:
            self._negotiate()
        except BaseException:
            self.worker.disconnect_coordinator()
            self.coordinator.disconnect_worker(self.name)
            self.link.close()
            raise

    def _negotiate(self):
        self.owner = (self.link.receiver, self.generation, self.generation)
        self.coordinator.mark_worker_connection_pending(
            self.name, connection_generation=self.generation)
        caps = {"stage_types": ["layer_forward"], "engines": ["llama_cpp"],
                "models": [], "max_concurrency": 1}
        if self.advertised is not None:
            caps["binary_hidden"] = self.advertised
        hello = self.worker.begin_worker_hello(
            node_id=self.name, capabilities=caps, registration_owner=self.owner)
        incoming = self.link.exchange(build_frame(MessageType.TASK_WORKER, hello.snapshot()))
        with _master_switch(self.master_enabled):
            ack = self.coordinator.receive_on_coordinator(
                self.name, incoming["data"], coordinator_node_id="comm_wire_master",
                connection_generation=self.generation)
        incoming = self.link.exchange(build_frame(MessageType.TASK_WORKER, ack.snapshot()),
                                      reverse=True)
        self.worker.receive_on_worker(incoming["data"], registration_owner=self.owner)
        self.coordinator.resolve_worker_connection_pending(
            self.name, connection_generation=self.generation)
        self.last_ack = ack.snapshot()
        self.enabled = self.worker.worker_binary_hidden_enabled(self.owner)
        if (not ack.payload["accepted"] or self.enabled != (
                self.coordinator.worker_snapshot(self.name).get("binary_hidden") is True)):
            raise BenchError("both peers must agree on binary negotiation")

    def reconnect(self):
        old_owner, old_ack = self.owner, self.last_ack
        self.worker.disconnect_coordinator()
        self.coordinator.disconnect_worker(self.name)
        if self.worker.worker_binary_hidden_enabled(old_owner):
            raise BenchError("disconnected owner kept binary capability")
        self.link.close()
        self.connect()
        if self.worker.worker_binary_hidden_enabled(old_owner):
            raise BenchError("replaced socket kept binary capability")
        try:
            self.worker.receive_on_worker(old_ack, registration_owner=old_owner)
        except WorkerProtocolError as exc:
            if exc.code != "stale_hello_ack":
                raise
        else:
            raise BenchError("retired owner ACK was accepted")
        return {"generation": self.generation, "binary_hidden": self.enabled,
                "retired_owner_enabled": False, "retired_ack": "stale_hello_ack"}

    def close(self):
        self.worker.disconnect_coordinator()
        self.coordinator.disconnect_worker(self.name)
        self.link.close()


def _message(raw, kind="stage_offer", *, n_embd=4096, n_past=0,
             model_identity=None, layer_range=(28, 32), binary_hidden=False):
    n_tokens = len(raw) // (n_embd * 4)
    if n_tokens < 1 or len(raw) != n_tokens * n_embd * 4:
        raise BenchError("hidden bytes must fit a nonempty float32 matrix")
    identity = {"workflow_id": "wf_commwire0001", "stage_id": "tail_segment",
                "attempt_id": "att_commwire0001", "lease_id": "lease_commwire0001",
                "lease_epoch": 1, "provider_id": "remote_comm_wire_worker"}
    encoded, digest = base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()
    if kind == "stage_offer":
        root = {"hidden_f32": encoded, "want_hidden": False}
        payload = {**identity, "request_id": "req_commwire0001", "stage_type": "layer_forward",
                   "lease_expires_at_ms": int(time.time() * 1000) + 600_000,
                   "root_input": root, "dependencies": {},
                   "input_sha256": stage_input_sha256(root, {}),
                   "model_identity": model_identity or {
                       "model_id": "comm_wire_fixture", "engine": "llama_cpp", "format": "gguf",
                       "revision": "wire_only", "sha256": "b" * 64},
                   "layer_range": list(layer_range), "handoff_at": layer_range[0],
                   "hidden_sha256": digest,
                   "hidden_spec": {"n_tokens": n_tokens, "n_embd": n_embd, "dtype": "float32"},
                   "positions": list(range(n_past, n_past + n_tokens)), "seq_ids": [0] * n_tokens}
    elif kind == "stage_result":
        output = {"hidden_out_f32": encoded, "hidden_out_sha256": digest}
        payload = {**identity, "output": output, "output_sha256": stage_output_sha256(output),
                   "metadata": {}}
    elif kind in ("stage_chunk", "stage_result_chunk"):
        payload = {**identity, "chunk_index": 0, "chunk_count": 1,
                   "payload_b64": encoded, "payload_sha256": digest, "total_bytes": len(raw)}
    else:
        raise BenchError(f"unsupported fixture kind: {kind}")
    return build_message(kind, payload, message_id="msg_commwire0001",
                         sent_at_ms=1, binary_hidden=binary_hidden)


def _process_cycles():
    """Windows CPU cycles avoid GetProcessTimes' coarse small-frame granularity."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    query = getattr(_process_cycles, "query", None)
    if query is None:
        query = ctypes.WinDLL("kernel32", use_last_error=True).QueryProcessCycleTime
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_ulonglong)]
        query.restype = wintypes.BOOL
        _process_cycles.query = query
    value = ctypes.c_ulonglong()
    if not query(wintypes.HANDLE(-1), ctypes.byref(value)):
        raise ctypes.WinError(ctypes.get_last_error())
    return value.value


def transfer(peer, raw, kind="stage_offer", **fields):
    """Include existing base64 staging, validation, framing and restoration."""
    cycles = _process_cycles()
    wall, cpu = time.perf_counter_ns(), time.process_time_ns()
    message = _message(raw, kind, binary_hidden=peer.enabled, **fields)
    original = message.snapshot()
    snapshot, body = split_inline_hidden(original, enabled=peer.enabled)
    packet = build_frame(MessageType.TASK_WORKER, snapshot, binary_body=body)
    frame = peer.link.exchange(packet, reverse=kind in ("stage_result", "stage_result_chunk"))
    if frame["format"] != ("binary" if peer.enabled else "json"):
        raise BenchError("wire format differs from negotiation")
    restored = restore_inline_hidden(frame["data"], frame.get("binary"))
    decoded = decode_message(restored, binary_hidden=frame.get("binary") is not None)
    payload = decoded.payload
    if kind == "stage_offer":
        restored_b64 = payload["root_input"]["hidden_f32"]
    elif kind == "stage_result":
        restored_b64 = payload["output"]["hidden_out_f32"]
    else:
        restored_b64 = payload["payload_b64"]
    received = base64.b64decode(restored_b64, validate=True)
    timing = {"wall_ms": (time.perf_counter_ns() - wall) / 1e6,
              "process_cpu_ms": (time.process_time_ns() - cpu) / 1e6,
              "frame_bytes": len(packet), "hidden_raw_bytes": len(raw),
              "hidden_wire_bytes": len(body) if body is not None else len(restored_b64),
              "format": frame["format"]}
    if cycles is not None:
        timing["process_cpu_cycles"] = _process_cycles() - cycles
    # Evidence assertions are outside the measured production codec path.
    if restored != original:
        raise BenchError("message metadata changed in production frame roundtrip")
    if received != raw:
        raise BenchError("hidden bytes changed in production frame roundtrip")
    return received, timing


def _summary(samples):
    keys = ["wall_ms", "process_cpu_ms", "frame_bytes", "hidden_wire_bytes"]
    if all("process_cpu_cycles" in row for row in samples):
        keys.append("process_cpu_cycles")
    return {key: {"median": statistics.median(row[key] for row in samples),
                  "mean": statistics.mean(row[key] for row in samples),
                  "stdev": statistics.stdev(row[key] for row in samples) if len(samples) > 1 else 0,
                  "min": min(row[key] for row in samples),
                  "max": max(row[key] for row in samples)} for key in keys}


def wire_bench(*, samples=21, warmup=3, sizes=(16384, 65536, 1048576)):
    coordinator = TaskWorkerControlPlane()
    peers = []
    try:
        for name, advertised, enabled in (("wire_new", True, True), ("wire_old", None, True),
                                          ("wire_disabled", True, False)):
            peers.append(Peer(coordinator, name, advertised, master_enabled=enabled))
        checks = []
        for peer in peers:
            for kind in ("stage_offer", "stage_result", "stage_chunk", "stage_result_chunk"):
                _, row = transfer(peer, bytes(16384), kind)
                checks.append({"peer": peer.name, "kind": kind, **row})
        # Old and new peers remain simultaneously registered in one coordinator.
        if [p.enabled for p in peers] != [True, False, False]:
            raise BenchError("mixed capability peers selected unexpected formats")
        reconnects = [peers[0].reconnect()]
        peers[0].master_enabled = False
        reconnects.append(peers[0].reconnect())
        _, disabled_after_reconnect = transfer(peers[0], bytes(16384))
        if disabled_after_reconnect["format"] != "json":
            raise BenchError("reconnect with disabled master did not fall back")
        peers[0].master_enabled = True
        reconnects.append(peers[0].reconnect())
        cases = []
        for size in sizes:
            raw = bytes((index % 251 for index in range(size)))
            for kind in ("stage_offer", "stage_result", "stage_chunk", "stage_result_chunk"):
                rows = {True: [], False: []}
                for iteration in range(warmup + samples):
                    # Alternate pair order to reduce temperature/order bias.
                    for binary in ((True, False) if iteration % 2 else (False, True)):
                        _, row = transfer(peers[0 if binary else 1], raw, kind)
                        if iteration >= warmup:
                            rows[binary].append(row)
                cases.append({"kind": kind, "raw_bytes": size,
                              "binary": {"summary": _summary(rows[True]), "samples": rows[True]},
                              "json": {"summary": _summary(rows[False]), "samples": rows[False]}})
                print(f"wire {kind} {size}: {samples} pairs", flush=True)
        return {"passed": True, "transport": "IPv4 TCP loopback; TCP_NODELAY",
                "samples_per_format": samples, "warmup_per_format": warmup,
                "scope": "base64 staging + message validation + framing + TCP + restore + validation + raw decode",
                "mixed_peers": checks, "reconnects": reconnects, "cases": cases}
    finally:
        for peer in peers:
            peer.close()


def compare_passes(passes, steps):
    """Incomplete or divergent runs must never produce a numerical gate pass."""
    names = [item["name"] for item in passes]
    if (steps < 32 or sorted(names) != ["binary", "json", "mixed"]
            or any(len(item["steps"]) != steps for item in passes)):
        raise BenchError("numerical gate requires three complete json/binary/mixed >=32-step passes")
    for item in passes:
        for index, row in enumerate(item["steps"]):
            binary = item["name"] == "binary" or (item["name"] == "mixed" and index % 2 == 1)
            if row["wire"]["format"] != ("binary" if binary else "json"):
                raise BenchError("numerical gate did not exercise the required wire formats")
    reference = passes[0]["steps"]
    for candidate in passes[1:]:
        for index, (left, right) in enumerate(zip(reference, candidate["steps"])):
            if left["token_argmax"] != right["token_argmax"]:
                raise BenchError(f"argmax differs in {candidate['name']} at step {index}")
            if left["hidden_sha256"] != right["hidden_sha256"]:
                raise BenchError(f"upstream hidden differs in {candidate['name']} at step {index}")
    return {"passed": True, "steps_per_pass": steps,
            "argmax_equal": True, "hidden_bytes_equal": True,
            "reference": passes[0]["name"], "compared": [p["name"] for p in passes[1:]]}


def _native_contract(runner, info, declaration, vocabulary):
    import ctypes
    if getattr(runner, "_worker", None) is not None:
        raise BenchError("native gate requires direct handles in this dedicated benchmark process")
    if not (declaration["hidden_size"] == info["actual"]["n_embd"] == runner.n_embd):
        raise BenchError("manifest/header/native hidden dimensions differ")
    if runner.n_layer != info["layers"]:
        raise BenchError("native layer count differs from verified segment")
    lib = runner._lib
    if not all(hasattr(lib, name) for name in ("qlh_kh_n_vocab", "qlh_kh_finite_logits_guard")):
        raise BenchError("native gate requires vocabulary and finite-logits guards; rebuild shim")
    lib.qlh_kh_n_vocab.argtypes = [ctypes.c_void_p]
    lib.qlh_kh_n_vocab.restype = ctypes.c_int32
    lib.qlh_kh_finite_logits_guard.argtypes = []
    lib.qlh_kh_finite_logits_guard.restype = ctypes.c_int32
    native_vocab = int(lib.qlh_kh_n_vocab(runner._handle))
    if native_vocab <= 0 or native_vocab != vocabulary or lib.qlh_kh_finite_logits_guard() != 1:
        raise BenchError("native vocabulary/finite-logits guard differs from the required contract")
    return {"n_embd": runner.n_embd, "n_layer": runner.n_layer,
            "native_n_vocab": native_vocab, "finite_logits_guard": True}


def native_bench(args, progress):
    import numpy as np
    from edge_bench import _inspect, _protected_model, _sha256, _json, _expect
    from keep_head_shim import resolve_keep_head_shim
    from llama_keep_head import KeepHeadUpstream

    if args.steps < 32:
        raise BenchError("--steps must be >=32 for the numerical gate")
    # edge_bench protects Windows inputs with FILE_SHARE_READ only. Its POSIX
    # helper copies the artifact, so refuse that branch for this no-copy tool.
    if os.name != "nt":
        raise BenchError("native no-copy file protection currently requires Windows")
    with ExitStack() as stack:
        models = [stack.enter_context(_protected_model(Path(path)))
                  for path in (args.head, args.tail)]
        manifests = [Path(args.head_manifest), Path(args.tail_manifest)]
        expectations = [_expect(args.head_expect), _expect(args.tail_expect)]
        info = [_inspect(path, manifest, expect, bench=True)
                for path, manifest, expect in zip(models, manifests, expectations)]
        declarations = [_json(path) for path in manifests]
        progress.update(segments=info, segment_declarations=declarations,
                        explicit_expectations=expectations, passes=[])
        if any(item["differences"] for item in info):
            raise BenchError("segment metadata differs from manifest")
        head, tail = info
        if (head["role"] != "head" or tail["role"] != "tail"
                or head["layer_range"][1] != tail["layer_range"][0]
                or head["total_layers"] != tail["total_layers"]):
            raise BenchError("head and tail must cover contiguous proper decoder ranges")
        for key in ("source_model_sha256", "tokenizer_sha256", "architecture", "hidden_size"):
            if not declarations[0].get(key) or declarations[0][key] != declarations[1].get(key):
                raise BenchError(f"segment {key} identities differ or are missing")
        shim = Path(args.shim or resolve_keep_head_shim()).resolve()
        binaries = {str(path): _sha256(path) for path in sorted(shim.parent.glob("*.dll"))}
        progress.update(segments=info, segment_declarations=declarations,
                        binaries=binaries, shim=str(shim), passes=[])
        # Keep pip's tokenizer DLL group out of the process owning patched
        # native handles. This also permits checking the actual loaded guard.
        tokenizer_code = """import json,sys
import llama_cpp
model = llama_cpp.Llama(model_path=sys.argv[1], vocab_only=True, verbose=False)
try:
    print(json.dumps(dict(tokens=model.tokenize(sys.argv[2].encode('utf-8'), add_bos=True),
                          vocabulary=model.n_vocab(), eos=model.token_eos(),
                          binding_version=llama_cpp.__version__)))
finally:
    model.close()
"""
        child = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", tokenizer_code,
                                str(models[0]), args.prompt], text=True, capture_output=True,
                               check=True, timeout=60)
        tokenizer = json.loads(child.stdout)
        tokens, vocabulary, eos = tokenizer["tokens"], tokenizer["vocabulary"], tokenizer["eos"]
        progress["tokenizer"] = tokenizer
        if not tokens or len(tokens) + args.steps > args.ctx:
            raise BenchError("prompt and all decode steps must fit context")
        progress.update(prompt=args.prompt, prompt_tokens=tokens, vocabulary_size=vocabulary,
                        eos_token=eos, steps=args.steps, n_ctx=args.ctx, n_threads=args.threads,
                        scope="fixed-length greedy continuation; EOS is recorded and does not stop this numerical probe")
        if any(name == "llama_cpp" or name.startswith("llama_cpp.") for name in sys.modules):
            raise BenchError("patched native benchmark process must not load pip llama_cpp")
        runners = []
        for model in models:
            runner = KeepHeadUpstream(shim, model, mode="nextn", n_ctx=args.ctx,
                                      n_threads=args.threads, n_batch=args.ctx, n_seq_max=1,
                                      _worker_process=True)
            runners.append(runner)
            stack.callback(runner.close)
        progress["native_contracts"] = [
            _native_contract(runner, item, declaration, vocabulary)
            for runner, item, declaration in zip(runners, info, declarations)]
        progress["materialization"] = [runner.materialization_observation() for runner in runners]
        coordinator = TaskWorkerControlPlane()
        peers = []
        for name, advertised in (("native_json", None), ("native_binary", True)):
            peer = Peer(coordinator, name, advertised)
            stack.callback(peer.close)
            peers.append(peer)
        identity = {"model_id": declarations[0]["source_model_id"], "engine": "llama_cpp",
                    "format": "gguf", "revision": "comm_wire_ab",
                    "sha256": declarations[0]["source_model_sha256"]}
        for mode in ("json", "binary", "mixed"):
            for runner in runners:
                runner.reset()
            current, past = list(tokens), 0
            result = {"name": mode, "steps": []}
            progress["passes"].append(result)
            for step in range(args.steps):
                started = time.perf_counter_ns()
                hidden = runners[0].forward_tokens_to_hidden(current, n_past=past)
                if not np.isfinite(hidden).all():
                    raise BenchError("native upstream returned non-finite hidden")
                raw = np.asarray(hidden, dtype="<f4", order="C").tobytes()
                binary = mode == "binary" or (mode == "mixed" and step % 2 == 1)
                received, timing = transfer(peers[int(binary)], raw,
                    n_embd=runners[0].n_embd, n_past=past, model_identity=identity,
                    layer_range=tail["layer_range"])
                incoming = np.frombuffer(received, dtype="<f4").reshape(hidden.shape)
                token = runners[1].forward_hidden_to_token(incoming, n_past=past,
                    positions=list(range(past, past + len(current))), seq_ids=[0] * len(current))
                if type(token) is not int or not 0 <= token < vocabulary:
                    raise BenchError("native argmax outside verified vocabulary")
                result["steps"].append({"index": step, "n_past": past,
                    "input_tokens": list(current), "token_argmax": token,
                    "hidden_sha256": hashlib.sha256(raw).hexdigest(), "wire": timing,
                    "step_wall_ms": (time.perf_counter_ns() - started) / 1e6})
                past += len(current)
                current = [token]
                if (step + 1) % 8 == 0:
                    print(f"native {mode}: {step + 1}/{args.steps}", flush=True)
        progress["gate"] = compare_passes(progress["passes"], args.steps)
        if binaries != {str(path): _sha256(path) for path in sorted(shim.parent.glob("*.dll"))}:
            raise BenchError("native binaries changed during the comparison")
        return progress


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--head")
    parser.add_argument("--head-manifest")
    parser.add_argument("--tail")
    parser.add_argument("--tail-manifest")
    parser.add_argument("--head-expect", help="explicit missing metadata; cannot override manifest")
    parser.add_argument("--tail-expect", help="explicit missing metadata; cannot override manifest")
    parser.add_argument("--shim")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prompt", default="Q: What is two plus two? Explain briefly.\nA:")
    args = parser.parse_args(argv)
    if args.samples < 1 or args.warmup < 0 or args.threads < 1 or args.ctx < 1:
        parser.error("samples/threads/ctx must be positive and warmup nonnegative")
    if args.native and not all((args.head, args.head_manifest, args.tail, args.tail_manifest)):
        parser.error("--native requires --head/--tail and both manifests")
    if args.json_out.suffix.lower() != ".json" or args.json_out.resolve() in {
            Path(path).resolve() for path in (args.head, args.tail, args.head_manifest,
                                               args.tail_manifest, args.shim) if path}:
        parser.error("--json-out must be a JSON report distinct from native inputs")
    from edge_bench import _sha256, _environment
    sources = ("scripts/comm_wire_bench.py", "scripts/edge_bench.py",
               "src/tcp_comm.py", "src/task_worker_protocol.py", "src/task_worker_adapter.py",
               "src/llama_keep_head.py", "src/llama_keep_head_worker.py")
    report = {"schema": SCHEMA, "started_at": datetime.now(timezone.utc).isoformat(),
              "invocation": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
              "source_sha256": {path: _sha256(ROOT / path) for path in sources},
              "resource_snapshot": _environment(),
              "environment": {"platform": platform.platform(), "python": sys.version,
                              "executable": sys.executable, "cpu_count": os.cpu_count()},
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                  text=True).strip(), "passed": False}
    code = 1
    try:
        report["wire"] = wire_bench(samples=args.samples, warmup=args.warmup)
        if args.native:
            report["native"] = {}
            native_bench(args, report["native"])
        report["passed"] = True
        code = 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
