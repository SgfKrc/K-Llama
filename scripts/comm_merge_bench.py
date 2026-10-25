#!/usr/bin/env python
"""COMM-MERGE-01: real provider/mixin/TCP loopback, injected model admission.

Run baseline and candidate in separate processes against their production source.
No model, service, external network, or production monkeypatch is used. Timings
include observer overhead; post-offer-send time is not an isolated json.dumps cost.
"""
from __future__ import annotations

import argparse
import base64
import collections
from contextlib import contextmanager
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from execution_provider import ModelIdentity, StageAttempt, StageRequest
from scheduler_task_worker import SchedulerTaskWorkerMixin
from scheduler_types import NodeRole
from task_worker_adapter import RemoteFullWorkerProvider, TaskWorkerControlPlane
from task_worker_protocol import MAX_STAGE_PAYLOAD_BYTES, split_inline_hidden
from tcp_comm import MessageType, TCPClient

SCHEMA = "qlh.comm_merge_bench.v1"
WORKER = "comm_merge_worker"
MODEL = ModelIdentity("comm_merge_fixture", "llama_cpp", "gguf", "injected", "b" * 64)
TOKEN = 42


class BenchError(RuntimeError):
    pass


class Trace:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def mark(self, event, **fields):
        with self.lock:
            self.events.append(dict(event=event, ns=time.perf_counter_ns(),
                                    thread=threading.current_thread().name, **fields))


class Endpoint(TCPClient):
    """Observe calls to the unmodified production send_data implementation."""
    def __init__(self, side, stream, trace):
        super().__init__(client_id="comm-merge-" + side)
        self.side, self.sock, self.trace = side, stream, trace
        # TCP registration/admission is fixture state, hello negotiation is real.
        self._registered = True
        self._registration_epoch = 1

    def send_data(self, data, msg_type=MessageType.TENSOR, **kwargs):
        fields = dict(side=self.side, message_type=data.get("message_type"),
                      message_id=data.get("message_id"),
                      attempt_id=data.get("payload", {}).get("attempt_id"),
                      binary_bytes=len(kwargs.get("binary_body") or b""))
        if fields["message_type"] == "stage_offer":
            fields["chunked"] = "hidden_ref" in data["payload"]["root_input"]
        self.trace.mark("frame_send_start", **fields)
        if fields["message_type"] == "stage_accept":
            self.trace.mark("accept_write_start", **fields)
        try:
            return super().send_data(data, msg_type, **kwargs)
        finally:
            if fields["message_type"] == "stage_accept":
                self.trace.mark("accept_write_end", **fields)
            self.trace.mark("frame_send_end", **fields)


class ObservedProvider(RemoteFullWorkerProvider):
    def __init__(self, trace, **kwargs):
        self.trace = trace
        super().__init__(**kwargs)

    def handle_message(self, raw, **kwargs):
        fields = dict(message_type=raw.get("message_type"),
                      attempt_id=raw.get("payload", {}).get("attempt_id"))
        self.trace.mark("provider_response_start", **fields)
        try:
            return super().handle_message(raw, **kwargs)
        finally:
            self.trace.mark("provider_response_end", **fields)


class Fixture(SchedulerTaskWorkerMixin):
    """Minimal scheduler; prepared/model/deployment admission is injected."""
    def __init__(self, side, link):
        self.side, self.link = side, link
        self._task_worker_control = TaskWorkerControlPlane()
        self._task_worker_stage_lock = threading.RLock()
        self._task_worker_refresh_lock = threading.RLock()
        self._task_worker_refresh_requested = False
        self._task_worker_active_attempts = {}
        self._task_worker_seen_messages = {}
        self._task_worker_seen_order = collections.deque()
        self._remote_task_worker_providers = {}
        self._nodes_lock = threading.RLock()
        self.nodes = {WORKER: SimpleNamespace(node_type="pc", role=NodeRole.CLIENT)}
        self._tcp_server = None
        self._tcp_client = link.worker_tcp if side == "worker" else None
        self._host = SimpleNamespace(full_chat_execution_lock=threading.Lock())
        self._inference_lock = threading.Lock()
        self._worker_prepare = SimpleNamespace(check_stage=self._prepared)

    def _effective_role(self):
        return "master" if self.side == "master" else "client"

    def get_effective_node_id(self):
        return "comm_merge_master" if self.side == "master" else WORKER

    def get_distributed_inference_enabled(self):
        return True

    def _runtime_setting(self, name):
        return name == "TASK_WORKER_EXPERIMENTAL_ENABLED"

    def _assignment_connection_generation(self, _node):
        return 1

    def _queue_task_worker_hello_followup(self, *_args):
        # Service scheduling/deployment is outside this loopback experiment.
        pass

    def _send_task_worker_to_node(self, _node, message):
        snapshot, binary = split_inline_hidden(
            message.snapshot(), enabled=self._task_worker_control.worker_snapshot(
                WORKER).get("binary_hidden") is True)
        kwargs = {} if binary is None else {"binary_body": binary}
        self.link.master_tcp.send_data(snapshot, MessageType.TASK_WORKER, **kwargs)

    def _prepared(self, offer):
        if offer["model_identity"] != MODEL.snapshot() or offer["layer_range"] != [0, 1]:
            raise BenchError("injected prepared identity mismatch")
        self.link.trace.mark("injected_prepare_check", attempt_id=offer["attempt_id"])

    def _require_callbacks(self):
        return SimpleNamespace(execute_task_worker_stage=self.link.compute)

    def _handle_task_worker_stage_offer(self, message):
        self.link.own_current_thread()
        attempt_id = message.payload["attempt_id"]
        try:
            return super()._handle_task_worker_stage_offer(message)
        finally:
            self.link.trace.mark("worker_stage_end", attempt_id=attempt_id)
            self.link.stage_done.set()

    def _watch_task_worker_lease(self, attempt_id):
        self.link.own_current_thread()
        return super()._watch_task_worker_lease(attempt_id)


class Loopback:
    def __init__(self, *, binary, compute_delay_ms, master_accept_delay_ms, timeout):
        self.trace = Trace()
        self.stop = threading.Event()
        self.stage_done = threading.Event()
        self.response_done = threading.Event()
        self.hello_done = threading.Event()
        self.threads = []
        self.threads_lock = threading.Lock()
        self.errors = []
        self.compute_delay_ms = compute_delay_ms
        self.master_accept_delay_ms = master_accept_delay_ms
        self.timeout = timeout
        self.closed = False
        self.streams = []
        self.provider = None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                listener.settimeout(timeout)
                sender = socket.create_connection(listener.getsockname(), timeout=timeout)
                self.streams.append(sender)
                receiver, _ = listener.accept()
                self.streams.append(receiver)
            for stream in self.streams:
                stream.settimeout(timeout)
                stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.master_tcp = Endpoint("master", sender, self.trace)
            self.worker_tcp = Endpoint("worker", receiver, self.trace)
            self.master, self.worker = Fixture("master", self), Fixture("worker", self)
            self.provider = ObservedProvider(
                self.trace, node_id=WORKER,
                peer_snapshot=lambda: self.master._task_worker_control.worker_snapshot(WORKER),
                send_message=lambda message: self.master._send_task_worker_to_node(WORKER, message),
                accept_timeout_seconds=timeout)
            self.master._remote_task_worker_providers[WORKER] = self.provider
            self.master._task_worker_control.mark_worker_connection_pending(
                WORKER, connection_generation=1)
            for side in ("master", "worker"):
                thread = threading.Thread(target=self.receive, args=(side,),
                                          name="comm-merge-recv-" + side, daemon=True)
                self.threads.append(thread)
                thread.start()
            caps = dict(stage_types=["layer_forward"], engines=["llama_cpp"],
                        models=[MODEL.snapshot()], max_concurrency=1,
                        layer_ranges=[[0, 1]], stage_chunked_input=True, binary_hidden=binary)
            hello = self.worker._task_worker_control.begin_worker_hello(
                node_id=WORKER, capabilities=caps,
                registration_owner=self.worker_tcp.capture_registration_owner())
            self.worker._send_task_worker_to_master(hello)
            self.wait(self.hello_done, "hello negotiation")
            coordinator = self.worker._task_worker_control.coordinator_snapshot()
            if not coordinator.get("accepted"):
                raise BenchError("production hello was rejected")
            self.binary = self.worker._task_worker_binary_hidden_enabled()
            if self.binary != binary:
                raise BenchError("binary negotiation disagrees with requested switch")
        except BaseException:
            self.close()
            raise

    def own_current_thread(self):
        with self.threads_lock:
            self.threads.append(threading.current_thread())

    def receive(self, side):
        endpoint = self.master_tcp if side == "master" else self.worker_tcp
        fixture = self.master if side == "master" else self.worker
        try:
            while not self.stop.is_set():
                start = time.perf_counter_ns()
                frame = endpoint.recv_data()
                if frame is None:
                    if not self.stop.is_set():
                        raise BenchError("unexpected loopback EOF")
                    return
                raw = frame["data"]
                fields = dict(side=side, message_type=raw["message_type"],
                              message_id=raw["message_id"],
                              attempt_id=raw.get("payload", {}).get("attempt_id"))
                self.trace.mark("frame_decoded", read_started_ns=start, **fields)
                self.trace.mark("frame_receive_end", binary_bytes=len(frame.get("binary") or b""),
                                **fields)
                if side == "master" and raw["message_type"] == "stage_accept":
                    self.trace.mark("master_accept_delay_start", **fields)
                    self.stop.wait(self.master_accept_delay_ms / 1000)
                    self.trace.mark("master_accept_delay_end", **fields)
                fixture._handle_task_worker_message(WORKER if side == "master" else "master", frame)
                if raw["message_type"] == "hello_ack":
                    self.hello_done.set()
                if side == "master" and raw["message_type"] in {"stage_result", "stage_error"}:
                    self.response_done.set()
        except BaseException as exc:
            if not self.stop.is_set():
                self.errors.append(f"{side}: {type(exc).__name__}: {exc}")
                self.provider.notify_disconnect()

    def wait(self, event, description):
        if not event.wait(self.timeout):
            raise BenchError(f"timed out waiting for {description}; {self.errors}")
        if self.errors:
            raise BenchError(str(self.errors))

    def compute(self, request, cancel):
        attempt_id = "att_merge_" + request.request_id.zfill(8)
        self.trace.mark("executor_start", attempt_id=attempt_id)
        try:
            raw = base64.b64decode(request.root_input["hidden_f32"], validate=True)
            if hashlib.sha256(raw).hexdigest() != request.stage_fields["hidden_sha256"]:
                raise BenchError("callback received different hidden")
            if cancel.wait(self.compute_delay_ms / 1000):
                raise BenchError("callback cancelled")
            return {"token_argmax": TOKEN}
        finally:
            self.trace.mark("executor_end", attempt_id=attempt_id)

    def sample(self, index, raw):
        self.stage_done.clear()
        self.response_done.clear()
        attempt_id = f"att_merge_{index:08d}"
        request = StageRequest(
            workflow_id=f"wf_merge_{index:08d}", request_id=str(index), stage_id="tail",
            stage_type="layer_forward", provider_id=self.provider.provider_id,
            dependencies={}, root_input=dict(hidden_f32=base64.b64encode(raw).decode("ascii"),
                                             want_hidden=False, pos_base=0),
            model_identity=MODEL,
            stage_fields=dict(layer_range=[0, 1], handoff_at=1,
                              hidden_sha256=hashlib.sha256(raw).hexdigest(),
                              hidden_spec=dict(n_tokens=1, n_embd=len(raw) // 4, dtype="float32")))
        reservation = self.provider.reserve(request)
        attempt = StageAttempt(attempt_id, request, self.provider.provider_id,
                               lease_id=f"lease_merge_{index:08d}", lease_epoch=1,
                               lease_expires_at=time.time() + self.timeout,
                               accept_timeout_seconds=self.timeout)
        self.trace.mark("provider_execute_start", attempt_id=attempt_id)
        try:
            result = self.provider.execute(attempt, reservation, threading.Event())
        finally:
            self.trace.mark("provider_execute_end", attempt_id=attempt_id)
            self.provider.release(reservation.reservation_id)
        self.wait(self.stage_done, "worker stage completion")
        self.wait(self.response_done, "provider response processing")
        events = [e for e in self.trace.events if e.get("attempt_id") == attempt_id]
        events.sort(key=lambda event: event["ns"])
        def stamp(name, message_type=None):
            return next(e["ns"] for e in events if e["event"] == name and (
                message_type is None or e.get("message_type") == message_type))
        sends = [e for e in events if e["event"] == "frame_send_start" and e["side"] == "master"]
        offers = [e for e in sends if e["message_type"] == "stage_offer"]
        accept_end = stamp("accept_write_end")
        executor_start = stamp("executor_start")
        accept_handle = stamp("provider_response_start", "stage_accept")
        checks = dict(accept_write_before_executor=accept_end <= executor_start,
                      one_offer=len(offers) == 1,
                      no_second_master_execution_frame=all(e["message_type"] == "stage_chunk"
                          for e in sends if e["message_type"] != "stage_offer"),
                      scalar_result=result.output == {"token_argmax": TOKEN},
                      result_identity=result.attempt_id == attempt_id and result.lease_epoch == 1)
        if not all(checks.values()):
            raise BenchError(f"sample checks failed: {checks}")
        return dict(index=index, attempt_id=attempt_id, output=result.output,
                    execute_total_ns=stamp("provider_execute_end") - stamp("provider_execute_start"),
                    execute_to_offer_send_ns=stamp("frame_send_start", "stage_offer") - stamp("provider_execute_start"),
                    offer_send_ns=stamp("frame_send_end", "stage_offer") - stamp("frame_send_start", "stage_offer"),
                    post_offer_send_to_execute_end_ns=stamp("provider_execute_end") - stamp("frame_send_end", "stage_offer"),
                    executor_ns=stamp("executor_end") - executor_start,
                    executor_before_master_accept_handling=executor_start < accept_handle,
                    executor_before_master_accept_receive=executor_start < stamp("frame_receive_end", "stage_accept"),
                    input_chunks=sum(e["message_type"] == "stage_chunk" for e in sends),
                    checks=checks, events=events)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        if hasattr(self, "worker"):
            with self.worker._task_worker_stage_lock:
                for active in self.worker._task_worker_active_attempts.values():
                    active.cancel_event.set()
        if self.provider is not None:
            self.provider.close()
        for stream in self.streams:
            try:
                stream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            stream.close()
        deadline = time.monotonic() + 5
        for thread in list(self.threads):
            if thread is not threading.current_thread():
                thread.join(max(0, deadline - time.monotonic()))
        self.cleanup = dict(sockets_closed=all(s.fileno() == -1 for s in self.streams),
                            live_threads=[t.name for t in self.threads if t.is_alive()])
        if self.cleanup["live_threads"]:
            raise BenchError(f"owned threads did not stop: {self.cleanup}")


@contextmanager
def binary_switch(enabled):
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


class PerfCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.rows = []
        self.sample_index = None

    def emit(self, record):
        text = record.getMessage()
        if text.startswith(("perf2 ", "perf3 ")) and record.thread == threading.main_thread().ident:
            self.rows.append(dict(index=self.sample_index, ns=time.perf_counter_ns(), text=text))


def source_fingerprints():
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
        "src/task_worker_adapter.py", "src/scheduler_task_worker.py", "src/tcp_comm.py",
        "src/task_worker_protocol.py", "src/task_worker_chunks.py", "scripts/comm_merge_bench.py")}


def run_bench(*, samples=30, hidden_size=16384, compute_delay_ms=0,
              binary=True, master_accept_delay_ms=0, label="baseline", timeout=15):
    if type(samples) is not int or samples < 1:
        raise BenchError("samples must be a positive integer")
    if type(hidden_size) is not int or not 4 <= hidden_size <= MAX_STAGE_PAYLOAD_BYTES or hidden_size % 4:
        raise BenchError("hidden_size must be a multiple of 4 within the production payload budget")
    if any(not math.isfinite(v) or v < 0 for v in (compute_delay_ms, master_accept_delay_ms)):
        raise BenchError("delays must be finite and nonnegative")
    if not math.isfinite(timeout) or timeout <= (compute_delay_ms + master_accept_delay_ms) / 1000 + 1:
        raise BenchError("timeout must exceed injected delays by more than one second")
    fingerprint = source_fingerprints()
    capture = PerfCapture()
    logger = logging.getLogger("task_worker_adapter")
    previous_level, previous_disabled = logger.level, logger.disabled
    logger.setLevel(logging.INFO)
    logger.disabled = False
    logger.addHandler(capture)
    rows = []
    link = None
    try:
        with binary_switch(binary):
            link = Loopback(binary=binary, compute_delay_ms=compute_delay_ms,
                            master_accept_delay_ms=master_accept_delay_ms, timeout=timeout)
            raw = b"\x00\x00\x80\x3f" * (hidden_size // 4)
            for index in range(samples):
                capture.sample_index = index
                row = link.sample(index, raw)
                # Keep the original key for baseline readers; both lists include perf3.
                row["perf_logs"] = [r for r in capture.rows if r["index"] == index]
                row["perf2_logs"] = row["perf_logs"]
                rows.append(row)
    finally:
        try:
            if link is not None:
                link.close()
        finally:
            logger.removeHandler(capture)
            logger.setLevel(previous_level)
            logger.disabled = previous_disabled
    if source_fingerprints() != fingerprint:
        raise BenchError("production sources changed during run; repeat this capture")
    totals = sorted(row["execute_total_ns"] for row in rows)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return dict(schema=SCHEMA, label=label, passed=True,
                scope="same-process loopback; model/deployment/registration admission and executor are injected; not device end-to-end",
                clock="time.perf_counter_ns; one process", source_sha256=fingerprint,
                revision=revision, python=platform.python_version(),
                config=dict(samples=samples, hidden_size_bytes=hidden_size, compute_delay_ms=compute_delay_ms,
                            binary_requested=binary, binary_negotiated=link.binary,
                            master_accept_delay_ms=master_accept_delay_ms, debug_logging=False),
                fingerprint_notes="Source digests include the benchmark script; record tool differences when comparing historical captures that only hashed production files.",
                event_notes="frame_decoded.ns and frame_receive_end.ns follow full recv_data decoding. frame_decoded.read_started_ns is the recv call start, which may include idle wait. master_accept_delay starts after decode and before provider response processing; it proves independence from master handling, not an unobserved TCP receive.",
                timing_notes="Observers add overhead. post_offer_send_to_execute_end_ns includes debug argument work, scheduling and waits; not isolated json.dumps cost. Compare separate baseline/candidate runs with identical config.",
                summary=dict(execute_total_ns=dict(min=min(totals), median=statistics.median(totals),
                             p95=totals[math.ceil(len(totals) * .95) - 1], max=max(totals)),
                             executor_before_master_accept_handling=sum(r["executor_before_master_accept_handling"] for r in rows)),
                handshake_events=[e for e in link.trace.events if not e.get("attempt_id")],
                cleanup=link.cleanup, samples=rows)


def parse_size(value):
    text = value.strip().lower()
    for suffix, multiplier in (("kib", 1024), ("mib", 1024 ** 2), ("k", 1024), ("m", 1024 ** 2)):
        if text.endswith(suffix):
            return int(text[:-len(suffix)]) * multiplier
    return int(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--hidden-size", type=parse_size, default=16384, help="raw f32 bytes, e.g. 16KiB, 256KiB, 4MiB")
    parser.add_argument("--compute-delay-ms", type=float, default=0)
    parser.add_argument("--master-accept-delay-ms", "--master-accept-delayed", type=float, default=0)
    parser.add_argument("--binary", choices=("on", "off"), default="on")
    parser.add_argument("--label", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--output", type=Path, help="JSON path; omit for stdout")
    args = parser.parse_args(argv)
    try:
        report = run_bench(samples=args.samples, hidden_size=args.hidden_size,
                           compute_delay_ms=args.compute_delay_ms, binary=args.binary == "on",
                           master_accept_delay_ms=args.master_accept_delay_ms,
                           label=args.label, timeout=args.timeout)
        code = 0
    except Exception as exc:
        report = dict(schema=SCHEMA, label=args.label, passed=False,
                      error=f"{type(exc).__name__}: {exc}")
        code = 1
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print(json.dumps(dict(output=str(args.output), passed=report["passed"],
                              summary=report.get("summary"), error=report.get("error"))))
    else:
        print(encoded, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
