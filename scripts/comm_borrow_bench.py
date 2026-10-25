#!/usr/bin/env python
"""22e: local provider/mixin/TCP timings with an injected Python stage executor.

The hidden fixture echoes validated input; scalar returns the merge fixture token.
Model/deployment/registration admission is fixture state. No model or native
computation runs. Each invocation measures one isolation/binary configuration.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import comm_merge_bench as merge

SCHEMA = "qlh.comm_borrow_bench.v1"
IO_ENV = "QLH_TASK_WORKER_IO_ISOLATION"
# Stay strictly below 64 MiB, including when production allows larger stages.
MAX_HIDDEN_BYTES = min(merge.MAX_STAGE_PAYLOAD_BYTES, 64 * 1024 * 1024 - 4)
BenchError = merge.BenchError
SOURCE_FILES = (
    "scripts/comm_borrow_bench.py", "scripts/comm_merge_bench.py",
    "src/task_worker_io.py", "src/scheduler.py", "src/scheduler_task_worker.py",
    "src/task_worker_adapter.py", "src/task_worker_protocol.py",
    "src/task_worker_chunks.py", "src/tcp_comm.py", "src/execution_provider.py",
    "src/qlh_core/contracts/token_output.py",
)
SENDER_COUNTERS = (
    "submitted", "completed", "failed", "retired", "cancelled", "queue_wait_ns",
    "action_wall_ns", "action_thread_cpu_ns",
)


class CountedSocket:
    """Pass bytes directly to a real socket; retain only successful byte counts."""

    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.Lock()
        self.sent_on_thread = threading.local()
        self.counts = dict(sent_bytes=0, received_bytes=0, sendall_calls=0, recv_calls=0)

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def sendall(self, data, *args, **kwargs):
        result = self.stream.sendall(data, *args, **kwargs)
        with self.lock:
            self.counts["sent_bytes"] += len(data)
            self.counts["sendall_calls"] += 1
            self.sent_on_thread.bytes = getattr(self.sent_on_thread, "bytes", 0) + len(data)
        return result

    def recv(self, *args, **kwargs):
        data = self.stream.recv(*args, **kwargs)
        with self.lock:
            self.counts["received_bytes"] += len(data)
            self.counts["recv_calls"] += 1
        return data

    def sent_bytes_on_thread(self):
        return getattr(self.sent_on_thread, "bytes", 0)

    def snapshot(self):
        with self.lock:
            return dict(self.counts)


class Trace(merge.Trace):
    """Clocks are read on the thread that emits each event, under one order lock."""

    def __init__(self):
        super().__init__()
        self.condition = threading.Condition(self.lock)
        self.streams = {}
        self.send_starts = {}
        self.receive_ends = collections.defaultdict(int)

    def mark(self, event, **fields):
        with self.lock:
            side = fields.get("side")
            if side in self.streams:
                counts = self.streams[side].snapshot()
                if event == "frame_send_start":
                    self.send_starts[(side, fields["message_id"])] = self.streams[side].sent_bytes_on_thread()
                elif event == "frame_send_end":
                    before = self.send_starts.pop((side, fields["message_id"]))
                    fields["wire_bytes"] = self.streams[side].sent_bytes_on_thread() - before
                elif event == "frame_receive_end":
                    fields["wire_bytes"] = counts["received_bytes"] - self.receive_ends[side]
                    self.receive_ends[side] = counts["received_bytes"]
            self.events.append(dict(
                event=event, seq=len(self.events), ns=time.perf_counter_ns(),
                process_time_ns=time.process_time_ns(), thread_time_ns=time.thread_time_ns(),
                thread=threading.current_thread().name, thread_id=threading.get_ident(),
                **fields,
            ))
            self.condition.notify_all()

    def wait_handshake(self, timeout):
        # hello_done is set by the receiver: it does not order the sending
        # thread's sendall return, counter publication or frame_send_end.
        expected = {
            ("frame_send_end", "worker", "hello"),
            ("frame_receive_end", "master", "hello"),
            ("frame_send_end", "master", "hello_ack"),
            ("frame_receive_end", "worker", "hello_ack"),
        }

        def complete():
            observed = {(e["event"], e.get("side"), e.get("message_type")) for e in self.events}
            return expected <= observed

        with self.condition:
            if not self.condition.wait_for(complete, timeout):
                raise BenchError("timed out waiting for handshake send/receive accounting")
        self.mark("handshake_accounting_done")


@contextmanager
def isolation_switch(mode):
    previous = os.environ.get(IO_ENV)
    os.environ[IO_ENV] = "1" if mode == "on" else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(IO_ENV, None)
        else:
            os.environ[IO_ENV] = previous


@contextmanager
def quiet_logs():
    loggers = [logging.getLogger(name) for name in
               ("task_worker_adapter", "scheduler_task_worker", "tcp_comm")]
    previous = [(logger.level, logger.disabled) for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.WARNING)
            logger.disabled = False
        yield
    finally:
        for logger, (level, disabled) in zip(loggers, previous):
            logger.setLevel(level)
            logger.disabled = disabled


class Loopback(merge.Loopback):
    """Reuse merge transport/fixtures; customize observation and injected compute."""

    # The inherited constructor assigns Trace before constructing either endpoint.
    # This local descriptor supplies all clocks during hello as well as samples;
    # no production class, transport method, or global trace factory is patched.
    @property
    def trace(self):
        return self._borrow_trace

    @trace.setter
    def trace(self, _trace):
        self._borrow_trace = Trace()

    def observe_endpoint(self, endpoint):
        # The inherited constructor publishes each endpoint before registration
        # ownership is captured or receive threads start. Only its socket is
        # decorated; the production TCPClient methods stay inherited and intact.
        endpoint.sock = CountedSocket(endpoint.sock)
        self.trace.streams[endpoint.side] = endpoint.sock
        return endpoint

    @property
    def master_tcp(self):
        return self._borrow_master_tcp

    @master_tcp.setter
    def master_tcp(self, endpoint):
        self._borrow_master_tcp = self.observe_endpoint(endpoint)

    @property
    def worker_tcp(self):
        return self._borrow_worker_tcp

    @worker_tcp.setter
    def worker_tcp(self, endpoint):
        self._borrow_worker_tcp = self.observe_endpoint(endpoint)

    def wire_snapshot(self):
        return {side: stream.snapshot() for side, stream in self.trace.streams.items()}

    def __init__(self, *, mode, output_kind, **kwargs):
        self.mode = mode
        self.output_kind = output_kind
        try:
            super().__init__(**kwargs)
            self.trace.wait_handshake(self.timeout)
        except BaseException:
            self.close()
            raise

    def sender_snapshot(self):
        sender = getattr(getattr(self, "worker", None), "_task_worker_io_sender", None)
        return sender.snapshot() if sender is not None else None

    def wait_sender_idle(self):
        deadline = time.monotonic() + self.timeout
        while True:
            snapshot = self.sender_snapshot()
            if snapshot is None:
                if self.mode == "on":
                    raise BenchError("mode on did not create the production worker IO sender")
                return None
            if self.mode == "off":
                raise BenchError("mode off unexpectedly created a worker IO sender")
            if snapshot["pending_items"] == 0 and snapshot["pending_bytes"] == 0:
                if any(snapshot[name] for name in ("failed", "retired", "cancelled")):
                    raise BenchError(f"worker IO sender has unsuccessful actions: {snapshot}")
                return snapshot
            if time.monotonic() >= deadline:
                raise BenchError(f"timed out waiting for worker IO sender idle: {snapshot}")
            self.stop.wait(0.001)

    def compute(self, request, cancel):
        """Injected Python validation/delay/echo; this is not a native executor."""
        attempt_id = "att_merge_" + request.request_id.zfill(8)
        self.trace.mark("executor_start", attempt_id=attempt_id)
        try:
            encoded = request.root_input["hidden_f32"]
            raw = base64.b64decode(encoded, validate=True)
            digest = hashlib.sha256(raw).hexdigest()
            if digest != request.stage_fields["hidden_sha256"]:
                raise BenchError("injected callback received different hidden")
            if cancel.wait(self.compute_delay_ms / 1000):
                raise BenchError("injected callback cancelled")
            if self.output_kind == "scalar":
                return {"token_argmax": merge.TOKEN}
            # Echo the existing string; actual result validation, chunk planning,
            # serialization and transport remain in the production mixin/provider.
            return {"hidden_out_f32": encoded, "hidden_out_sha256": digest}
        finally:
            self.trace.mark("executor_end", attempt_id=attempt_id)

    def sample(self, index, raw):
        self.stage_done.clear()
        self.response_done.clear()
        before = self.sender_snapshot()
        wire_before = self.wire_snapshot()
        attempt_id = f"att_merge_{index:08d}"
        digest = hashlib.sha256(raw).hexdigest()
        request = merge.StageRequest(
            workflow_id=f"wf_merge_{index:08d}", request_id=str(index), stage_id="tail",
            stage_type="layer_forward", provider_id=self.provider.provider_id,
            dependencies={}, root_input=dict(hidden_f32=base64.b64encode(raw).decode("ascii"),
                                             want_hidden=self.output_kind == "hidden", pos_base=0),
            model_identity=merge.MODEL,
            stage_fields=dict(layer_range=[0, 1], handoff_at=1, hidden_sha256=digest,
                              hidden_spec=dict(n_tokens=1, n_embd=len(raw) // 4, dtype="float32")),
        )
        reservation = self.provider.reserve(request)
        attempt = merge.StageAttempt(
            attempt_id, request, self.provider.provider_id,
            lease_id=f"lease_merge_{index:08d}", lease_epoch=1,
            lease_expires_at=time.time() + self.timeout, accept_timeout_seconds=self.timeout,
        )
        self.trace.mark("provider_execute_start", attempt_id=attempt_id)
        try:
            result = self.provider.execute(attempt, reservation, threading.Event())
        finally:
            self.trace.mark("provider_execute_end", attempt_id=attempt_id)
            self.provider.release(reservation.reservation_id)
        # Async sends can finish after stage_done; provider completion can race
        # recv-loop bookkeeping. Both barriers precede the sender idle snapshot.
        self.wait(self.stage_done, "worker stage completion")
        self.wait(self.response_done, "provider response processing")
        after = self.wait_sender_idle()
        self.trace.mark("sender_idle", attempt_id=attempt_id)
        wire_after = self.wire_snapshot()
        wire = {side: {key: value - wire_before[side][key] for key, value in counts.items()}
                for side, counts in wire_after.items()}
        events = [e for e in self.trace.events if e.get("attempt_id") == attempt_id]

        def event(name, message_type=None):
            return next(e for e in events if e["event"] == name and
                        (message_type is None or e.get("message_type") == message_type))

        start, end = event("provider_execute_start"), event("provider_execute_end")
        executor_start, executor_end = event("executor_start"), event("executor_end")
        starts = [e for e in events if e["event"] == "frame_send_start"]
        ends = [e for e in events if e["event"] == "frame_send_end"]
        receives = [e for e in events if e["event"] == "frame_receive_end"]
        master = [e for e in starts if e["side"] == "master"]
        worker = [e for e in starts if e["side"] == "worker"]
        counts = {side: dict(collections.Counter(e["message_type"] for e in frames))
                  for side, frames in (("master", master), ("worker", worker))}
        input_chunks = counts["master"].get("stage_chunk", 0)
        result_chunks = counts["worker"].get("stage_result_chunk", 0)
        if self.output_kind == "hidden":
            returned = base64.b64decode(result.output["hidden_out_f32"], validate=True)
            output_ok = (len(returned) == len(raw) and hashlib.sha256(returned).hexdigest() == digest
                         and result.output.get("hidden_out_sha256") == digest)
            output = dict(kind="hidden", hidden_size_bytes=len(returned), hidden_sha256=digest)
        else:
            output_ok = result.output == {"token_argmax": merge.TOKEN}
            output = dict(result.output)
        def frame_ids(frames):
            return collections.Counter(e["message_id"] for e in frames)

        checks = dict(
            one_offer=counts["master"].get("stage_offer") == 1,
            one_accept=counts["worker"].get("stage_accept") == 1,
            one_result=counts["worker"].get("stage_result") == 1,
            master_order=[e["message_type"] for e in master] == ["stage_chunk"] * input_chunks + ["stage_offer"],
            worker_order=[e["message_type"] for e in worker] == ["stage_accept"] + ["stage_result_chunk"] * result_chunks + ["stage_result"],
            frames_complete=frame_ids(starts) == frame_ids(ends) == frame_ids(receives),
            wire_bytes_match=(wire["master"]["sent_bytes"] == wire["worker"]["received_bytes"]
                              and wire["worker"]["sent_bytes"] == wire["master"]["received_bytes"]),
            wire_messages_match=all(wire[side]["sendall_calls"] == len(frames)
                                    for side, frames in (("master", master), ("worker", worker))),
            accept_write_before_executor=event("accept_write_end")["ns"] <= executor_start["ns"],
            result_after_executor=all(e["ns"] >= executor_end["ns"] for e in worker
                                      if e["message_type"] != "stage_accept"),
            worker_send_thread=all(e["thread"] == ("task-worker-io" if self.mode == "on"
                                                   else executor_start["thread"]) for e in worker),
            result_identity=result.attempt_id == attempt_id and result.lease_epoch == 1,
            output_matches=output_ok,
            sender_idle=after is None or (after["pending_items"] == after["pending_bytes"] == 0),
        )
        if not all(checks.values()):
            raise BenchError(f"sample checks failed: {checks}")
        delta = None if after is None else {name: after[name] - (before or {}).get(name, 0)
                                           for name in SENDER_COUNTERS}
        return dict(
            index=index, attempt_id=attempt_id, output=output,
            execute_ns=end["ns"] - start["ns"],
            process_time_ns=end["process_time_ns"] - start["process_time_ns"],
            thread_time_ns=end["thread_time_ns"] - start["thread_time_ns"],
            executor_ns=executor_end["ns"] - executor_start["ns"],
            executor_thread_time_ns=executor_end["thread_time_ns"] - executor_start["thread_time_ns"],
            wire_bytes=wire,
            application_messages=dict(total=len(starts), by_sender=counts,
                                      input_chunks=input_chunks, result_chunks=result_chunks),
            queue_stats=dict(available=after is not None, before=before, after=after, delta=delta),
            checks=checks, events=events,
        )

    def close(self):
        if self.closed:
            return
        self.stop.set()
        worker = getattr(self, "worker", None)
        sender = None
        if worker is not None:
            with worker._task_worker_stage_lock:
                worker._task_worker_io_stopped = True
                sender = getattr(worker, "_task_worker_io_sender", None)
                for active in worker._task_worker_active_attempts.values():
                    active.cancel_event.set()
        # Interrupt a blocked TCP send before joining the separate IO thread.
        for stream in self.streams:
            try:
                stream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            stream.close()
        try:
            if sender is not None:
                sender.close(wait=True)
        finally:
            super().close()
        snapshot = sender.snapshot() if sender is not None else None
        self.cleanup.update(
            sender_close_wait=sender is not None, sender_snapshot=snapshot,
            close_order=["sockets", "sender", "stage_and_receive_threads"],
        )
        if snapshot is not None and (snapshot["thread_alive"] or snapshot["pending_items"]
                                     or snapshot["pending_bytes"] or not snapshot["closed"]):
            raise BenchError(f"worker IO sender did not stop: {self.cleanup}")


def source_fingerprints():
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_FILES}


def validate_config(*, samples, hidden_size, compute_delay_ms, master_accept_delay_ms,
                    timeout, mode, binary, output_kind):
    if type(samples) is not int or samples < 1:
        raise BenchError("samples must be a positive integer")
    if type(hidden_size) is not int or not 4 <= hidden_size <= MAX_HIDDEN_BYTES or hidden_size % 4:
        raise BenchError(f"hidden_size must be a multiple of 4 between 4 and {MAX_HIDDEN_BYTES} bytes (below 64 MiB)")
    if mode not in ("off", "on"):
        raise BenchError("mode must be off or on")
    if type(binary) is not bool:
        raise BenchError("binary must be a bool")
    if output_kind not in ("scalar", "hidden"):
        raise BenchError("output_kind must be scalar or hidden")
    for name, value in (("compute_delay_ms", compute_delay_ms),
                        ("master_accept_delay_ms", master_accept_delay_ms), ("timeout", timeout)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise BenchError(f"{name} must be finite and nonnegative")
    if timeout <= (compute_delay_ms + master_accept_delay_ms) / 1000 + 1:
        raise BenchError("timeout must exceed injected delays by more than one second")


def clock_metadata():
    result = {}
    for label, name in (("wall", "perf_counter"), ("process", "process_time"), ("thread", "thread_time")):
        info = time.get_clock_info(name)
        result[label] = dict(implementation=info.implementation, monotonic=info.monotonic,
                             adjustable=info.adjustable, resolution_seconds=info.resolution,
                             resolution_ns=info.resolution * 1_000_000_000)
    return result


def cpu_zero_deltas(rows):
    return dict(process_time_ns=sum(row["process_time_ns"] == 0 for row in rows),
                thread_time_ns=sum(row["thread_time_ns"] == 0 for row in rows),
                sender_action_thread_cpu_ns=(sum(row["queue_stats"]["delta"]["action_thread_cpu_ns"] == 0
                                                for row in rows)
                                             if rows[0]["queue_stats"]["available"] else None))


def run_bench(*, samples=30, hidden_size=16384, compute_delay_ms=0,
              mode="off", binary=True, output_kind="hidden", master_accept_delay_ms=0, timeout=15):
    validate_config(samples=samples, hidden_size=hidden_size, compute_delay_ms=compute_delay_ms,
                    master_accept_delay_ms=master_accept_delay_ms, timeout=timeout,
                    mode=mode, binary=binary, output_kind=output_kind)
    fingerprint = source_fingerprints()
    rows, link = [], None
    with isolation_switch(mode), merge.binary_switch(binary), quiet_logs():
        try:
            link = Loopback(mode=mode, output_kind=output_kind, binary=binary,
                            compute_delay_ms=compute_delay_ms,
                            master_accept_delay_ms=master_accept_delay_ms, timeout=timeout)
            raw = b"\x00\x00\x80\x3f" * (hidden_size // 4)
            for index in range(samples):
                rows.append(link.sample(index, raw))
            final_sender = link.wait_sender_idle()
        finally:
            if link is not None:
                link.close()
    if source_fingerprints() != fingerprint:
        raise BenchError("sources changed during run; repeat this capture")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                          text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None

    def summary(field):
        values = sorted(row[field] for row in rows)
        return dict(min=min(values), median=statistics.median(values),
                    p95=values[math.ceil(len(values) * .95) - 1], max=max(values))

    return dict(
        schema=SCHEMA, passed=True, revision=revision, python=platform.python_version(),
        scope=dict(transport="same-process persistent IPv4 TCP loopback; production provider/mixin/framing/hello",
                   admission="injected registration, prepared model identity and deployment",
                   executor="injected Python hidden validation/delay and echo or scalar token",
                   native_compute=False, multi_machine=False),
        config=dict(samples=samples, hidden_size_bytes=hidden_size, compute_delay_ms=compute_delay_ms,
                    master_accept_delay_ms=master_accept_delay_ms, timeout_seconds=timeout,
                    mode=mode, isolation_env=IO_ENV, isolation_env_value="1" if mode == "on" else "0",
                    binary_requested=binary, binary_negotiated=link.binary, output_kind=output_kind),
        clocks=dict(wall="time.perf_counter_ns", process="time.process_time_ns",
                    thread="time.thread_time_ns"),
        clock_info=clock_metadata(), cpu_zero_deltas=cpu_zero_deltas(rows),
        clock_notes="clock_info records time.get_clock_info implementation and declared resolution, not measured effective CPU accounting granularity. Nanosecond units do not imply nanosecond precision. CPU zero deltas are retained unchanged and counted; a zero reading does not establish zero CPU work. Sender action_thread_cpu_ns uses the same thread_time clock.",
        timing_notes="execute_ns covers the complete provider.execute call, excluding reserve, release and post-execute idle barriers. Process CPU includes all threads; thread CPU belongs to the emitting thread and deltas require the same thread_id. Event seq is observer lock order. Observers add overhead; injected delays are waits. Native segment interference requires a separate measurement.",
        message_notes="Application message counts count frame_send_start calls once and exclude hello/hello_ack. Sample baselines follow both handshake send-end and receive-end observations; per-frame send bytes are attributed by message_id and emitting thread. wire_bytes counts successful real socket sendall/recv bytes, including JSON/base64, length headers and optional binary bodies; it excludes IP/TCP headers, ACKs and retransmissions. Per-frame wire_bytes is on frame_send_end/frame_receive_end. Hidden output is recorded only as size/digest.",
        sender_notes="Snapshots follow stage_done, response_done and pending_items/pending_bytes == 0. Queue wait ends at owner-check start; action timings and CPU use the sender's production metrics. Per-stage deltas include accept and one result batch when isolation is on.",
        source_sha256=fingerprint,
        summary={name: summary(name) for name in ("execute_ns", "process_time_ns", "thread_time_ns")},
        handshake_events=[e for e in link.trace.events if not e.get("attempt_id")],
        queue_stats=final_sender, cleanup=link.cleanup, samples=rows,
    )


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise BenchError(message)


def prepare_output(output, output_dir):
    """Reserve a fresh JSON file; never replace existing files or clear directories."""
    if output_dir is not None:
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise BenchError("output directory must be new or empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "report.json"
    if output is None:
        return None, None
    if output.suffix.lower() != ".json":
        raise BenchError("output must have a .json suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output, output.open("x", encoding="utf-8", newline="\n")


def main(argv=None):
    parser = JSONArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--hidden-size", type=merge.parse_size, default=16384,
                        help="raw f32 bytes, e.g. 16KiB, 256KiB, 9MiB; strictly below 64MiB")
    parser.add_argument("--compute-delay-ms", type=float, default=0)
    parser.add_argument("--master-accept-delay-ms", type=float, default=0)
    parser.add_argument("--mode", choices=("off", "on"), default="off")
    parser.add_argument("--binary", choices=("off", "on"), default="on")
    parser.add_argument("--output-kind", choices=("scalar", "hidden"), default="hidden")
    parser.add_argument("--timeout", type=float, default=15)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", "--json-out", type=Path, help="new JSON file; parents are created")
    output_group.add_argument("--output-dir", "--out-dir", type=Path, help="new or empty directory for report.json")
    path, stream, stage = None, None, "arguments"
    try:
        args = parser.parse_args(argv)
        stage = "output_prepare"
        path, stream = prepare_output(args.output, args.output_dir)
        stage = "benchmark"
        report = run_bench(samples=args.samples, hidden_size=args.hidden_size,
                           compute_delay_ms=args.compute_delay_ms,
                           master_accept_delay_ms=args.master_accept_delay_ms,
                           mode=args.mode, binary=args.binary == "on",
                           output_kind=args.output_kind, timeout=args.timeout)
    except Exception as exc:
        report = dict(schema=SCHEMA, passed=False, error_stage=stage,
                      error=f"{type(exc).__name__}: {exc}")
    try:
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        if stream is not None:
            stream.write(encoded)
            stream.close()
            print(json.dumps(dict(schema=SCHEMA, output=str(path), passed=report["passed"],
                                  summary=report.get("summary"), error=report.get("error")),
                             ensure_ascii=False, allow_nan=False))
        else:
            print(encoded, end="")
    except Exception as exc:
        if stream is not None:
            stream.close()
        print(json.dumps(dict(schema=SCHEMA, passed=False, error_stage="output_write",
                              error=f"{type(exc).__name__}: {exc}"), ensure_ascii=False))
        return 1
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
