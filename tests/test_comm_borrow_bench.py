"""22e tool contracts over real local provider/mixin/TCP and injected compute.

These verify evidence accounting and resource ownership. Native interference and
receive-loop nonblocking behavior have separate integration coverage.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "comm_borrow_bench.py"
SPEC = importlib.util.spec_from_file_location("comm_borrow_bench", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def event(row, name, message_type=None, side=None):
    return next(e for e in row["events"] if e["event"] == name
                and (message_type is None or e.get("message_type") == message_type)
                and (side is None or e.get("side") == side))


def assert_clean(cleanup, mode):
    assert cleanup["sockets_closed"] is True
    assert cleanup["live_threads"] == []
    assert cleanup["close_order"] == ["sockets", "sender", "stage_and_receive_threads"]
    assert cleanup["sender_close_wait"] is (mode == "on")
    snapshot = cleanup["sender_snapshot"]
    if mode == "off":
        assert snapshot is None
    else:
        assert snapshot["closed"] is True
        assert snapshot["thread_alive"] is False
        assert snapshot["pending_items"] == snapshot["pending_bytes"] == 0


def assert_wire_accounting(row):
    starts = [e for e in row["events"] if e["event"] == "frame_send_start"]
    ends = [e for e in row["events"] if e["event"] == "frame_send_end"]
    receives = [e for e in row["events"] if e["event"] == "frame_receive_end"]
    assert row["application_messages"]["total"] == len(starts) == len(ends) == len(receives)
    for start in starts:
        end, = [e for e in ends if e["message_id"] == start["message_id"]]
        receive, = [e for e in receives if e["message_id"] == start["message_id"]]
        assert end["wire_bytes"] == receive["wire_bytes"] > start["binary_bytes"]
        assert start["ns"] <= end["ns"] and start["ns"] <= receive["ns"]
        assert receive["side"] != start["side"]
        assert end["message_type"] == receive["message_type"] == start["message_type"]
    for side, counts in row["wire_bytes"].items():
        assert counts["sent_bytes"] == sum(e["wire_bytes"] for e in ends if e["side"] == side)
        assert counts["received_bytes"] == sum(e["wire_bytes"] for e in receives if e["side"] == side)
        assert counts["sendall_calls"] == len([e for e in starts if e["side"] == side])
        assert counts["recv_calls"] >= len([e for e in receives if e["side"] == side]) * 2


@pytest.mark.parametrize("mode", ["off", "on"])
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("output_kind", ["scalar", "hidden"])
def test_real_persistent_loopback_full_execute_clocks_threads_wire_and_queue(mode, binary, output_kind):
    report = bench.run_bench(samples=2, mode=mode, binary=binary, output_kind=output_kind,
                             hidden_size=16384, compute_delay_ms=1)
    assert report["passed"] is True
    assert_clean(report["cleanup"], mode)
    assert report["config"]["mode"] == mode
    assert report["config"]["binary_requested"] is report["config"]["binary_negotiated"] is binary
    assert report["scope"]["native_compute"] is False
    assert report["scope"]["multi_machine"] is False
    assert "injected" in report["scope"]["executor"]
    assert "injected" in report["scope"]["admission"]
    hello = [e for e in report["handshake_events"] if e["event"] == "frame_send_start"]
    assert [(e["side"], e["message_type"]) for e in hello] == [
        ("worker", "hello"), ("master", "hello_ack")]
    handshake_done, = [e for e in report["handshake_events"] if e["event"] == "handshake_accounting_done"]
    for label, name in (("wall", "perf_counter"), ("process", "process_time"), ("thread", "thread_time")):
        actual = bench.time.get_clock_info(name)
        info = report["clock_info"][label]
        assert info["implementation"] == actual.implementation
        assert info["resolution_seconds"] == actual.resolution
        assert info["resolution_ns"] == actual.resolution * 1_000_000_000
        assert info["monotonic"] is True
    assert report["cpu_zero_deltas"] == bench.cpu_zero_deltas(report["samples"])
    sender_ids = set()
    for row in report["samples"]:
        assert all(row["checks"].values())
        assert_wire_accounting(row)
        first, last = event(row, "provider_execute_start"), event(row, "provider_execute_end")
        assert row["execute_ns"] == last["ns"] - first["ns"]
        assert row["process_time_ns"] == last["process_time_ns"] - first["process_time_ns"] >= 0
        assert row["thread_time_ns"] == last["thread_time_ns"] - first["thread_time_ns"] >= 0
        assert first["thread_id"] == last["thread_id"] == threading.get_ident()
        assert handshake_done["seq"] < first["seq"]
        assert row["execute_ns"] >= row["executor_ns"] >= 1_000_000
        assert row["executor_thread_time_ns"] >= 0
        assert row["application_messages"] == dict(
            total=3, by_sender={"master": {"stage_offer": 1},
                                "worker": {"stage_accept": 1, "stage_result": 1}},
            input_chunks=0, result_chunks=0)
        accept_end = event(row, "accept_write_end")
        executor_start = event(row, "executor_start")
        assert accept_end["ns"] <= executor_start["ns"]
        assert event(row, "executor_end")["ns"] <= event(row, "frame_send_start", "stage_result")["ns"]
        assert event(row, "worker_stage_end")["ns"] <= event(row, "sender_idle")["ns"]
        assert event(row, "provider_response_end", "stage_result")["ns"] <= event(row, "sender_idle")["ns"]
        assert event(row, "frame_send_start", "stage_offer")["binary_bytes"] == (16384 if binary else 0)
        expected_result_bytes = 16384 if binary and output_kind == "hidden" else 0
        assert event(row, "frame_send_start", "stage_result")["binary_bytes"] == expected_result_bytes
        if output_kind == "scalar":
            assert row["output"] == {"token_argmax": 42}
        else:
            assert row["output"] == dict(kind="hidden", hidden_size_bytes=16384,
                hidden_sha256=hashlib.sha256(b"\x00\x00\x80\x3f" * 4096).hexdigest())
        seqs = [e["seq"] for e in row["events"]]
        assert seqs == sorted(set(seqs))
        for item in row["events"] + report["handshake_events"]:
            assert item["ns"] > 0 and item["process_time_ns"] >= 0 and item["thread_time_ns"] >= 0
            assert item["thread"] and isinstance(item["thread_id"], int)
        worker_sends = [e for e in row["events"] if e["event"] == "frame_send_start" and e["side"] == "worker"]
        if mode == "on":
            assert {e["thread"] for e in worker_sends} == {"task-worker-io"}
            assert all(e["thread_id"] != executor_start["thread_id"] for e in worker_sends)
            sender_ids.update(e["thread_id"] for e in worker_sends)
            stats = row["queue_stats"]
            assert stats["available"] is True
            assert stats["delta"]["submitted"] == stats["delta"]["completed"] == 2
            assert stats["delta"]["failed"] == stats["delta"]["retired"] == stats["delta"]["cancelled"] == 0
            # Windows GetThreadTimes may resolve no CPU increment for short
            # actions. Check raw counter monotonicity/differencing; thread
            # routing, completed actions and real bytes prove work took place.
            cpu_before = (stats["before"] or {}).get("action_thread_cpu_ns", 0)
            cpu_after = stats["after"]["action_thread_cpu_ns"]
            assert isinstance(cpu_after, int) and cpu_after >= cpu_before
            assert stats["delta"]["action_thread_cpu_ns"] == cpu_after - cpu_before >= 0
            assert stats["delta"]["action_wall_ns"] > 0
            assert stats["delta"]["queue_wait_ns"] > 0
            snapshot = stats["after"]
            assert snapshot["pending_items"] == snapshot["pending_bytes"] == snapshot["queued_items"] == 0
            assert 0 < snapshot["peak_items"] <= snapshot["max_items"]
            assert 0 < snapshot["peak_bytes"] <= snapshot["max_bytes"]
            assert snapshot["submitted"] == snapshot["completed"]
        else:
            assert {e["thread_id"] for e in worker_sends} == {executor_start["thread_id"]}
            assert row["queue_stats"] == dict(available=False, before=None, after=None, delta=None)
    if mode == "on":
        assert len(sender_ids) == 1
        assert report["queue_stats"]["submitted"] == report["queue_stats"]["completed"] == 4
    assert "hidden_out_f32" not in json.dumps(report)


def test_overlapping_frame_start_observers_keep_bytes_on_the_sending_thread():
    # TCP serializes writes after Endpoint has emitted frame_send_start. Start
    # A first but finish B first, without relying on scheduler timing or sleep.
    sink = SimpleNamespace(sendall=lambda _data: None)
    counted = bench.CountedSocket(sink)
    trace = bench.Trace()
    trace.streams["master"] = counted
    a_started, b_finished = threading.Event(), threading.Event()
    errors = []

    def send_a():
        try:
            trace.mark("frame_send_start", side="master", message_id="a")
            a_started.set()
            if not b_finished.wait(5):
                raise AssertionError("B did not finish")
            counted.sendall(b"aaa")
            trace.mark("frame_send_end", side="master", message_id="a")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=send_a, name="borrow-overlap-a", daemon=True)
    thread.start()
    try:
        assert a_started.wait(5)
        trace.mark("frame_send_start", side="master", message_id="b")
        counted.sendall(b"bbbbbbb")
        trace.mark("frame_send_end", side="master", message_id="b")
    finally:
        b_finished.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    ends = {e["message_id"]: e["wire_bytes"] for e in trace.events if e["event"] == "frame_send_end"}
    assert ends == {"a": 3, "b": 7}
    assert counted.snapshot()["sent_bytes"] == 10
    assert counted.snapshot()["sendall_calls"] == 2


def test_first_sample_waits_for_hello_ack_sendall_counter_publication(monkeypatch):
    # Deliver hello_ack over the real socket, then hold the raw sendall return.
    # The worker processes the ACK while CountedSocket still has not published
    # the master's send bytes/call. This is the original first-sample race.
    release, ack_delivered = threading.Event(), threading.Event()
    barrier_entered, sample_started = threading.Event(), threading.Event()
    reports, errors = [], []
    observe = bench.Loopback.observe_endpoint
    wait_handshake = bench.Trace.wait_handshake
    sample = bench.Loopback.sample

    class HeldAckReturn:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def sendall(self, data, *args, **kwargs):
            result = self.stream.sendall(data, *args, **kwargs)
            if b'"message_type": "hello_ack"' in data:
                ack_delivered.set()
                if not release.wait(10):
                    raise AssertionError("test did not release hello_ack sendall")
            return result

    def hold_ack(self, endpoint):
        endpoint = observe(self, endpoint)
        if endpoint.side == "master":
            endpoint.sock.stream = HeldAckReturn(endpoint.sock.stream)
        return endpoint

    def observed_barrier(self, timeout):
        barrier_entered.set()
        return wait_handshake(self, timeout)

    def observed_sample(self, *args):
        sample_started.set()
        return sample(self, *args)

    def run():
        try:
            reports.append(bench.run_bench(samples=2, mode="on", binary=False, output_kind="scalar"))
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(bench.Loopback, "observe_endpoint", hold_ack)
    monkeypatch.setattr(bench.Trace, "wait_handshake", observed_barrier)
    monkeypatch.setattr(bench.Loopback, "sample", observed_sample)
    thread = threading.Thread(target=run, name="borrow-held-ack", daemon=True)
    thread.start()
    try:
        assert ack_delivered.wait(5)
        # Entering the new barrier proves the inherited hello negotiation has
        # completed, while the real sendall's accounting is still withheld.
        assert barrier_entered.wait(5)
        assert not sample_started.wait(0.05)
    finally:
        release.set()
        thread.join(20)
    assert not thread.is_alive() and not errors
    report, = reports
    assert_clean(report["cleanup"], "on")
    assert sample_started.is_set()
    for row in report["samples"]:
        assert all(row["checks"].values())
        assert_wire_accounting(row)
        assert row["wire_bytes"]["master"]["sendall_calls"] == 1
        assert row["wire_bytes"]["worker"]["sendall_calls"] == 2


def test_cpu_zero_deltas_remain_raw_and_report_clock_resolution():
    rows = [dict(process_time_ns=0, thread_time_ns=0,
                 queue_stats=dict(available=True, delta=dict(action_thread_cpu_ns=0))),
            dict(process_time_ns=100, thread_time_ns=0,
                 queue_stats=dict(available=True, delta=dict(action_thread_cpu_ns=100)))]
    original = json.dumps(rows, sort_keys=True)
    assert bench.cpu_zero_deltas(rows) == dict(process_time_ns=1, thread_time_ns=2,
                                             sender_action_thread_cpu_ns=1)
    assert json.dumps(rows, sort_keys=True) == original
    rows[0]["queue_stats"] = rows[1]["queue_stats"] = dict(available=False, delta=None)
    assert bench.cpu_zero_deltas(rows)["sender_action_thread_cpu_ns"] is None
    for info in bench.clock_metadata().values():
        assert info["implementation"]
        assert info["resolution_seconds"] > 0
        assert info["resolution_ns"] == info["resolution_seconds"] * 1_000_000_000


@pytest.mark.parametrize("mode", ["off", "on"])
@pytest.mark.parametrize("binary", [False, True])
def test_chunked_hidden_input_and_return_preserve_single_offer_accept_result(mode, binary):
    size = 9 * 1024 * 1024
    report = bench.run_bench(samples=1, hidden_size=size, mode=mode, binary=binary)
    assert_clean(report["cleanup"], mode)
    row, = report["samples"]
    assert all(row["checks"].values())
    assert row["application_messages"] == dict(
        total=21, by_sender={"master": {"stage_chunk": 9, "stage_offer": 1},
                             "worker": {"stage_accept": 1, "stage_result_chunk": 9, "stage_result": 1}},
        input_chunks=9, result_chunks=9)
    assert row["output"]["hidden_size_bytes"] == size
    assert_wire_accounting(row)
    if mode == "on":
        # All return chunks and terminal result are one ordered sender action.
        assert row["queue_stats"]["delta"]["submitted"] == 2
    assert len(json.dumps(report)) < 150_000  # Evidence never embeds the 9 MiB hidden.


def test_wire_bytes_measure_real_base64_expansion_in_both_directions():
    reports = [bench.run_bench(samples=1, hidden_size=256 * 1024, mode="on", binary=flag)
               for flag in (False, True)]
    for side in ("master", "worker"):
        json_bytes, binary_bytes = [report["samples"][0]["wire_bytes"][side]["sent_bytes"]
                                    for report in reports]
        assert json_bytes - binary_bytes > 80_000
        assert binary_bytes > 256 * 1024
    for report in reports:
        assert_clean(report["cleanup"], "on")
        assert_wire_accounting(report["samples"][0])


@pytest.mark.parametrize("kwargs", [
    {"samples": 0}, {"samples": -1}, {"samples": 1.5}, {"samples": True},
    {"hidden_size": 0}, {"hidden_size": 7}, {"hidden_size": True},
    {"hidden_size": 64 * 1024 * 1024}, {"hidden_size": bench.MAX_HIDDEN_BYTES + 4},
    {"compute_delay_ms": -1}, {"compute_delay_ms": float("nan")},
    {"compute_delay_ms": "1"}, {"master_accept_delay_ms": float("inf")},
    {"timeout": 0}, {"timeout": None}, {"timeout": 1.1, "compute_delay_ms": 100},
    {"mode": "auto"}, {"binary": "on"}, {"output_kind": "native"},
])
def test_invalid_parameters_fail_before_source_reads_or_socket_creation(kwargs, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid parameters reached benchmark setup")
    monkeypatch.setattr(bench, "Loopback", forbidden)
    monkeypatch.setattr(bench, "source_fingerprints", forbidden)
    with pytest.raises(bench.BenchError):
        bench.run_bench(**kwargs)


@pytest.mark.parametrize("initial", [None, "outer-setting"])
@pytest.mark.parametrize("fail", [False, True])
def test_restores_both_environment_switches_and_loggers_and_closes_sender(initial, fail, monkeypatch):
    keys = (bench.IO_ENV, "QLH_TASK_WORKER_BINARY_HIDDEN")
    for key in keys:
        if initial is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, initial)
    loggers = [logging.getLogger(name) for name in
               ("task_worker_adapter", "scheduler_task_worker", "tcp_comm")]
    for logger in loggers:
        monkeypatch.setattr(logger, "level", logging.ERROR)
        monkeypatch.setattr(logger, "disabled", True)
    handlers = [list(logger.handlers) for logger in loggers]
    sample, links = bench.Loopback.sample, []

    def observed_sample(self, *args):
        links.append(self)
        row = sample(self, *args)
        if fail:
            raise bench.BenchError("injected tool failure after sender use")
        return row

    monkeypatch.setattr(bench.Loopback, "sample", observed_sample)
    if fail:
        with pytest.raises(bench.BenchError, match="injected tool failure"):
            bench.run_bench(samples=1, mode="on")
    else:
        bench.run_bench(samples=1, mode="on")
    link, = links
    assert_clean(link.cleanup, "on")
    assert not any(thread.is_alive() for thread in link.threads)
    link.close()  # Idempotent; no resource gets reopened.
    assert all(os.environ.get(key) == initial for key in keys)
    for logger, previous_handlers in zip(loggers, handlers):
        assert logger.level == logging.ERROR and logger.disabled is True
        assert logger.handlers == previous_handlers


def test_injected_executor_failure_uses_real_stage_error_and_still_cleans_up(monkeypatch):
    from execution_provider import ProviderExecutionError
    links = []

    def fail_compute(self, *_args):
        links.append(self)
        raise bench.BenchError("injected executor failure")

    monkeypatch.setattr(bench.Loopback, "compute", fail_compute)
    with pytest.raises(ProviderExecutionError, match="injected executor failure"):
        bench.run_bench(samples=1, mode="on")
    link, = links
    assert_clean(link.cleanup, "on")
    assert any(e.get("message_type") == "stage_error" for e in link.trace.events)


def test_cleanup_interrupts_sockets_before_explicit_sender_wait(monkeypatch):
    with bench.isolation_switch("on"), bench.merge.binary_switch(True):
        link = bench.Loopback(mode="on", output_kind="hidden", binary=True,
                              compute_delay_ms=0, master_accept_delay_ms=0, timeout=15)
        calls = []
        try:
            link.sample(0, b"\x00\x00\x80\x3f" * 16)
            sender = link.worker._task_worker_io_sender
            original = sender.close

            def checked_close(*, wait=False):
                assert all(stream.fileno() == -1 for stream in link.streams)
                calls.append(wait)
                return original(wait=wait)

            monkeypatch.setattr(sender, "close", checked_close)
        finally:
            link.close()
        assert calls == [True]
        assert_clean(link.cleanup, "on")


def test_idle_snapshot_waits_for_pending_items_and_bytes(monkeypatch):
    link = object.__new__(bench.Loopback)
    link.mode, link.timeout, link.stop = "on", 1, threading.Event()
    snapshots = iter([dict(pending_items=1, pending_bytes=4),
                      dict(pending_items=0, pending_bytes=4),
                      dict(pending_items=0, pending_bytes=0, failed=0, retired=0, cancelled=0)])
    monkeypatch.setattr(link, "sender_snapshot", lambda: next(snapshots))
    assert link.wait_sender_idle()["pending_bytes"] == 0


def test_idle_rejects_missing_sender_when_isolation_requested():
    link = object.__new__(bench.Loopback)
    link.mode, link.timeout, link.stop = "on", 1, threading.Event()
    link.worker = SimpleNamespace()
    with pytest.raises(bench.BenchError, match="did not create"):
        link.wait_sender_idle()


def test_source_fingerprints_cover_tool_loopback_sender_and_scheduler():
    fingerprints = bench.source_fingerprints()
    assert {"scripts/comm_borrow_bench.py", "scripts/comm_merge_bench.py", "src/task_worker_io.py",
            "src/scheduler.py", "src/scheduler_task_worker.py", "src/task_worker_protocol.py",
            "src/task_worker_adapter.py", "src/task_worker_chunks.py", "src/tcp_comm.py"} <= fingerprints.keys()
    for path, digest in fingerprints.items():
        assert digest == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_completed_capture_rejected_if_sources_change_and_resources_already_closed(monkeypatch):
    fingerprints = bench.source_fingerprints()
    calls, links = [], []
    sample = bench.Loopback.sample

    def observed_sample(self, *args):
        links.append(self)
        return sample(self, *args)

    def changed_sources():
        calls.append(True)
        return fingerprints if len(calls) == 1 else {**fingerprints, "src/task_worker_io.py": "0" * 64}

    monkeypatch.setattr(bench.Loopback, "sample", observed_sample)
    monkeypatch.setattr(bench, "source_fingerprints", changed_sources)
    with pytest.raises(bench.BenchError, match="sources changed during run"):
        bench.run_bench(samples=1, mode="on")
    assert len(calls) == 2
    assert_clean(links[0].cleanup, "on")


@pytest.mark.parametrize("args", [
    ["--samples", "no"], ["--binary", "auto"], ["--mode", "auto"],
    ["--hidden-size", "badMiB"], ["--unknown"], ["--samples"],
    ["--samples", "0"], ["--hidden-size", "64MiB"], ["--compute-delay-ms", "nan"],
])
def test_cli_errors_are_json_and_nonzero_before_opening_transport(args, monkeypatch, capsys):
    def forbidden(**_kwargs):
        pytest.fail("invalid CLI reached loopback")
    monkeypatch.setattr(bench, "Loopback", forbidden)
    assert bench.main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == bench.SCHEMA and report["passed"] is False
    assert report["error"] and report["error_stage"] in {"arguments", "benchmark"}


def test_cli_failed_capture_writes_failure_json_to_fresh_nested_path(tmp_path, capsys):
    output = tmp_path / "nested" / "failure.json"
    assert bench.main(["--samples", "0", "--output", str(output)]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is False and "samples" in report["error"]
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["output"] == str(output) and receipt["passed"] is False


@pytest.mark.parametrize("kind", ["existing_file", "nonempty_dir", "file_as_dir", "wrong_suffix", "both"])
def test_output_management_refuses_overwrite_and_bad_directories_before_run(kind, tmp_path, monkeypatch, capsys):
    protected = tmp_path / "protected.json"
    protected.write_text("keep this", encoding="utf-8")
    if kind == "existing_file":
        args = ["--output", str(protected)]
    elif kind == "nonempty_dir":
        args = ["--output-dir", str(tmp_path)]
    elif kind == "file_as_dir":
        args = ["--output-dir", str(protected)]
    elif kind == "wrong_suffix":
        args = ["--output", str(tmp_path / "wrong.txt")]
    else:
        args = ["--output", str(tmp_path / "other.json"), "--output-dir", str(tmp_path / "other")]

    def forbidden(**_kwargs):
        pytest.fail("invalid output destination reached run_bench")
    monkeypatch.setattr(bench, "run_bench", forbidden)
    assert bench.main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False and report["error_stage"] in {"arguments", "output_prepare"}
    assert protected.read_text(encoding="utf-8") == "keep this"
    assert list(tmp_path.iterdir()) == [protected]


def test_standalone_cli_works_outside_repo_and_creates_compact_evidence_directory(tmp_path):
    output_dir = tmp_path / "evidence" / "on-binary-off"
    completed = subprocess.run([
        sys.executable, str(SCRIPT), "--samples", "1", "--hidden-size", "4KiB",
        "--compute-delay-ms", "1", "--mode", "on", "--binary", "off",
        "--output-kind", "hidden", "--output-dir", str(output_dir)],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    receipt = json.loads(completed.stdout)
    assert receipt["passed"] is True
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["schema"] == bench.SCHEMA
    assert report["config"]["hidden_size_bytes"] == 4096
    assert report["config"]["binary_negotiated"] is False
    assert_clean(report["cleanup"], "on")
    assert list(output_dir.iterdir()) == [output_dir / "report.json"]
    assert report["source_sha256"]["scripts/comm_borrow_bench.py"]
    assert_wire_accounting(report["samples"][0])
