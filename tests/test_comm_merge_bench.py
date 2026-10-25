"""COMM-MERGE-01 tool contracts; production adversarial coverage lives elsewhere.

These tests exercise the benchmark's injected fixture over real persistent TCP.
They do not replace production provider, stage handlers, framing, or wire limits.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "comm_merge_bench.py"
SPEC = importlib.util.spec_from_file_location("comm_merge_bench", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def event(row, name, message_type=None, side=None):
    return next(e for e in row["events"] if e["event"] == name
                and (message_type is None or e.get("message_type") == message_type)
                and (side is None or e.get("side") == side))


def assert_clean(report):
    assert report["passed"] is True
    assert report["cleanup"] == {"sockets_closed": True, "live_threads": []}


def assert_frames_match(row):
    starts = [e for e in row["events"] if e["event"] == "frame_send_start"]
    ends = [e for e in row["events"] if e["event"] == "frame_send_end"]
    receives = [e for e in row["events"] if e["event"] == "frame_receive_end"]
    assert len(starts) == len(ends) == len(receives)
    for start in starts:
        end, = [e for e in ends if e["message_id"] == start["message_id"]]
        receive, = [e for e in receives if e["message_id"] == start["message_id"]]
        assert start["ns"] <= end["ns"]
        assert start["ns"] <= receive["ns"]
        assert receive["side"] != start["side"]
        assert receive["message_type"] == start["message_type"]
    for decoded in (e for e in row["events"] if e["event"] == "frame_decoded"):
        # The recv call can begin while idle, before the sample or frame send.
        assert decoded["read_started_ns"] <= decoded["ns"]
        assert "frame_receive_start" not in {e["event"] for e in row["events"]}


@pytest.mark.parametrize("binary", [False, True])
def test_inline_stage_real_hello_causal_order_result_and_persistent_full_duplex(binary):
    report = bench.run_bench(samples=2, hidden_size=16384, binary=binary,
                             compute_delay_ms=1)
    assert_clean(report)
    assert report["config"]["binary_requested"] is binary
    assert report["config"]["binary_negotiated"] is binary
    assert report["config"]["debug_logging"] is False
    assert "injected" in report["scope"] and "not device end-to-end" in report["scope"]
    handshake = [e for e in report["handshake_events"] if e["event"] == "frame_send_start"]
    assert [(e["side"], e["message_type"]) for e in handshake] == [
        ("worker", "hello"), ("master", "hello_ack")]
    assert len({row["attempt_id"] for row in report["samples"]}) == 2
    receiver_threads = set()
    for row in report["samples"]:
        assert all(row["checks"].values())
        assert row["output"] == {"token_argmax": 42}
        assert row["input_chunks"] == 0
        assert row["execute_total_ns"] == (
            event(row, "provider_execute_end")["ns"] - event(row, "provider_execute_start")["ns"])
        assert row["execute_total_ns"] >= row["executor_ns"] >= 1_000_000
        assert row["offer_send_ns"] > 0
        master_sends = [e["message_type"] for e in row["events"]
                        if e["event"] == "frame_send_start" and e["side"] == "master"]
        worker_sends = [e["message_type"] for e in row["events"]
                        if e["event"] == "frame_send_start" and e["side"] == "worker"]
        assert master_sends == ["stage_offer"]
        assert worker_sends == ["stage_accept", "stage_result"]
        assert event(row, "accept_write_start")["ns"] <= event(row, "accept_write_end")["ns"]
        assert event(row, "accept_write_end")["ns"] <= event(row, "executor_start")["ns"]
        assert event(row, "executor_end")["ns"] <= event(row, "frame_send_start", "stage_result")["ns"]
        for kind in ("stage_accept", "stage_result"):
            assert event(row, "frame_receive_end", kind)["ns"] <= event(
                row, "provider_response_start", kind)["ns"] <= event(
                row, "provider_response_end", kind)["ns"]
        offer = event(row, "frame_send_start", "stage_offer")
        assert offer["binary_bytes"] == (16384 if binary else 0)
        assert row["perf_logs"] == row["perf2_logs"]
        assert any(record["text"].startswith("perf2 ") for record in row["perf_logs"])
        assert all(record["index"] == row["index"] for record in row["perf_logs"])
        assert_frames_match(row)
        receiver_threads.update(e["thread"] for e in row["events"]
                                if e["event"] == "frame_receive_end")
    assert receiver_threads == {"comm-merge-recv-master", "comm-merge-recv-worker"}


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("size", [256 * 1024, 4 * 1024 * 1024])
def test_large_inline_frames_use_production_transport_and_scalar_return(binary, size):
    report = bench.run_bench(samples=1, hidden_size=size, binary=binary)
    assert_clean(report)
    row, = report["samples"]
    assert row["input_chunks"] == 0
    assert event(row, "frame_send_start", "stage_offer")["chunked"] is False
    assert event(row, "frame_receive_end", "stage_offer")["binary_bytes"] == (size if binary else 0)
    assert row["output"] == {"token_argmax": 42}
    assert_frames_match(row)


def test_accept_delay_probe_computes_before_master_handles_accept_without_second_send():
    report = bench.run_bench(samples=2, hidden_size=16384, binary=True,
                             master_accept_delay_ms=80)
    assert_clean(report)
    assert report["summary"]["executor_before_master_accept_handling"] == 2
    for row in report["samples"]:
        assert row["executor_before_master_accept_handling"] is True
        delay_start = event(row, "master_accept_delay_start")["ns"]
        delay_end = event(row, "master_accept_delay_end")["ns"]
        assert delay_end - delay_start >= 80_000_000
        assert event(row, "executor_start")["ns"] <= event(row, "executor_end")["ns"] < delay_end
        assert delay_end <= event(row, "provider_response_start", "stage_accept")["ns"]
        master_frames = [e for e in row["events"] if e["event"] == "frame_send_start"
                         and e["side"] == "master"]
        assert [e["message_type"] for e in master_frames] == ["stage_offer"]
        # Receipt decoding already happened before the injected handling delay.
        assert event(row, "frame_receive_end", "stage_accept")["ns"] <= delay_start
        assert row["checks"]["no_second_master_execution_frame"]


@pytest.mark.parametrize("binary", [False, True])
def test_over_budget_hidden_uses_real_chunks_before_single_offer_and_is_reassembled(binary):
    size = 9 * 1024 * 1024
    report = bench.run_bench(samples=1, hidden_size=size, binary=binary)
    assert_clean(report)
    row, = report["samples"]
    master_frames = [e for e in row["events"] if e["event"] == "frame_send_start"
                     and e["side"] == "master"]
    assert [e["message_type"] for e in master_frames] == ["stage_chunk"] * 9 + ["stage_offer"]
    assert row["input_chunks"] == 9
    assert master_frames[-1]["chunked"] is True
    assert all(e["binary_bytes"] == (1024 * 1024 if binary else 0)
               for e in master_frames[:-1])
    assert master_frames[-1]["binary_bytes"] == 0
    offer_received = event(row, "frame_receive_end", "stage_offer")["ns"]
    chunk_receipts = [e for e in row["events"] if e["event"] == "frame_receive_end"
                      and e["message_type"] == "stage_chunk"]
    assert len(chunk_receipts) == 9
    assert max(e["ns"] for e in chunk_receipts) < offer_received
    # The injected callback hashes the reassembled hidden against the request.
    assert row["output"] == {"token_argmax": 42}
    assert all(row["checks"].values())
    assert_frames_match(row)


def test_perf_capture_accepts_perf2_and_perf3_and_ignores_other_threads_and_logs():
    capture = bench.PerfCapture()
    capture.sample_index = 7
    for text, thread in (("perf2 a=1 r=2", threading.main_thread().ident),
                         ("perf3 p=1 u=2 s=3 t=4", threading.main_thread().ident),
                         ("perf3 other request", threading.main_thread().ident + 1),
                         ("unrelated info", threading.main_thread().ident)):
        capture.emit(logging.makeLogRecord(dict(msg=text, args=(), thread=thread)))
    assert [r["text"] for r in capture.rows] == ["perf2 a=1 r=2", "perf3 p=1 u=2 s=3 t=4"]
    assert all(r["index"] == 7 and r["ns"] > 0 for r in capture.rows)


def test_fingerprints_cover_transport_protocol_chunks_and_tool():
    import hashlib
    fingerprints = bench.source_fingerprints()
    assert set(fingerprints) == {
        "src/task_worker_adapter.py", "src/scheduler_task_worker.py", "src/tcp_comm.py",
        "src/task_worker_protocol.py", "src/task_worker_chunks.py", "scripts/comm_merge_bench.py"}
    for path, digest in fingerprints.items():
        assert digest == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


@pytest.mark.parametrize("kwargs", [
    {"samples": 0}, {"samples": 1.5}, {"hidden_size": 0}, {"hidden_size": 7},
    {"hidden_size": bench.MAX_STAGE_PAYLOAD_BYTES + 4},
    {"compute_delay_ms": -1}, {"compute_delay_ms": float("nan")},
    {"master_accept_delay_ms": float("inf")}, {"timeout": 0},
    {"timeout": 1.1, "compute_delay_ms": 100},
])
def test_invalid_parameters_fail_before_opening_sockets(kwargs, monkeypatch):
    def forbidden(**_kwargs):
        pytest.fail("invalid parameters must be rejected before constructing loopback")
    monkeypatch.setattr(bench, "Loopback", forbidden)
    with pytest.raises(bench.BenchError):
        bench.run_bench(**kwargs)


def test_failed_sample_closes_sockets_joins_threads_and_restores_environment_and_logging(monkeypatch):
    logger = logging.getLogger("task_worker_adapter")
    monkeypatch.setattr(logger, "level", logging.ERROR)
    monkeypatch.setattr(logger, "disabled", True)
    monkeypatch.setenv("QLH_TASK_WORKER_BINARY_HIDDEN", "outer-value")
    previous_handlers = list(logger.handlers)
    links = []
    def fail_sample(self, *_args):
        links.append(self)
        raise bench.BenchError("injected tool failure")
    monkeypatch.setattr(bench.Loopback, "sample", fail_sample)
    with pytest.raises(bench.BenchError, match="injected tool failure"):
        bench.run_bench(samples=1)
    link, = links
    assert link.cleanup == {"sockets_closed": True, "live_threads": []}
    assert not any(thread.is_alive() for thread in link.threads)
    link.close()  # cleanup is idempotent
    assert os.environ["QLH_TASK_WORKER_BINARY_HIDDEN"] == "outer-value"
    assert logger.level == logging.ERROR and logger.disabled is True
    assert logger.handlers == previous_handlers


def test_failure_inside_injected_executor_still_joins_stage_and_lease_threads(monkeypatch):
    from execution_provider import ProviderExecutionError
    links = []
    def fail_compute(self, *_args):
        links.append(self)
        raise bench.BenchError("injected executor failure")
    monkeypatch.setattr(bench.Loopback, "compute", fail_compute)
    with pytest.raises(ProviderExecutionError):
        bench.run_bench(samples=1)
    link, = links
    assert link.cleanup == {"sockets_closed": True, "live_threads": []}
    assert not any(thread.is_alive() for thread in link.threads)


def test_source_drift_rejects_completed_capture_without_editing_production(monkeypatch):
    fingerprints = bench.source_fingerprints()
    calls = []
    def changed_fingerprints():
        calls.append(True)
        return fingerprints if len(calls) == 1 else {**fingerprints, "scripts/comm_merge_bench.py": "0" * 64}
    monkeypatch.setattr(bench, "source_fingerprints", changed_fingerprints)
    with pytest.raises(bench.BenchError, match="sources changed during run"):
        bench.run_bench(samples=1)
    assert len(calls) == 2


def test_standalone_cli_supports_label_sizes_binary_and_json_output(tmp_path):
    output = tmp_path / "candidate.json"
    completed = subprocess.run([
        sys.executable, str(SCRIPT), "--samples", "1", "--hidden-size", "16KiB",
        "--binary", "off", "--compute-delay-ms", "0", "--label", "candidate",
        "--output", str(output)], cwd=tmp_path, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    assert_clean(report)
    assert report["label"] == "candidate"
    assert report["config"]["hidden_size_bytes"] == 16384
    assert report["config"]["binary_negotiated"] is False
    assert report["source_sha256"]["scripts/comm_merge_bench.py"]
    assert "read_started_ns" in report["event_notes"]
    assert json.loads(completed.stdout)["passed"] is True


def test_cli_invalid_run_outputs_failed_json_with_nonzero_exit_status(tmp_path):
    output = tmp_path / "failed.json"
    assert bench.main(["--samples", "0", "--output", str(output)]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == bench.SCHEMA
    assert report["passed"] is False
    assert "samples" in report["error"]
