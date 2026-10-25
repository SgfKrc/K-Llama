"""Receipt timing includes a blocking send, independently of waiter scheduling."""
from __future__ import annotations

import logging
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import task_worker_adapter as adapter
from execution_provider import ModelIdentity, StageAttempt, StageRequest
from task_worker_protocol import build_message, canonical_sha256


@pytest.mark.parametrize("upload_ms", [0, 70])
def test_receipts_inside_send_preserve_causal_intervals(monkeypatch, caplog, upload_ms):
    # Only the diagnostic clock is controlled. Lease/deadline clocks stay real.
    clock = [10.0]
    diagnostic_time = SimpleNamespace(
        time=time.time, monotonic=time.monotonic,
        perf_counter=lambda: clock[0],
    )
    monkeypatch.setattr(adapter, "time", diagnostic_time)
    identity = ModelIdentity(model_id="timing-model", engine="pytorch",
        format="safetensors", revision="r1", sha256="a" * 64)
    snapshot = {"healthy": True, "manual_stage_dispatch_enabled": True,
        "selected_version": 3, "worker_kind": "pc_full_worker",
        "capabilities": {"stage_types": ["full_inference"],
            "engines": ["pytorch"], "models": [identity.snapshot()],
            "max_concurrency": 1}}
    outbound = []
    def send(offer):
        outbound.append(offer.message_type)
        fields = {key: offer.payload[key] for key in (
            "workflow_id", "stage_id", "attempt_id", "lease_id",
            "lease_epoch", "provider_id")}
        clock[0] += 0.020
        provider.handle_message(build_message("stage_accept", {
            **fields, "accepted": True, "reason_code": "", "retryable": False},
            message_id="msg_timing_accept01", sent_at_ms=int(time.time() * 1000)).snapshot())
        clock[0] += 0.030
        output = {"content": "answer"}
        provider.handle_message(build_message("stage_result", {
            **fields, "output": output, "output_sha256": canonical_sha256(output),
            "metadata": {}}, message_id="msg_timing_result01",
            sent_at_ms=int(time.time() * 1000)).snapshot())
        # The waiting thread cannot run until transport returns; a/r must retain
        # receipt times rather than this later return/wake time.
        clock[0] += 0.040
    provider = adapter.RemoteFullWorkerProvider(node_id="timing_worker",
        peer_snapshot=lambda: snapshot, send_message=send)
    request = StageRequest(workflow_id="wf_timing01", request_id="request-timing",
        stage_id="candidate_a", stage_type="full_inference",
        provider_id=provider.provider_id, root_input={"message": "hello"},
        dependencies={}, model_identity=identity)
    reservation = provider.reserve(request)
    attempt = StageAttempt(attempt_id="att_timing01", request=request,
        provider_id=provider.provider_id, lease_id="lease_timing01", lease_epoch=1,
        lease_expires_at=time.time() + 10)
    serialized_messages = []
    original_dumps = adapter.json.dumps
    def diagnostic_dumps(value, *args, **kwargs):
        if isinstance(value, adapter.WorkerMessage):
            serialized_messages.append(value.message_type)
        return original_dumps(value, *args, **kwargs)
    monkeypatch.setattr(adapter.json, "dumps", diagnostic_dumps)
    original_upload = provider._maybe_send_stage_chunks
    def upload(attempt):
        clock[0] += upload_ms / 1000
        return original_upload(attempt)
    monkeypatch.setattr(provider, "_maybe_send_stage_chunks", upload)
    try:
        with caplog.at_level(logging.INFO, logger="task_worker_adapter"):
            result = provider.execute(attempt, reservation, threading.Event())
        assert result.output == {"content": "answer"}
        assert outbound == ["stage_offer"]
        assert serialized_messages == []
        messages = [record.getMessage() for record in caplog.records]
        assert "perf2 a=20 r=30" in messages
        assert f"perf3 p=0 u={upload_ms} s=90 t={90 + upload_ms}" in messages
        provider.release(reservation.reservation_id)
        assert provider.inspect().active_reservations == 0
    finally:
        provider.close()
