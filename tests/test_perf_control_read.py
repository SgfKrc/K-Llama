"""The performance collector rejects incomplete stages and preserves read boundaries."""
import datetime
import json

import pytest

from scripts import perf_control_read
from scripts.perf_control_read import pair_stages, tsv_slice, window_reads


def stage_events():
    events = []
    for step in range(2):
        for index, node in enumerate(("middle", "tail")):
            instant = datetime.datetime(2026, 10, 25, 12, 0, step, index * 200000)
            stage_id = f"req:attempt:step:{step}"
            stamp = instant.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            events.append({"text": f"{stamp} Route-A stage handoff: node={node} stage=layer_forward tokens=1 payload=8 stage_request_id={stage_id}"})
            receipt = {"node_id": node, "stage_request_id": stage_id,
                       "completed_at": instant.astimezone().timestamp() + .1}
            events.append({"text": f"{stamp} Route-A stage completed: {json.dumps(receipt)}"})
    for index, node in enumerate(("middle", "tail")):
        instant = datetime.datetime(2026, 10, 25, 12, 0, 2, index * 200000)
        stamp = instant.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        stage_id = "req:attempt:cleanup"
        events.append({"text": f"{stamp} Route-A stage handoff: node={node} stage=cleanup tokens=1 payload=8 stage_request_id={stage_id}"})
        receipt = {"node_id": node, "stage_request_id": stage_id,
                   "completed_at": instant.astimezone().timestamp() + .1}
        events.append({"text": f"{stamp} Route-A stage completed: {json.dumps(receipt)}"})
    return events


def test_pairing_measures_only_adjacent_tail_completion_intervals():
    events = stage_events()
    receipts, paired, result = pair_stages(events, expected_steps=2)
    assert result["valid"] and not result["errors"]
    assert len(receipts) == len(paired) == 6
    assert result["decode_step_intervals"]["n"] == 1
    assert result["decode_step_intervals"]["mean_s"] == pytest.approx(1)
    assert result["between_steps_outside_remote_window"]["mean_s"] == pytest.approx(.7)


@pytest.mark.parametrize("fault", ["missing_receipt", "conflicting_receipt", "missing_step"])
def test_incomplete_or_conflicting_stage_evidence_is_invalid(fault):
    events = stage_events()
    if fault == "missing_receipt":
        events.pop(1)
    elif fault == "conflicting_receipt":
        damaged = events[1]["text"].split("Route-A stage completed: ", 1)
        receipt = json.loads(damaged[1])
        receipt["completed_at"] += .01
        events.append({"text": damaged[0] + "Route-A stage completed: " + json.dumps(receipt)})
    else:
        events = [event for event in events if ":step:1" not in event["text"]]
    assert pair_stages(events, expected_steps=2)[2]["valid"] is False


def test_profiler_slices_require_complete_rows_and_half_open_time_windows(tmp_path):
    path = tmp_path / "control-reads.tsv"
    payload = b"1.000000\trow\t100\texecutor\n2.000000\trow\t200\texecutor\n"
    path.write_bytes(payload)
    raw, records = tsv_slice(path, 0, len(payload))
    assert raw == payload
    assert window_reads(records, 1, 2)["executor"]["payload_length_sum"] == 100
    assert window_reads(records, 2, 3)["executor"]["payload_length_sum"] == 200
    with pytest.raises(RuntimeError, match="incomplete_profiler_slice"):
        tsv_slice(path, 0, len(payload) - 1)


def test_atomic_ack_retries_windows_sharing_conflict(tmp_path, monkeypatch):
    destination = tmp_path / "ack.json"
    replace = perf_control_read.os.replace
    attempts = []
    def temporarily_locked(source, target):
        attempts.append(target)
        if len(attempts) < 4:
            raise PermissionError("reader temporarily holds the previous ACK")
        replace(source, target)
    monkeypatch.setattr(perf_control_read.os, "replace", temporarily_locked)
    monkeypatch.setattr(perf_control_read.time, "sleep", lambda _seconds: None)
    perf_control_read.atomic_json(destination, {"status": "ok"})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "ok"}
    assert len(attempts) == 4
    assert not list(tmp_path.glob("*.tmp"))


def test_controller_retries_command_when_ack_publication_fails(tmp_path, monkeypatch):
    controller = perf_control_read.Controller(tmp_path, "legacy")
    command = {"command_id": "one", "launch_id": controller.launch_id}
    (tmp_path / "arm-control.json").write_text(json.dumps(command), encoding="utf-8")
    waits = iter((False, False, True))
    monkeypatch.setattr(controller.stopping, "wait", lambda _seconds: next(waits))
    monkeypatch.setattr(controller, "apply", lambda _command: {"status": "ok"})
    atomic = perf_control_read.atomic_json
    attempts = []
    def failed_ack(path, value):
        if path.name == "arm-ack.json":
            attempts.append(path)
            if len(attempts) == 1:
                raise PermissionError("persistent sharing conflict")
        atomic(path, value)
    monkeypatch.setattr(perf_control_read, "atomic_json", failed_ack)
    controller.loop()
    assert len(attempts) == 2 and controller.seen == {"one"}
    assert json.loads((tmp_path / "arm-ack.json").read_text())["status"] == "ok"

