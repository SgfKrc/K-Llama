"""All-position output must survive execution and bind to the negotiated offer."""
from __future__ import annotations
import base64
import copy
import hashlib
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qlh_core.contracts.token_output import (ALL_POSITIONS, INT32_MAX, TokenOutputError,
    validate_token_output, validate_token_output_request, tail_token)
from task_worker_protocol import build_message, stage_input_sha256, canonical_sha256, WorkerProtocolError
from tests.test_engine_host_layer_forward import _request, _host, _FakeUpstream
from tests.test_task_worker_layer_forward import _offer_payload, _layer_fields


def output():
    return {"token_argmax": 4, "token_argmax_by_position": [2, 3, 4],
            "positions": [41, 42, 43], "n_tokens": 3, "n_vocab": 8}


def offer():
    payload = _offer_payload(**_layer_fields(hidden_spec={"n_tokens": 3, "n_embd": 4, "dtype": "float32"},
        positions=[41, 42, 43], seq_ids=[0, 0, 0], token_output=ALL_POSITIONS))
    payload["root_input"] = {"want_hidden": False}
    payload["input_sha256"] = stage_input_sha256(payload["root_input"], payload["dependencies"])
    return payload


def result_payload(value):
    payload = offer()
    return {**{key: payload[key] for key in ("workflow_id", "stage_id", "attempt_id", "lease_id", "lease_epoch", "provider_id")},
            "output": value, "output_sha256": canonical_sha256(value),
            "metadata": {"usage": {}, "usage_estimated": True, "tokens_per_second": 0.0, "model": "fixture"}}


@pytest.mark.parametrize("value", [output(), {"token_argmax": 4}])
def test_protocol_accepts_group_and_legacy_scalar(value):
    envelope = build_message("stage_result", result_payload(value), message_id="msg_multi_result", sent_at_ms=1000, version=3)
    assert envelope.payload["output"] == value


@pytest.mark.parametrize("field,value", [
    ("positions", [41, 43, 44]), ("positions", [41, 42]), ("positions", [41, True, 43]),
    ("positions", [INT32_MAX, INT32_MAX + 1, INT32_MAX + 2]),
    ("n_tokens", 2), ("n_tokens", True), ("n_tokens", 0), ("n_tokens", INT32_MAX + 1),
    ("n_vocab", 4), ("n_vocab", True), ("n_vocab", 0), ("n_vocab", INT32_MAX + 1),
    ("token_argmax_by_position", [2, -1, 4]), ("token_argmax_by_position", [2, 3.5, 4]),
    ("token_argmax_by_position", [2, True, 4]), ("token_argmax_by_position", [2, 8, 4]),
    ("token_argmax_by_position", [2, 4]), ("token_argmax", 3), ("token_argmax", True),
    ("hidden_out_f32", "AAAA"), ("hidden_out_ref", {}),
])
def test_digest_correct_invalid_output_is_rejected(field, value):
    data = output(); data[field] = value
    with pytest.raises(WorkerProtocolError) as error:
        build_message("stage_result", result_payload(data), message_id="msg_multi_bad", sent_at_ms=1000, version=3)
    assert error.value.code == "invalid_token_output"


@pytest.mark.parametrize("field", ["token_argmax_by_position", "positions", "n_vocab", "n_tokens", "token_argmax"])
def test_partial_group_is_rejected(field):
    data = output(); del data[field]
    with pytest.raises(WorkerProtocolError):
        build_message("stage_result", result_payload(data), message_id="msg_multi_partial", sent_at_ms=1000, version=3)


@pytest.mark.parametrize("field,value", [
    ("token_output", "last"), ("token_output", True), ("positions", None),
    ("positions", [41, 43, 44]), ("positions", [-1, 0, 1]),
    ("positions", [41, 42, INT32_MAX + 1]), ("seq_ids", [0, 1, 0]),
    ("seq_ids", [0, True, 0]), ("seq_ids", [INT32_MAX + 1] * 3),
])
def test_protocol_rejects_invalid_explicit_request(field, value):
    data = offer(); data[field] = value
    with pytest.raises(WorkerProtocolError):
        build_message("stage_offer", data, message_id="msg_multi_offer", sent_at_ms=1000, version=3)


def test_request_cannot_mix_hidden_and_tokens():
    data = offer(); data["root_input"]["want_hidden"] = True
    data["input_sha256"] = stage_input_sha256(data["root_input"], data["dependencies"])
    with pytest.raises(WorkerProtocolError, match="token_output"):
        build_message("stage_offer", data, message_id="msg_multi_mixed", sent_at_ms=1000, version=3)


def test_request_explicit_positions_and_single_nonzero_sequence():
    data = offer(); data["seq_ids"] = [7] * 3
    assert build_message("stage_offer", data, message_id="msg_multi_valid", sent_at_ms=1000, version=3).payload["token_output"] == ALL_POSITIONS


@pytest.mark.parametrize("changes", [
    {"n_tokens": 2, "positions": [41, 42], "token_argmax_by_position": [2, 4]},
    {"positions": [42, 43, 44]},
])
def test_pipeline_rejects_self_consistent_result_for_different_offer(changes):
    from scheduler_pipeline import _layer_stage_result_to_pipeline_value
    data = output(); data.update(changes)
    with pytest.raises(TokenOutputError, match="request_mismatch"):
        _layer_stage_result_to_pipeline_value(data, hidden_spec={"n_tokens": 3, "n_embd": 4},
            token_output=ALL_POSITIONS, positions=[41, 42, 43])


def test_pipeline_rejects_missing_and_unsolicited_groups():
    from scheduler_pipeline import _layer_stage_result_to_pipeline_value
    spec = {"n_tokens": 3, "n_embd": 4}
    with pytest.raises(TokenOutputError, match="missing_token_output"):
        _layer_stage_result_to_pipeline_value({"token_argmax": 4}, hidden_spec=spec, token_output=ALL_POSITIONS, positions=[41, 42, 43])
    with pytest.raises(TokenOutputError, match="unexpected_token_output"):
        _layer_stage_result_to_pipeline_value(output(), hidden_spec=spec)
    mapped = _layer_stage_result_to_pipeline_value(output(), hidden_spec=spec, token_output=ALL_POSITIONS, positions=[41, 42, 43])
    assert mapped["token_argmax_by_position"] == [2, 3, 4]
    assert tail_token(mapped) == 4


class MultiUpstream(_FakeUpstream):
    supports_multi_position_argmax = True
    n_vocab = 8
    def forward_hidden_to_tokens(self, hidden, **kwargs):
        self.multi_call = (hidden.copy(), kwargs)
        return [2, 3, 4]


def test_pc_stage_uses_one_multi_forward_and_keeps_scalar_path():
    upstream = MultiUpstream(); host = _host(upstream)
    request = _request(want_hidden=False, positions=[41, 42, 43], seq_ids=[0] * 3)
    request.stage_fields["token_output"] = ALL_POSITIONS
    value = host._execute_layer_forward_stage(request, threading.Event())
    assert value == output()
    assert upstream.token_calls == [] and upstream.reset_count == 0
    assert upstream.multi_call[1]["positions"] == [41, 42, 43]
    del request.stage_fields["token_output"]
    assert host._execute_layer_forward_stage(request, threading.Event()) == {"token_argmax": 42}


def test_old_native_rejects_unavailable_multi_before_reset_or_decode():
    upstream = _FakeUpstream(); host = _host(upstream)
    request = _request(want_hidden=False, positions=[0, 1, 2]); request.stage_fields["token_output"] = ALL_POSITIONS
    with pytest.raises(Exception, match="multi_position_argmax_unavailable"):
        host._execute_layer_forward_stage(request, threading.Event())
    assert upstream.reset_count == 0 and upstream.token_calls == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 8])
def test_pc_rejects_invalid_native_group(bad):
    upstream = MultiUpstream()
    upstream.forward_hidden_to_tokens = lambda *_a, **_kw: [bad, 3, 4]
    request = _request(want_hidden=False, positions=[41, 42, 43]); request.stage_fields["token_output"] = ALL_POSITIONS
    with pytest.raises(TokenOutputError):
        _host(upstream)._execute_layer_forward_stage(request, threading.Event())


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf")])
def test_pytorch_tail_returns_every_row_and_rejects_nonfinite(monkeypatch, bad):
    import torch
    from inference_service.engine_host import EngineHost
    host = EngineHost.__new__(EngineHost)
    manifest = {"layer_range": [1, 2], "total_layers": 2, "hidden_size": 4, "has_lm_head": True}
    selected = {"manifest": manifest}
    logits = torch.tensor([[[1., 5., 0.], [7., 0., 1.], [0., 1., 8.]]])
    if bad is not None:
        logits[0, 0, 0] = bad
    seen = []
    def head(hidden):
        seen.append(tuple(hidden.shape)); return logits
    manager = SimpleNamespace(forward_layers=lambda **kw: {"hidden_states": kw["hidden_states"], "cache": object()}, forward_lm_head=head)
    host._supplied_layer_manager_for = lambda *_a: manager
    request = SimpleNamespace(stage_fields={"layer_range": [1, 2]}, workflow_id="wf_multipos001")
    kwargs = dict(want_hidden=False, n_past=0, positions=[0, 1, 2], seq_ids=[0, 0, 0],
                  provider_cancel_event=threading.Event(), want_all_tokens=True)
    if bad is not None:
        with pytest.raises(Exception, match="supplied_logits_invalid"):
            host._execute_supplied_layer_forward(selected, request, np.zeros((3, 4), np.float32), **kwargs)
        assert host._supplied_layer_caches == {}
    else:
        value = host._execute_supplied_layer_forward(selected, request, np.zeros((3, 4), np.float32), **kwargs)
        assert value == {"token_argmax": 2, "token_argmax_by_position": [1, 0, 2], "positions": [0, 1, 2], "n_vocab": 3, "n_tokens": 3}
        assert seen == [(1, 3, 4)]


@pytest.mark.parametrize("declared", [True, False, None])
def test_scheduler_negotiates_multi_and_uses_scalar_fallback(declared):
    from scheduler_pipeline import SchedulerPipelineMixin
    from execution_provider import ModelIdentity
    captured = []
    def execute(attempt, _reservation, _cancel):
        captured.append(attempt.request)
        return SimpleNamespace(output=output() if declared is True else {"token_argmax": 4})
    provider = SimpleNamespace(reserve=lambda r: SimpleNamespace(reservation_id="res_multi"), execute=execute, release=lambda *_a, **_kw: None)
    caps = {"multi_position_argmax": declared} if declared is not None else {}
    sched = SimpleNamespace(_task_worker_control=SimpleNamespace(status=lambda **_kw: {"workers": [{"node_id": "tail", "capabilities": caps}]}),
        _worker_prepare_assignment_loaded=lambda *_a: (True, ""), _ensure_remote_task_worker_provider=lambda *_a: provider)
    assignment = {"start_layer": 1, "end_layer": 2, "worker_prepare_identity": {"operation_id": "op-multi", "config_id": "cfg-multi", "generation": 1,
        "assignment_id": "asg-multi", "artifact_sha256": "a" * 64}}
    result = SchedulerPipelineMixin._execute_layer_stage_offer(sched, node_id="tail", assignment=assignment,
        hidden_states=np.zeros((3, 4), np.float32), model_identity=ModelIdentity(model_id="fixture", engine="llama_cpp", format="gguf", revision="r1", sha256="a" * 64),
        workflow_id="wf_multipos001", request_id="req-multi", stage_id="tail", context_size=256,
        want_hidden=False, want_all_tokens=True, positions=[41, 42, 43], seq_ids=[0] * 3)
    assert (captured[0].stage_fields.get("token_output") == ALL_POSITIONS) is (declared is True)
    assert result["token_argmax"] == 4
    assert ("token_argmax_by_position" in result) is (declared is True)


@pytest.mark.parametrize("mode", ["valid", "shifted", "short", "missing", "unsolicited"])
def test_remote_provider_binds_output_to_pending_offer(mode):
    from execution_provider import StageRequest, StageAttempt, ModelIdentity
    from task_worker_adapter import RemoteFullWorkerProvider, _PendingRemoteAttempt
    fields = _layer_fields(hidden_spec={"n_tokens": 3, "n_embd": 4, "dtype": "float32"},
                           positions=[41, 42, 43], seq_ids=[0] * 3)
    if mode != "unsolicited":
        fields["token_output"] = ALL_POSITIONS
    request = StageRequest(workflow_id="wf_multipos001", request_id="req-multi", stage_id="tail",
        stage_type="layer_forward", provider_id="remote_tail", dependencies={}, root_input={"want_hidden": False},
        model_identity=ModelIdentity(model_id="fixture", engine="llama_cpp", format="gguf", revision="r1", sha256="a" * 64), stage_fields=fields)
    attempt = StageAttempt(attempt_id="att_multipos001", request=request, provider_id="remote_tail",
        lease_id="lease_multipos001", lease_epoch=1, lease_expires_at=time.time() + 30)
    pending = _PendingRemoteAttempt(attempt=attempt, lease_expires_at=attempt.lease_expires_at,
                                    lease_deadline_monotonic=time.monotonic() + 30)
    pending.accepted = True
    provider = RemoteFullWorkerProvider(node_id="tail", peer_snapshot=lambda: {}, send_message=lambda _m: None)
    provider._pending[attempt.attempt_id] = pending
    data = output()
    if mode == "shifted": data["positions"] = [42, 43, 44]
    if mode == "short": data.update(n_tokens=2, positions=[41, 42], token_argmax_by_position=[2, 4])
    if mode == "missing": data = {"token_argmax": 4}
    payload = result_payload(data)
    payload.update(workflow_id="wf_multipos001", stage_id="tail", attempt_id="att_multipos001", lease_id="lease_multipos001", provider_id="remote_tail")
    message = build_message("stage_result", payload, message_id="msg_multi_bound", sent_at_ms=1000, version=3)
    returned = provider.handle_message(message.snapshot())
    assert returned == message
    assert pending.result_event.is_set() and pending.physical_result_received
    kind, result, error = pending.terminal_snapshot()
    if mode == "valid":
        assert kind == "stage_result" and result.output == data and error is None
    else:
        assert kind == "invalid_token_output" and result is None
        assert error.code in {"token_output_request_mismatch", "missing_token_output", "unexpected_token_output"}
    # Duplicate invalid receipts stay idempotent and cannot change the decision.
    assert provider.handle_message(message.snapshot()) == message
    assert pending.terminal_snapshot()[0] == kind


@pytest.mark.parametrize("declaration", [True, False, None, 1, "true"])
def test_capability_is_optional_strict_boolean(declaration):
    import json
    fixture = json.loads((Path(__file__).parent / "fixtures/task_worker_protocol_v1.json").read_text(encoding="utf-8"))
    hello = next(m for m in fixture["messages"] if m["message_type"] == "hello")
    payload = copy.deepcopy(hello["payload"])
    if declaration is not None:
        payload["capabilities"]["multi_position_argmax"] = declaration
    if declaration is not None and type(declaration) is not bool:
        with pytest.raises(WorkerProtocolError):
            build_message("hello", payload, message_id="msg_multi_cap", sent_at_ms=1000, version=3)
    else:
        assert build_message("hello", payload, message_id="msg_multi_cap", sent_at_ms=1000, version=3)


@pytest.mark.parametrize("capability", [True, False, "yes", None])
def test_scheduler_probes_explicit_execution_owner_callback(capability):
    from scheduler_task_worker import SchedulerTaskWorkerMixin
    callbacks = SimpleNamespace(supports_layer_multi_position_argmax=lambda: capability)
    scheduler = SimpleNamespace(_require_callbacks=lambda: callbacks)
    assert SchedulerTaskWorkerMixin._task_worker_multi_position_argmax_available(scheduler) is (capability is True)


def test_api_composition_wires_real_layer_stage_owner(monkeypatch):
    import api_server
    owner = SimpleNamespace(supports_layer_multi_position_argmax=lambda: True)
    monkeypatch.setattr(api_server, "_get_layer_stage_host", lambda: owner)
    assert api_server._scheduler_callbacks.supports_layer_multi_position_argmax() is True


@pytest.mark.parametrize("enabled", [True, False])
def test_production_serial_loop_consumes_last_verified_row_and_keeps_decode_positions(tmp_path, enabled):
    from tests.helpers.linear_application import make_application
    from tests.test_refactor_execution_integration import Tokenizer
    from scheduler_pipeline import _layer_stage_result_to_pipeline_value
    class PromptTokenizer(Tokenizer):
        def __call__(self, *_a, **_kw):
            return {"input_ids": np.array([[1, 2, 3]], dtype=np.int64)}
    def forward(**kwargs):
        count = kwargs["input_ids"].shape[1]
        return {"hidden_states": np.zeros((1, count, 4), np.float32), "cache": object()}
    manager = SimpleNamespace(tokenizer=PromptTokenizer(), engine_type="llama_cpp", forward_layers=forward, clear_pipeline_kv=lambda: None)
    seen = []
    def stage(**kwargs):
        if kwargs.get("cleanup_only"):
            return {"kind": "cleanup", "kv_cleared": True}
        assert kwargs["want_all_tokens"] is enabled
        positions = kwargs["positions"]; seen.append(positions)
        tokens = [42, 43, 42] if positions[0] == 0 else [999]
        if not enabled:
            return {"kind": "token", "token_argmax": tokens[-1]}
        data = {"token_argmax": tokens[-1], "token_argmax_by_position": tokens,
                "positions": positions, "n_tokens": len(tokens), "n_vocab": 1000}
        return _layer_stage_result_to_pipeline_value(data, hidden_spec={"n_tokens": len(tokens), "n_embd": 4},
                                                    token_output=ALL_POSITIONS, positions=positions)
    callbacks = SimpleNamespace(thinking_system_prompt="", build_model_chat_prompt=lambda *_a, **_kw: "prompt", format_model_response=lambda text, *_a, **_kw: (text, None))
    app, _, _ = make_application(tmp_path / "requests.sqlite3", manager, stage, callbacks, boundaries=(0, 4, 8))
    try:
        result = app.run("hello", max_new_tokens=4, **({"_all_position_argmax": True} if enabled else {}))
        assert result["response"] == "A", result
        assert result["metrics"]["stop_reason"] == "eos_token"
        assert seen == [[0, 1, 2], [3]]
        assert result["metrics"]["request"]["execution_state"] == "stopped"
    finally:
        app.requests.close()


def _pending_multi_receipt(stage_type="layer_forward"):
    from execution_provider import StageRequest, StageAttempt, ModelIdentity
    from task_worker_adapter import RemoteFullWorkerProvider, _PendingRemoteAttempt
    payload = offer()
    fields = {key: value for key, value in payload.items() if key in _layer_fields() or key in {"token_output", "positions", "seq_ids"}}
    request = StageRequest(workflow_id=payload["workflow_id"], request_id="req-multi", stage_id=payload["stage_id"],
        stage_type=stage_type, provider_id=payload["provider_id"], dependencies={}, root_input={"want_hidden": False},
        model_identity=ModelIdentity(model_id="fixture", engine="llama_cpp", format="gguf", revision="r1", sha256="a" * 64), stage_fields=fields)
    attempt = StageAttempt(attempt_id=payload["attempt_id"], request=request, provider_id=payload["provider_id"],
        lease_id=payload["lease_id"], lease_epoch=payload["lease_epoch"], lease_expires_at=time.time() + 30)
    pending = _PendingRemoteAttempt(attempt=attempt, lease_expires_at=attempt.lease_expires_at,
        lease_deadline_monotonic=time.monotonic() + 30)
    pending.accepted = True
    provider = RemoteFullWorkerProvider(node_id="tail", peer_snapshot=lambda: {}, send_message=lambda _m: None)
    provider._pending[attempt.attempt_id] = pending
    raw = build_message("stage_result", result_payload(output()), message_id="msg_multi_malformed", sent_at_ms=1000, version=3).snapshot()
    return provider, pending, raw


@pytest.mark.parametrize("change", [
    {"n_vocab": 4}, {"token_argmax": 3}, {"positions": [41, 43, 44]},
    {"token_argmax_by_position": [2, True, 4]}, {"token_argmax_by_position": [2, -1, 4]},
    {"hidden_out_ref": {}}, {"hidden_out_f32": "AAAA"}, {"n_tokens": 2},
])
def test_invalid_wire_group_finishes_matching_pending_without_timeout(change):
    provider, pending, raw = _pending_multi_receipt()
    raw["payload"]["output"].update(change)
    raw["payload"]["output_sha256"] = canonical_sha256(raw["payload"]["output"])
    receipt = provider.handle_message(raw)
    assert receipt.snapshot() == raw
    assert pending.result_event.is_set() and pending.accept_event.is_set()
    assert pending.physical_result_received
    kind, result, error = pending.terminal_snapshot()
    assert kind == "invalid_token_output" and result is None and error.code == "invalid_token_output"
    assert provider.handle_message(raw) == receipt
    changed = copy.deepcopy(raw); changed["payload"]["output"]["n_vocab"] = 1
    changed["payload"]["output_sha256"] = canonical_sha256(changed["payload"]["output"])
    with pytest.raises(WorkerProtocolError, match="message_id was reused"):
        provider.handle_message(changed)


@pytest.mark.parametrize("fault", ["digest", "metadata", "identity", "attempt", "version", "envelope", "size"])
def test_invalid_group_does_not_bypass_receipt_or_pending_fences(fault):
    import task_worker_protocol as protocol
    provider, pending, raw = _pending_multi_receipt()
    raw["payload"]["output"]["n_vocab"] = 4
    raw["payload"]["output_sha256"] = canonical_sha256(raw["payload"]["output"])
    if fault == "digest": raw["payload"]["output_sha256"] = "0" * 64
    if fault == "metadata": raw["payload"]["metadata"] = {"usage": []}
    if fault == "identity": raw["payload"]["lease_epoch"] += 1
    if fault == "attempt": raw["payload"]["attempt_id"] = "att_other"
    if fault == "version": raw["version"] = 99
    if fault == "envelope": raw["unexpected"] = True
    if fault == "size":
        raw["payload"]["output"]["extra"] = "x" * protocol.MAX_MESSAGE_BYTES
        raw["payload"]["output_sha256"] = canonical_sha256(raw["payload"]["output"])
    with pytest.raises(WorkerProtocolError):
        provider.handle_message(raw)
    assert not pending.result_event.is_set() and not getattr(pending, "physical_result_received", False)


def test_invalid_group_keeps_prior_cancellation_terminal():
    provider, pending, raw = _pending_multi_receipt()
    from execution_provider import ProviderExecutionError
    pending.decide_terminal("provider_cancelled", error=ProviderExecutionError("cancelled", code="provider_cancelled", provider_id=provider.provider_id))
    raw["payload"]["output"]["n_vocab"] = 4
    raw["payload"]["output_sha256"] = canonical_sha256(raw["payload"]["output"])
    provider.handle_message(raw)
    assert pending.terminal_snapshot()[0] == "provider_cancelled"
    assert provider.cancel_diagnostics()["late_stage_responses_absorbed"] == 1


@pytest.mark.parametrize("admitted", [True, False])
def test_scheduler_routes_invalid_receipt_only_to_admitted_provider(admitted):
    from scheduler_task_worker import SchedulerTaskWorkerMixin
    from scheduler_types import NodeRole
    provider, pending, raw = _pending_multi_receipt()
    raw["payload"]["output"]["n_vocab"] = 4
    raw["payload"]["output_sha256"] = canonical_sha256(raw["payload"]["output"])
    sched = SimpleNamespace(_effective_role=lambda: "master", _nodes_lock=threading.Lock(),
        nodes={"tail": SimpleNamespace(node_type="pc", role=NodeRole.CLIENT)} if admitted else {},
        _remote_task_worker_providers={"tail": provider},
        _task_worker_control=SimpleNamespace(record_rejection=lambda: None))
    SchedulerTaskWorkerMixin._handle_task_worker_message(sched, "tail", {"data": raw})
    assert pending.result_event.is_set() is admitted
    if admitted: assert pending.terminal_snapshot()[0] == "invalid_token_output"


def test_invalid_wire_group_cannot_be_accepted_by_full_inference_provider():
    provider, pending, raw = _pending_multi_receipt("full_inference")
    raw["payload"]["output"]["n_vocab"] = 4
    raw["payload"]["output_sha256"] = canonical_sha256(raw["payload"]["output"])
    provider.handle_message(raw)
    kind, result, error = pending.terminal_snapshot()
    assert kind == "invalid_token_output" and result is None and error.code == "invalid_token_output"
