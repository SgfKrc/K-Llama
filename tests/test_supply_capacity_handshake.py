"""Supplied-file availability must wait for its exact accepted capability hello."""
from copy import deepcopy

import pytest

import model_supply_capacity as capacity
from model_supply import ModelSupplyError
from tests.test_supply_capacity_evidence import (
    _accept_supply_hello,
    _invalidate,
    _supplied_capabilities,
    _waiting_refresh,
    capacity_case,  # noqa: F401 -- shared real coordinator/control fixture
)


def _old_capabilities(case, field="range", node_id="worker"):
    """Build a protocol-valid hello for an asset other than this assignment."""
    capabilities = _supplied_capabilities(case, node_id)
    artifact = capabilities["layer_artifacts"][0]
    model = capabilities["models"][0]
    if field == "artifact_sha256":
        artifact["artifact_sha256"] = model["sha256"] = "c" * 64
    elif field in {"source_model_sha256", "tokenizer_sha256"}:
        artifact[field] = "c" * 64
    elif field == "range":
        start, end = artifact["layer_range"]
        artifact["layer_range"] = [start - 1, end]
        capabilities["layer_ranges"] = [[start - 1, end]]
    elif field in {"engine", "linked_model"}:
        selected = model["engine"]
        alternate = "pytorch" if selected == "llama_cpp" else "llama_cpp"
        capabilities["engines"] = [selected, alternate]
        if field == "linked_model":
            # Merely finding a correct engine/digest elsewhere in models must
            # not satisfy the artifact's model_id reference.
            decoy = deepcopy(model)
            decoy["model_id"] = "unrelated-correct-model"
            capabilities["models"].append(decoy)
        model["engine"] = alternate
        model["format"] = "safetensors" if alternate == "pytorch" else "gguf"
    elif field == "missing_artifact":
        capabilities.pop("layer_artifacts")
    elif field == "missing_source":
        artifact.pop("source_model_sha256")
    elif field == "missing_tokenizer":
        # The protocol permits the contract extension to be wholly absent.
        for key in ("source_model_id", "hidden_size", "tokenizer_sha256"):
            artifact.pop(key)
    else:
        raise AssertionError(field)
    return capabilities


def _assert_no_probe(case, state):
    assert case.c.get(case.op)["status"] == "available"
    assert not case.server.sent, "resource probe escaped the supplied-capability gate"
    assert not case.local_calls, "local budget was sampled before capability admission"
    assert not state.done.is_set()


def _reply_to_probe(case):
    def send(node_id, request):
        case.reply(node_id, request)
    case.on_send = send


@pytest.mark.parametrize("worker_engine", ["llama_cpp", "pytorch"])
def test_available_with_old_accepted_hello_waits_before_fresh_probe_and_restarts_budget_ttl(
        capacity_case, monkeypatch, worker_engine):
    case = capacity_case(worker_engine=worker_engine)
    connection = deepcopy(case.server.clients["worker"])
    old_hello = _accept_supply_hello(case, capabilities=_old_capabilities(case))
    assert old_hello["accepted"] and old_hello["manual_stage_dispatch_enabled"]
    assert case.c._worker_handshake_ready("worker", case.server.get_client_info("worker"))
    with _waiting_refresh(case, monkeypatch) as state:
        _assert_no_probe(case, state)
        # A fresh response on the old supply operation cannot replace the
        # asynchronously refreshed capability hello.
        case.reply("worker", {"operation_id": case.op})
        _assert_no_probe(case, state)
        with case.c._condition:
            case.clock.elapsed = capacity.MAX_BUDGET_AGE_SECONDS - 0.5
            accepted_at = 1800000000.0 + case.clock.elapsed

            def fresh_probe(node_id, request):
                current = case.c.scheduler._task_worker_control.worker_snapshot(node_id)
                assert current["capabilities"]["layer_ranges"] == [[2, 4]]
                # Sampling after a long hello wait still gets its own 30s TTL.
                case.clock.elapsed += 2
                case.reply(node_id, request)
            case.on_send = fresh_probe
            _accept_supply_hello(case)
        # No supply condition notification accompanies the control hello.
        assert state.done.wait(2)
        assert state.error is None, state.error
        projected, provenance = state.result
        assert projected[1]["layer_ranges"] == [[2, 4]]
        assert projected[1]["capacity_bytes"] == 256
        assert all(item["requested_at"] == accepted_at for item in provenance)
        assert provenance[1]["received_at"] == accepted_at + 2
    assert case.server.clients["worker"] == connection
    assert not case.c._capacity_queries


@pytest.mark.parametrize("field", [
    "artifact_sha256", "source_model_sha256", "tokenizer_sha256", "range",
    "engine", "linked_model", "missing_artifact", "missing_source", "missing_tokenizer",
])
def test_only_accepted_capabilities_for_the_exact_supplied_assignment_release_probe(
        capacity_case, monkeypatch, field):
    case = capacity_case()
    old_hello = _accept_supply_hello(case, capabilities=_old_capabilities(case, field))
    assert old_hello["accepted"] and old_hello["manual_stage_dispatch_enabled"]
    with _waiting_refresh(case, monkeypatch) as state:
        _assert_no_probe(case, state)
        _reply_to_probe(case)
        _accept_supply_hello(case)
        assert state.done.wait(2)
        assert state.error is None, state.error
        assert state.result[0][1]["capacity_bytes"] == 256


@pytest.mark.parametrize("late_node", ["worker", "tail"])
def test_all_remote_assignments_must_publish_their_matching_hello_before_any_probe(
        capacity_case, monkeypatch, late_node):
    case = capacity_case(tail=True)
    _accept_supply_hello(case, late_node, _old_capabilities(case, node_id=late_node))
    with _waiting_refresh(case, monkeypatch) as state:
        _assert_no_probe(case, state)
        _reply_to_probe(case)
        _accept_supply_hello(case, late_node)
        assert state.done.wait(2)
        assert state.error is None, state.error
        assert [record["node_id"] for record in state.result[0]] == ["master", "worker", "tail"]


@pytest.mark.parametrize("change,code", [
    ("cancel", "SUPPLY_PREPARE_CANCELLED"),
    ("expired", "SUPPLY_RUNTIME_CAPABILITY_TIMEOUT"),
    ("connected_at", "SUPPLY_WORKER_DISCONNECTED"),
    ("epoch", "SUPPLY_RUNTIME_CONNECTION_CHANGED"),
    ("superseded", "SUPPLY_OPERATION_STALE"),
    ("plan", "SUPPLY_OPERATION_STALE"),
])
def test_capability_wait_obeys_cancellation_deadline_connection_and_prepare_owner(
        capacity_case, monkeypatch, change, code):
    case = capacity_case()
    _accept_supply_hello(case, capabilities=_old_capabilities(case))
    with _waiting_refresh(case, monkeypatch) as state:
        _assert_no_probe(case, state)
        with case.c._condition:
            if change == "plan":
                case.c._operations[case.op]["plan"]["assignments"][1]["start_layer"] = 1
            else:
                _invalidate(case, change)
        assert state.done.wait(2), "capability wait ignored an invalidated owner/deadline"
        assert isinstance(state.error, ModelSupplyError), state.error
        assert state.error.code == code
        assert state.result is None
        assert not case.server.sent and not case.local_calls
    assert not case.c._capacity_queries


def test_expired_capability_wait_cannot_be_rescued_by_a_late_matching_hello(capacity_case, monkeypatch):
    case = capacity_case()
    _accept_supply_hello(case, capabilities=_old_capabilities(case))
    with _waiting_refresh(case, monkeypatch) as state:
        _assert_no_probe(case, state)
        _reply_to_probe(case)
        with case.c._condition:
            case.clock.elapsed += capacity.MAX_BUDGET_AGE_SECONDS + 1
            _accept_supply_hello(case)
        assert state.done.wait(2)
        assert isinstance(state.error, ModelSupplyError), state.error
        assert state.error.code == "SUPPLY_RUNTIME_CAPABILITY_TIMEOUT"
        assert not case.server.sent and not case.local_calls


@pytest.mark.parametrize("field", ["artifact_sha256", "source_model_sha256", "tokenizer_sha256",
                                   "range", "linked_model"])
def test_same_connection_hello_rollback_is_rechecked_on_each_projection(capacity_case, field):
    case = capacity_case()
    connection = deepcopy(case.server.clients["worker"])
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        assert evidence.project(case.records)[0][1]["capacity_bytes"] == 256
        _accept_supply_hello(case, capabilities=_old_capabilities(case, field))
        assert case.c._worker_handshake_ready("worker", case.server.get_client_info("worker"))
        with pytest.raises(ModelSupplyError) as error:
            evidence.project(case.records)
        assert error.value.code == "SUPPLY_RUNTIME_CAPABILITY_CHANGED"
    assert case.server.clients["worker"] == connection


def test_old_record_snapshot_is_rejected_even_after_current_hello_returns_to_the_supplied_asset(capacity_case):
    case = capacity_case()
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        old = _accept_supply_hello(case, capabilities=_old_capabilities(case))
        stale_records = deepcopy(case.records)
        for field in ("models", "layer_artifacts", "layer_ranges"):
            stale_records[1][field] = deepcopy(old["capabilities"][field])
        before = deepcopy(stale_records)
        _accept_supply_hello(case)
        with pytest.raises(ModelSupplyError) as error:
            evidence.project(stale_records)
        assert error.value.code == "SUPPLY_RUNTIME_CAPABILITY_CHANGED"
        assert stale_records == before
        assert evidence.project(case.records)[0][1]["layer_ranges"] == [[2, 4]]


@pytest.mark.parametrize("field", ["model_id", "engine", "sha256", "format", "missing_models"])
def test_record_model_must_reference_the_same_artifact_with_its_selected_engine_and_format(capacity_case, field):
    case = capacity_case()
    records = deepcopy(case.records)
    if field == "missing_models":
        records[1].pop("models")
    else:
        records[1]["models"][0][field] = {
            "model_id": "unrelated-model", "engine": "pytorch",
            "sha256": "c" * 64, "format": "safetensors",
        }[field]
    before = deepcopy(records)
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        with pytest.raises(ModelSupplyError) as error:
            evidence.project(records)
        assert error.value.code == "SUPPLY_RUNTIME_CAPABILITY_CHANGED"
    assert records == before


def test_physical_artifact_alias_can_change_when_hello_and_record_keep_the_exact_link(capacity_case):
    case = capacity_case()
    capabilities = _supplied_capabilities(case)
    capabilities["models"][0]["model_id"] = "renamed-physical-artifact"
    capabilities["layer_artifacts"][0]["model_id"] = "renamed-physical-artifact"
    _accept_supply_hello(case, capabilities=capabilities)
    records = deepcopy(case.records)
    records[1]["models"] = deepcopy(capabilities["models"])
    records[1]["layer_artifacts"] = deepcopy(capabilities["layer_artifacts"])
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        projected, provenance = evidence.project(records)
        assert projected[1]["models"][0]["model_id"] == "renamed-physical-artifact"
        assert provenance[1]["artifact_sha256"] == case.c.get(case.op)["nodes"]["worker"]["artifact_sha256"]
