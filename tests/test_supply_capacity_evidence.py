"""Operation-bound runtime inventory over the real supply/control entry points."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import threading
from types import SimpleNamespace
import uuid

import pytest

import model_supply_capacity as capacity
from model_supply import ModelSupplyError
from tests.test_model_supply import coordinator, node, worker_hello
from task_worker_adapter import TaskWorkerControlPlane
from transport_port import MessageType


def _inventory(engine, available):
    # The llama_cpp shape is the existing Android inventory response.
    result = {
        "distributed": True, "disk_free_bytes": 10**12,
        "memory_available_bytes": available, "local_cut": False,
        "engines": [engine], "sources": [],
    }
    if engine == "pytorch":
        result["pytorch_runtime"] = {
            "execution_device": "cpu", "target_element_size": 4,
            "available_bytes": available,
        }
    return result


def _supplied_capabilities(case, node_id="worker"):
    """Advertise the physical supplied asset through the real v3 hello shape."""
    operation = case.c.get(case.op)
    assignment = next(a for a in operation["plan"]["assignments"] if a["node_id"] == node_id)
    start, end = assignment["start_layer"], assignment["end_layer"]
    engine = assignment["engine"]
    digest = operation["nodes"][node_id]["artifact_sha256"]
    model_id = "supply-" + digest[:24]
    return {
        "stage_types": ["layer_forward"], "engines": [engine],
        "models": [{"model_id": model_id, "engine": engine,
                    "format": "gguf" if engine == "llama_cpp" else "safetensors",
                    "revision": "local-supply", "sha256": digest}],
        "max_concurrency": 1, "layer_worker": True,
        "layer_ranges": [[start, end]],
        "layer_artifacts": [{
            "layer_range": [start, end],
            "segment_mode": "head" if start == 0 else "tail" if end == operation["plan"]["total_layers"] else "middle",
            "model_id": model_id, "artifact_sha256": digest,
            "source_model_sha256": operation["source_model_sha256"],
            "source_model_id": operation["intent"]["model_id"], "hidden_size": 16,
            "tokenizer_sha256": operation["tokenizer_sha256"],
        }],
    }


def _accept_supply_hello(case, node_id="worker", capabilities=None):
    # Capability refresh is a hello on the existing registration, unlike the
    # worker_hello helper used to exercise reconnect/epoch invalidation.
    if capabilities is None:
        capabilities = _supplied_capabilities(case, node_id)
    epoch = case.server.get_authenticated_peer_epoch(node_id)
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(node_id=node_id, capabilities=deepcopy(capabilities))
    assert hello is not None
    control = case.c.scheduler._task_worker_control
    ack = control.receive_on_coordinator(node_id, hello.snapshot(),
        coordinator_node_id="master", connection_generation=epoch)
    assert ack.payload["accepted"] is True
    assert worker.receive_on_worker(ack.snapshot()).payload["accepted"] is True
    assert case.server.get_authenticated_peer_epoch(node_id) == epoch
    return control.worker_snapshot(node_id)


@pytest.fixture
def capacity_case(tmp_path, monkeypatch):
    # Replace only this module's clock; real control hello timestamps and
    # threading.Condition keep their normal clocks.
    clock = SimpleNamespace(elapsed=0.0)
    monkeypatch.setattr(capacity, "time", SimpleNamespace(
        monotonic=lambda: 1000.0 + clock.elapsed,
        time=lambda: 1800000000.0 + clock.elapsed,
    ))
    cases = []

    def make(*, master_engine="llama_cpp", worker_engine="llama_cpp", tail=False):
        root = tmp_path / str(len(cases))
        root.mkdir()
        c, op, file = coordinator(root)
        server = c.scheduler._tcp_server
        if tail:
            server.clients["tail"] = {
                "connected_at": 1.5, "registration_epoch": 1,
                "registration_confirmed": True,
            }
            worker_hello(c.scheduler, "tail")
            c._connections[(op, "tail")] = 1.5
        engines = {"master": master_engine, "worker": worker_engine}
        if tail:
            engines["tail"] = worker_engine
        assignments = [
            {"node_id": key, "engine": engine, "start_layer": index * 2,
             "end_layer": index * 2 + 2, "layers_count": 2}
            for index, (key, engine) in enumerate(engines.items())
        ]
        c._operations[op].update(
            status="available", master_model_path=str(file), loaded_ready=False,
            source_model_sha256="a" * 64, tokenizer_sha256="b" * 64,
            plan={"admitted": True, "assignments": assignments,
                  "total_layers": len(assignments) * 2},
            nodes={a["node_id"]: {
                "status": "available", "artifact_sha256": str(index + 1) * 64,
                "layer_range": [a["start_layer"], a["end_layer"]],
            } for index, a in enumerate(assignments)},
        )
        c._leases.clear()
        c.scheduler._model_supply_worker.selected_model_path = lambda: file
        inventories = {key: _inventory(engine, 128 + index * 128)
                       for index, (key, engine) in enumerate(engines.items())}
        records = [{**node(a["node_id"], capacity=10**12), "engine": a["engine"],
                    "capacity_source": "cached.profile", "execution_device": "cpu",
                    "layer_ranges": [[a["start_layer"], a["end_layer"]]],
                    "layer_artifacts": [{"artifact_sha256": c._operations[op]["nodes"][a["node_id"]]["artifact_sha256"],
                                         "source_layer_range": [a["start_layer"], a["end_layer"]]}]}
                   for a in assignments]
        profiles = {key: {"ram": {"available_gb": 999},
                          "gpu": {"vram_free_gb": 999}} for key in engines}
        c.scheduler._local_device_profile = profiles["master"]
        c.scheduler.nodes = {key: SimpleNamespace(device_info=value)
                             for key, value in profiles.items()}
        case = SimpleNamespace(c=c, op=op, file=file, server=server, clock=clock,
                               inventories=inventories, records=records,
                               profiles=profiles, on_send=None, on_local=None,
                               local_calls=[])
        for record in records:
            capabilities = _supplied_capabilities(case, record["node_id"])
            record["models"] = deepcopy(capabilities["models"])
            record["layer_artifacts"][0].update(deepcopy(capabilities["layer_artifacts"][0]))
            if record["node_id"] != "master":
                _accept_supply_hello(case, record["node_id"], capabilities)
        case.records_before = deepcopy(records)
        case.profiles_before = deepcopy(profiles)

        def reply(node_id, request=None, /, *, peer=None, **overrides):
            if request is None:
                request = _request(case, node_id)
            payload = {**deepcopy(inventories[node_id]), "action": "inventory_result",
                       "operation_id": request["operation_id"], "node_id": node_id,
                       **overrides}
            c.handle(peer or node_id, payload)
            return payload
        case.reply = reply
        original_send = server.send_to_client

        def send(node_id, payload, message_type):
            original_send(node_id, payload, message_type)
            if payload["action"] == "inventory":
                if case.on_send is not None:
                    return case.on_send(node_id, payload)
                reply(node_id, payload)
            return None
        monkeypatch.setattr(server, "send_to_client", send)

        def local_inventory(**kwargs):
            case.local_calls.append(kwargs)
            assert kwargs == {"model_id": ""}
            if case.on_local is not None:
                case.on_local()
            return inventories["master"]
        c.scheduler._model_supply_worker.inventory = local_inventory
        cases.append(case)
        return case

    yield make
    for case in cases:
        assert not case.c._capacity_queries, "pending capacity query leaked"
        assert case.c._preparing == "", "prepare guard leaked"
        assert case.records == case.records_before
        assert case.profiles == case.profiles_before


def _request(case, node_id="worker"):
    return next(payload for peer, payload, _ in reversed(case.server.sent)
                if peer == node_id and payload["action"] == "inventory")


def _invalidate(case, change):
    if change == "connected_at":
        case.server.clients["worker"]["connected_at"] = 2.0
        worker_hello(case.c.scheduler, "worker")
    elif change == "epoch":
        # Same TCP timestamp, fresh real hello/ACK: checking connected_at alone
        # would accept the old evidence.
        worker_hello(case.c.scheduler, "worker")
    elif change == "ack_pending":
        worker_hello(case.c.scheduler, "worker", acknowledge=False)
    elif change == "unauthenticated":
        case.server.clients["worker"]["registration_confirmed"] = False
    elif change == "cancel":
        case.c.cancel(case.op)
    elif change == "expired":
        case.clock.elapsed += capacity.MAX_BUDGET_AGE_SECONDS + 1
    elif change == "superseded":
        case.c._active = "supply_" + "b" * 32
    else:
        raise AssertionError(change)


_INVALIDATIONS = [
    ("connected_at", "SUPPLY_WORKER_DISCONNECTED"),
    ("epoch", "SUPPLY_RUNTIME_CONNECTION_CHANGED"),
    ("unauthenticated", "SUPPLY_RUNTIME_WORKER_UNAVAILABLE"),
    ("ack_pending", "SUPPLY_RUNTIME_WORKER_UNAVAILABLE"),
    ("cancel", "SUPPLY_PREPARE_CANCELLED"),
    ("expired", "SUPPLY_RUNTIME_BUDGET_EXPIRED"),
    ("superseded", "SUPPLY_OPERATION_STALE"),
]


@contextmanager
def _waiting_refresh(case, monkeypatch):
    """Observe the real condition wait, and always release the test thread."""
    state = SimpleNamespace(waiting=threading.Event(), done=threading.Event(),
                            result=None, error=None)
    original_wait = case.c._condition.wait

    def wait(timeout=None):
        state.waiting.set()
        return original_wait(timeout)
    monkeypatch.setattr(case.c._condition, "wait", wait)
    case.on_send = lambda node_id, payload: None

    def run():
        try:
            with case.c.prepare_guard(case.op):
                evidence = case.c.prepare_capacity_evidence(case.op)
                state.result = evidence.project(case.records)
        except BaseException as exc:
            state.error = exc
        finally:
            state.done.set()
    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    try:
        assert state.waiting.wait(2), state.error
        yield state
    finally:
        if runner.is_alive():
            case.c.cancel(case.op)
            with case.c._condition:
                case.c._condition.notify_all()
        runner.join(2)
        assert not runner.is_alive(), "capacity waiter survived test cleanup"


def test_legacy_android_wire_uses_fresh_supply_uuid_and_current_operation(capacity_case):
    case = capacity_case()
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        projected, provenance = evidence.project(case.records)
        assert not case.c._capacity_queries
    request = _request(case)
    wire_id = request["operation_id"]
    assert wire_id != case.op and wire_id == "supply_" + uuid.UUID(wire_id[7:]).hex
    assert request == {"action": "inventory", "operation_id": wire_id,
                       "model_id": "", "source_model_sha256": ""}
    assert case.server.sent[0][2] == MessageType.MODEL_SUPPLY
    # No query_id, timestamp, epoch or runtime extension is required in Android's reply.
    assert "pytorch_runtime" not in case.inventories["worker"]
    assert [r["capacity_bytes"] for r in projected] == [128, 256]
    assert case.local_calls == [{"model_id": ""}]
    for item, assignment in zip(provenance, case.c.get(case.op)["plan"]["assignments"]):
        assert item["operation_id"] == case.op and item["query_id"] == wire_id
        assert item["layer_range"] == [assignment["start_layer"], assignment["end_layer"]]
        assert item["artifact_sha256"] == case.c.get(case.op)["nodes"][item["node_id"]]["artifact_sha256"]
        assert item["requested_at"] <= item["received_at"]
    assert provenance[1]["registration_epoch"] == case.server.get_authenticated_peer_epoch("worker")
    assert case.c.get(case.op)["loaded_ready"] is False


@pytest.mark.parametrize("bad_reply", ["supply_operation", "previous_query", "unknown_query",
                                       "wrong_node", "wrong_peer", "unselected_peer", "wrong_action"])
def test_old_or_misaddressed_reply_cannot_satisfy_a_new_query(capacity_case, bad_reply):
    case = capacity_case()
    with case.c.prepare_guard(case.op):
        case.c.prepare_capacity_evidence(case.op)
        old_request = deepcopy(_request(case))
        # The normal supply response cache is not fresh runtime evidence either.
        case.c._responses[(case.op, "worker", "inventory_result")] = {
            **deepcopy(case.inventories["worker"]), "action": "inventory_result",
            "operation_id": case.op, "node_id": "worker", "memory_available_bytes": 10**15,
        }

        def send(node_id, request):
            changes = {"memory_available_bytes": 10**15}
            peer = node_id
            if bad_reply == "supply_operation":
                changes["operation_id"] = case.op
            elif bad_reply == "previous_query":
                changes["operation_id"] = old_request["operation_id"]
            elif bad_reply == "unknown_query":
                changes["operation_id"] = "supply_" + "c" * 32
            elif bad_reply == "wrong_node":
                changes["node_id"] = "other"
            elif bad_reply == "wrong_peer":
                peer = "other"
            elif bad_reply == "unselected_peer":
                peer = changes["node_id"] = "other"
            else:
                changes["action"] = "result"
            case.reply(node_id, request, peer=peer, **changes)
            assert not case.c._capacity_queries[request["operation_id"]]["replies"]
            case.clock.elapsed += 0.5
            case.reply(node_id, request)
        case.on_send = send
        evidence = case.c.prepare_capacity_evidence(case.op)
        projected, provenance = evidence.project(case.records)
        assert _request(case)["operation_id"] != old_request["operation_id"]
        assert projected[1]["capacity_bytes"] == 256
        assert provenance[1]["query_id"] == _request(case)["operation_id"]
        assert provenance[1]["received_at"] > provenance[1]["requested_at"]


@pytest.mark.parametrize("change,code", _INVALIDATIONS)
@pytest.mark.parametrize("boundary", ["waiting", "buffered", "project"])
def test_invalidation_fails_closed_and_cleans_pending_queries(capacity_case, monkeypatch,
                                                             change, code, boundary):
    case = capacity_case()
    if boundary == "waiting":
        with _waiting_refresh(case, monkeypatch) as state:
            _invalidate(case, change)
            # No supply callback/notification: the bounded poll must notice it.
            assert state.done.wait(2), "invalidated budget waiter did not stop"
            assert isinstance(state.error, ModelSupplyError), state.error
            assert state.error.code == code
            assert state.result is None
    else:
        with case.c.prepare_guard(case.op):
            if boundary == "buffered":
                def invalidate_after_reply():
                    assert set(case.c._capacity_queries[_request(case)["operation_id"]]["replies"]) == {"worker"}
                    _invalidate(case, change)
                case.on_local = invalidate_after_reply
                with pytest.raises(ModelSupplyError) as error:
                    case.c.prepare_capacity_evidence(case.op)
            else:
                evidence = case.c.prepare_capacity_evidence(case.op)
                _invalidate(case, change)
                with pytest.raises(ModelSupplyError) as error:
                    evidence.project(case.records)
            assert error.value.code == code
    assert not case.c._capacity_queries
    late_reply = case.reply("worker")
    assert not case.c._responses
    assert late_reply["operation_id"] not in case.c._capacity_queries


def test_every_selected_worker_must_reply_from_its_own_transport(capacity_case, monkeypatch):
    case = capacity_case(tail=True)
    with _waiting_refresh(case, monkeypatch) as state:
        case.reply("worker")
        case.reply("tail", peer="worker", memory_available_bytes=10**15)
        with case.c._condition:
            pending = case.c._capacity_queries[_request(case)["operation_id"]]
            assert set(pending["replies"]) == {"worker"}
            assert not state.done.is_set()
        case.reply("tail")
        assert state.done.wait(2)
        assert state.error is None
        projected, provenance = state.result
        assert [r["node_id"] for r in projected] == ["master", "worker", "tail"]
        assert [r["capacity_bytes"] for r in projected] == [128, 256, 384]
        assert len({item["query_id"] for item in provenance}) == 1


@pytest.mark.parametrize("proof", ["register_pending", "ack_pending", "stale_hello", "bool_epoch"])
def test_initial_query_requires_current_authenticated_hello_ack(capacity_case, proof):
    case = capacity_case()
    if proof == "register_pending":
        case.server.clients["worker"]["registration_confirmed"] = False
    elif proof == "ack_pending":
        worker_hello(case.c.scheduler, "worker", acknowledge=False)
    elif proof == "stale_hello":
        hello = case.c.scheduler._task_worker_control.worker_snapshot("worker")
        case.server.clients["worker"]["connected_at"] = hello["hello_received_at"] + 1
    else:
        case.server.clients["worker"]["registration_epoch"] = True
    with case.c.prepare_guard(case.op):
        with pytest.raises(ModelSupplyError) as error:
            case.c.prepare_capacity_evidence(case.op)
        assert error.value.code == "SUPPLY_RUNTIME_WORKER_UNAVAILABLE"
    assert not case.server.sent and not case.local_calls


@pytest.mark.parametrize("engine,node_id", [("llama_cpp", "master"), ("llama_cpp", "worker"),
                                          ("pytorch", "master"), ("pytorch", "worker")])
@pytest.mark.parametrize("budget", [True, -1, 0, None, "missing"])
def test_invalid_or_missing_live_budget_never_uses_cached_capacity(capacity_case, engine, node_id, budget):
    case = capacity_case(master_engine=engine, worker_engine=engine)
    target = case.inventories[node_id]
    key = "memory_available_bytes"
    if engine == "pytorch":
        target = target["pytorch_runtime"]
        key = "available_bytes"
    if budget == "missing":
        target.pop(key)
    else:
        target[key] = budget
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        with pytest.raises(ModelSupplyError) as error:
            evidence.project(case.records)
        assert error.value.code == "SUPPLY_RUNTIME_BUDGET_MISSING"


@pytest.mark.parametrize("node_id", ["master", "worker"])
@pytest.mark.parametrize("runtime", [pytest.param("missing", id="absent"), None, {},
    {"execution_device": "cuda", "available_bytes": 512},
    {"execution_device": "cuda", "available_bytes": 512, "target_element_size": True},
    {"execution_device": "unknown", "available_bytes": 512, "target_element_size": 2},
])
def test_pytorch_without_valid_runtime_cannot_fall_back_to_large_host_ram(capacity_case, node_id, runtime):
    case = capacity_case(master_engine="pytorch", worker_engine="pytorch")
    case.inventories[node_id]["memory_available_bytes"] = 10**15
    if runtime == "missing":
        case.inventories[node_id].pop("pytorch_runtime")
    else:
        case.inventories[node_id]["pytorch_runtime"] = runtime
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        with pytest.raises(ModelSupplyError) as error:
            evidence.project(case.records)
        assert error.value.code == "SUPPLY_RUNTIME_BUDGET_MISSING"


@pytest.mark.parametrize("device,source,multiplier", [
    ("cuda", "inventory.cuda.mem_get_info", 1.0),
    ("cpu", "inventory.ram.available_bytes_live", 2.0),
])
def test_selected_runtime_budget_does_not_mix_vram_host_ram_or_cached_profiles(capacity_case,
                                                                            device, source, multiplier):
    case = capacity_case(master_engine="pytorch")
    case.inventories["master"].update(memory_available_bytes=31, pytorch_runtime={
        "execution_device": device, "available_bytes": 512, "target_element_size": 2,
    })
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        projected, provenance = evidence.project(case.records)
        assert projected[0]["capacity_bytes"] == provenance[0]["capacity_bytes"] == 512
        assert projected[0]["capacity_source"] == provenance[0]["capacity_source"] == source
        assert projected[0]["execution_device"] == device
        assert projected[0]["runtime_multiplier"] == multiplier
        assert projected[1]["capacity_bytes"] == 256
        assert projected[1]["execution_device"] == "cpu"
        for result, original in zip(projected, case.records):
            for field in ("layer_ranges", "layer_artifacts", "reserve_bytes", "score"):
                assert result[field] == original[field]
        # Neither caller-owned output nor later inventory mutation can alter
        # the captured evidence, the source records or cached profile aliases.
        projected[0]["layer_ranges"][0][0] = 99
        projected[0]["layer_artifacts"][0]["artifact_sha256"] = "changed"
        provenance[0]["layer_range"][0] = 99
        case.inventories["master"]["pytorch_runtime"]["available_bytes"] = 10**15
        again, again_provenance = evidence.project(case.records)
        assert again[0]["capacity_bytes"] == 512
        assert again[0]["layer_ranges"] == [[0, 2]]
        assert again_provenance[0]["layer_range"] == [0, 2]


def test_first_reply_is_snapshotted_and_duplicates_cannot_inflate_it(capacity_case):
    case = capacity_case()

    def send(node_id, request):
        first = case.reply(node_id, request)
        first["memory_available_bytes"] = 10**15
        case.reply(node_id, request, memory_available_bytes=10**15)
    case.on_send = send
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        assert evidence.project(case.records)[0][1]["capacity_bytes"] == 256


@pytest.mark.parametrize("failure", ["send_false", "send_error", "local_inventory_error"])
def test_transport_and_local_sampling_errors_always_remove_pending_query(capacity_case, failure):
    case = capacity_case(tail=True)
    if failure == "local_inventory_error":
        def fail_local():
            raise RuntimeError("local inventory failed")
        case.on_local = fail_local
        expected = RuntimeError
    else:
        def send(node_id, request):
            if node_id == "worker":
                case.reply(node_id, request)
            elif failure == "send_false":
                return False
            else:
                raise OSError("transport closed")
        case.on_send = send
        expected = ModelSupplyError
    with case.c.prepare_guard(case.op):
        with pytest.raises(expected) as error:
            case.c.prepare_capacity_evidence(case.op)
        if failure in {"send_false", "send_error"}:
            assert error.value.code == "SUPPLY_RUNTIME_WORKER_UNAVAILABLE"
        assert not case.c._capacity_queries
        case.reply("worker")
        assert not case.c._responses


@pytest.mark.parametrize("change,code", [
    ("plan", "SUPPLY_OPERATION_STALE"),
    ("eligibility", "SUPPLY_RUNTIME_NODE_UNAVAILABLE"),
    ("engine", "SUPPLY_RUNTIME_ENGINE_UNAVAILABLE"),
    ("opt_out", "SUPPLY_RUNTIME_ENGINE_UNAVAILABLE"),
])
def test_projection_keeps_selection_eligibility_and_runtime_gates(capacity_case, change, code):
    case = capacity_case()
    if change == "engine":
        case.inventories["worker"]["engines"] = ["pytorch"]
    elif change == "opt_out":
        case.inventories["worker"]["distributed"] = False
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        records = case.records
        if change == "plan":
            case.c._operations[case.op]["plan"]["assignments"][1]["start_layer"] = 1
        elif change == "eligibility":
            records = case.records[:1]
        with pytest.raises(ModelSupplyError) as error:
            evidence.project(records)
        assert error.value.code == code


def test_prepare_ownership_is_required_both_before_query_and_at_projection(capacity_case):
    case = capacity_case()
    with pytest.raises(ModelSupplyError) as error:
        case.c.prepare_capacity_evidence(case.op)
    assert error.value.code == "SUPPLY_PREPARE_REQUIRED"
    assert not case.server.sent
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
    with pytest.raises(ModelSupplyError) as error:
        evidence.project(case.records)
    assert error.value.code == "SUPPLY_OPERATION_STALE"
