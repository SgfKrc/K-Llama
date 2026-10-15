"""Reconnect cleanup must remain behind the current HELLO acknowledgement."""
from copy import deepcopy
import time

import pytest

from tests.test_pipeline_retirement import retirement_case  # noqa: F401
from tests.test_worker_prepare_coordinator import stage_case, identity  # noqa: F401
from tests.test_worker_prepare_races import race_case  # noqa: F401
from task_worker_protocol import build_message, PROTOCOL_VERSION


@pytest.fixture
def reconnect_case(race_case, monkeypatch):
    import scheduler
    case = race_case
    master = case.master
    monkeypatch.setattr(scheduler, "TASK_WORKER_EXPERIMENTAL_ENABLED", True)
    monkeypatch.setattr(master, "_push_node_list_to_client", lambda *_: None)
    monkeypatch.setattr(master, "_ensure_remote_task_worker_provider", lambda *_: None)
    case.pushes = []
    monkeypatch.setattr(master, "push_layer_config_to_clients", lambda: case.pushes.append(True))
    monkeypatch.setattr(master, "request_authoritative_layer_sync", lambda **_: case.pushes.append(True))
    return case


def debt(case, node):
    value = identity(node, [4, 6] if node == "worker" else [6, 8])
    with case.master._layer_config_lock:
        record = case.master._record_pipeline_model_cleanup_locked(
            node_id=node, candidate_config_id=value["config_id"],
            generation=value["generation"] + 1, action="release",
            assignment_id=value["assignment_id"],
            payload={"release": True, "worker_prepare_identity": value},
        )
        case.master._persist_pipeline_lifecycle_locked()
    return record


def hello(master, node):
    message = build_message("hello", {
        "node_id": node,
        "worker_kind": "android_full_worker" if node == "android" else "pc_full_worker",
        "min_version": PROTOCOL_VERSION, "max_version": PROTOCOL_VERSION,
        "capabilities": {"stage_types": ["aggregate"], "engines": ["pytorch"],
                         "models": [], "max_concurrency": 1},
    }, message_id="msg_reconnect_" + node, sent_at_ms=int(time.time() * 1000))
    master._handle_task_worker_message(node, {"data": message.snapshot()})


@pytest.mark.parametrize("node", ["worker", "android"])
@pytest.mark.parametrize("recovery_pending", [False, True])
def test_cleanup_waits_for_hello_ack_then_replays_before_replanning(reconnect_case, node, recovery_pending):
    case = reconnect_case
    master = case.master
    record = debt(case, node)
    master._pipeline_recovery_pending = recovery_pending
    master._on_tcp_registration_confirmed(node)
    # Registration and concurrent retry scans must preserve unsent cleanup debt.
    assert node in master._task_worker_control.pending_worker_ids()
    assert case.sent == []
    assert master._dispatch_pipeline_model_cleanup(record) is False
    assert master._retry_pending_layer_configs(now=time.monotonic() + 120) == 0
    assert case.sent == []
    assert record["cleanup_id"] in master._pipeline_model_cleanup_ledger
    # A retry racing capability validation/ACK emission is still fenced.
    send = master._tcp_server.send_to_client
    def send_with_retry(target, payload, kind):
        if kind.value == "task_worker":
            assert payload["message_type"] == "hello_ack"
            assert master._dispatch_pipeline_model_cleanup(record) is False
        return send(target, payload, kind)
    master._tcp_server.send_to_client = send_with_retry
    hello(master, node)
    assert [packet[2] for packet in case.sent] == ["task_worker", "model_prepare"]
    assert case.sent[0][1]["payload"]["accepted"] is True
    cleanup = case.sent[1][1]
    assert cleanup["action"] == "release" and cleanup["cleanup_id"] == record["cleanup_id"]
    assert cleanup["identity"] == record["payload"]["worker_prepare_identity"]
    assert case.pushes == []
    # Successful delivery is not cleanup completion: only the matching ACK retires debt.
    master._handle_worker_prepare_ack(node, {
        "action": "ack", "phase": "release", "status": "released", "loaded": False,
        "identity": deepcopy(cleanup["identity"]), "cleanup_id": cleanup["cleanup_id"],
    })
    assert record["cleanup_id"] not in master._pipeline_model_cleanup_ledger
    assert case.pushes == [True]


@pytest.mark.parametrize("failure", ["false", "exception"])
def test_failed_hello_ack_does_not_release_cleanup_barrier(reconnect_case, failure):
    case = reconnect_case
    master = case.master
    record = debt(case, "android")
    master._on_tcp_registration_confirmed("android")
    def failed_send(*_):
        if failure == "exception":
            raise ConnectionError("ACK socket failed")
        return False
    master._tcp_server.send_to_client = failed_send
    hello(master, "android")
    assert "android" in master._task_worker_control.pending_worker_ids()
    assert master._dispatch_pipeline_model_cleanup(record) is False
    assert case.sent == [] and case.pushes == []


def test_old_hello_ack_cannot_open_successor_registration(reconnect_case):
    case = reconnect_case
    master = case.master
    record = debt(case, "android")
    master._on_tcp_registration_confirmed("android")
    send = master._tcp_server.send_to_client
    def replace_after_ack(node, payload, kind):
        result = send(node, payload, kind)
        if kind.value == "task_worker":
            master._task_worker_control.mark_worker_connection_pending(
                node, connection_generation=12,
            )
        return result
    master._tcp_server.send_to_client = replace_after_ack
    hello(master, "android")
    assert [packet[2] for packet in case.sent] == ["task_worker"]
    assert "android" in master._task_worker_control.pending_worker_ids()
    assert master._dispatch_pipeline_model_cleanup(record) is False
    assert case.pushes == []


@pytest.mark.parametrize("replacement_confirmed", [False, True])
def test_cleanup_send_is_bound_to_checked_registration(reconnect_case, replacement_confirmed):
    from tcp_comm import TCPServer, ClientConn
    from tests.test_task_worker_hello_owner import RecordingSocket
    case = reconnect_case
    master = case.master
    record = debt(case, "android")
    server = TCPServer()
    old = ClientConn("android", RecordingSocket(), ("127.0.0.1", 1),
                     registration_confirmed=True, registration_epoch=11)
    new = ClientConn("android", RecordingSocket(), ("127.0.0.1", 2),
                     registration_confirmed=replacement_confirmed, registration_epoch=12)
    server.clients["android"] = old
    server._running = True
    master._tcp_server = server
    pending_ids = master._task_worker_control.pending_worker_ids
    def replace_before_send():
        server.clients["android"] = new
        return pending_ids()
    # Deterministic replacement after epoch/fence check, before socket lookup.
    master._task_worker_control.pending_worker_ids = replace_before_send
    assert master._dispatch_pipeline_model_cleanup(record) is False
    assert old.sock.packets == new.sock.packets == []
    assert record["cleanup_id"] in master._pipeline_model_cleanup_ledger


def test_rejected_hello_never_receives_native_cleanup(reconnect_case, monkeypatch):
    case = reconnect_case
    master = case.master
    record = debt(case, "android")
    master._on_tcp_registration_confirmed("android")
    receive = master._task_worker_control.receive_on_coordinator
    def rejected(node, raw, **kwargs):
        raw = deepcopy(raw)
        raw["payload"]["min_version"] = raw["payload"]["max_version"] = 1
        return receive(node, raw, **kwargs)
    monkeypatch.setattr(master._task_worker_control, "receive_on_coordinator", rejected)
    hello(master, "android")
    assert case.sent[0][1]["payload"]["accepted"] is False
    assert [packet[2] for packet in case.sent] == ["task_worker"]
    assert master._dispatch_pipeline_model_cleanup(record) is False
    assert record["cleanup_id"] in master._pipeline_model_cleanup_ledger


def test_superseded_inbound_hello_cannot_ack_new_socket(reconnect_case):
    from tcp_comm import TCPServer, ClientConn
    from tests.test_task_worker_hello_owner import RecordingSocket
    case = reconnect_case
    master = case.master
    record = debt(case, "android")
    server = TCPServer()
    old = ClientConn("android", RecordingSocket(), ("127.0.0.1", 1),
                     registration_confirmed=True, registration_epoch=11)
    new = ClientConn("android", RecordingSocket(), ("127.0.0.1", 2),
                     registration_confirmed=True, registration_epoch=12)
    server.clients["android"] = new
    server._registration_context.client = old
    server._running = True
    master._tcp_server = server
    master._task_worker_control.mark_worker_connection_pending("android", connection_generation=12)
    hello(master, "android")
    assert old.sock.packets == new.sock.packets == []
    assert "android" in master._task_worker_control.pending_worker_ids()
    assert record["cleanup_id"] in master._pipeline_model_cleanup_ledger
    assert case.pushes == []


def _superseded_callback_server(master):
    from tcp_comm import TCPServer, ClientConn
    from tests.test_task_worker_hello_owner import RecordingSocket
    server = TCPServer()
    old = ClientConn("android", RecordingSocket(), ("127.0.0.1", 1),
                     registration_confirmed=True, registration_epoch=11)
    new = ClientConn("android", RecordingSocket(), ("127.0.0.1", 2),
                     registration_confirmed=True, registration_epoch=12)
    server.clients["android"] = new
    server._registration_context.client = old
    server._running = True
    master._tcp_server = server
    return server, old, new


def test_old_hello_cannot_overwrite_successor_rejection(reconnect_case):
    case = reconnect_case
    master = case.master
    record = debt(case, "android")
    server, old, new = _superseded_callback_server(master)
    control = master._task_worker_control
    control.mark_worker_connection_pending("android", connection_generation=12)
    rejected = build_message("hello", {
        "node_id": "android", "worker_kind": "android_full_worker",
        "min_version": 1, "max_version": 1,
        "capabilities": {"stage_types": ["aggregate"], "engines": ["pytorch"],
                         "models": [], "max_concurrency": 1},
    }, message_id="msg_new_rejected", sent_at_ms=int(time.time() * 1000))
    assert not control.receive_on_coordinator(
        "android", rejected.snapshot(), coordinator_node_id="master", connection_generation=12,
    ).payload["accepted"]
    assert control.resolve_worker_connection_pending("android", connection_generation=12)
    before = control.worker_snapshot("android")
    hello(master, "android")
    assert control.worker_snapshot("android") == before
    assert old.sock.packets == new.sock.packets == []
    del server._registration_context.client  # Retry runs outside the stale callback.
    assert master._dispatch_pipeline_model_cleanup(record) is False
    assert record["cleanup_id"] in master._pipeline_model_cleanup_ledger


def test_late_registration_callback_cannot_rearm_successor_pending(reconnect_case):
    case = reconnect_case
    master = case.master
    server, old, new = _superseded_callback_server(master)
    control = master._task_worker_control
    assert control.mark_worker_connection_pending("android", connection_generation=12)
    assert control.resolve_worker_connection_pending("android", connection_generation=12)
    master._on_tcp_registration_confirmed("android")
    assert control.pending_worker_ids() == set()
    # Even a stale direct mark must not replace the authoritative generation.
    assert control.mark_worker_connection_pending("android", connection_generation=0) is False
    assert control.resolve_worker_connection_pending("android", connection_generation=12)
    assert old.sock.packets == new.sock.packets == []
    assert case.pushes == []


def test_authenticated_epoch_cannot_mix_confirmation_and_successor(reconnect_case, monkeypatch):
    from tcp_comm import TCPServer, ClientConn
    from tests.test_task_worker_hello_owner import RecordingSocket
    case = reconnect_case
    master = case.master
    record = debt(case, "android")
    server = TCPServer()
    old = ClientConn("android", RecordingSocket(), ("127.0.0.1", 1),
                     registration_confirmed=True, registration_epoch=11)
    new = ClientConn("android", RecordingSocket(), ("127.0.0.1", 2),
                     registration_confirmed=False, registration_epoch=12)
    server.clients["android"] = old
    server._running = True
    master._tcp_server = server
    getter = server._get_client
    def replace_after_lookup(node):
        current = getter(node)
        if current is old:
            server.clients[node] = new
        return current
    monkeypatch.setattr(server, "_get_client", replace_after_lookup)
    # Confirmation belongs to A, so the authenticated result cannot be B's epoch.
    assert server.get_authenticated_peer_epoch("android") == 11
    assert server.get_authenticated_peer_epoch("android") is None
    server.clients["android"] = old
    def confirm_successor_before_send():
        new.registration_confirmed = True
        return set()
    monkeypatch.setattr(master._task_worker_control, "pending_worker_ids", confirm_successor_before_send)
    assert master._dispatch_pipeline_model_cleanup(record) is False
    assert old.sock.packets == new.sock.packets == []
    assert record["cleanup_id"] in master._pipeline_model_cleanup_ledger
