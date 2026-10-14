"""Intent admission uses real authenticated TCP owners, without network I/O."""
from __future__ import annotations

from copy import copy
from types import SimpleNamespace

from fastapi import HTTPException
import pytest
from starlette.requests import Request

import model_intent_access as access
from model_api_access import is_model_api_source_trusted

OPERATION = "supply_" + "a" * 32
PEER = "100.64.10.20"


class PeerSocket:
    """Only the OS socket operations used by registration and source evidence."""
    def __init__(self, host=PEER, port=41000):
        self.peer = (host, port)
        self.closed = False
        self.written = []
        self.on_peer = None

    def fileno(self):
        return -1 if self.closed else 17

    def getpeername(self):
        if self.closed:
            raise OSError("closed socket")
        if self.on_peer:
            self.on_peer()
        return self.peer

    def sendall(self, data):
        if self.closed:
            raise OSError("closed socket")
        self.written.append(data)

    def shutdown(self, *_args):
        pass

    def close(self):
        self.closed = True


def make_request(host=PEER, path="/api/models/intent/catalog", method="GET", headers=None):
    return Request({"type": "http", "method": method, "scheme": "http", "path": path,
                    "query_string": b"", "server": ("100.64.1.2", 18000), "client": (host, 50000),
                    "headers": [(name.lower().encode(), value.encode()) for name, value in (headers or {}).items()]})


@pytest.fixture
def peer(monkeypatch):
    import tcp_comm

    monkeypatch.delenv("QLH_MODEL_API_TRUSTED_CIDRS", raising=False)
    monkeypatch.delenv("QLH_ALLOW_UNPROTECTED_CLUSTER_TCP", raising=False)
    monkeypatch.setattr(tcp_comm, "_get_cluster_secret", lambda: "authenticated-bootstrap-credential-32")
    monkeypatch.setattr(tcp_comm, "_get_cluster_secret_epoch", lambda: 7)
    server = tcp_comm.TCPServer(host="100.64.1.2", port=18888)
    server._running = True
    sock = PeerSocket()
    node_id = "worker-intent-test"
    server._handle_registration(sock, sock.peer, "temporary", {"data": {
        "client_id": node_id, "role": "client", "node_type": "pc",
        "advertised_host": "100.64.99.99", "auth": tcp_comm.build_auth_signature(node_id),
    }})
    assert server.confirm_registration(node_id)
    connection = server.clients[node_id]
    scheduler = SimpleNamespace(
        role="master", enabled=True, run_mode="distributed", tcp_server=server,
    )
    scheduler._effective_role = lambda: scheduler.role
    scheduler.get_distributed_inference_enabled = lambda: scheduler.enabled
    scheduler._scheduler_facade_global = lambda name: scheduler.run_mode if name == "RUN_MODE" else None
    return SimpleNamespace(scheduler=scheduler, server=server, sock=sock, connection=connection,
                           node_id=node_id, monkeypatch=monkeypatch, tcp=tcp_comm)


def denied(peer, request=None):
    with pytest.raises(HTTPException) as failed:
        access.require_model_intent_source(request or make_request(), peer.scheduler)
    assert failed.value.status_code == 403
    assert failed.value.detail["code"] == "MODEL_API_SOURCE_UNTRUSTED"


@pytest.mark.parametrize("method,suffix", [("GET", "catalog"), ("GET", "current"),
                                          ("POST", ""), ("GET", OPERATION), ("DELETE", OPERATION)])
def test_registered_tailnet_source_only_admits_fixed_intent_contract(peer, method, suffix):
    req = make_request(path="/api/models/intent" + ("/" + suffix if suffix else ""), method=method)
    assert not is_model_api_source_trusted(req)
    assert access.require_model_intent_source(req, peer.scheduler) == PEER
    assert access.require_model_intent_source(req, peer.scheduler) == PEER
    assert not is_model_api_source_trusted(req)


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/models/intent"), ("POST", "/api/models/intent/connect"),
    ("POST", "/api/models/intent/catalog"), ("PUT", "/api/models/intent/" + OPERATION),
    ("GET", "/api/models/intent/" + OPERATION + "/files/x/a"),
    ("GET", "/api/models/intent/" + OPERATION + "/prepare"),
    ("GET", "/api/models/intent/supply_" + "A" * 32),
    ("GET", "/api/models/intent/unknown"), ("GET", "/api/models/intent/catalog/"),
    ("GET", "/api/models/registry"), ("GET", "/api/models/downloads"),
    ("POST", "/api/models/load"), ("GET", "/api/models/current"),
    ("GET", "/api/models/supply/" + OPERATION), ("GET", "/api/cluster/nodes"),
    ("GET", "/api/models/intent-unsafe/catalog"),
])
def test_registered_peer_never_expands_other_source_boundaries(peer, method, path):
    req = make_request(path=path, method=method)
    assert not access.is_path_free_intent_request(req)
    denied(peer, req)


@pytest.mark.parametrize("host", ["100.64.10.21", "192.0.2.20", "fd7a:115c:a1e0::123", "unknown", "master.test"])
def test_tailnet_or_claimed_host_without_exact_registered_socket_peer_is_denied(peer, host):
    denied(peer, make_request(host=host, headers={
        "Host": PEER, "Forwarded": "for=" + PEER, "X-Forwarded-For": PEER,
        "X-Real-IP": PEER, "X-QLH-Intent-Hop": "1", "X-QLH-Node-ID": peer.node_id,
    }))


@pytest.mark.parametrize("host", ["fd7a:115c:a1e0::123", "::ffff:100.64.10.20"])
def test_real_tailnet_ipv6_and_mapped_ipv4_are_compared_as_ips(peer, host):
    peer.sock.peer = (host, 41000)
    peer.connection.addr = peer.sock.peer
    assert access.require_model_intent_source(make_request(host), peer.scheduler) == host


@pytest.mark.parametrize("host", ["192.168.1.20", "203.0.113.1", "fd00::1", "100.63.255.254", "fd7a:115c:a1e1::1"])
def test_registration_on_unprotected_or_non_tailnet_peer_does_not_grant_intent(peer, host):
    peer.sock.peer = (host, 41000)
    peer.connection.addr = peer.sock.peer
    peer.monkeypatch.setenv("QLH_ALLOW_UNPROTECTED_CLUSTER_TCP", "1")
    denied(peer, make_request(host))


@pytest.mark.parametrize("state", ["no_server", "stopped", "pending", "unconfirmed", "no_epoch", "bool_epoch",
                                   "no_socket", "closed", "no_peer_method", "wrong_peer", "wrong_port",
                                   "advertised_only", "id_mismatch", "removed", "client", "local", "single"])
def test_missing_or_stale_owner_evidence_fails_closed(peer, state):
    if state == "no_server":
        peer.scheduler.tcp_server = None
    elif state == "stopped":
        peer.server._running = False
    elif state == "pending":
        peer.server._pending_registrations[peer.node_id] = peer.connection
    elif state == "unconfirmed":
        peer.connection.registration_confirmed = False
    elif state == "no_epoch":
        peer.connection.registration_epoch = 0
    elif state == "bool_epoch":
        peer.connection.registration_epoch = True
    elif state == "no_socket":
        peer.connection.sock = None
    elif state == "closed":
        peer.sock.close()
    elif state == "no_peer_method":
        peer.connection.sock = SimpleNamespace(fileno=lambda: 12)
    elif state == "wrong_peer":
        peer.sock.peer = ("100.64.10.21", 41000)
    elif state == "wrong_port":
        peer.connection.addr = (PEER, 41001)
    elif state == "advertised_only":
        peer.connection.advertised_host = PEER
        peer.connection.addr = peer.sock.peer = ("100.64.10.21", 41000)
    elif state == "id_mismatch":
        peer.connection.client_id = "another-worker"
    elif state == "removed":
        peer.server.clients.clear()
    elif state == "client":
        peer.scheduler.role = "client"
        peer.scheduler._tcp_client = SimpleNamespace(_running=True, is_registered=True, sock=peer.sock)
    elif state == "local":
        peer.scheduler.enabled = False
    elif state == "single":
        peer.scheduler.run_mode = "single"
    denied(peer)


@pytest.mark.parametrize("change", ["connection", "socket", "epoch", "peer_port", "peer_ip", "removed", "server", "role", "closed"])
def test_same_http_request_cannot_adopt_a_reconnected_owner(peer, change):
    req = make_request()
    access.require_model_intent_source(req, peer.scheduler)
    if change == "connection":
        peer.server.clients[peer.node_id] = copy(peer.connection)
    elif change == "socket":
        peer.connection.sock = PeerSocket()
    elif change == "epoch":
        peer.connection.registration_epoch += 1
    elif change == "peer_port":
        peer.connection.addr = peer.sock.peer = (PEER, 41001)
    elif change == "peer_ip":
        peer.connection.addr = peer.sock.peer = ("100.64.10.21", 41000)
    elif change == "removed":
        peer.server.clients.clear()
    elif change == "server":
        peer.scheduler.tcp_server = copy(peer.server)
    elif change == "role":
        peer.scheduler.role = "client"
    elif change == "closed":
        peer.sock.close()
    denied(peer, req)


def test_new_http_request_can_use_fresh_confirmed_registration(peer):
    old = make_request()
    access.require_model_intent_source(old, peer.scheduler)
    new_sock = PeerSocket()
    peer.server._handle_registration(new_sock, new_sock.peer, "next", {"data": {
        "client_id": peer.node_id, "role": "client", "auth": peer.tcp.build_auth_signature(peer.node_id),
    }})
    denied(peer, old)
    denied(peer)  # new registration is still pending
    assert peer.server.confirm_registration(peer.node_id)
    assert access.require_model_intent_source(make_request(), peer.scheduler) == PEER
    denied(peer, old)


@pytest.mark.parametrize("change", ["socket", "epoch", "owner", "role", "addr", "closed"])
def test_peer_lookup_races_cannot_publish_stale_admission(peer, change):
    def mutate():
        if change == "socket":
            peer.connection.sock = PeerSocket()
        elif change == "epoch":
            peer.connection.registration_epoch += 1
        elif change == "owner":
            peer.server.clients[peer.node_id] = copy(peer.connection)
        elif change == "role":
            peer.scheduler.role = "client"
        elif change == "addr":
            peer.connection.addr = (PEER, 41001)
        elif change == "closed":
            peer.sock.close()
    peer.sock.on_peer = mutate
    denied(peer)


def test_bad_signature_never_enters_owner_map(peer):
    sock = PeerSocket(host="100.64.10.21")
    with pytest.raises(peer.tcp._RegistrationRejected):
        peer.server._handle_registration(sock, sock.peer, "invalid", {"data": {
            "client_id": "invalid", "role": "client", "auth": {"auth_version": 2, "auth_signature": "forged"},
        }})
    assert "invalid" not in peer.server.clients
    denied(peer, make_request("100.64.10.21"))


def test_credential_rotation_detaches_existing_authority(peer):
    req = make_request()
    access.require_model_intent_source(req, peer.scheduler)
    peer.server.disconnect_all_clients("credential rotation")
    denied(peer, req)
    denied(peer)


@pytest.mark.parametrize("host", ["127.0.0.1", "100.64.10.21"])
def test_original_explicit_source_policy_still_works_without_tcp_owner(peer, host):
    if host != "127.0.0.1":
        peer.monkeypatch.setenv("QLH_MODEL_API_TRUSTED_CIDRS", host + "/32")
    peer.scheduler.tcp_server = None
    req = make_request(host, path="/api/models/intent/connect", method="POST")
    assert access.require_model_intent_source(req, peer.scheduler) == host


def test_headers_cannot_install_an_internal_peer_snapshot(peer):
    peer.server.clients.clear()
    denied(peer, make_request(headers={access._PEER_KEY: "forged", "X-QLH-Intent-Hop": "1"}))


def test_missing_runtime_mode_authority_fails_closed(peer):
    peer.scheduler._scheduler_facade_global = lambda _name: (_ for _ in ()).throw(RuntimeError("mode unavailable"))
    denied(peer)


def test_direct_in_process_calls_keep_original_boundary(peer):
    assert access.require_model_intent_source(None, peer.scheduler) == "unknown"
