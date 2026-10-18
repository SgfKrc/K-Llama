"""Discovery contract tests; fake scheduler, no API/scheduler/model startup.

Isolated invocation (avoids the repository's runtime-oriented conftest):
  .venv-test/Scripts/python.exe -B -m pytest --noconftest -o addopts= -p no:cacheprovider tests/test_model_intent_discovery.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from src import model_intent_discovery as discovery

NONCE = "0123456789abcdef" * 2
PEER = ("127.0.0.1", 45000)


class FakeScheduler:
    def __init__(self):
        self.role = "master"
        self.run_mode = "distributed"
        self.distributed = True
        self.node_id = "master-test"
        self._lan_ip = "100.100.10.20"
        self._tcp_server = SimpleNamespace(port=18888, _running=True)
        self.cluster_secret = "never-send-cluster-secret"
        self.api_token = "never-send-api-token"
        self.private_key = "never-send-private-key"

    def _effective_role(self):
        return self.role

    def _runtime_setting(self, name):
        assert name == "RUN_MODE"
        return self.run_mode

    def get_distributed_inference_enabled(self):
        return self.distributed

    def get_effective_node_id(self):
        return self.node_id

    def get_status(self):
        pytest.fail("discovery must not serialize a runtime status or credentials")

    get_invite_info = get_status


def probe(nonce=NONCE, **extra):
    return json.dumps({"protocol": discovery.PROTOCOL, "nonce": nonce, **extra}).encode()


@pytest.fixture
def responder(monkeypatch):
    monkeypatch.setattr(discovery, "_interface_addresses", lambda: [])
    return discovery.MasterDiscoveryResponder(FakeScheduler(), 18000, port=0, lan_interface="127.0.0.1")


@pytest.fixture
def listening(responder):
    responder.scheduler._lan_ip = "127.0.0.1"
    responder.start()
    try:
        assert responder._ready.wait(2), "listener setup did not finish"
        assert responder.running
        assert responder.bound_port is not None
        yield responder
    finally:
        responder.stop()
        assert not responder._thread.is_alive()
        assert not responder.running
        assert responder.bound_port is None


def test_protocol_nonce_ports_and_no_secrets(responder):
    payload = responder._response(probe(), PEER)
    assert payload is not None and len(payload) <= 1024
    assert json.loads(payload) == {
        "protocol": "qlh-master-discovery-v1", "nonce": NONCE,
        "master_api_host": "100.100.10.20", "master_api_port": 18000,
        "master_tcp_port": 18888, "node_id": "master-test",
    }
    for secret in (responder.scheduler.cluster_secret, responder.scheduler.api_token,
                   responder.scheduler.private_key, "identity_verified", "credential"):
        assert secret.encode() not in payload


def test_callbacks_are_live_not_startup_snapshots(responder):
    api_port = [18001]
    responder.api_port = lambda: api_port[0]
    assert json.loads(responder._response(probe(), PEER))["master_api_port"] == 18001
    api_port[0] = 18002
    responder.scheduler._tcp_server.port = 18889
    responder.scheduler.node_id = "current-master"
    current = json.loads(responder._response(probe("a" * 32), PEER))
    assert (current["master_api_port"], current["master_tcp_port"], current["node_id"]) == (
        18002, 18889, "current-master",
    )


@pytest.mark.parametrize("role,mode,distributed", [
    ("client", "distributed", True), ("auto", "distributed", True),
    ("unknown", "distributed", True), ("master", "local", True),
    ("master", "distributed", False), ("master", "local", False),
    ("master", "distributed", "true"),
])
def test_only_actual_distributed_master_replies(responder, role, mode, distributed):
    responder.scheduler.role = role
    responder.scheduler.run_mode = mode
    responder.scheduler.distributed = distributed
    assert responder._response(probe(), PEER) is None


@pytest.mark.parametrize("nonce", ["", "a" * 31, "a" * 33, "g" * 32, "a" * 32 + "\n", 123, None, []])
def test_invalid_nonce_is_silent(responder, nonce):
    assert responder._response(probe(nonce), PEER) is None
    assert responder._sources == {}


def test_uppercase_hex_nonce_is_echoed_exactly(responder):
    assert json.loads(responder._response(probe("ABCDEF01" * 4), PEER))["nonce"] == "ABCDEF01" * 4


@pytest.mark.parametrize("data", [
    b"", b"not json", b"null", b"[]", b"42", b"{}", b"\xff", b"{",
    b'{"protocol":"other","nonce":"' + NONCE.encode() + b'"}',
    b'{"protocol":"qlh-master-discovery-v1"}',
])
def test_invalid_json_or_protocol_is_silent(responder, data):
    assert responder._response(data, PEER) is None


def test_request_boundary_and_extra_fields(responder):
    request = probe()
    assert responder._response(request + b" " * (1024 - len(request)), PEER) is not None
    assert responder._response(request + b" " * (1025 - len(request)), PEER) is None
    assert responder._response(probe(master_api_host="http://evil.example/path"), PEER) is None
    assert responder._response(probe(cluster_secret="never-echo"), PEER) is None


@pytest.mark.parametrize("field,value", [
    ("api_port", 0), ("api_port", 65536), ("api_port", True), ("api_port", "18000"),
    ("tcp_port", 0), ("tcp_port", 65536), ("tcp_port", False), ("tcp_port", "18888"),
    ("node_id", ""), ("node_id", "x" * 129), ("node_id", "bad\nnode"), ("node_id", None),
])
def test_invalid_runtime_fields_do_not_advertise(responder, field, value):
    if field == "api_port":
        responder.api_port = value
    elif field == "tcp_port":
        responder.scheduler._tcp_server.port = value
    else:
        responder.scheduler.node_id = value
    assert responder._response(probe(), PEER) is None


def test_tcp_listener_must_exist_and_be_running(responder):
    responder.scheduler._tcp_server._running = False
    assert responder._response(probe(), PEER) is None
    responder.scheduler._tcp_server = None
    assert responder._response(probe(), PEER) is None


def test_response_byte_limit_is_not_just_a_character_limit(responder):
    responder.scheduler.node_id = "\U0001f600" * 128
    payload = responder._response(probe(), PEER)
    assert payload is not None and len(payload) <= 1024
    # Exercise the final serialization guard even if future fields grow.
    monkey_host = "x" * 1024
    responder._advertised_host = lambda peer: monkey_host
    assert responder._response(probe(), PEER) is None


@pytest.mark.parametrize("host", [
    "http://192.168.1.2:8000/path", "https://evil.example/", "user@192.168.1.2",
    "192.168.1.2:8000", "192.168.1.2/path", "example.ts.net", "[::1]",
    " 192.168.1.2", "0.0.0.0", "::", "239.255.81.76", "255.255.255.255",
    "fe80::1%Ethernet", None, 123,
])
def test_hosts_are_ip_literals_never_arbitrary_urls(host):
    assert discovery._ip_literal(host) is None


@pytest.mark.parametrize("host", ["192.168.1.2", "127.0.0.1", "100.64.1.2", "fd7a:115c:a1e0::2"])
def test_valid_advertised_ip_formats(host):
    assert discovery._ip_literal(host) == host


@pytest.mark.parametrize("addresses,expected", [
    (["192.168.1.5", "100.64.2.3"], "100.64.2.3"),
    (["192.168.1.5", "fd7a:115c:a1e0::5"], "fd7a:115c:a1e0::5"),
    (["fd7a:115c:a1e0::5", "100.64.2.3"], "100.64.2.3"),
    (["192.168.1.5"], "192.168.1.4"),
])
def test_tailnet_interfaces_win_over_scheduler_lan(responder, monkeypatch, addresses, expected):
    responder.scheduler._lan_ip = "192.168.1.4"
    monkeypatch.setattr(discovery, "_interface_addresses", lambda: addresses)
    assert json.loads(responder._response(probe(), PEER))["master_api_host"] == expected


def test_bad_scheduler_host_falls_back_to_local_route(responder):
    responder.scheduler._lan_ip = "http://evil.example/secret"
    assert json.loads(responder._response(probe(), PEER))["master_api_host"] == "127.0.0.1"


def test_psutil_is_optional(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert discovery._interface_addresses() == []


def test_interface_enumeration_filters_urls_and_handles_failure(monkeypatch, caplog):
    psutil = SimpleNamespace(net_if_addrs=lambda: {"LAN": [
        SimpleNamespace(family=socket.AF_INET, address="192.168.1.4"),
        SimpleNamespace(family=socket.AF_INET, address="http://evil.example/"),
        SimpleNamespace(family=socket.AF_INET6, address="fd7a:115c:a1e0::1"),
        SimpleNamespace(family=socket.AF_INET, address="192.168.1.4"),
    ]})
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    assert discovery._interface_addresses() == ["192.168.1.4", "fd7a:115c:a1e0::1"]
    def fail():
        raise OSError("interface enumeration unavailable")
    psutil.net_if_addrs = fail
    with caplog.at_level(logging.WARNING, logger="model_intent_discovery"):
        assert discovery._interface_addresses() == []
    assert "master_discovery_interfaces_failed" in caplog.text


def test_multicast_joins_lan_interfaces_and_has_default_fallback(responder, monkeypatch):
    memberships = []
    sock = SimpleNamespace(setsockopt=lambda level, option, value: memberships.append((level, option, value)))
    responder.lan_interface = None
    monkeypatch.setattr(discovery, "_interface_addresses", lambda: ["127.0.0.1", "100.64.2.3", "192.168.1.4"])
    responder._join_multicast(sock)
    assert memberships == [(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                            socket.inet_aton(discovery.MULTICAST_GROUP) + socket.inet_aton("192.168.1.4"))]
    responder._addresses_at = float("-inf")
    monkeypatch.setattr(discovery, "_interface_addresses", lambda: [])
    responder._join_multicast(sock)
    assert memberships[-1][2][-4:] == socket.inet_aton("0.0.0.0")


def test_rate_limit_tracks_source_ip_not_source_port(responder, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(discovery.time, "monotonic", lambda: clock[0])
    assert responder._response(probe(), PEER) is not None
    assert responder._response(probe(), (PEER[0], PEER[1] + 1)) is not None
    assert responder._response(probe(), (PEER[0], PEER[1] + 2)) is None
    for index in range(1, discovery.GLOBAL_REPLIES_PER_SECOND - 1):
        assert responder._response(probe(), (f"192.168.1.{index}", 45000)) is not None
    for index in range(50, 200):
        assert responder._response(probe(), (f"192.168.1.{index}", 45000)) is None
    assert len(responder._sources) <= discovery.GLOBAL_REPLIES_PER_SECOND
    clock[0] += 1.0
    assert responder._response(probe(), PEER) is not None
    assert responder._sources == {PEER[0]: 1}


@pytest.mark.parametrize("peer", [("0.0.0.0", 45000), ("239.1.1.1", 45000),
                                  ("255.255.255.255", 45000), ("127.0.0.1", 53)])
def test_no_reflection_to_multicast_broadcast_or_service_ports(responder, peer):
    assert responder._response(probe(), peer) is None


def test_actual_loopback_probe_and_dynamic_role_local_gates(listening):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.bind(("127.0.0.1", 0))
        client.settimeout(0.15)
        target = ("127.0.0.1", listening.bound_port)
        assert listening._socket.getsockname()[0] == "0.0.0.0"
        client.sendto(probe(), target)
        payload, sender = client.recvfrom(1025)
        assert sender == target
        reply = json.loads(payload)
        assert reply["nonce"] == NONCE
        assert reply["master_api_host"] == "127.0.0.1"
        assert len(payload) <= 1024
        # A new window for each role check ensures silence is caused by the gate.
        listening.scheduler.role = "client"
        listening._window_at = float("-inf")
        client.sendto(probe("b" * 32), target)
        with pytest.raises(socket.timeout):
            client.recvfrom(1025)
        listening.scheduler.role = "master"
        listening.scheduler.distributed = False
        listening._window_at = float("-inf")
        client.sendto(probe("c" * 32), target)
        with pytest.raises(socket.timeout):
            client.recvfrom(1025)
        listening.scheduler.distributed = True
        listening._window_at = float("-inf")
        client.sendto(probe("d" * 32), target)
        assert json.loads(client.recvfrom(1025)[0])["nonce"] == "d" * 32


def test_oversized_real_datagram_does_not_kill_listener(listening):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.settimeout(1)
        target = ("127.0.0.1", listening.bound_port)
        client.sendto(probe() + b" " * 2048, target)
        client.sendto(probe("e" * 32), target)
        assert json.loads(client.recvfrom(1025)[0])["nonce"] == "e" * 32
        assert listening.running


def test_start_stop_idempotence_releases_socket_and_port(listening):
    thread, sock, port = listening._thread, listening._socket, listening.bound_port
    listening.start()
    assert listening._thread is thread
    listening.stop()
    listening.stop()
    assert not thread.is_alive()
    assert sock.fileno() == -1
    assert listening.bound_port is None
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reused:
        reused.bind(("0.0.0.0", port))
    listening.start()
    assert listening._ready.wait(2)
    assert listening.running
    assert listening._thread is not thread


def test_stop_before_start_and_immediate_stop_are_real_cleanup(responder):
    responder.stop()
    responder.start()
    thread = responder._thread
    responder.stop()
    assert not thread.is_alive()
    assert responder._socket is None
    assert responder.bound_port is None


def test_start_does_not_wait_for_interface_probe(responder, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    responder.lan_interface = None
    def blocked_probe():
        entered.set()
        assert release.wait(3)
        return []
    monkeypatch.setattr(discovery, "_interface_addresses", blocked_probe)
    started = time.monotonic()
    responder.start()
    elapsed = time.monotonic() - started
    try:
        assert elapsed < 0.5, "start must not wait for interface enumeration"
        assert entered.wait(1), "start must return while the worker is probing interfaces"
    finally:
        release.set()
        responder.stop()
    assert not responder._thread.is_alive()


def test_bind_conflict_is_logged_and_does_not_raise(responder, caplog):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as occupied:
        occupied.bind(("0.0.0.0", 0))
        responder.port = occupied.getsockname()[1]
        with caplog.at_level(logging.WARNING, logger="model_intent_discovery"):
            responder.start()
            assert responder._ready.wait(2)
            responder.stop()
        assert not responder.running
        assert not responder._thread.is_alive()
        assert responder.bound_port is None
        assert "master_discovery_bind_failed" in caplog.text
    responder.start()
    try:
        assert responder._ready.wait(2)
        assert responder.running
    finally:
        responder.stop()


def test_unexpected_socket_close_is_logged_and_thread_exits(listening, caplog):
    with caplog.at_level(logging.WARNING, logger="model_intent_discovery"):
        listening._socket.close()
        listening._thread.join(timeout=2)
    assert not listening._thread.is_alive()
    assert not listening.running
    assert listening.bound_port is None
    assert "master_discovery_receive_failed" in caplog.text


def test_multicast_failure_preserves_unicast(listening, caplog):
    listening.stop()
    # A local but unassigned interface makes membership fail on both platforms.
    listening.lan_interface = "192.0.2.123"
    with caplog.at_level(logging.WARNING, logger="model_intent_discovery"):
        listening.start()
        assert listening._ready.wait(2)
    assert listening.running
    assert "master_discovery_multicast_join_failed" in caplog.text
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.settimeout(1)
        client.sendto(probe(), ("127.0.0.1", listening.bound_port))
        assert json.loads(client.recvfrom(1025)[0])["nonce"] == NONCE


def test_callback_failure_is_logged_and_listener_recovers(listening, caplog):
    def fail():
        raise RuntimeError("scheduler snapshot unavailable")
    original = listening.scheduler.get_effective_node_id
    listening.scheduler.get_effective_node_id = fail
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.settimeout(0.15)
        target = ("127.0.0.1", listening.bound_port)
        with caplog.at_level(logging.WARNING, logger="model_intent_discovery"):
            client.sendto(probe(), target)
            with pytest.raises(socket.timeout):
                client.recvfrom(1025)
        assert "master_discovery_snapshot_failed" in caplog.text
        listening.scheduler.get_effective_node_id = original
        client.sendto(probe("f" * 32), target)
        assert json.loads(client.recvfrom(1025)[0])["nonce"] == "f" * 32


def test_import_does_not_load_runtime_and_supports_top_level_import():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; sys.path.insert(0, 'src'); import model_intent_discovery; "
        "assert not {'api_server', 'scheduler', 'config', 'torch', 'psutil'} & set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_actual_multicast_loopback_probe_receives_unicast_reply(listening):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.bind(("127.0.0.1", 0))
        client.settimeout(1)
        client.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton("127.0.0.1"))
        client.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        client.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        client.sendto(probe("9" * 32), (discovery.MULTICAST_GROUP, listening.bound_port))
        payload, sender = client.recvfrom(1025)
        assert sender == ("127.0.0.1", listening.bound_port)
        assert json.loads(payload)["nonce"] == "9" * 32


def test_invalid_multicast_interface_fails_with_named_log_and_cleanup(responder, caplog):
    responder.lan_interface = "http://evil.example/interface"
    with caplog.at_level(logging.WARNING, logger="model_intent_discovery"):
        responder.start()
        assert responder._ready.wait(2)
        responder.stop()
    assert "master_discovery_multicast_setup_failed" in caplog.text
    assert responder._socket is None
    assert not responder._thread.is_alive()


def test_thread_start_failure_is_observable_without_raising(responder, monkeypatch, caplog):
    def fail_start(thread):
        raise RuntimeError("thread start unavailable")
    monkeypatch.setattr(discovery.threading.Thread, "start", fail_start)
    with caplog.at_level(logging.WARNING, logger="model_intent_discovery"):
        responder.start()
        responder.stop()
    assert "master_discovery_thread_start_failed" in caplog.text
    assert responder._thread is None
    assert not responder.running


def test_socket_creation_failure_is_observable_without_raising(responder, monkeypatch, caplog):
    def fail_socket(*args, **kwargs):
        raise OSError("socket creation unavailable")
    monkeypatch.setattr(discovery.socket, "socket", fail_socket)
    with caplog.at_level(logging.WARNING, logger="model_intent_discovery"):
        responder.start()
        assert responder._ready.wait(2)
        responder.stop()
    assert "master_discovery_socket_failed" in caplog.text
    assert responder._socket is None
    assert not responder._thread.is_alive()


def test_socket_close_failure_is_named(caplog):
    def fail_close():
        raise OSError("close unavailable")
    with caplog.at_level(logging.WARNING, logger="model_intent_discovery"):
        discovery.MasterDiscoveryResponder._close(SimpleNamespace(close=fail_close))
    assert "master_discovery_socket_close_failed" in caplog.text
