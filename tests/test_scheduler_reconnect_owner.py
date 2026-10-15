"""Scheduler observation cannot compete with a live TCP retry intent."""
from types import SimpleNamespace
import pytest


def worker(monkeypatch):
    from scheduler import Scheduler
    from tcp_comm import TCPClient
    sched = Scheduler(); sched._role_override = "client"
    monkeypatch.setattr(sched, "get_distributed_inference_enabled", lambda: True)
    monkeypatch.setattr(sched, "get_effective_node_id", lambda: "worker-one")
    client = TCPClient(server_host="127.0.0.1", server_port=8888, client_id="worker-one")
    client._running = client._registered = False
    sched._tcp_client = client
    return sched, client


@pytest.mark.parametrize("state", ["idle", "ready", "connecting", "backing_off", "retry_exhausted"])
def test_automatic_observer_reuses_tcp_intent_in_all_handoff_states(monkeypatch, state):
    sched, client = worker(monkeypatch)
    client.reconnect_state = state
    client.reconnect_failure_count = 8 if state == "retry_exhausted" else 3
    monkeypatch.setattr(sched, "_connect_to_master_locked", lambda *a, **kw: pytest.fail("second client"))
    result = sched.connect_to_master("127.0.0.1", 8888, _automatic=True)
    assert result["status"] == ("retry_exhausted" if state == "retry_exhausted" else "reconnecting")
    assert result["reused"] is True and sched._tcp_client is client
    assert client._reconnect_enabled and client.reconnect_failure_count == (8 if state == "retry_exhausted" else 3)


def test_new_automatic_endpoint_stops_previous_intent_before_connect(monkeypatch):
    sched, client = worker(monkeypatch)
    intent = client._reconnect_intent_epoch
    def fresh(host, port, **kw):
        assert not client._reconnect_enabled and client._reconnect_intent_epoch > intent
        assert host == "127.0.0.2" and port == 8889
        return {"status": "connected"}
    monkeypatch.setattr(sched, "_connect_to_master_locked", fresh)
    assert sched.connect_to_master("127.0.0.2", 8889, _automatic=True)["status"] == "connected"


def test_explicit_retry_may_restart_exhausted_intent(monkeypatch):
    sched, client = worker(monkeypatch)
    client.reconnect_state = "retry_exhausted"
    def fresh(*a, **kw):
        assert client.reconnect_state == "stopped" and not client._reconnect_enabled
        return {"status": "connected"}
    monkeypatch.setattr(sched, "_connect_to_master_locked", fresh)
    assert sched.connect_to_master("127.0.0.1", 8888)["status"] == "connected"


def test_startup_observer_does_not_reset_existing_backoff(monkeypatch):
    sched, client = worker(monkeypatch)
    sched._running = True
    monkeypatch.setattr(sched, "_startup_cancel_event", SimpleNamespace(wait=lambda **kw: False))
    monkeypatch.setattr(sched, "discover_master", lambda: pytest.fail("transport already owns retry"))
    sched._auto_connect_on_startup()
    assert sched._tcp_client is client and client._reconnect_enabled


def test_health_monitor_handoff_cannot_spawn_a_competing_client(monkeypatch):
    import scheduler_cluster
    sched, client = worker(monkeypatch)
    sched._running = sched._client_reconnect_enabled = True
    sched._client_master_was_online = True
    sched._client_master_down_since = 0.0
    sched._client_last_reconnect_attempt = 0.0
    monkeypatch.setattr(sched, "discover_master", lambda: {"master_host": "127.0.0.1", "master_port": 8888})
    monkeypatch.setattr(sched, "_connect_to_master_locked", lambda *a, **kw: pytest.fail("competing client"))
    monkeypatch.setattr(sched, "_discover_master_fallbacks", lambda *a: pytest.fail("retry owner reset by fallback"))
    monkeypatch.setattr(scheduler_cluster.time, "sleep", lambda _: setattr(sched, "_running", False))
    sched._client_health_monitor_loop()
    assert sched._tcp_client is client and client._reconnect_enabled
