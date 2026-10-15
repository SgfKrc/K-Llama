import threading
from types import SimpleNamespace
import pytest


def _worker(monkeypatch):
    from scheduler import Scheduler
    import scheduler as scheduler_module
    monkeypatch.setattr(scheduler_module, "TASK_WORKER_EXPERIMENTAL_ENABLED", True)
    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker-reconnect")
    monkeypatch.setattr(scheduler, "get_distributed_inference_enabled", lambda: True)
    sent = []
    from tcp_comm import TCPClient
    client = TCPClient(client_id="worker-reconnect")
    with client._disconnect_callback_lock:
        client._registered = client._running = True
        client._registration_epoch = 1
        client.sock = SimpleNamespace(close=lambda: None, shutdown=lambda *_: None)
    monkeypatch.setattr(client, "send_data", lambda *args, **kwargs: sent.append((args, kwargs)))
    scheduler._tcp_client = client
    capabilities = {"stage_types": ["aggregate"], "engines": ["pytorch"], "models": [],
                    "max_concurrency": 1, "compute_score": 1.0,
                    "battery_pct": None, "memory_available_mb": 1024,
                    "memory_total_mb": 2048}
    return scheduler, client, sent, capabilities


@pytest.mark.parametrize("change", ["epoch", "client", "disconnected"])
def test_slow_capability_inspection_cannot_publish_into_new_registration(monkeypatch, change):
    scheduler, client, sent, capabilities = _worker(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def inspect():
        entered.set()
        assert release.wait(5)
        return capabilities
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", inspect)
    result = []
    thread = threading.Thread(target=lambda: result.append(scheduler._send_task_worker_hello(client)))
    thread.start()
    try:
        assert entered.wait(5)
        if change == "epoch":
            with client._disconnect_callback_lock:
                client._registration_epoch += 1
        elif change == "client":
            scheduler._tcp_client = SimpleNamespace(is_registered=True)
        else:
            client.disconnect()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert result == [False]
    assert sent == []
    assert scheduler._task_worker_control._worker_hello_pending is False


def test_scheduler_auto_reconnect_binding_refreshes_only_current_client(monkeypatch):
    import config
    import scheduler as scheduler_module
    from scheduler import Scheduler
    monkeypatch.setattr(config, "CLUSTER_SECRET", "existing-secret")
    monkeypatch.setattr(scheduler_module, "_configured_node_id", lambda: "worker-reconnect")
    monkeypatch.setattr(scheduler_module, "_sync_runtime_node_config", lambda **kwargs: None)
    class Client:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.is_registered = self._running = False
        def connect(self, on_message=None):
            self.is_registered = self._running = True
            return True
        def send_data(self, *args):
            pass
    monkeypatch.setattr("transport_port.create_client", lambda **kwargs: Client(**kwargs))
    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "_report_local_device_profile", lambda *args: None)
    monkeypatch.setattr(scheduler, "_send_task_worker_hello", lambda *args: True)
    calls, disconnects = [], []
    monkeypatch.setattr(scheduler, "refresh_task_worker_capabilities", lambda: calls.append(True))
    monkeypatch.setattr(scheduler, "_on_master_connection_lost", lambda client: disconnects.append(client))
    assert scheduler.connect_to_master("127.0.0.1", 8888)["status"] == "connected"
    client = scheduler._tcp_client
    client.on_reconnect()
    assert calls == [True]
    # A delayed disconnect callback must not clear a registered successor.
    client.on_disconnect()
    assert disconnects == []
    client.is_registered = False
    client.on_disconnect()
    assert disconnects == [client]
    scheduler._tcp_client = object()
    client.on_reconnect()
    assert calls == [True]
    client.on_disconnect()
    assert disconnects == [client]
