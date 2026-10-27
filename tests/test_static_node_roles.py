import pytest


@pytest.mark.parametrize("configured_role", ["master", "client"])
def test_scheduler_uses_configured_static_role(monkeypatch, configured_role):
    import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "NODE_ROLE", configured_role)
    scheduler = scheduler_module.Scheduler()
    scheduler._role_override = "client" if configured_role == "master" else "master"

    assert scheduler._effective_role() == configured_role


def test_configured_master_is_never_provisional_or_joinable(monkeypatch):
    import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "NODE_ROLE", "master")
    scheduler = scheduler_module.Scheduler()
    scheduler._master_identity_verified = False
    scheduler._master_identity_reason = "mac_mismatch"
    monkeypatch.setattr(
        scheduler,
        "discover_master",
        lambda **kwargs: pytest.fail("a static master must not discover-and-demote"),
    )

    role = scheduler.get_my_role()

    assert role["node_role"] == "master"
    assert role["is_master"] is True
    assert role["is_provisional"] is False
    assert role["can_join_existing_master"] is False
    assert scheduler.can_join_existing_master() is False


def test_master_startup_identity_mismatch_does_not_demote_or_discover(monkeypatch):
    import scheduler as scheduler_module
    import transport_port

    monkeypatch.setattr(scheduler_module, "NODE_ROLE", "master")
    monkeypatch.setattr(transport_port, "detect_lan_ip", lambda: "100.64.0.10")
    monkeypatch.setattr(transport_port, "get_mac_addresses", lambda: ["001122334455"])
    scheduler = scheduler_module.Scheduler()
    scheduler._running = True
    scheduler._lan_ip = ""
    scheduler._mac_addresses = []
    scheduler._startup_cancel_event.clear()
    monkeypatch.setattr(
        scheduler,
        "_verify_master_identity",
        lambda: setattr(scheduler, "_master_identity_reason", "mac_mismatch"),
    )
    monkeypatch.setattr(
        scheduler,
        "discover_master",
        lambda **kwargs: pytest.fail("identity mismatch must not trigger role discovery"),
    )

    scheduler._initialize_network_identity()

    assert scheduler._effective_role() == "master"
    assert scheduler._master_identity_reason == "mac_mismatch"


def test_client_disconnect_keeps_client_role(monkeypatch):
    import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "NODE_ROLE", "client")
    scheduler = scheduler_module.Scheduler()
    monkeypatch.setattr(scheduler, "_cleanup_task_worker_prepare", lambda *a, **kw: None)

    scheduler._on_master_connection_lost()

    assert scheduler._effective_role() == "client"


def test_scheduler_http_master_connect_rejects_role_switch(monkeypatch):
    from fastapi.testclient import TestClient
    import scheduler as scheduler_module
    import scheduler_svc_http

    monkeypatch.setattr(scheduler_module, "NODE_ROLE", "master")
    scheduler = scheduler_module.Scheduler()
    client = TestClient(scheduler_svc_http.build_scheduler_app(scheduler))
    try:
        response = client.post(
            "/cluster/connect",
            json={"master_host": "127.0.0.1", "master_port": 8888, "switch_to_client": True},
        )
    finally:
        scheduler_svc_http.reset_scheduler()

    assert response.status_code == 403
    assert scheduler._effective_role() == "master"

