from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(monkeypatch):
    import api_server

    role = {"value": "client"}
    monkeypatch.setenv("QLH_AUTH_REQUIRED", "1")
    monkeypatch.setattr(
        api_server,
        "scheduler",
        SimpleNamespace(
            _effective_role=lambda: role["value"],
            get_my_role=lambda: {"role": role["value"], "is_master": role["value"] == "master"},
            get_effective_node_id=lambda: "test-node",
            connect_to_master=lambda *args, **kwargs: pytest.fail("master must not connect as a client"),
        ),
    )
    client = TestClient(api_server.app)
    try:
        yield client, role, api_server
    finally:
        client.close()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/auth/capability"),
        ("POST", "/api/auth/login"),
        ("GET", "/api/users"),
        ("POST", "/api/users"),
        ("POST", "/api/cluster/join/request"),
        ("POST", "/api/cluster/join/grant"),
        ("POST", "/api/cluster/join/consume"),
        ("POST", "/api/cluster/credentials/rotate"),
        ("POST", "/api/cluster/control-plane/certificate"),
        ("POST", "/api/cluster/quorum/voter"),
        ("POST", "/api/cluster/transfer-master"),
        ("GET", "/api/cluster/spare-master"),
        ("GET", "/api/cluster/review/tickets"),
    ],
)
def test_retired_account_and_ha_routes_return_404(api_client, method, path):
    client, _role, _api = api_client

    response = client.request(method, path, json={})

    assert response.status_code == 404


def test_auth_required_environment_does_not_gate_api_requests(api_client):
    client, _role, _api = api_client

    response = client.get("/api/cluster/my-role")

    assert response.status_code == 200
    assert response.json()["role"] == "client"


@pytest.mark.parametrize(
    ("role", "leader_id"),
    [("master", "test-node"), ("client", None)],
)
def test_control_plane_reports_static_role_authority(api_client, role, leader_id):
    client, configured_role, _api = api_client
    configured_role["value"] = role

    response = client.get("/api/cluster/control-plane")

    assert response.status_code == 200
    assert response.json() == {
        "enabled": False,
        "available": True,
        "authority": "static_role",
        "role": role,
        "leader_id": leader_id,
    }


def test_static_master_cannot_connect_to_another_master(api_client):
    client, role, _api = api_client
    role["value"] = "master"

    response = client.post(
        "/api/cluster/connect",
        json={
            "master_host": "100.64.0.20",
            "master_port": 8888,
            "switch_to_client": True,
        },
    )

    assert response.status_code == 403
    assert "静态主节点不能切换" in response.json()["detail"]


def test_reset_identity_route_remains_master_only(api_client):
    client, _role, _api = api_client

    response = client.post("/api/cluster/reset-identity", json={"confirm": "reset"})

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "not_master"


def test_user_settings_put_remains_available_without_account_login(api_client, monkeypatch):
    client, _role, api_server = api_client
    persisted = {}
    monkeypatch.setattr(api_server._local_store, "set_local_user_settings", persisted.update)
    monkeypatch.setattr(
        api_server._local_store,
        "get_local_user_settings",
        lambda: dict(persisted),
    )

    response = client.put("/api/user/settings", json={"settings": {"theme": "system"}})

    assert response.status_code == 200
    assert response.json()["synced_fields"] == ["theme"]
    assert client.get("/api/user/settings").json()["settings"] == {"theme": "system"}
