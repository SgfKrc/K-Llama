from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "src")


@pytest.fixture
def join_api(monkeypatch, tmp_path):
    import auth_app
    import auth_service
    import api_server
    from auth_store import ROLE_ADMIN

    monkeypatch.setenv("QLH_SQLITE_PATH", str(tmp_path / "control.sqlite3"))
    monkeypatch.setenv("QLH_CLUSTER_ID", "cluster-test")
    auth_service._reset_for_tests()
    store = auth_service.get_auth_store()
    store.create_user("root", "password123", role=ROLE_ADMIN)
    secret = "JBSWY3DPEHPK3PXP"
    store.bind_totp("root", secret)
    token, _ = store.issue_session("root")
    monkeypatch.setattr(api_server, "_join_ledger_instance", None)
    role = {"value": "client"}
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: role["value"])
    monkeypatch.setattr(api_server.scheduler, "get_effective_node_id", lambda: "client-test")
    monkeypatch.setattr(api_server.scheduler, "can_join_existing_master", lambda: True)
    monkeypatch.setattr(
        api_server.scheduler,
        "connect_to_master",
        lambda host, port, **kwargs: {
            "status": "connected", "master_host": host, "master_port": port,
        },
    )
    with TestClient(api_server.app) as client:
        client.headers.update({"Authorization": f"Bearer {token}"})
        yield client, role, secret
    monkeypatch.setattr(api_server, "_join_ledger_instance", None)
    auth_service._reset_for_tests()


def test_join_request_issue_consume_and_replay_rejected(join_api):
    import auth_app

    client, role, secret = join_api
    created = client.post(
        "/api/cluster/join/request",
        json={"master_endpoint": "[fd7a:115c::1]:8888", "target_node_id": "client-test"},
    )
    assert created.status_code == 200, created.text
    request_code = created.json()["request_code"]
    assert request_code.startswith("qlhjoinreq1.")

    role["value"] = "master"
    denied = client.post(
        "/api/cluster/join/grant",
        json={"request_code": request_code},
    )
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "totp_required"

    issued = client.post(
        "/api/cluster/join/grant",
        json={"request_code": request_code, "otp_code": auth_app.totp(secret)},
    )
    assert issued.status_code == 200, issued.text
    grant_code = issued.json()["grant_code"]
    assert grant_code.startswith("qlhjoin1.")
    assert issued.json()["qr_payload"] == grant_code

    role["value"] = "client"
    consumed = client.post("/api/cluster/join/consume", json={"grant_code": grant_code})
    assert consumed.status_code == 200, consumed.text
    assert consumed.json()["role"] == "client"

    replay = client.post("/api/cluster/join/consume", json={"grant_code": grant_code})
    assert replay.status_code == 409
    assert replay.json()["detail"]["code"] == "request_not_found"


def test_join_grant_requires_master_role(join_api):
    import auth_app

    client, role, secret = join_api
    response = client.post(
        "/api/cluster/join/grant",
        json={"request_code": "qlhjoinreq1.invalid", "otp_code": auth_app.totp(secret)},
    )
    assert response.status_code == 403
    role["value"] = "master"
    response = client.post(
        "/api/cluster/join/grant",
        json={"request_code": "qlhjoinreq1.invalid", "otp_code": auth_app.totp(secret)},
    )
    assert response.status_code == 400


def test_join_grant_without_totp_binding_fails_closed(join_api):
    import auth_service

    client, role, _secret = join_api
    role["value"] = "master"
    assert auth_service.get_auth_store().clear_totp("root") is True
    response = client.post(
        "/api/cluster/join/grant",
        json={"request_code": "qlhjoinreq1.invalid", "otp_code": "123456"},
    )
    assert response.status_code == 501
    assert response.json()["detail"]["code"] == "auth_control_plane_unavailable"


def test_join_grant_rejects_invalid_and_replayed_totp(join_api):
    import auth_app

    client, role, secret = join_api
    role["value"] = "master"
    invalid = client.post(
        "/api/cluster/join/grant",
        json={"request_code": "qlhjoinreq1.invalid", "otp_code": "000000"},
    )
    assert invalid.status_code == 403
    assert invalid.json()["detail"]["code"] == "totp_invalid"

    created = client.post(
        "/api/cluster/join/request",
        json={"master_endpoint": "[fd7a:115c::1]:8888", "target_node_id": "client-test"},
    )
    request_code = created.json()["request_code"]
    code = auth_app.totp(secret)
    issued = client.post(
        "/api/cluster/join/grant",
        json={"request_code": request_code, "otp_code": code},
    )
    assert issued.status_code == 200, issued.text
    replay = client.post(
        "/api/cluster/join/grant",
        json={"request_code": request_code, "otp_code": code},
    )
    assert replay.status_code == 403
    assert replay.json()["detail"]["code"] == "totp_replayed"


def test_join_grant_requires_named_bearer_principal_even_when_auth_is_optional(join_api):
    import auth_app

    client, role, secret = join_api
    role["value"] = "master"
    client.headers.pop("Authorization", None)
    response = client.post(
        "/api/cluster/join/grant",
        json={"request_code": "qlhjoinreq1.invalid", "otp_code": auth_app.totp(secret)},
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "auth_required"


def test_join_grant_rejects_legacy_boolean_field(join_api):
    client, role, _secret = join_api
    role["value"] = "master"
    response = client.post(
        "/api/cluster/join/grant",
        json={"request_code": "qlhjoinreq1.invalid", "auth_verified": True},
    )
    assert response.status_code == 422


def test_cluster_secret_rotation_requires_master_and_tailnet_revocation(join_api):
    import auth_app

    client, role, secret = join_api
    code = auth_app.totp(secret)
    denied_role = client.post(
        "/api/cluster/credentials/rotate",
        json={"otp_code": code, "tailnet_revocation_completed": True},
    )
    assert denied_role.status_code == 403

    role["value"] = "master"
    missing_revocation = client.post(
        "/api/cluster/credentials/rotate",
        json={"otp_code": code, "tailnet_revocation_completed": False},
    )
    assert missing_revocation.status_code == 409
    assert missing_revocation.json()["detail"]["code"] == "tailnet_revocation_required"


def test_cluster_secret_rotation_never_returns_secret_and_disconnects_peers(
    monkeypatch,
):
    import api_server
    import node_config
    from types import SimpleNamespace

    calls = []

    class FakeTCPServer:
        def disconnect_all_clients(self, reason, *, before_disconnect):
            calls.append(reason)
            return before_disconnect(), 3

    monkeypatch.setattr(api_server, "scheduler", SimpleNamespace(
        _effective_role=lambda: "master",
        _tcp_server=FakeTCPServer(),
    ))
    monkeypatch.setattr(
        api_server.auth_service,
        "verify_totp_confirmation",
        lambda principal, code, source: None,
    )
    monkeypatch.setattr(
        node_config,
        "rotate_local_cluster_secret",
        lambda: ("new-secret-must-not-leak", 7),
    )

    response = __import__("asyncio").run(api_server.rotate_cluster_secret(
        api_server.ClusterSecretRotateRequest(
            otp_code="123456",
            tailnet_revocation_completed=True,
        ),
        SimpleNamespace(client=SimpleNamespace(host="127.0.0.1")),
        principal=SimpleNamespace(username="root"),
    ))

    assert response == {
        "status": "rotated",
        "cluster_secret_epoch": 7,
        "disconnected_clients": 3,
        "rebootstrap_required": True,
    }
    assert "new-secret-must-not-leak" not in str(response)
    assert calls == ["cluster credential rotation"]


def test_totp_failures_for_one_account_do_not_lock_another(join_api):
    import auth_app
    import auth_service
    from auth_store import ROLE_ADMIN

    client, role, _secret = join_api
    store = auth_service.get_auth_store()
    store.create_user("second-admin", "password123", role=ROLE_ADMIN)
    second_secret = "MFRGGZDFMZTWQ2LK"
    store.bind_totp("second-admin", second_secret)
    for _ in range(auth_app.TOTP_MAX_FAILURES):
        failed = client.post(
            "/api/auth/login",
            json={
                "username": "root",
                "password": "password123",
                "totp_code": "000000",
            },
        )
        assert failed.status_code == 401
    recovered = client.post(
        "/api/auth/login",
        json={
            "username": "second-admin",
            "password": "password123",
            "totp_code": auth_app.totp(second_secret),
        },
    )
    assert recovered.status_code == 200, recovered.text


def test_provisional_master_request_uses_future_client_id(join_api, monkeypatch):
    client, role, _secret = join_api
    monkeypatch.setattr("socket.gethostname", lambda: "provisional-box")
    import api_server
    monkeypatch.setattr(api_server.scheduler, "get_effective_node_id", lambda: "master")
    # The fixture's role is master for the API decision, while the scheduler
    # reports the pre-switch identity as master.
    role["value"] = "master"
    response = client.post(
        "/api/cluster/join/request",
        json={"master_endpoint": "100.64.0.10:8888"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["target_node_id"] == "client_provisional-box"


def test_join_grant_replay_is_rejected_before_connection_side_effect(join_api, monkeypatch):
    import auth_app
    import api_server

    client, role, secret = join_api
    created = client.post(
        "/api/cluster/join/request",
        json={"master_endpoint": "100.64.0.10:8888", "target_node_id": "client-test"},
    )
    role["value"] = "master"
    issued = client.post(
        "/api/cluster/join/grant",
        json={
            "request_code": created.json()["request_code"],
            "otp_code": auth_app.totp(secret),
        },
    )
    role["value"] = "client"

    entered = threading.Event()
    release = threading.Event()
    calls = []

    def blocking_connect(host, port, **kwargs):
        calls.append((host, port))
        entered.set()
        assert release.wait(timeout=5)
        return {"status": "connected", "master_host": host, "master_port": port}

    monkeypatch.setattr(api_server.scheduler, "connect_to_master", blocking_connect)
    grant_code = issued.json()["grant_code"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            client.post, "/api/cluster/join/consume", json={"grant_code": grant_code},
        )
        assert entered.wait(timeout=5)
        replay = pool.submit(
            client.post, "/api/cluster/join/consume", json={"grant_code": grant_code},
        ).result(timeout=5)
        release.set()
        accepted = first.result(timeout=5)

    assert accepted.status_code == 200
    assert replay.status_code == 409
    assert replay.json()["detail"]["code"] in {"nonce_replayed", "request_not_found"}
    assert calls == [("100.64.0.10", 8888)]
