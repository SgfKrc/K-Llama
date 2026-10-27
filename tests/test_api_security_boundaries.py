"""Security rejection contracts for legacy single-process API routes."""

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server


def test_retired_email_surface_has_no_api_routes():
    paths = set(api_server.app.openapi()["paths"])
    assert paths.isdisjoint({
        "/api/cluster/email-test",
        "/api/cluster/email-config",
        "/api/cluster/review/mail-poll",
    })


@pytest.mark.parametrize("method,path,payload", [
    ("post", "/api/chat", {"message": "boundary"}),
    ("post", "/api/chat/stream", {"message": "boundary"}),
    ("post", "/api/cluster/nodes/worker-1/deregister", None),
    ("put", "/api/cluster/config/max-nodes", {"max_nodes": 2}),
])
def test_common_api_boundary_rejects_untrusted_remote_before_handler(
    method, path, payload, monkeypatch,
):
    monkeypatch.delenv("QLH_AUTH_REQUIRED", raising=False)
    with TestClient(api_server.app, client=("192.0.2.10", 50000)) as client:
        response = getattr(client, method)(path, json=payload)
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "MODEL_API_SOURCE_UNTRUSTED"


def test_first_connect_returns_only_target_encrypted_cluster_credential(
    monkeypatch, tmp_path,
):
    import bootstrap_credentials
    import node_config
    from bootstrap_nonce_ledger import BootstrapRequestNonceLedger

    class MasterScheduler:
        _lan_ip = "100.64.0.1"
        tcp_server = SimpleNamespace(port=18888)

        @staticmethod
        def _effective_role():
            return "master"

        @staticmethod
        def manual_register_node(**kwargs):
            return {"status": "registered", "node_id": kwargs["node_id"]}

    monkeypatch.setattr(api_server, "scheduler", MasterScheduler())
    monkeypatch.setattr(api_server, "_bootstrap_cluster_credential", lambda: ("s" * 32, 5))
    monkeypatch.setattr(
        api_server,
        "_get_bootstrap_nonce_ledger",
        lambda: BootstrapRequestNonceLedger(tmp_path / "bootstrap.sqlite3"),
    )
    exchange = bootstrap_credentials.create_bootstrap_credential_request()
    response = asyncio.run(api_server.first_connect_bootstrap(
        api_server.FirstConnectBootstrapRequest(
            node_id="client-secure",
            node_type="pc",
            credential_public_key=exchange.public_key,
            credential_request_nonce=exchange.request_nonce,
            credential_requested_at=exchange.requested_at,
        ),
        _request("100.64.0.20"),
    ))

    assert "cluster_secret" not in response["cluster"]
    assert "s" * 32 not in str(response)
    credential = bootstrap_credentials.open_bootstrap_credential(
        response["credential_envelope"],
        private_key=exchange.private_key,
        expected_request_nonce=exchange.request_nonce,
        expected_cluster_id=response["cluster"]["cluster_id"],
        expected_node_id="client-secure",
    )
    assert credential == {
        "cluster_secret": "s" * 32,
        "cluster_secret_epoch": 5,
        "issued_at": credential["issued_at"],
        "expires_at": credential["expires_at"],
    }


def test_first_connect_rejects_custom_plain_lan_even_if_cidr_is_trusted(
    monkeypatch,
):
    import bootstrap

    monkeypatch.setattr(bootstrap, "is_trusted_bootstrap_source", lambda host: True)
    monkeypatch.setattr(bootstrap, "is_secure_bootstrap_source", lambda host: False)
    exchange = __import__("bootstrap_credentials").create_bootstrap_credential_request()
    with pytest.raises(HTTPException) as rejected:
        asyncio.run(api_server.first_connect_bootstrap(
            api_server.FirstConnectBootstrapRequest(
                node_id="client-lan",
                credential_public_key=exchange.public_key,
                credential_request_nonce=exchange.request_nonce,
                credential_requested_at=exchange.requested_at,
            ),
            _request("192.168.1.20"),
        ))
    assert rejected.value.status_code == 403
    assert rejected.value.detail["code"] == "protected_bootstrap_transport_required"


def _request(host: str, *, headers: dict[str, str] | None = None) -> Request:
    encoded_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in (headers or {}).items()
    ]
    return Request({
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": "/security-test",
        "raw_path": b"/security-test",
        "query_string": b"",
        "headers": encoded_headers,
        "client": (host, 32123),
        "server": ("testserver", 80),
    })


def test_system_shutdown_rejects_remote_missing_and_wrong_tokens(monkeypatch):
    remote = _request("192.0.2.10")
    request = api_server.SystemShutdownRequest(reason="test")

    monkeypatch.setattr(api_server, "_SHUTDOWN_TOKEN", "")
    with pytest.raises(HTTPException) as missing:
        asyncio.run(api_server.system_shutdown(request, remote))
    assert missing.value.status_code == 403

    monkeypatch.setattr(api_server, "_SHUTDOWN_TOKEN", "expected-token")
    with pytest.raises(HTTPException) as wrong:
        asyncio.run(api_server.system_shutdown(
            request,
            _request("192.0.2.10", headers={
                "X-QLH-Shutdown-Token": "wrong-token",
            }),
        ))
    assert wrong.value.status_code == 403


def test_system_shutdown_accepts_only_the_configured_remote_token(monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, *, target, daemon, name):
            self._target = target
            self.daemon = daemon
            self.name = name

        def start(self):
            started.append(self._target)

    monkeypatch.setattr(api_server, "_SHUTDOWN_TOKEN", "expected-token")
    monkeypatch.setattr(api_server.threading, "Thread", FakeThread)

    result = asyncio.run(api_server.system_shutdown(
        api_server.SystemShutdownRequest(reason="authorized-test"),
        _request("192.0.2.10", headers={
            "X-QLH-Shutdown-Token": "expected-token",
        }),
    ))

    assert result["ok"] is True
    assert started == [api_server._graceful_exit]


@pytest.mark.parametrize("filename", [
    "../model.gguf",
    "nested/model.gguf",
    "..\\model.gguf",
    "model.bin",
])
def test_model_download_rejects_paths_and_non_gguf_filenames(filename):
    with pytest.raises(HTTPException) as rejected:
        asyncio.run(api_server.download_model_file(filename))

    assert rejected.value.status_code == 400


def test_gguf_catalog_and_download_supports_range_without_local_path_leak(
    monkeypatch,
    tmp_path,
):
    from fastapi.testclient import TestClient

    payload = b"0123456789"
    model_path = tmp_path / "tiny.gguf"
    model_path.write_bytes(payload)
    monkeypatch.setattr(api_server, "_MODELS_DIR", str(tmp_path))

    with TestClient(api_server.app) as client:
        catalog = client.get("/api/models/gguf")
        assert catalog.status_code == 200
        body = catalog.json()
        assert body["models"] == [{
            "filename": "tiny.gguf",
            "size_bytes": len(payload),
            "size_mb": 0.0,
            "sha256": __import__("hashlib").sha256(payload).hexdigest(),
            "download_url": "/api/models/download/tiny.gguf",
        }]
        assert "directory" not in body

        downloaded = client.get(
            "/api/models/download/tiny.gguf",
            headers={"Range": "bytes=4-7"},
        )
        assert downloaded.status_code == 206
        assert downloaded.content == b"4567"
        assert downloaded.headers["accept-ranges"] == "bytes"
        assert downloaded.headers["content-range"] == "bytes 4-7/10"


def test_downloadable_model_rejects_untrusted_peer(monkeypatch):
    import bootstrap

    monkeypatch.setattr(
        bootstrap, "is_trusted_bootstrap_source", lambda host: False,
    )

    with pytest.raises(HTTPException) as rejected:
        asyncio.run(api_server.downloadable_pytorch_model(
            _request("192.0.2.10"),
        ))

    assert rejected.value.status_code == 403


def test_log_download_rejects_remote_peer_without_admin_token(monkeypatch):
    monkeypatch.setattr(
        api_server, "_get_request_client", lambda request: "192.0.2.10",
    )
    monkeypatch.delenv("QLH_LOG_ADMIN_TOKEN", raising=False)

    with pytest.raises(HTTPException) as rejected:
        asyncio.run(api_server.download_log_file(
            _request("192.0.2.10"), "example.log",
        ))

    assert rejected.value.status_code == 403


def test_cluster_mutation_endpoints_map_scheduler_denials_to_forbidden(
    monkeypatch,
):
    class DeniedScheduler:
        @staticmethod
        def _effective_role():
            return "client"

        @staticmethod
        def reset_master_identity():
            return {"status": "denied", "reason": "client denied"}

        @staticmethod
        def connect_to_master(
            host, port, *, force_bootstrap=False, persist_preference=False,
        ):
            del host, port, force_bootstrap, persist_preference
            return {"status": "denied", "reason": "client denied"}

    monkeypatch.setattr(api_server, "scheduler", DeniedScheduler())

    with pytest.raises(HTTPException) as reset:
        asyncio.run(api_server.reset_master_identity(
            api_server.ResetIdentityRequest(confirm="reset"),
        ))
    with pytest.raises(HTTPException) as connect:
        asyncio.run(api_server.connect_to_master(
            api_server.ConnectToMasterRequest(master_host="100.64.0.10"),
        ))
    assert reset.value.status_code == 403
    assert connect.value.status_code == 403


def test_manual_cluster_connect_persists_the_explicit_endpoint_preference(
    monkeypatch,
):
    calls = []

    class ClientScheduler:
        @staticmethod
        def _effective_role():
            return "client"

        @staticmethod
        def connect_to_master(
            host, port, *, force_bootstrap=False, persist_preference=False,
        ):
            calls.append((host, port, force_bootstrap, persist_preference))
            return {"status": "connected", "master": f"{host}:{port}"}

    monkeypatch.setattr(api_server, "scheduler", ClientScheduler())

    result = asyncio.run(api_server.connect_to_master(
        api_server.ConnectToMasterRequest(
            master_host="fd7a:115c:a1e0::8d01:4cc5",
            master_port=8888,
        ),
    ))

    assert result["status"] == "connected"
    assert calls == [
        ("fd7a:115c:a1e0::8d01:4cc5", 8888, False, True),
    ]
