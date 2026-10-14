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


def test_common_api_boundary_enforces_bearer_when_enabled(monkeypatch):
    monkeypatch.setenv("QLH_AUTH_REQUIRED", "1")
    with TestClient(api_server.app, client=("127.0.0.1", 50000)) as client:
        response = client.get("/api/cluster/nodes")
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "auth_required"


@pytest.mark.parametrize("method,path,payload", [
    ("post", "/api/cluster/nodes/worker-1/deregister", None),
    ("delete", "/api/cluster/nodes/worker-1", None),
    ("post", "/api/cluster/reset-identity", {"confirm": "reset"}),
])
def test_destructive_cluster_endpoints_require_named_admin_when_auth_is_optional(
    method, path, payload, monkeypatch,
):
    monkeypatch.delenv("QLH_AUTH_REQUIRED", raising=False)
    with TestClient(api_server.app, client=("127.0.0.1", 50000)) as client:
        response = client.request(method.upper(), path, json=payload)
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "auth_required"


def test_first_user_bootstrap_remains_open_when_auth_is_forced(monkeypatch):
    monkeypatch.setenv("QLH_AUTH_REQUIRED", "1")
    monkeypatch.setattr(api_server.auth_service, "is_bootstrap_open", lambda: True)
    principal = api_server.auth_service.require_session(authorization=None)
    assert principal.username == "anonymous"
    assert principal.is_admin is True


def test_first_connect_returns_only_target_encrypted_cluster_credential(
    monkeypatch, tmp_path,
):
    import bootstrap_credentials
    import node_config
    from cluster_join import JoinGrantLedger

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
    monkeypatch.setattr(node_config, "ensure_local_cluster_secret", lambda: "s" * 32)
    monkeypatch.setattr(node_config, "get_local_cluster_secret_epoch", lambda: 5)
    monkeypatch.setattr(
        api_server,
        "_get_join_ledger",
        lambda: JoinGrantLedger(tmp_path / "bootstrap.sqlite3"),
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

        @staticmethod
        def transfer_master_role(target_node_id):
            del target_node_id
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
    with pytest.raises(HTTPException) as transfer:
        asyncio.run(api_server.transfer_master_role(
            api_server.TransferMasterRequest(target_node_id="worker-next"),
        ))

    assert {
        reset.value.status_code,
        connect.value.status_code,
        transfer.value.status_code,
    } == {403}


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


def test_review_create_and_vote_keep_their_role_and_eligibility_gates(
    monkeypatch,
):
    class UntrustedReviewScheduler:
        @staticmethod
        def _effective_role():
            return "client"

        @staticmethod
        def get_effective_node_id():
            return "worker-no-gpu"

        @staticmethod
        def can_node_vote(node_id):
            assert node_id == "worker-no-gpu"
            return False, "not eligible"

    monkeypatch.setattr(api_server, "scheduler", UntrustedReviewScheduler())

    with pytest.raises(HTTPException) as create:
        asyncio.run(api_server.create_review_ticket(
            api_server.CreateReviewRequest(target_node_id="worker-next"),
        ))
    with pytest.raises(HTTPException) as vote:
        asyncio.run(api_server.cast_review_vote(
            api_server.CastVoteRequest(ticket_id="review-test", vote=1),
        ))

    assert create.value.status_code == 403
    assert vote.value.status_code == 403


def test_review_side_effect_endpoints_require_master_role(monkeypatch):
    class ClientScheduler:
        @staticmethod
        def _effective_role():
            return "client"

    monkeypatch.setattr(api_server, "scheduler", ClientScheduler())

    endpoints = (
        lambda: api_server.trigger_expire_check(),
        lambda: api_server.delete_review_ticket("review-test"),
        api_server.delete_resolved_review_tickets,
    )
    for endpoint in endpoints:
        with pytest.raises(HTTPException) as rejected:
            asyncio.run(endpoint())
        assert rejected.value.status_code == 403
        assert rejected.value.detail == "仅主节点可执行此操作"


def test_review_side_effect_endpoints_execute_for_master(monkeypatch):
    import review

    class MasterScheduler:
        @staticmethod
        def _effective_role():
            return "master"

    class FakeReviewManager:
        def resolve_expired(self):
            return ["review-expired"]

        def delete_ticket(self, ticket_id):
            assert ticket_id == "review-test"
            return True

        def delete_resolved(self):
            return 2

    monkeypatch.setattr(api_server, "scheduler", MasterScheduler())
    monkeypatch.setattr(review, "ReviewManager", FakeReviewManager)
    assert asyncio.run(api_server.trigger_expire_check()) == {
        "expired": ["review-expired"],
        "count": 1,
    }
    assert asyncio.run(api_server.delete_review_ticket("review-test")) == {
        "status": "deleted",
        "ticket_id": "review-test",
    }
    assert asyncio.run(api_server.delete_resolved_review_tickets()) == {
        "status": "deleted",
        "count": 2,
    }


# ================================================================
# T-3 读端点开放契约（§4.2.1 矩阵 3/4/5）：网络信任域内读开放，
# 不因部署角色变化而 403（写端点的 master 门在
# test_review_side_effect_endpoints_* 已覆盖）
# ================================================================

def test_review_read_endpoints_stay_open_for_non_master(monkeypatch):
    class ClientScheduler:
        @staticmethod
        def _effective_role():
            return "client"

        @staticmethod
        def get_effective_node_id():
            return "node-client"

        @staticmethod
        def can_node_vote(node_id):
            return True, ""

    monkeypatch.setattr(api_server, "scheduler", ClientScheduler())
    # 列表：非 master 角色必须可读（不 403）
    result = asyncio.run(api_server.list_review_tickets())
    assert "tickets" in result and "count" in result
    # can-vote 自查询：开放
    result = asyncio.run(api_server.check_can_vote())
    assert "can_vote" in result


def test_review_summary_is_bounded_and_does_not_return_comments(monkeypatch):
    import review

    class FakeReviewManager:
        def list_tickets(self, status=None):
            return [SimpleNamespace(
                ticket_id=f"review-{index}",
                status=SimpleNamespace(value="pending"),
                created_at=float(index),
                target_node_id="node-target",
                score=0,
                expires_at=100.0,
                resolved_at=None,
                votes=[SimpleNamespace(comment="secret") for _ in range(20)],
                transfer_reason="private reason",
            ) for index in range(20)]

    monkeypatch.setattr(review, "ReviewManager", FakeReviewManager)
    result = asyncio.run(api_server.list_review_tickets(limit=50, summary=True))
    assert result["count"] == 8
    assert len(result["tickets"]) == 8
    assert result["tickets"][0]["vote_count"] == 20
    assert "transfer_reason" not in result["tickets"][0]
    assert "votes" not in result["tickets"][0]


def test_review_ticket_detail_missing_is_404_not_403_for_non_master(monkeypatch):
    class ClientScheduler:
        @staticmethod
        def _effective_role():
            return "client"

    monkeypatch.setattr(api_server, "scheduler", ClientScheduler())
    with pytest.raises(HTTPException) as missing:
        asyncio.run(api_server.get_review_ticket("ticket-does-not-exist"))
    # 非 master 读不存在工单 → 404（资源不存在优先于角色拒绝，契约 4）
    assert missing.value.status_code == 404
