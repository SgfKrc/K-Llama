"""Public intent entry, source boundary and master fence contracts."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from starlette.responses import Response
from tests.test_model_intent import native_intent, stage_case, retirement_case  # noqa: F401


@pytest.fixture
def api(monkeypatch):
    import api_server
    from api import routes_model_intent
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "master")
    fake = SimpleNamespace(current=lambda: None, get=lambda op_id: {"operation_id": op_id, "status": "ready"},
                           create=AsyncMock(return_value={"operation_id": "supply_" + "a" * 32, "status": "queued"}),
                           cancel=lambda op_id: {"operation_id": op_id, "status": "cancelled"})
    monkeypatch.setattr(routes_model_intent, "coordinator", lambda: fake)
    monkeypatch.setattr(routes_model_intent, "get_binding", lambda *_: {"status": "connected", "is_master": True})
    return api_server, routes_model_intent, fake


def test_catalog_is_master_owned_and_path_free(api, monkeypatch):
    server, _, _ = api
    monkeypatch.setattr(server, "_get_all_model_configs", lambda: [object()])
    monkeypatch.setattr(server, "_model_api_payload", lambda _: {"model_id": "fixture", "name": "fixture", "model_path": "H:/private/model", "gguf_path": "H:/private/model.gguf", "has_gguf": True})
    value = TestClient(server.app).get("/api/models/intent/catalog")
    assert value.status_code == 200
    assert value.json()["models"] == [{"model_id": "fixture", "name": "fixture", "has_gguf": True}]
    assert value.json()["active_model_id"] is None


def test_create_has_one_entry_and_rejects_local_paths_and_manual_layers(api):
    server, _, fake = api
    client = TestClient(server.app)
    good = client.post("/api/models/intent", json={"model_id": "fixture", "engine": "llama_cpp"})
    assert good.status_code == 200 and good.json()["status"] == "queued"
    assert fake.create.await_args.args[0]["strategy"] == "auto"
    for extra in ({"model_path": "H:/other"}, {"layer_range": [0, 32]}, {"master_host": "evil"}):
        assert client.post("/api/models/intent", json={"model_id": "fixture", **extra}).status_code == 422
    assert fake.create.await_count == 1


@pytest.mark.parametrize("path,method", [("/api/models/intent/catalog", "get"), ("/api/models/intent/current", "get"),
                                         ("/api/models/intent", "post"), ("/api/models/intent/connect", "post")])
def test_untrusted_source_is_blocked_before_intent_work(api, path, method):
    server, _, fake = api
    client = TestClient(server.app, client=("192.0.2.50", 4567))
    result = getattr(client, method)(path, **({"json": {"model_id": "fixture"}} if method == "post" else {}))
    assert result.status_code == 403
    assert result.json()["detail"]["code"] == "MODEL_API_SOURCE_UNTRUSTED"
    assert fake.create.await_count == 0


def test_worker_catalog_preserves_master_auth_error_response(api, monkeypatch):
    server, routes, _ = api
    monkeypatch.setattr(server.scheduler, "_effective_role", lambda: "client")
    monkeypatch.setattr(routes, "forward_intent", AsyncMock(return_value=Response(content='{"detail":{"code":"auth_required"}}', status_code=401, media_type="application/json")))
    response = TestClient(server.app).get("/api/models/intent/catalog")
    assert response.status_code == 401 and response.json()["detail"]["code"] == "auth_required"


def test_master_intent_write_is_a_control_fence_action_but_worker_relay_is_not(api, monkeypatch):
    server, _, _ = api
    from starlette.requests import Request
    request = Request({"type": "http", "method": "POST", "path": "/api/models/intent", "headers": []})
    assert server._is_control_write_path(request)
    monkeypatch.setattr(server.scheduler, "_effective_role", lambda: "client")
    assert not server._is_control_write_path(request)


def test_unknown_operations_do_not_fall_through_to_catalog(api):
    server, routes, fake = api
    op = "supply_" + "b" * 32
    response = TestClient(server.app).get("/api/models/intent/" + op)
    assert response.status_code == 200 and response.json()["operation_id"] == op
    assert TestClient(server.app).get("/api/models/intent/current").json() is None


@pytest.fixture
def registered_api(api, monkeypatch):
    """Publish a real HMAC/ACK TCP registration without opening a socket."""
    import tcp_comm
    from cluster_fence import ControlFence

    server, routes, fake = api
    monkeypatch.delenv("QLH_MODEL_API_TRUSTED_CIDRS", raising=False)
    monkeypatch.setattr(server.auth_service, "auth_required", lambda: False)
    monkeypatch.setattr(server, "control_fence", ControlFence(required=False))
    monkeypatch.setattr(tcp_comm, "_get_cluster_secret", lambda: "intent-api-bootstrap-credential-32")
    monkeypatch.setattr(tcp_comm, "_get_cluster_secret_epoch", lambda: 3)
    tcp_server = tcp_comm.TCPServer(host="100.64.1.2", port=18888)
    tcp_server._running = True
    peer = ("100.64.10.20", 41000)
    sock = SimpleNamespace(getpeername=lambda: peer, fileno=lambda: 17, sendall=lambda _data: None)
    node_id = "intent-api-worker"
    tcp_server._handle_registration(sock, peer, "temporary", {"data": {
        "client_id": node_id, "role": "client", "node_type": "pc", "advertised_host": "100.64.99.99",
        "auth": tcp_comm.build_auth_signature(node_id),
    }})
    assert tcp_server.confirm_registration(node_id)
    monkeypatch.setattr(server.scheduler, "_tcp_server_default", tcp_server)
    monkeypatch.setattr(server.scheduler, "get_distributed_inference_enabled", lambda: True)
    original_facade = server.scheduler._scheduler_facade_global
    monkeypatch.setattr(server.scheduler, "_scheduler_facade_global",
                        lambda name: "distributed" if name == "RUN_MODE" else original_facade(name))
    monkeypatch.setattr(server, "_get_all_model_configs", lambda: [])
    return SimpleNamespace(server=server, routes=routes, coordinator=fake, tcp=tcp_server,
                           sock=sock, node_id=node_id, peer=peer[0], monkeypatch=monkeypatch)


@pytest.mark.parametrize("node_type", ["pc", "android"])
@pytest.mark.parametrize("method,suffix", [("GET", "catalog"), ("GET", "current"), ("POST", ""),
                                          ("GET", "supply_" + "b" * 32), ("DELETE", "supply_" + "b" * 32)])
def test_registered_tailnet_devices_can_use_intent_without_manual_cidr(registered_api, node_type, method, suffix):
    owner = registered_api
    owner.tcp.clients[owner.node_id].node_type = node_type
    client = TestClient(owner.server.app, client=(owner.peer, 50000))
    path = "/api/models/intent" + ("/" + suffix if suffix else "")
    value = client.request(method, path, **({"json": {"model_id": "fixture"}} if method == "POST" else {}))
    assert value.status_code == 200, value.text


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/models/registry"), ("GET", "/api/models/downloads"),
    ("GET", "/api/models/available"), ("GET", "/api/models/current"),
    ("POST", "/api/models/load"), ("POST", "/api/models/prepare-pipeline"),
    ("POST", "/api/models/supply"), ("GET", "/api/models/supply/supply_" + "b" * 32),
    ("GET", "/api/cluster/nodes"), ("POST", "/api/models/intent/connect"),
    ("GET", "/api/models/intent/supply_" + "B" * 32), ("GET", "/api/models/intent/catalog/"),
    ("GET", "/api/models/intent/catalog/registry"), ("PUT", "/api/models/intent"),
])
def test_registered_intent_source_cannot_access_general_management(registered_api, method, path):
    owner = registered_api
    value = TestClient(owner.server.app, client=(owner.peer, 50000)).request(method, path)
    assert value.status_code == 403
    assert value.json()["detail"]["code"] == "MODEL_API_SOURCE_UNTRUSTED"
    assert owner.coordinator.create.await_count == 0


@pytest.mark.parametrize("field,value", [("model_path", "H:/private"), ("layer_range", [0, 32]),
                                        ("master_host", "100.64.1.2"), ("source", "http://other")])
def test_new_source_admission_keeps_intent_schema_path_free(registered_api, field, value):
    owner = registered_api
    result = TestClient(owner.server.app, client=(owner.peer, 50000)).post(
        "/api/models/intent", json={"model_id": "fixture", field: value})
    assert result.status_code == 422
    assert owner.coordinator.create.await_count == 0


def test_registered_intent_source_still_requires_bearer(registered_api):
    owner = registered_api
    owner.monkeypatch.setattr(owner.server.auth_service, "auth_required", lambda: True)
    owner.monkeypatch.setattr(owner.server.auth_service, "resolve_bearer",
                              lambda value: object() if value == "Bearer valid-session" else None)
    client = TestClient(owner.server.app, client=(owner.peer, 50000))
    for headers in ({}, {"X-QLH-Intent-Hop": "1"}, {"Authorization": "Bearer invalid"}):
        result = client.post("/api/models/intent", json={"model_id": "fixture"}, headers=headers)
        assert result.status_code == 401 and result.json()["detail"]["code"] == "auth_required"
    assert owner.coordinator.create.await_count == 0
    assert client.post("/api/models/intent", json={"model_id": "fixture"},
                       headers={"Authorization": "Bearer valid-session"}).status_code == 200
    assert owner.coordinator.create.await_count == 1


def test_registered_intent_write_keeps_real_control_certificate_validation(registered_api):
    import base64
    import time
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    from cluster_control_contract import ControlPlaneAuthority, VoterIdentity, VoterSet, QuorumCertificate, sign_certificate
    from cluster_fence import ControlFence, encode_certificate_header

    owner = registered_api
    key = Ed25519PrivateKey.generate()
    public = base64.urlsafe_b64encode(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode().rstrip("=")
    voters = VoterSet(cluster_id="intent-source-test", voter_set_epoch=1,
                      voters=[VoterIdentity(voter_id="master", public_key=public)])
    unsigned = QuorumCertificate(cluster_id=voters.cluster_id, voter_set_epoch=1, term=1, leader_id="master",
                                 lease_id="intent-lease", expires_at_ms=int(time.time() * 1000) + 60000, signatures=())
    certificate = unsigned.with_signatures((sign_certificate(unsigned, voter_id="master", private_key=key),))
    fence = ControlFence(ControlPlaneAuthority(cluster_id=voters.cluster_id, voter_set=voters), required=True)
    fence.install_certificate(certificate)
    owner.monkeypatch.setattr(owner.server, "control_fence", fence)
    client = TestClient(owner.server.app, client=(owner.peer, 50000))
    missing = client.post("/api/models/intent", json={"model_id": "fixture"}, headers={"X-QLH-Intent-Hop": "1"})
    assert missing.status_code == 409 and missing.json()["detail"]["code"] == "control_certificate_missing"
    invalid = client.post("/api/models/intent", json={"model_id": "fixture"},
                          headers={"X-QLH-Control-Certificate": "forged"})
    assert invalid.status_code == 409 and invalid.json()["detail"]["code"] == "control_certificate_invalid"
    assert owner.coordinator.create.await_count == 0
    admitted = client.post("/api/models/intent", json={"model_id": "fixture"},
                           headers={"X-QLH-Control-Certificate": encode_certificate_header(certificate)})
    assert admitted.status_code == 200 and owner.coordinator.create.await_count == 1
    assert fence.current_permit() is None


@pytest.mark.parametrize("change", ["remove", "replace", "stop", "unconfirm"])
def test_reconnect_between_middleware_and_handler_cannot_invoke_intent(registered_api, change):
    from copy import copy
    owner = registered_api
    owner.monkeypatch.setattr(owner.server.auth_service, "auth_required", lambda: True)

    def resolve(_authorization):
        if change == "remove":
            owner.tcp.clients.clear()
        elif change == "replace":
            owner.tcp.clients[owner.node_id] = copy(owner.tcp.clients[owner.node_id])
        elif change == "stop":
            owner.tcp._running = False
        elif change == "unconfirm":
            owner.tcp.clients[owner.node_id].registration_confirmed = False
        return object()
    owner.monkeypatch.setattr(owner.server.auth_service, "resolve_bearer", resolve)
    value = TestClient(owner.server.app, client=(owner.peer, 50000)).post(
        "/api/models/intent", json={"model_id": "fixture"}, headers={"Authorization": "Bearer valid"})
    assert value.status_code == 403 and value.json()["detail"]["code"] == "MODEL_API_SOURCE_UNTRUSTED"
    assert owner.coordinator.create.await_count == 0


def test_disconnect_during_handler_discards_stale_success_response(registered_api):
    owner = registered_api
    async def create(_intent):
        owner.tcp.clients.clear()
        return {"status": "queued"}
    owner.coordinator.create.side_effect = create
    value = TestClient(owner.server.app, client=(owner.peer, 50000)).post(
        "/api/models/intent", json={"model_id": "fixture"})
    assert value.status_code == 403 and value.json()["detail"]["code"] == "MODEL_API_SOURCE_UNTRUSTED"
    assert owner.coordinator.create.await_count == 1


def test_unregistered_tailnet_address_and_spoofed_proxy_headers_never_grant_source(registered_api):
    owner = registered_api
    value = TestClient(owner.server.app, client=("100.64.10.21", 50000)).get(
        "/api/models/intent/catalog", headers={"Host": owner.peer, "X-Forwarded-For": owner.peer,
        "X-Real-IP": owner.peer, "X-QLH-Intent-Hop": "1", "X-QLH-Node-ID": owner.node_id})
    assert value.status_code == 403 and value.json()["detail"]["code"] == "MODEL_API_SOURCE_UNTRUSTED"


def test_loopback_worker_relay_reaches_registered_master_without_manual_cidr(registered_api):
    import asyncio
    import httpx
    from starlette.requests import Request
    import model_intent_gateway as gateway
    owner = registered_api
    worker = SimpleNamespace(_effective_role=lambda: "client", get_distributed_inference_enabled=lambda: True,
        _scheduler_facade_global=lambda _name: "distributed",
        _tcp_client=SimpleNamespace(_running=True, is_registered=True, sock=object(), registration_epoch=1,
                                   server_host="100.64.1.2", server_port=18888, client_id=owner.node_id))
    owner.monkeypatch.setattr(gateway.node_config, "load_node_config", lambda: {
        "bootstrapped": True, "node": {"node_id": owner.node_id}, "cluster": {
            "cluster_id": "intent-source-test", "master_tcp_host": "100.64.1.2", "master_tcp_port": 18888,
            "master_api_host": "100.64.1.2", "master_api_port": 18000}})
    incoming = Request({"type": "http", "method": "POST", "scheme": "http", "path": "/api/models/intent",
                        "query_string": b"", "client": ("127.0.0.1", 50000), "headers": []})
    class Raw(httpx.AsyncByteStream):
        def __init__(self, raw):
            self.raw = raw
        async def __aiter__(self):
            yield self.raw
    async def master(req):
        result = TestClient(owner.server.app, client=(owner.peer, 50000)).request(
            req.method, req.url.path, content=req.content, headers=dict(req.headers))
        return httpx.Response(result.status_code, headers=result.headers, stream=Raw(result.content))
    client_type = httpx.AsyncClient
    owner.monkeypatch.setattr(gateway.httpx, "AsyncClient", lambda **kwargs: client_type(
        transport=httpx.MockTransport(master), **kwargs))
    result = asyncio.run(gateway.forward_intent(worker, incoming, "", {"model_id": "fixture"}))
    assert result["status"] == "queued" and owner.coordinator.create.await_count == 1


@pytest.mark.parametrize("result", [{"success": True, "loaded": False, "unloaded": True},
                                    {"success": True, "loaded": False, "unloaded": False}])
def test_intent_retire_callback_uses_existing_unload_owner(retirement_case, monkeypatch, result):
    import asyncio
    import api_server
    from api import routes_model_intent
    from tests.test_pipeline_retirement import _publish
    case = retirement_case
    _publish(case)
    calls = []
    def unload(*, retire_worker_runtime):
        assert retire_worker_runtime is False
        assert case.host.full_chat_execution_lock._is_owned()
        assert case.master._layer_config_model_change_depth == 1
        calls.append("unload")
        return result
    monkeypatch.setattr(api_server, "_wait_for_worker_prepare_retirement", lambda reason: calls.append(reason))
    monkeypatch.setattr(api_server, "_unload_model_under_model_lock", unload)
    asyncio.run(routes_model_intent._retire_native_plan())
    assert calls == ["local_model_load", "unload"]
    assert case.master._layer_config_model_change_depth == 0


@pytest.mark.parametrize("result", [None, {}, {"success": False, "loaded": False},
                                    {"success": True, "loaded": True}])
def test_intent_retire_callback_rejects_unproven_local_unload(retirement_case, monkeypatch, result):
    import asyncio
    import api_server
    from api import routes_model_intent
    from model_supply import ModelSupplyError
    from tests.test_pipeline_retirement import _publish
    _publish(retirement_case)
    monkeypatch.setattr(api_server, "_wait_for_worker_prepare_retirement", lambda reason: None)
    monkeypatch.setattr(api_server, "_unload_model_under_model_lock", lambda **kwargs: result)
    with pytest.raises(ModelSupplyError) as error:
        asyncio.run(routes_model_intent._retire_native_plan())
    assert error.value.code == "INTENT_RETIRE_FAILED"


def test_production_intent_coordinator_injects_native_retirement(monkeypatch):
    import api_server
    from api import routes_model_intent
    monkeypatch.setattr(routes_model_intent, "_coordinator", None)
    coordinator = routes_model_intent.coordinator()
    assert coordinator.scheduler is api_server.scheduler
    assert coordinator.retire is routes_model_intent._retire_native_plan
    assert coordinator.retire_timeout == 300


def test_ui_switch_waits_identity_bound_native_release_before_supply(native_intent, monkeypatch):
    import asyncio
    from api import routes_model_intent
    from tests.test_worker_prepare_coordinator import ack, prepare_both
    from tests.test_worker_prepare_races import _finish_finalizes
    async def scenario():
        case = native_intent
        c, master, supply = case.coordinator, case.master, case.supply
        op = (await c.create(case.intent))["operation_id"]
        task = c._tasks[op]
        case.available(op)
        from tests.test_model_intent import wait_until
        await wait_until(lambda: master._pipeline_load_transaction is not None)
        prepare_both(case)
        ack(master, "worker", "commit"); ack(master, "android", "commit")
        await task
        _finish_finalizes(master)
        assert c.current()["loaded_ready"] and not master._pipeline_model_cleanup_ledger
        c.retire = routes_model_intent._retire_native_plan
        # The fixture's fake model manager is unloaded by the real API owner.
        selection = asyncio.create_task(c.create({**case.intent, "model_id": "replacement"}))
        await wait_until(lambda: master._pipeline_load_transaction is None)
        await asyncio.sleep(0.01)
        releases = list(master._pipeline_model_cleanup_ledger.values())
        assert {item["node_id"] for item in releases} == {"worker", "android"}
        assert all(item["action"] == "release" for item in releases)
        assert not selection.done() and len(supply._operations) == 1
        # Ordinary prepare shares the native cleanup fence despite bypassing the intent lane.
        blocked = master.start_pipeline_model_transaction(candidate_id="ordinary-prepare",
            descriptor={}, model_path=str(case.model_path), quant_type="fp16")
        assert blocked["admitted"] is False and blocked["reason_code"] == "MODEL_TXN_CLEANUP_PENDING"
        assert master._pipeline_load_transaction is None
        # A wrong native identity cannot drain the original cleanup ledger.
        from copy import deepcopy
        first = releases[0]
        identity = deepcopy(first["payload"]["worker_prepare_identity"])
        identity["artifact_sha256"] = "f" * 64
        master._handle_worker_prepare_ack(first["node_id"], {"action": "ack", "phase": "release", "identity": identity,
            "cleanup_id": first["cleanup_id"], "status": "released", "loaded": False})
        assert len(master._pipeline_model_cleanup_ledger) == 2 and not selection.done()
        for record in releases:
            master._handle_worker_prepare_ack(record["node_id"], {"action": "ack", "phase": "release",
                "identity": record["payload"]["worker_prepare_identity"], "cleanup_id": record["cleanup_id"],
                "status": "released", "loaded": False})
        assert not master._pipeline_model_cleanup_ledger
        fresh = await selection
        assert fresh["operation_id"] != op and len(supply._operations) == 2
        assert not c.get(op)["loaded_ready"]
        c.cancel(fresh["operation_id"])
        await c.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("replacement", ["native", "local"])
def test_late_intent_retirement_preserves_newer_committed_model(retirement_case, monkeypatch, replacement):
    import asyncio
    import threading
    import api_server
    from api import routes_model_intent
    from model_supply import ModelSupplyError
    from tests.test_pipeline_retirement import _publish, _ack_cleanup, _start_candidate, _worker_ack, _Manager
    from tests.test_model_intent import wait_until
    case = retirement_case
    _publish(case)
    for record in list(case.master._pipeline_model_cleanup_ledger.values()):
        assert _ack_cleanup(case.master, record)
    entered, released = threading.Event(), threading.Event()
    caller = threading.current_thread()
    def wait(reason):
        if threading.current_thread() is caller:
            return
        entered.set()
        assert released.wait(3), "retirement test gate was not released"
    monkeypatch.setattr(api_server, "_wait_for_worker_prepare_retirement", wait)

    async def scenario():
        retirement = asyncio.create_task(routes_model_intent._retire_native_plan())
        try:
            await wait_until(entered.is_set)
            if replacement == "native":
                plan = _start_candidate(case, "candidate-newer")
                assert plan["admitted"]
                _worker_ack(case.master, "prepared")
                _worker_ack(case.master, "ready")
                assert case.master._pipeline_load_transaction["phase"] == "ready"
                newer = case.host.peek_manager()
                expected = "INTENT_RETIRE_SUPERSEDED"
            else:
                api_server._set_node_inference_mode("local")
                newer = _Manager(loaded=True)
                def load():
                    # Model I/O fake; the original local-change owner publishes it.
                    case.host._manager = newer
                    case.host.model_loaded = True
                    case.host.current_quant = "fp16"
                    return {"success": True}
                api_server._run_exclusive_model_change(load, release_worker_reservation=True)
                expected = "SUPPLY_LOCAL_MODE"
            projection = (case.host.model_loaded, case.host.current_quant)
            released.set()
            with pytest.raises(ModelSupplyError) as error:
                await retirement
            assert error.value.code == expected
            assert case.host.peek_manager() is newer and newer.is_loaded
            assert newer.unload_attempts == 0
            assert (case.host.model_loaded, case.host.current_quant) == projection
            assert case.master._layer_config_model_change_depth == 0
            if replacement == "native":
                assert case.master._pipeline_load_transaction["candidate_id"] == "candidate-newer"
                assert case.master._active_pipeline_capacity_plan["transaction_phase"] == "ready"
        finally:
            released.set()
            await asyncio.gather(retirement, return_exceptions=True)
    asyncio.run(scenario())


def test_intent_retirement_transition_fences_ordinary_prepare(retirement_case, monkeypatch):
    import asyncio
    import api_server
    from api import routes_model_intent
    from tests.test_pipeline_retirement import _publish
    case = retirement_case
    _publish(case)
    original = api_server._unload_model_under_model_lock
    attempts = []
    def unload(**kwargs):
        attempt = case.master.start_pipeline_model_transaction(candidate_id="concurrent-prepare",
            descriptor={}, model_path=str(case.model_path), quant_type="fp16")
        attempts.append(attempt)
        return original(**kwargs)
    # Remove finalize debt to prove admission is rejected by the transition itself.
    from tests.test_pipeline_retirement import _ack_cleanup
    for record in list(case.master._pipeline_model_cleanup_ledger.values()):
        assert _ack_cleanup(case.master, record)
    monkeypatch.setattr(api_server, "_wait_for_worker_prepare_retirement", lambda reason: None)
    monkeypatch.setattr(api_server, "_unload_model_under_model_lock", unload)
    asyncio.run(routes_model_intent._retire_native_plan())
    assert attempts[0]["admitted"] is False and attempts[0]["reason_code"] == "MODEL_TXN_CONFLICT"
    assert case.master._pipeline_load_transaction is None
    assert case.master._layer_config_model_change_depth == 0


def test_cleanup_only_retry_does_not_issue_a_second_unload(retirement_case, monkeypatch):
    import asyncio
    import api_server
    from api import routes_model_intent
    case = retirement_case
    case.master._pipeline_model_cleanup_ledger["pending"] = {"state": "sent"}
    monkeypatch.setattr(api_server, "_unload_model_under_model_lock", lambda **kw: pytest.fail("late duplicate unload"))
    asyncio.run(routes_model_intent._retire_native_plan())
    assert case.active.is_loaded and case.active.unload_attempts == 0
