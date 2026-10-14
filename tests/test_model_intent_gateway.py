"""Model intent relay tests use real TCP field names and mocked HTTP only."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

import model_intent_gateway as gateway

OPERATION = "supply_" + "a1" * 16
_REAL_ASYNC_CLIENT = httpx.AsyncClient


class Scheduler:
    def __init__(self):
        self.role = "client"
        self.enabled = True
        self.run_mode = "distributed"
        self._tcp_client = SimpleNamespace(
            _running=True, is_registered=True, registration_epoch=1, sock=object(),
            server_host="100.64.2.3", server_port=18888, client_id="worker-pc",
        )

    def _effective_role(self):
        return self.role

    def get_distributed_inference_enabled(self):
        return self.enabled

    def _scheduler_facade_global(self, name):
        assert name == "RUN_MODE"
        return self.run_mode


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False
        self.consumed = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk

    async def aclose(self):
        self.closed = True


def response(status=200, payload=None, *, raw=None, headers=None, stream=None):
    if stream is None:
        stream = Chunks([raw if raw is not None else json.dumps(payload).encode()])
    return httpx.Response(status, headers={"Content-Type": "application/json", **(headers or {})}, stream=stream)


def request(method="GET", *, headers=None, peer="127.0.0.1"):
    return Request({
        "type": "http", "method": method, "scheme": "http", "path": "/api/models/intent",
        "query_string": b"", "server": ("127.0.0.1", 18001), "client": (peer, 12345),
        "headers": [(name.lower().encode(), value.encode()) for name, value in (headers or {}).items()],
    })


@pytest.fixture
def setup(monkeypatch):
    scheduler = Scheduler()
    config = {
        "bootstrapped": True,
        "node": {"node_id": "worker-pc", "role": "client"},
        "cluster": {"cluster_id": "cluster-a", "cluster_secret_epoch": 1,
                    "master_tcp_host": "100.64.2.3", "master_tcp_port": 18888,
                    "master_api_host": "master-api.tailnet.test", "master_api_port": 18000},
    }
    monkeypatch.setattr(gateway.node_config, "load_node_config", lambda: deepcopy(config))
    # Source tests must not inherit a broad CIDR opt-in from the operator's env.
    monkeypatch.delenv("QLH_MODEL_API_TRUSTED_CIDRS", raising=False)
    calls = []
    options = []

    def install(handler=None, *, enter=None, exit=None):
        if handler is None:
            handler = lambda req: response(payload={"ok": True})

        async def dispatch(req):
            calls.append(req)
            value = handler(req)
            return await value if hasattr(value, "__await__") else value

        class Client(_REAL_ASYNC_CLIENT):
            async def __aenter__(self):
                result = await super().__aenter__()
                if enter:
                    enter()
                return result

            async def __aexit__(self, *args):
                result = await super().__aexit__(*args)
                if exit:
                    exit()
                return result

        def factory(**kwargs):
            options.append(kwargs)
            return Client(transport=httpx.MockTransport(dispatch), **kwargs)

        monkeypatch.setattr(gateway.httpx, "AsyncClient", factory)

    install()
    return SimpleNamespace(scheduler=scheduler, config=config, calls=calls,
                           options=options, install=install, monkeypatch=monkeypatch)


def forward(setup, method="GET", suffix="catalog", *, body=None, headers=None, peer="127.0.0.1"):
    return asyncio.run(gateway.forward_intent(setup.scheduler, request(method, headers=headers, peer=peer), suffix, body))


def rejected(setup, code, status, **kwargs):
    with pytest.raises(HTTPException) as failed:
        forward(setup, **kwargs)
    assert failed.value.status_code == status
    assert failed.value.detail["code"] == code
    return failed.value


def test_binding_uses_bootstrap_api_port_and_not_local_or_tcp_port(setup):
    setup.monkeypatch.setenv("QLH_MASTER_API_PORT", "9999")
    binding = gateway.get_binding(setup.scheduler, 18001)
    assert binding["status"] == "connected" and not binding["is_master"]
    assert (binding["master_host"], binding["master_api_port"]) == ("master-api.tailnet.test", 18000)
    assert binding["master_tcp_port"] == 18888
    assert binding["cluster_id"] == "cluster-a"


def test_master_binding_never_reads_tcp_client_or_node_config(setup):
    class Master(Scheduler):
        @property
        def _tcp_client(self):
            raise AssertionError("master must not inspect a worker connection")

        @_tcp_client.setter
        def _tcp_client(self, value):
            pass

    scheduler = Master()
    scheduler.role = "master"
    setup.monkeypatch.setattr(gateway.node_config, "load_node_config", lambda: pytest.fail("master must not read bootstrap"))
    assert gateway.get_binding(scheduler, 18001) == {
        "status": "connected", "is_master": True, "master_host": "127.0.0.1",
        "master_api_port": 18001, "reason_code": "", "reason": "",
    }


@pytest.mark.parametrize("field,value", [("enabled", False), ("run_mode", "single")])
@pytest.mark.parametrize("role", ["master", "client"])
def test_local_mode_blocks_even_a_saved_live_connection(setup, field, value, role):
    setup.scheduler.role = role
    setattr(setup.scheduler, field, value)
    setup.monkeypatch.setattr(gateway.node_config, "load_node_config", lambda: pytest.fail("local mode must not read bootstrap"))
    binding = gateway.get_binding(setup.scheduler, 18001)
    assert binding["status"] == "local" and binding["reason_code"] == "INTENT_LOCAL_MODE"
    rejected(setup, "INTENT_LOCAL_MODE", 409)
    assert not setup.calls and not setup.options


@pytest.mark.parametrize("field,value", [("_running", False), ("is_registered", False), ("sock", None)])
def test_actual_tcp_registration_and_socket_are_required(setup, field, value):
    setattr(setup.scheduler._tcp_client, field, value)
    setup.scheduler._tcp_client.connected = True
    assert gateway.get_binding(setup.scheduler, 18001)["status"] == "not_found"
    rejected(setup, "INTENT_MASTER_DISCONNECTED", 503)
    assert not setup.calls


def test_missing_client_does_not_fall_back_to_discovery_or_environment(setup):
    setup.scheduler._tcp_client = None
    setup.scheduler.discover_master = lambda: pytest.fail("relay must not discover a different master")
    setup.monkeypatch.setenv("QLH_MASTER_API_HOST", "other.test")
    rejected(setup, "INTENT_MASTER_DISCONNECTED", 503)
    assert not setup.calls


@pytest.mark.parametrize("role", ["master", "unknown", "auto"])
def test_forwarding_role_gate(setup, role):
    setup.scheduler.role = role
    rejected(setup, "INTENT_CLIENT_REQUIRED", 403)
    assert not setup.calls


def test_non_bootstrapped_connection_is_not_an_api_binding(setup):
    setup.config["bootstrapped"] = False
    rejected(setup, "INTENT_MASTER_NOT_FOUND", 503)
    assert not setup.calls


@pytest.mark.parametrize("field,value", [
    ("master_api_host", "http://evil.test"), ("master_api_host", "master.test@evil.test"),
    ("master_api_host", "master.test/evil"), ("master_api_host", "master.test:18000"),
    ("master_api_host", "master.test?next=evil"), ("master_api_host", "0.0.0.0"),
    ("master_api_host", "::"), ("master_api_host", ""),
    ("master_api_port", None), ("master_api_port", 65536),
    ("master_api_port", True), ("master_api_port", 18000.0),
    ("master_tcp_port", 0),
])
def test_invalid_bootstrap_endpoint_never_opens_http(setup, field, value):
    setup.config["cluster"][field] = value
    rejected(setup, "INTENT_MASTER_BINDING_INVALID", 409)
    assert not setup.calls


@pytest.mark.parametrize("field,value", [("master_tcp_host", "100.64.2.4"), ("master_tcp_port", 28888)])
def test_current_master_must_match_bootstrap_tcp_binding(setup, field, value):
    setup.config["cluster"][field] = value
    rejected(setup, "INTENT_MASTER_BINDING_MISMATCH", 409)
    assert not setup.calls


def test_preferred_ipv6_peer_cannot_inherit_old_bootstrap_api_port(setup):
    setup.scheduler._tcp_client.server_host = "fd7a:115c:a1e0::4"
    setup.config["cluster"]["preferred_master_endpoint"] = {"host": "fd7a:115c:a1e0::4", "port": 18888}
    rejected(setup, "INTENT_MASTER_BINDING_MISMATCH", 409)
    assert not setup.calls


def test_bootstrap_fallback_connection_is_valid_even_with_a_different_preference(setup):
    setup.config["cluster"]["preferred_master_endpoint"] = {"host": "100.64.2.4", "port": 28888}
    assert forward(setup) == {"ok": True}
    assert setup.calls[0].url.port == 18000


def test_registered_node_must_match_bootstrap_identity(setup):
    setup.config["node"]["node_id"] = "old-worker"
    rejected(setup, "INTENT_MASTER_BINDING_MISMATCH", 409)
    assert not setup.calls


def test_invalid_config_or_config_read_failure_is_named(setup):
    def fail():
        raise OSError("private config path must not leak")
    setup.monkeypatch.setattr(gateway.node_config, "load_node_config", fail)
    failed = rejected(setup, "INTENT_MASTER_BINDING_INVALID", 409)
    assert "private config" not in str(failed.detail)


def test_ipv6_endpoint_is_canonical_and_url_bracketed(setup):
    setup.scheduler._tcp_client.server_host = "[fd7a:115c:a1e0::4]"
    setup.config["cluster"]["master_tcp_host"] = "fd7a:115c:a1e0:0:0:0:0:4"
    setup.config["cluster"]["master_api_host"] = "[fd7a:115c:a1e0::5]"
    assert forward(setup) == {"ok": True}
    assert str(setup.calls[0].url) == "http://[fd7a:115c:a1e0::5]:18000/api/models/intent/catalog"


@pytest.mark.parametrize("method,suffix", [
    ("GET", "catalog"), ("GET", "current"), ("POST", ""),
    ("GET", OPERATION), ("DELETE", OPERATION),
])
def test_only_router_contract_is_forwarded(setup, method, suffix):
    body = {"model_id": "model-a"} if method == "POST" else None
    result = forward(setup, method, suffix, body=body)
    assert result == {"ok": True} and len(setup.calls) == 1
    outgoing = setup.calls[0]
    assert outgoing.method == method
    assert outgoing.url.path == "/api/models/intent" + ("/" + suffix if suffix else "")
    assert outgoing.url.host == "master-api.tailnet.test" and outgoing.url.port == 18000
    if body is not None:
        assert json.loads(outgoing.content) == body
        assert outgoing.headers["Content-Type"].startswith("application/json")
    assert setup.options[0]["trust_env"] is False and setup.options[0]["follow_redirects"] is False
    assert setup.options[0]["timeout"].connect == 5.0


@pytest.mark.parametrize("method,suffix", [
    ("GET", ""), ("POST", "catalog"), ("DELETE", "current"), ("POST", OPERATION),
    ("PUT", OPERATION), ("GET", "connect"), ("GET", "/catalog"),
    ("GET", "https://evil.test"), ("GET", "//evil.test"),
    ("GET", "catalog?url=evil"), ("GET", "../current"),
    ("GET", OPERATION + "/prepare"), ("GET", OPERATION + "?query=1"),
    ("GET", "supply_" + "A" * 32), ("GET", "supply_" + "a" * 31),
    ("GET", "supply_" + "a" * 33), ("GET", OPERATION + "\n"),
    ("GET", "catalog%2f..%2fcurrent"), ("GET", None), ("GET", []),
])
def test_unapproved_suffix_or_method_is_rejected_before_http(setup, method, suffix):
    rejected(setup, "INTENT_FORWARD_ROUTE_INVALID", 400, method=method, suffix=suffix)
    assert not setup.calls and not setup.options


def test_get_body_is_not_forwarded(setup):
    rejected(setup, "INTENT_FORWARD_ROUTE_INVALID", 400, body={"model_id": "model-a"})
    assert not setup.calls


def test_only_authorization_trace_and_certificate_from_caller_are_forwarded(setup):
    headers = {
        "Authorization": "Bearer caller-token", "X-Request-ID": "request-1",
        "X-QLH-Control-Certificate": "certificate-exact",
        "Host": "attacker.test", "Content-Length": "999", "Cookie": "session=secret",
        "Forwarded": "for=127.0.0.1", "X-Forwarded-For": "127.0.0.1",
        "X-Forwarded-Host": "attacker.test", "X-Forwarded-Proto": "https",
        "Connection": "keep-alive, X-QLH-Supply-Token", "Proxy-Authorization": "secret",
        "X-QLH-Supply-Token": "file-lease", "X-QLH-Cluster-Secret": "cluster-secret",
    }
    setup.config["cluster"]["cluster_secret"] = "never-send-bootstrap-secret"
    assert forward(setup, headers=headers) == {"ok": True}
    sent = setup.calls[0].headers
    for name in gateway._REQUEST_HEADERS:
        assert sent[name] == headers[name]
    assert sent[gateway.HOP_HEADER] == "1"
    assert sent["Host"] == "master-api.tailnet.test:18000"
    assert sent["Accept-Encoding"] == "identity"
    for name in ("Cookie", "Forwarded", "X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto",
                 "Proxy-Authorization", "X-QLH-Supply-Token", "X-QLH-Cluster-Secret", "Content-Length"):
        assert name not in sent
    assert "never-send-bootstrap-secret" not in str(sent)


def test_no_bearer_or_control_certificate_is_synthesized(setup):
    forward(setup)
    assert "Authorization" not in setup.calls[0].headers
    assert "X-QLH-Control-Certificate" not in setup.calls[0].headers


@pytest.mark.parametrize("value", ["", "1", "2", "bogus"])
def test_any_existing_hop_marker_blocks_a_second_relay(setup, value):
    rejected(setup, "INTENT_FORWARD_LOOP", 508, headers={gateway.HOP_HEADER: value})
    assert not setup.calls


def test_hop_or_forwarded_headers_cannot_bypass_local_source_gate(setup):
    rejected(setup, "MODEL_API_SOURCE_UNTRUSTED", 403, peer="192.0.2.3",
             headers={gateway.HOP_HEADER: "1", "X-Forwarded-For": "127.0.0.1"})
    assert not setup.calls


@pytest.mark.parametrize("status,raw", [
    (401, b'{"detail":{"code":"auth_required","message":"login"}}'),
    (403, b'{"detail":{"code":"MODEL_API_SOURCE_UNTRUSTED"},"extra":true}'),
    (409, b'{"detail":{"code":"control_certificate_stale"}}'),
    (500, b"upstream error with original formatting\n"),
])
def test_master_errors_keep_original_body_status_and_safe_response_headers(setup, status, raw):
    upstream = response(status, raw=raw, headers={
        "Content-Type": "application/problem+json", "X-Request-ID": "master-id",
        "X-QLH-Error-Code": "master-error", "Retry-After": "3",
        "Set-Cookie": "must-not-install-cookie", "Location": "http://evil.test",
    })
    setup.install(lambda req: upstream)
    result = forward(setup)
    assert isinstance(result, Response) and result.status_code == status and result.body == raw
    assert result.headers["content-type"] == "application/problem+json"
    assert result.headers["x-request-id"] == "master-id"
    assert result.headers["retry-after"] == "3"
    assert "set-cookie" not in result.headers and "location" not in result.headers
    assert upstream.is_closed


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_redirect_is_rejected_without_following_or_returning_location(setup, status):
    upstream = response(status, headers={"Location": "http://evil.test/api/models/intent"})
    setup.install(lambda req: upstream)
    rejected(setup, "INTENT_MASTER_REDIRECT_FORBIDDEN", 502)
    assert len(setup.calls) == 1 and upstream.is_closed


@pytest.mark.parametrize("exc_type,code,status", [
    (httpx.ConnectTimeout, "INTENT_MASTER_TIMEOUT", 504),
    (httpx.ReadTimeout, "INTENT_MASTER_TIMEOUT", 504),
    (httpx.ConnectError, "INTENT_MASTER_UNREACHABLE", 502),
    (httpx.ReadError, "INTENT_MASTER_UNREACHABLE", 502),
])
def test_transport_failure_is_named_and_never_retries_writes(setup, exc_type, code, status):
    def handler(req):
        raise exc_type("private upstream diagnostics", request=req)
    setup.install(handler)
    failed = rejected(setup, code, status, method="POST", suffix="", body={"model_id": "model-a"})
    assert "private upstream" not in str(failed.detail)
    assert len(setup.calls) == 1


def test_total_deadline_bounds_a_stream_that_never_finishes(setup):
    class Stalled(Chunks):
        async def __aiter__(self):
            yield b"{"
            await asyncio.Event().wait()
    stream = Stalled([])
    setup.install(lambda req: response(stream=stream))
    setup.monkeypatch.setattr(gateway, "FORWARD_TIMEOUT_SECONDS", 0.02)
    rejected(setup, "INTENT_MASTER_TIMEOUT", 504)
    assert stream.closed


@pytest.mark.parametrize("headers", [{}, {"Content-Length": "1"}])
def test_streamed_response_limit_applies_even_without_an_honest_length(setup, headers):
    stream = Chunks([b"a" * (64 * 1024)] * 40)
    upstream = response(stream=stream, headers=headers)
    setup.install(lambda req: upstream)
    rejected(setup, "INTENT_MASTER_RESPONSE_TOO_LARGE", 502)
    assert stream.consumed == 33 and stream.closed


def test_advertised_oversize_is_rejected_before_body_consumption(setup):
    stream = Chunks([b"not consumed"])
    setup.install(lambda req: response(stream=stream, headers={"Content-Length": str(gateway.MAX_RESPONSE_BYTES + 1)}))
    rejected(setup, "INTENT_MASTER_RESPONSE_TOO_LARGE", 502)
    assert stream.consumed == 0 and stream.closed


def test_exactly_two_mib_json_is_accepted(setup):
    raw = b'"' + b"a" * (gateway.MAX_RESPONSE_BYTES - 2) + b'"'
    setup.install(lambda req: response(raw=raw))
    assert len(forward(setup)) == gateway.MAX_RESPONSE_BYTES - 2


def test_compressed_response_is_rejected_before_decompression(setup):
    stream = Chunks([b"compressed"])
    setup.install(lambda req: response(stream=stream, headers={"Content-Encoding": "gzip"}))
    rejected(setup, "INTENT_MASTER_RESPONSE_INVALID", 502)
    assert stream.consumed == 0 and stream.closed


@pytest.mark.parametrize("raw", [b"not json", b"\xff"], ids=["not-json", "invalid-encoding"])
def test_invalid_success_body_is_named(setup, raw):
    setup.install(lambda req: response(raw=raw))
    rejected(setup, "INTENT_MASTER_RESPONSE_INVALID", 502)


def test_excessive_json_nesting_is_named(setup):
    # This limit is a gateway contract, independent of pytest/plugin recursion
    # settings. The bytes are valid JSON and far below the wire-body limit.
    depth = gateway.MAX_JSON_NESTING_DEPTH + 1
    raw = b"[" * depth + b"0" + b"]" * depth
    assert len(raw) < gateway.MAX_RESPONSE_BYTES
    setup.install(lambda req: response(raw=raw))
    failed = rejected(setup, "INTENT_MASTER_RESPONSE_INVALID", 502)
    assert "nesting limit" in failed.detail["message"]


@pytest.mark.parametrize("opening,closing", [("[", "]"), ('{"a":', "}"), ('[{"a":', "}]")])
@pytest.mark.parametrize("extra_depth", [0, 2])
def test_json_nesting_checks_arrays_objects_and_mixed_containers(setup, opening, closing, extra_depth):
    depth = gateway.MAX_JSON_NESTING_DEPTH + extra_depth
    count = depth // len(closing)
    raw = (opening * count + "0" + closing * count).encode()
    setup.install(lambda req: response(raw=raw))
    if extra_depth:
        failed = rejected(setup, "INTENT_MASTER_RESPONSE_INVALID", 502)
        assert "nesting limit" in failed.detail["message"]
    else:
        value = forward(setup)
        for _ in range(count):
            if opening.startswith("["):
                value = value[0]
            if "{" in opening:
                value = value["a"]
        assert value == 0


@pytest.mark.parametrize("value", ['[{]}"' * 100, '\\"[{]}\\' * 100, '\u015b\u017b' * 100], ids=["brackets", "escapes", "unicode"])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-16-be", "utf-32", "utf-32-be"])
def test_json_depth_ignores_escaped_strings_and_preserves_byte_encodings(setup, value, encoding):
    payload = {"value": value, "siblings": [[1]] * 100}
    raw = json.dumps(payload, ensure_ascii=False).encode(encoding)
    setup.install(lambda req: response(raw=raw))
    assert forward(setup) == payload


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-16-be", "utf-32", "utf-32-be"])
def test_json_nesting_cannot_bypass_limit_with_an_alternate_encoding(setup, encoding):
    depth = gateway.MAX_JSON_NESTING_DEPTH + 1
    raw = ("[" * depth + "0" + "]" * depth).encode(encoding)
    setup.install(lambda req: response(raw=raw))
    failed = rejected(setup, "INTENT_MASTER_RESPONSE_INVALID", 502)
    assert "nesting limit" in failed.detail["message"]


def test_deep_upstream_error_body_is_preserved_without_json_parsing(setup):
    depth = gateway.MAX_JSON_NESTING_DEPTH + 1
    raw = b"[" * depth + b"0" + b"]" * depth
    setup.install(lambda req: response(403, raw=raw))
    result = forward(setup)
    assert isinstance(result, Response) and result.status_code == 403 and result.body == raw


def test_null_current_is_preserved(setup):
    setup.install(lambda req: response(payload=None))
    assert forward(setup, suffix="current") is None


def test_connection_replaced_before_http_send_never_sends(setup):
    def swap():
        setup.scheduler._tcp_client = deepcopy(setup.scheduler._tcp_client)
    setup.install(enter=swap)
    rejected(setup, "INTENT_MASTER_CONNECTION_CHANGED", 409)
    assert not setup.calls


@pytest.mark.parametrize("change,code,status", [
    ("client", "INTENT_MASTER_CONNECTION_CHANGED", 409),
    ("sock", "INTENT_MASTER_CONNECTION_CHANGED", 409),
    ("running", "INTENT_MASTER_DISCONNECTED", 503),
    ("registered", "INTENT_MASTER_DISCONNECTED", 503),
    ("role", "INTENT_CLIENT_REQUIRED", 403),
    ("local", "INTENT_LOCAL_MODE", 409),
    ("port", "INTENT_MASTER_BINDING_MISMATCH", 409),
    ("api_port", "INTENT_MASTER_CONNECTION_CHANGED", 409),
    ("epoch", "INTENT_MASTER_CONNECTION_CHANGED", 409),
    ("registration", "INTENT_MASTER_CONNECTION_CHANGED", 409),
])
def test_connection_or_binding_changed_after_send_never_returns_stale_success(setup, change, code, status):
    upstream = response(payload={"ready": True})
    def handler(req):
        client = setup.scheduler._tcp_client
        if change == "client":
            setup.scheduler._tcp_client = deepcopy(client)
        elif change == "sock":
            client.sock = object()
        elif change == "running":
            client._running = False
        elif change == "registered":
            client.is_registered = False
        elif change == "role":
            setup.scheduler.role = "master"
        elif change == "local":
            setup.scheduler.enabled = False
        elif change == "port":
            client.server_port = 28888
        elif change == "api_port":
            setup.config["cluster"]["master_api_port"] = 18002
        elif change == "epoch":
            setup.config["cluster"]["cluster_secret_epoch"] = 2
        elif change == "registration":
            client.registration_epoch += 1
        return upstream
    setup.install(handler)
    rejected(setup, code, status)
    assert len(setup.calls) == 1 and upstream.is_closed


def test_disconnect_mid_body_stops_reading_and_discards_response(setup):
    class Disconnecting(Chunks):
        async def __aiter__(self):
            yield b" " * (64 * 1024)
            setup.scheduler._tcp_client.sock = None
            yield b"{}"
    stream = Disconnecting([])
    setup.install(lambda req: response(stream=stream))
    rejected(setup, "INTENT_MASTER_DISCONNECTED", 503)
    assert stream.closed


def test_connection_rechecked_after_http_client_cleanup(setup):
    setup.install(exit=lambda: setattr(setup.scheduler._tcp_client, "sock", object()))
    rejected(setup, "INTENT_MASTER_CONNECTION_CHANGED", 409)


def test_binding_resolution_rechecks_socket_after_loading_config(setup):
    def load():
        setup.scheduler._tcp_client.sock = object()
        return deepcopy(setup.config)
    setup.monkeypatch.setattr(gateway.node_config, "load_node_config", load)
    rejected(setup, "INTENT_MASTER_DISCONNECTED", 503)
    assert not setup.calls


def test_binding_resolution_rechecks_tcp_endpoint_after_loading_config(setup):
    def load():
        setup.scheduler._tcp_client.server_host = "100.64.2.4"
        return deepcopy(setup.config)
    setup.monkeypatch.setattr(gateway.node_config, "load_node_config", load)
    rejected(setup, "INTENT_MASTER_DISCONNECTED", 503)
    assert not setup.calls


def test_oversized_decimal_content_length_never_needs_unbounded_integer_conversion(setup):
    stream = Chunks([])
    setup.install(lambda req: response(stream=stream, headers={"Content-Length": "9" * 5000}))
    rejected(setup, "INTENT_MASTER_RESPONSE_TOO_LARGE", 502)
    assert stream.closed


def test_real_tcp_client_registration_fields_are_authoritative(setup):
    from tcp_comm import TCPClient

    client = TCPClient(server_host="100.64.2.3", server_port=18888, client_id="worker-pc")
    # Construct only: do not connect or create any real sockets/threads.
    client._running = True
    client.sock = object()
    client._registered = True
    client._registration_epoch = 1
    setup.scheduler._tcp_client = client
    assert forward(setup) == {"ok": True}
    client._registered = False
    assert gateway.get_binding(setup.scheduler, 18001)["reason_code"] == "INTENT_MASTER_DISCONNECTED"
    rejected(setup, "INTENT_MASTER_DISCONNECTED", 503)
    assert len(setup.calls) == 1


@pytest.mark.parametrize("field,value,code,status", [
    ("role", "master", "INTENT_CLIENT_REQUIRED", 403),
    ("enabled", False, "INTENT_LOCAL_MODE", 409),
    ("run_mode", "single", "INTENT_LOCAL_MODE", 409),
])
def test_role_and_mode_are_rechecked_after_loading_config(setup, field, value, code, status):
    def load():
        setattr(setup.scheduler, field, value)
        return deepcopy(setup.config)
    setup.monkeypatch.setattr(gateway.node_config, "load_node_config", load)
    rejected(setup, code, status)
    assert not setup.calls and not setup.options


@pytest.mark.parametrize("field,value,code,status", [
    ("client_id", "another-worker", "INTENT_MASTER_BINDING_MISMATCH", 409),
    ("registration_epoch", 2, "INTENT_MASTER_DISCONNECTED", 503),
])
def test_registration_identity_is_rechecked_after_loading_config(setup, field, value, code, status):
    def load():
        setattr(setup.scheduler._tcp_client, field, value)
        return deepcopy(setup.config)
    setup.monkeypatch.setattr(gateway.node_config, "load_node_config", load)
    rejected(setup, code, status)
    assert not setup.calls and not setup.options


def test_renewed_registration_before_http_send_never_sends(setup):
    setup.install(enter=lambda: setattr(setup.scheduler._tcp_client, "registration_epoch", 2))
    rejected(setup, "INTENT_MASTER_CONNECTION_CHANGED", 409)
    assert not setup.calls
