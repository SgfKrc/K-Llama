"""Product runtime diagnostic stays opt-in, loopback-only and on the common API boundary."""
from types import SimpleNamespace

from fastapi import HTTPException
from starlette.requests import Request
import pytest
import api_server as api
import product_runtime_receipt as runtime


def request(host="127.0.0.1", forwarded=None):
    return Request({"type":"http", "method":"GET", "path":"/api/diagnostics/product-runtime",
        "headers": [] if forwarded is None else [(b"x-forwarded-for",forwarded.encode())],
        "query_string":b"", "client":(host,50123), "server":("127.0.0.1",8000), "scheme":"http"})


@pytest.mark.parametrize("enabled,host", [(False,"127.0.0.1"),(True,"100.1.2.3"),(True,"testclient")])
def test_default_and_network_requests_do_not_touch_runtime_or_sources(monkeypatch,enabled,host):
    monkeypatch.setenv("QLH_PRODUCT_OBSERVATION","1" if enabled else "0")
    monkeypatch.setattr(runtime,"capture_runtime",lambda *_:pytest.fail("denied request read model"),raising=False)
    monkeypatch.setattr(api._product_runtime_source,"observe_source",lambda *_:pytest.fail("denied request read source"))
    with pytest.raises(HTTPException) as error:
        api.product_runtime_observation(request(host,"127.0.0.1"),"a"*32)
    assert error.value.status_code == 404


@pytest.mark.parametrize("nonce", ["a"*31,"A"*32,"a"*33,"../file","a\n"])
def test_nonce_is_validated_before_any_observation(monkeypatch,nonce):
    monkeypatch.setenv("QLH_PRODUCT_OBSERVATION","1")
    monkeypatch.setattr(runtime,"capture_runtime",lambda *_:pytest.fail("invalid nonce touched runtime"),raising=False)
    with pytest.raises(HTTPException) as error:
        api.product_runtime_observation(request(),nonce)
    assert error.value.status_code == 422


@pytest.mark.parametrize("host", ["127.0.0.1","::1","::ffff:127.0.0.1"])
def test_valid_request_returns_actual_owner_receipt_and_source_without_constructing_manager(monkeypatch,host):
    import psutil
    monkeypatch.setenv("QLH_PRODUCT_OBSERVATION","1")
    observed={"node_id":"master","operation_id":"op","generation":2}
    source={"source_dirty":False,"manifest_sha256":"b"*64}
    calls=[]
    monkeypatch.setattr(runtime,"capture_runtime",lambda module:calls.append(module) or observed,raising=False)
    monkeypatch.setattr(api._product_runtime_source,"observe_source",lambda role:calls.append(role) or source)
    monkeypatch.setattr(api.scheduler,"_effective_role",lambda:"master")
    result=api.product_runtime_observation(request(host),"a"*32)
    assert result["nonce"]=="a"*32 and result["runtime"] is observed and result["source"] is source
    assert result["pid"]==psutil.Process().pid and result["created_at"]==psutil.Process().create_time()
    assert calls==[api,"master"] and result["errors"]==[]


def test_runtime_missing_facts_return_specific_conflict(monkeypatch):
    monkeypatch.setenv("QLH_PRODUCT_OBSERVATION","1")
    def missing(_): raise ValueError("committed runtime missing")
    monkeypatch.setattr(runtime,"capture_runtime",missing,raising=False)
    with pytest.raises(HTTPException) as error:
        api.product_runtime_observation(request(),"a"*32)
    assert error.value.status_code==409
    assert error.value.detail=={"code":"PRODUCT_OBSERVATION_UNAVAILABLE","message":"committed runtime missing"}


def test_registered_readonly_route_uses_existing_common_middleware():
    routes=[route for route in api.app.routes if getattr(route,"path",None)=="/api/diagnostics/product-runtime"]
    assert len(routes)==1 and routes[0].methods=={"GET"}
    assert routes[0].endpoint is api.product_runtime_observation
    assert "/api/diagnostics/product-runtime" not in api._API_SOURCE_EXEMPT_PATHS
    assert "/api/diagnostics/product-runtime" not in api._API_AUTH_EXEMPT_PATHS
