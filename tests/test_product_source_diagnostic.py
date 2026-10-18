"""Source diagnostics work with no loaded model and retain the API boundary."""
import pytest
from fastapi import HTTPException
from starlette.requests import Request
import api_server as api
import product_runtime_receipt as runtime


def request(host="127.0.0.1"):
    return Request({"type": "http", "method": "GET", "path": "/api/diagnostics/product-source",
                    "headers": [(b"x-forwarded-for", b"127.0.0.1")], "query_string": b"",
                    "client": (host, 50123), "server": ("127.0.0.1", 8000), "scheme": "http"})


@pytest.mark.parametrize("enabled,host", [(False, "127.0.0.1"), (True, "100.1.2.3"),
                                          (True, "testclient"), (True, "")])
def test_denied_request_never_reads_source_or_model(monkeypatch, enabled, host):
    monkeypatch.setenv("QLH_PRODUCT_OBSERVATION", "1" if enabled else "0")
    monkeypatch.setattr(api._product_runtime_source, "observe_source",
                        lambda *_: pytest.fail("denied request touched source"))
    monkeypatch.setattr(runtime, "capture_runtime", lambda *_: pytest.fail("model read"))
    with pytest.raises(HTTPException) as error:
        api.product_source_observation(request(host), "a" * 32)
    assert error.value.status_code == 404


@pytest.mark.parametrize("nonce", ["a" * 31, "A" * 32, "a" * 33, "../file", "a\n"])
def test_invalid_nonce_is_rejected_before_source_read(monkeypatch, nonce):
    monkeypatch.setenv("QLH_PRODUCT_OBSERVATION", "1")
    monkeypatch.setattr(api._product_runtime_source, "observe_source",
                        lambda *_: pytest.fail("invalid nonce touched source"))
    with pytest.raises(HTTPException) as error:
        api.product_source_observation(request(), nonce)
    assert error.value.status_code == 422


@pytest.mark.parametrize("role,host", [("master", "127.0.0.1"), ("client", "::1"),
                                       ("client", "::ffff:127.0.0.1")])
def test_empty_runtime_can_still_observe_source(monkeypatch, role, host):
    import psutil
    monkeypatch.setenv("QLH_PRODUCT_OBSERVATION", "1")
    monkeypatch.setattr(runtime, "capture_runtime", lambda *_: pytest.fail("empty model read"))
    monkeypatch.setattr(api, "_peek_model_manager", lambda: pytest.fail("constructed/read model manager"))
    monkeypatch.setattr(api.scheduler, "_effective_role", lambda: role)
    source = {"kind": "installed_exe", "source_dirty": False}
    calls = []
    monkeypatch.setattr(api._product_runtime_source, "observe_source",
                        lambda actual_role: calls.append(actual_role) or source)
    result = api.product_source_observation(request(host), "a" * 32)
    assert calls == ["master" if role == "master" else "pc"]
    assert result["source"] is source and result["nonce"] == "a" * 32
    assert result["pid"] == psutil.Process().pid
    assert result["created_at"] == psutil.Process().create_time()
    assert result["errors"] == [] and "runtime" not in result


def test_source_error_is_visible_before_model_prepare(monkeypatch):
    monkeypatch.setenv("QLH_PRODUCT_OBSERVATION", "1")
    monkeypatch.setattr(runtime, "capture_runtime", lambda *_: pytest.fail("empty model read"))
    def unavailable(_):
        raise api._product_runtime_source.SourceObservationError("installed EXE inventory mismatch")
    monkeypatch.setattr(api._product_runtime_source, "observe_source", unavailable)
    with pytest.raises(HTTPException) as error:
        api.product_source_observation(request(), "a" * 32)
    assert error.value.status_code == 409
    assert error.value.detail == {"code": "PRODUCT_SOURCE_UNAVAILABLE",
                                  "message": "installed EXE inventory mismatch"}


def test_source_route_retains_common_auth_and_source_checks():
    path = "/api/diagnostics/product-source"
    routes = [route for module in api._api_route_modules for route in module.router.routes
              if getattr(route, "path", None) == path]
    assert len(routes) == 1 and routes[0].methods == {"GET"}
    assert routes[0].endpoint is api.product_source_observation
    assert set(api.app.openapi()["paths"][path]) == {"get"}
    assert path not in api._API_SOURCE_EXEMPT_PATHS
    assert path not in api._API_AUTH_EXEMPT_PATHS
