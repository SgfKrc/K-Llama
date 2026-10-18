"""The TUI intent transport remains standard-library only and uses local API relay."""
import io
import json
import urllib.error

import pytest

import tui_api as api_module
from tui_api import ApiClient, ApiError
from tui_shared import MODEL_INTENT_STATUSES, format_model_intent, model_intent_ready

OP = "supply_" + "a" * 32


def test_intent_transport_paths_payload_and_auth(monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append((request.method, request.full_url,
                      json.loads(request.data) if request.data else None, timeout,
                      dict(request.header_items())))
        return io.BytesIO(json.dumps({"schema_version": 2, "revision": 7, "identity": {"boot_id": "tui-test"}, "observed_at": 100, "validity": "valid", "operation_id": OP, "status": "queued"}).encode())

    monkeypatch.setattr(api_module.urllib.request, "urlopen", urlopen)
    api = ApiClient(auth_token="fixture-token")
    api_module.create_model_intent(api, "master-model", engine="llama_cpp", quant_type="Q4_K_M")
    api_module.get_model_intent_catalog(api)
    api_module.current_model_intent(api)
    api_module.get_model_intent(api, OP)
    api_module.cancel_model_intent(api, OP)
    api_module.connect_model_intent(api)
    assert [(method, url.removeprefix(api.base_url)) for method, url, *_ in calls] == [
        ("POST", "/api/models/intent"), ("GET", "/api/models/intent/catalog"),
        ("GET", "/api/models/intent/current"), ("GET", f"/api/models/intent/{OP}"),
        ("DELETE", f"/api/models/intent/{OP}"), ("POST", "/api/models/intent/connect"),
    ]
    assert calls[0][2] == {"model_id": "master-model", "engine": "llama_cpp", "quant_type": "Q4_K_M",
                           "strategy": "auto", "timeout_seconds": 3600}
    assert calls[-1][2] == {}
    assert all(headers.get("Authorization") == "Bearer fixture-token" for *_, headers in calls)
    assert all(url.startswith(api.base_url) for _, url, *_ in calls)


@pytest.mark.parametrize("bad", ["", "../master", "supply_" + "a" * 31,
                                  "supply_" + "g" * 32, "current", OP + "/prepare"])
def test_operation_id_rejected_before_request(bad):
    class NoNetwork:
        def get(self, *_args, **_kwargs):
            pytest.fail("invalid operation reached transport")
        delete = get
    for function in (api_module.get_model_intent, api_module.cancel_model_intent):
        with pytest.raises(ValueError):
            function(NoNetwork(), bad)


def test_http_error_retains_named_reason_for_retry(monkeypatch):
    def urlopen(*_args, **_kwargs):
        body = {"detail": {"code": "INTENT_MASTER_NOT_FOUND", "message": "Retry discovery"}}
        raise urllib.error.HTTPError("http://fixture", 409, "Conflict", {},
                                     io.BytesIO(json.dumps(body).encode()))
    monkeypatch.setattr(api_module.urllib.request, "urlopen", urlopen)
    with pytest.raises(ApiError) as caught:
        api_module.connect_model_intent(ApiClient())
    assert caught.value.status == 409
    assert caught.value.reason_code == "INTENT_MASTER_NOT_FOUND"
    assert "INTENT_MASTER_NOT_FOUND" in str(caught.value)


def test_local_full_load_requires_leave_and_successful_real_unload():
    class Control:
        def __init__(self, fail=None):
            self.calls = []
            self.fail = fail
        def request(self, method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            if path == self.fail:
                return {"success": False, "reason": "runtime still loaded"}
            return {"success": True}
    api = Control()
    api_module.load_local_model(api, "local-model")
    assert [(method, path) for method, path, _ in api.calls] == [
        ("PUT", "/cluster/config/distributed-inference"), ("POST", "/models/unload"),
        ("POST", "/models/load"),
    ]
    assert api.calls[0][2]["body"] == {"enabled": False}
    api = Control(fail="/models/unload")
    with pytest.raises(ApiError, match="runtime still loaded"):
        api_module.load_local_model(api, "local-model")
    assert all(path != "/models/load" for _, path, _ in api.calls)


@pytest.mark.parametrize("status", MODEL_INTENT_STATUSES)
def test_status_projection_reports_nodes_and_reasons(status):
    operation = {"schema_version": 2, "revision": 7, "identity": {"boot_id": "tui-test"}, "observed_at": 100, "validity": "valid", "runnable": status == "ready", "operation_id": OP, "status": status, "config_id": "config-3", "generation": 7,
                 "loaded_ready": status == "ready", "reason_code": "MODEL_REASON", "reason": "detail",
                 "nodes": {"worker-1": {"status": "verifying", "bytes_received": 33, "total_bytes": 44,
                                          "reason_code": "NODE_REASON", "reason": "receipt", "layer_range": [5, 9]}}}
    text = format_model_intent(operation)
    for fragment in (status, OP, "config-3", "7", "worker-1", "verifying", "33/44 bytes",
                     "[5, 9]", "MODEL_REASON", "detail", "NODE_REASON", "receipt"):
        assert fragment in text
    assert model_intent_ready(operation) is (status == "ready")
    operation["loaded_ready"] = False
    assert not model_intent_ready(operation)


def test_current_intent_accepts_no_selection():
    class Empty:
        def get(self, *_args):
            return None
    assert api_module.current_model_intent(Empty()) is None


def test_readonly_distributed_commands_use_master_catalog_and_intent(capsys):
    import argparse
    import tui_commands
    class ReadOnly:
        accept_snapshot = ApiClient.accept_snapshot
        _snapshot_cache = {}
        base_url = "http://local-worker:8000"
        def __init__(self):
            self.paths = []
        def get(self, path, **_kwargs):
            self.paths.append(path)
            return {
                "/health": {"status": "ok"}, "/status": {"run_mode": "distributed"},
                "/models/intent/catalog": {"models": [{"model_id": "master-model"}],
                                            "binding": {"status": "connected"}},
                "/models/intent/current": {"schema_version": 2, "revision": 7, "identity": {"boot_id": "tui-test"}, "observed_at": 100, "validity": "valid", "operation_id": OP, "status": "preparing",
                                            "intent": {"model_id": "master-model"}},
            }.get(path, {})
    api = ReadOnly()
    args = argparse.Namespace(json=False)
    assert tui_commands.cmd_models(api, args) == 0
    assert tui_commands.cmd_status(api, args) == 0
    text = capsys.readouterr().out
    assert "master-model" in text and "preparing" in text and OP in text
    assert "/models" not in api.paths and "/models/current" not in api.paths


@pytest.mark.parametrize("mode, expected", [(None, "interactive"), ("interactive", "interactive"), ("full", "full")])
def test_chat_streaming_mode_is_explicit_in_local_endpoint_request(monkeypatch, mode, expected):
    calls = []
    def urlopen(request, timeout):
        calls.append((request.method, request.full_url, json.loads(request.data), dict(request.header_items())))
        return io.BytesIO(b'data: {"done":true,"response":"forwarded-answer"}\n\n')
    monkeypatch.setattr(api_module.urllib.request, "urlopen", urlopen)
    api = ApiClient(host="local-worker", auth_token="fixture-token")
    assert list(api_module.iter_chat_payloads(
        api, "hello", streaming_mode=mode, routing_preference="distributed_required",
    )) == [{"done": True, "response": "forwarded-answer"}]
    method, url, body, headers = calls[0]
    assert (method, url) == ("POST", api.base_url + "/api/chat/stream")
    assert body["streaming_mode"] == expected
    assert body["routing_preference"] == "distributed_required"
    assert headers["Authorization"] == "Bearer fixture-token"
    assert "caller_host" not in body
    assert len(calls) == 1


def test_chat_invalid_streaming_mode_never_reaches_network(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("invalid streaming mode reached transport")
    monkeypatch.setattr(api_module.urllib.request, "urlopen", no_network)
    with pytest.raises(ValueError, match="streaming_mode"):
        list(api_module.iter_chat_payloads(ApiClient(), "hello", streaming_mode="fast"))

@pytest.mark.parametrize("mode, override, expected", [
    (None, None, 60.0), ("interactive", None, 60.0), ("full", None, 120.0),
    ("full", 3.0, 3.0), ("interactive", 4.0, 4.0),
])
def test_full_chat_uses_finite_prefill_budget_and_keeps_caller_override(monkeypatch, mode, override, expected):
    budgets = []
    class PrefillResponse(io.BytesIO):
        def read(self, size=-1):
            # Simulate an otherwise valid full response whose first answer needs 70s.
            if mode == "full" and override is None and budgets[-1] < 70:
                raise TimeoutError("prefill exceeded socket read budget")
            return super().read(size)
    def open_response(request, timeout):
        budgets.append(timeout)
        return PrefillResponse(b': processing\n\ndata: {"done":true,"response":"2"}\n\n')
    monkeypatch.setattr(api_module.urllib.request, "urlopen", open_response)
    options = {"streaming_mode": mode}
    if override is not None:
        options["read_timeout"] = override
    assert list(api_module.iter_chat_payloads(ApiClient(), "hello", **options)) == [
        {"done": True, "response": "2"},
    ]
    assert budgets == [expected]