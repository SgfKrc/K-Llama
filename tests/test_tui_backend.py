"""进程内 TUI 后端监督器测试。"""

import os
import sys
import logging
import threading
import types

import pytest

sys.path.insert(0, "src")

import tui_backend


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_existing_backend_is_reused(monkeypatch):
    monkeypatch.setattr(tui_backend.urllib.request, "urlopen",
                        lambda *args, **kwargs: _Response())
    supervisor = tui_backend.BackendSupervisor(startup_timeout=1)

    assert supervisor.ensure_ready() is False
    assert supervisor.started_here is False


@pytest.mark.parametrize("bind_host", [None, "0.0.0.0", "::"])
def test_remote_backend_is_never_started(monkeypatch, bind_host):
    def refused(*args, **kwargs):
        raise OSError("refused")

    monkeypatch.setattr(tui_backend.urllib.request, "urlopen", refused)
    supervisor = tui_backend.BackendSupervisor(host="100.100.52.106", bind_host=bind_host)

    with pytest.raises(tui_backend.BackendStartupError, match="远程后端"):
        supervisor.ensure_ready()


@pytest.mark.parametrize(
    "client_host, bind_host, environment_bind, expected_bind",
    [
        ("127.0.0.1", None, None, "127.0.0.1"),
        ("localhost", None, None, "127.0.0.1"),
        ("::1", None, None, "::1"),
        ("127.0.0.1", "0.0.0.0", None, "0.0.0.0"),
        ("::1", "::", None, "::"),
        ("127.0.0.1", None, "0.0.0.0", "0.0.0.0"),
        ("127.0.0.1", "127.0.0.1", "0.0.0.0", "127.0.0.1"),
    ],
)
def test_missing_local_backend_runs_uvicorn_in_daemon_thread(
    monkeypatch, client_host, bind_host, environment_bind, expected_bind,
):
    if environment_bind is None:
        monkeypatch.delenv("QLH_BACKEND_HOST", raising=False)
    else:
        monkeypatch.setenv("QLH_BACKEND_HOST", environment_bind)
    original_cidrs = os.environ.get("QLH_MODEL_API_TRUSTED_CIDRS")
    probes = iter([False, True])
    probe_timeouts = []

    def fake_probe(self, *, timeout=None):
        probe_timeouts.append(timeout)
        return next(probes)

    monkeypatch.setattr(tui_backend.BackendSupervisor, "probe",
                        fake_probe)
    servers = []

    class FakeServer:
        def __init__(self, config):
            self.should_exit = False
            self.config = config
            self.ran = threading.Event()
            servers.append(self)

        def run(self):
            self.ran.set()

    fake_api = types.SimpleNamespace(
        app=object(),
        register_uvicorn_server=lambda server: None,
    )
    fake_uvicorn = types.SimpleNamespace(
        Config=lambda *args, **kwargs: (args, kwargs),
        Server=FakeServer,
    )
    monkeypatch.setitem(sys.modules, "api_server", fake_api)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    supervisor = tui_backend.BackendSupervisor(
        host=client_host, bind_host=bind_host, startup_timeout=1,
    )
    assert supervisor.ensure_ready() is True
    assert supervisor.started_here is True
    assert supervisor.thread is not None
    assert supervisor.thread.daemon is True
    assert servers and servers[0].ran.is_set()
    assert servers[0].config[1]["log_level"] == "warning"
    assert servers[0].config[1]["access_log"] is False
    assert servers[0].config[1]["host"] == expected_bind
    assert supervisor.host == client_host
    assert os.environ.get("QLH_MODEL_API_TRUSTED_CIDRS") == original_cidrs
    assert probe_timeouts == [0.2, None]
    supervisor.stop()


def test_backend_console_handlers_are_silenced_but_file_handler_survives(
    monkeypatch, tmp_path,
):
    console = logging.StreamHandler()
    file_handler = logging.FileHandler(tmp_path / "backend.log", encoding="utf-8")
    root = types.SimpleNamespace(handlers=[console, file_handler])
    named = types.SimpleNamespace(handlers=[console])

    def fake_get_logger(name=None):
        return root if name is None else named

    fake_logging = types.SimpleNamespace(
        CRITICAL=logging.CRITICAL,
        StreamHandler=logging.StreamHandler,
        FileHandler=logging.FileHandler,
        getLogger=fake_get_logger,
    )
    monkeypatch.setattr(tui_backend, "logging", fake_logging)
    original_console_level = console.level
    tui_backend.BackendSupervisor._silence_console_logging()
    try:
        assert console.level > logging.CRITICAL
        assert file_handler.level == logging.NOTSET
    finally:
        console.setLevel(original_console_level)
        file_handler.close()


@pytest.mark.parametrize("bind_host", ["0.0.0.0", "::"])
def test_public_bind_preserves_model_api_cidr_boundary(monkeypatch, bind_host):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from model_api_access import require_model_api_source

    monkeypatch.delenv("QLH_MODEL_API_TRUSTED_CIDRS", raising=False)
    supervisor = tui_backend.BackendSupervisor(bind_host=bind_host)
    assert supervisor.bind_host == bind_host
    assert supervisor.health_url == "http://127.0.0.1:8000/api/health"
    app = FastAPI()

    @app.get("/model-control")
    def model_control(request: Request):
        return {"source": require_model_api_source(request)}

    with TestClient(app, client=("192.0.2.10", 50000)) as client:
        response = client.get("/model-control")
        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "MODEL_API_SOURCE_UNTRUSTED"
        monkeypatch.setenv("QLH_MODEL_API_TRUSTED_CIDRS", "192.0.2.0/24")
        assert client.get("/model-control").status_code == 200
    with TestClient(app, client=("198.51.100.10", 50000)) as client:
        assert client.get("/model-control").status_code == 403
