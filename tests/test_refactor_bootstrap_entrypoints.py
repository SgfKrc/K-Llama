"""R1 entrypoint wiring, with temporary settings and no real server/native model."""
from __future__ import annotations

import builtins
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import qlh
import tui_backend
from qlh_core.adapters.settings import IdentityStore, SettingsStore


def _unexpected_startup(*args, **kwargs):
    pytest.fail("listener discovery/reuse must not load startup credentials or components")


@pytest.fixture
def listener_inputs(monkeypatch, tmp_path):
    import node_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(node_config, "get_app_root", lambda: tmp_path)
    for name in ("QLH_BACKEND_HOST", "QLH_BACKEND_PORT", "QLH_API_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QLH_RELEASE_PROFILE_ENFORCE", "0")
    monkeypatch.setenv("QLH_NODE_ROLE", "client")
    monkeypatch.setenv("QLH_RUN_MODE", "distributed")
    # Always hit this candidate first; never read the checkout's .env.
    dotenv = tmp_path / ".env"
    dotenv.write_text("", encoding="utf-8")
    monkeypatch.setattr(SettingsStore, "load", _unexpected_startup)
    monkeypatch.setattr(IdentityStore, "credential", _unexpected_startup)
    return dotenv


@pytest.fixture
def shell(monkeypatch):
    seen = {}
    monkeypatch.setattr(qlh, "_textual_available", lambda: True)

    def run(host, port, **options):
        seen.update(host=host, port=port, **options)
        supervisor = options["supervisor"]
        if supervisor is not None:
            try:
                seen["started"] = supervisor.ensure_ready()
            finally:
                supervisor.stop()
        return 0

    monkeypatch.setitem(sys.modules, "tui_textual", SimpleNamespace(run=run))
    return seen


@pytest.fixture
def local_backend(monkeypatch, shell):
    events = []
    probes = iter([False, True])
    monkeypatch.setattr(tui_backend.BackendSupervisor, "probe", lambda self, **kwargs: next(probes))
    monkeypatch.setattr(tui_backend.BackendSupervisor, "_silence_console_logging", staticmethod(lambda: None))
    runtime = SimpleNamespace(configure_listener=lambda port: events.append(("configure", port)))

    def config(app, **options):
        events.append(("bind", options["host"], options["port"]))
        shell["listener"] = options
        return options

    class Server:
        def __init__(self, config):
            self.should_exit = False

        def run(self):
            pass

    monkeypatch.setitem(sys.modules, "api_server", SimpleNamespace(
        app=object(), _get_app_runtime=lambda: runtime,
        register_uvicorn_server=lambda server: None,
    ))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(Config=config, Server=Server))
    shell["events"] = events
    return shell


@pytest.mark.parametrize("command", [[], ["chat"], ["tui"], ["ui"]])
@pytest.mark.parametrize(
    "dotenv, explicit, arguments, client_host, bind_host, port",
    [
        ("QLH_BACKEND_HOST=0.0.0.0\nQLH_BACKEND_PORT=18101\nQLH_API_PORT=18100\n", {}, [], "127.0.0.1", "0.0.0.0", 18101),
        ("QLH_BACKEND_HOST=::\nQLH_BACKEND_PORT=18101\n", {"QLH_BACKEND_HOST": "127.0.0.1", "QLH_BACKEND_PORT": "18102"}, [], "127.0.0.1", "127.0.0.1", 18102),
        ("QLH_BACKEND_HOST=0.0.0.0\nQLH_API_PORT=18103\n", {}, ["--port", "18104", "--backend-host", "127.0.0.1"], "127.0.0.1", "127.0.0.1", 18104),
        ("QLH_BACKEND_HOST=0.0.0.0\nQLH_BACKEND_PORT=18101\n", {}, ["--host", "http://127.0.0.1:18105"], "127.0.0.1", "0.0.0.0", 18105),
        ("QLH_API_PORT=18106\n", {}, [], "127.0.0.1", "127.0.0.1", 18106),
        ("", {}, ["--host", "http://[::1]:18107"], "::1", "::1", 18107),
        ("", {}, [], "127.0.0.1", "127.0.0.1", 8000),
    ],
    ids=["dotenv", "explicit-env", "cli", "url-port-dotenv-host", "api-port", "ipv6-loopback", "defaults"],
)
def test_tui_entrypoint_resolves_listener_before_bind(
    monkeypatch, listener_inputs, local_backend, command,
    dotenv, explicit, arguments, client_host, bind_host, port,
):
    listener_inputs.write_text(dotenv, encoding="utf-8")
    for name, value in explicit.items():
        monkeypatch.setenv(name, value)
    load_listener = SettingsStore.load_listener
    calls = []

    def listener(self):
        snapshot = load_listener(self)
        calls.append(snapshot)
        # Both choices must already belong to the returned snapshot.
        monkeypatch.setenv("QLH_BACKEND_HOST", "192.0.2.99")
        monkeypatch.setenv("QLH_BACKEND_PORT", "19999")
        return snapshot

    monkeypatch.setattr(SettingsStore, "load_listener", listener)
    assert qlh.main([*command, *arguments]) == 0
    assert len(calls) == 1
    assert (local_backend["host"], local_backend["port"]) == (client_host, port)
    assert local_backend["listener"]["host"] == bind_host
    assert local_backend["listener"]["port"] == port
    assert local_backend["events"] == [("configure", port), ("bind", bind_host, port)]
    assert local_backend["started"] is True
    supervisor = local_backend["supervisor"]
    assert supervisor.server.should_exit is True
    if supervisor.thread is not None:
        supervisor.thread.join(timeout=1)
        assert not supervisor.thread.is_alive()
    probe_host = "[::1]" if client_host == "::1" else client_host
    assert supervisor.health_url == f"http://{probe_host}:{port}/api/health"


def test_tui_entrypoint_reads_public_master_port(listener_inputs, local_backend, monkeypatch):
    monkeypatch.delenv("QLH_NODE_ROLE")
    Path(os.environ["QLH_NODE_CONFIG_PATH"]).write_text(json.dumps({
        "node": {"role": "master", "role_confirmed": True},
        "cluster": {"master_api_port": 18108},
    }), encoding="utf-8")
    assert qlh.main([]) == 0
    assert local_backend["port"] == local_backend["listener"]["port"] == 18108
    assert local_backend["listener"]["host"] == "127.0.0.1"


@pytest.mark.parametrize("command", [[], ["chat"], ["tui"], ["ui"]])
@pytest.mark.parametrize("host", ["http://192.0.2.10:18109", "http://[2001:db8::1]:18109"])
def test_remote_tui_entrypoint_does_not_resolve_or_start_local_backend(monkeypatch, shell, command, host):
    monkeypatch.setattr(SettingsStore, "load_listener", _unexpected_startup)
    monkeypatch.setattr(tui_backend, "BackendSupervisor", _unexpected_startup)
    assert qlh.main([*command, "--host", host, "--backend-host", "0.0.0.0", "--auto-start"]) == 0
    assert shell["port"] == 18109
    assert shell["supervisor"] is None


@pytest.mark.parametrize("failure", ["missing-key", "malformed-store", "bad-ciphertext"])
def test_healthy_backend_reuse_ignores_broken_local_secret(
    monkeypatch, listener_inputs, shell, failure,
):
    from local_secret_store import LocalSecretStore

    listener_inputs.write_text("QLH_BACKEND_PORT=18110\n", encoding="utf-8")
    secret = Path(os.environ["QLH_NODE_SECRET_STORE_PATH"])
    key = secret.with_suffix(secret.suffix + ".key")
    if failure == "malformed-store":
        secret.write_text("{broken", encoding="utf-8")
        key.write_bytes(b"broken-wrapping-key")
    elif failure == "missing-key":
        secret.write_text('{"schema":"qlh.local-secret-store.v1","records":{}}', encoding="utf-8")
    else:
        LocalSecretStore(secret).set("cluster_secret", "fixture-credential")
        payload = json.loads(secret.read_text(encoding="utf-8"))
        payload["records"]["cluster_secret"]["ciphertext"] = "invalid"
        secret.write_text(json.dumps(payload), encoding="utf-8")
    before = (secret.read_bytes(), key.read_bytes() if key.exists() else None)
    for method in ("get", "set", "_load_payload", "_load_or_create_key"):
        monkeypatch.setattr(LocalSecretStore, method, _unexpected_startup)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"api_server", "uvicorn"}:
            _unexpected_startup()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    urls = []

    class Healthy:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(url, **kwargs):
        urls.append(url)
        return Healthy()

    monkeypatch.setattr(tui_backend.urllib.request, "urlopen", urlopen)
    assert qlh.main([]) == 0
    assert urls == ["http://127.0.0.1:18110/api/health"]
    assert shell["started"] is False
    assert shell["supervisor"].server is None
    assert before == (secret.read_bytes(), key.read_bytes() if key.exists() else None)


def _segment(directory, name, source_sha, span=(0, 4)):
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / (name + ".gguf")
    artifact.write_bytes(name.encode("ascii"))
    artifact.with_suffix(".manifest.json").write_text(json.dumps({
        "source_layer_range": list(span),
        "source_model_sha256": source_sha,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    return artifact


@pytest.mark.parametrize("mode", ["snapshot", "legacy-none", "empty-snapshot"])
def test_prepare_pipeline_model_uses_instance_layer_gguf(monkeypatch, tmp_path, mode):
    import model_supply_worker
    from llama_engine import LlamaCppEngine

    source = tmp_path / "source.gguf"
    source.write_bytes(b"source")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot_artifact = _segment(tmp_path / "snapshot", "head", source_sha)
    process_artifact = _segment(tmp_path / "process", "head", source_sha, (4, 8) if mode == "snapshot" else (0, 4))
    monkeypatch.setenv("QLH_LAYER_GGUF", str(process_artifact))
    monkeypatch.setattr(model_supply_worker, "selected_gguf_path", lambda: "")
    engine = LlamaCppEngine()
    engine._environment = {"QLH_LAYER_GGUF": str(snapshot_artifact)} if mode == "snapshot" else ({} if mode == "empty-snapshot" else None)
    monkeypatch.setattr(engine, "get_pipeline_descriptor", lambda: {
        "model_sha256": source_sha, "pipeline_runtime_supported": True,
    })
    if mode == "empty-snapshot":
        with pytest.raises(ValueError, match="layer_range"):
            engine.prepare_pipeline_model("fixture", str(source), layer_range=(0, 4))
    else:
        result = engine.prepare_pipeline_model("fixture", str(source), layer_range=(0, 4))
        assert result["model_sha256"] == source_sha
        assert engine.is_pipeline_prepared is True
        assert engine.is_loaded is False


@pytest.mark.parametrize("mode", ["snapshot", "legacy-none", "empty-snapshot"])
def test_find_layer_artifact_uses_instance_artifact_directory(monkeypatch, tmp_path, mode):
    import model_supply_worker
    from llama_engine import LlamaCppEngine

    source = tmp_path / "models" / "source.gguf"
    source.parent.mkdir()
    source.write_bytes(b"source")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    fallback = _segment(source.parent, "fallback", source_sha)
    snapshot_artifact = _segment(tmp_path / "snapshot", "snapshot-head", source_sha)
    process_artifact = _segment(tmp_path / "process", "process-head", source_sha)
    monkeypatch.setenv("QLH_LAYER_ARTIFACT_DIR", str(process_artifact.parent))
    monkeypatch.setattr(model_supply_worker, "selected_gguf_path", lambda: "")
    engine = LlamaCppEngine()
    engine._model_path = str(source)
    engine._pipeline_descriptor = {"model_path": str(source), "model_sha256": source_sha}
    engine._environment = {"QLH_LAYER_ARTIFACT_DIR": str(snapshot_artifact.parent)} if mode == "snapshot" else ({} if mode == "empty-snapshot" else None)
    expected = {"snapshot": snapshot_artifact, "legacy-none": process_artifact, "empty-snapshot": fallback}[mode]
    assert engine._find_layer_artifact(0, 4) == str(expected)
