"""R1 executable contracts: explicit construction, isolated IO, truthful close."""
from __future__ import annotations
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest

from qlh_core.adapters.settings import IdentityStore, SettingsStore
from qlh_core.bootstrap.runtime import AppRuntime

pytestmark = pytest.mark.refactor_contract


def settings(root, **overrides):
    environment = {"QLH_STATE_DIR": str(root / "state"),
                   "QLH_NODE_CONFIG_PATH": str(root / "node.json"),
                   "QLH_NODE_SECRET_STORE_PATH": str(root / "secrets.json"),
                   "QLH_NODE_ID": root.name, "QLH_NODE_ROLE": "client",
                   "QLH_RUN_MODE": "single", **overrides}
    return SettingsStore(environment, dotenv_path=root / "absent.env").load()


def test_cold_import_starts_no_resources(tmp_path):
    script = textwrap.dedent("""
        import logging, sqlite3, subprocess, threading
        from unittest.mock import patch
        from contextlib import ExitStack
        from qlh_core.adapters.settings import SettingsStore
        from local_secret_store import LocalSecretStore
        import runtime_source
        def forbidden(*args, **kwargs):
            raise AssertionError('resource started during import')
        with ExitStack() as stack:
            for owner, name in [(SettingsStore, 'load'), (LocalSecretStore, 'get'),
                                (LocalSecretStore, 'set'), (threading.Thread, 'start'),
                                (subprocess, 'Popen'), (sqlite3, 'connect'),
                                (logging.FileHandler, '__init__'), (runtime_source, 'initialize')]:
                stack.enter_context(patch.object(owner, name, forbidden))
            import api_server
            assert api_server._app_runtime._services is None
            assert api_server._app_runtime.config is None
        print('IMPORT_ZERO_RESOURCES')
    """)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
           "QLH_NODE_CONFIG_PATH": str(tmp_path / "node.json"),
           "QLH_NODE_SECRET_STORE_PATH": str(tmp_path / "secrets.json"),
           "QLH_STATE_DIR": str(tmp_path / "state")}
    result = subprocess.run([sys.executable, "-B", "-c", script], env=env, cwd=tmp_path,
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "IMPORT_ZERO_RESOURCES" in result.stdout
    assert "QLH_CLUSTER_SECRET 未设置" not in result.stderr
    assert not (tmp_path / "state").exists()


def test_cluster_secret_warning_uses_resolved_runtime_snapshot(tmp_path):
    import warnings

    from config import warn_if_cluster_secret_unavailable
    from local_secret_store import LocalSecretStore

    secret = "encrypted-fixture-credential-32chars"
    config = settings(tmp_path)
    LocalSecretStore(config.secret_store_path).set("cluster_secret", json.dumps({
        "schema": "qlh.cluster-credential.v1", "secret": secret, "epoch": 4,
    }))
    resolved = SettingsStore(
        config.environment_snapshot(), dotenv_path=tmp_path / "absent.env",
    ).load()
    runtime = AppRuntime(resolved)
    try:
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            runtime.open_identity()
            warn_if_cluster_secret_unavailable(runtime.config.environment_snapshot())
        assert not [warning for warning in emitted if issubclass(warning.category, RuntimeWarning)]
        assert runtime.config.environment_snapshot()["QLH_CLUSTER_SECRET"] == secret
    finally:
        runtime.close()

    missing = settings(tmp_path / "missing")
    with pytest.warns(RuntimeWarning, match="集群凭据未加载"):
        warn_if_cluster_secret_unavailable(missing.environment_snapshot())


def test_final_config_is_immutable_and_identity_projection_agrees(tmp_path):
    config = settings(tmp_path, QLH_NODE_ROLE="master", QLH_RELEASE_PROFILE_ENFORCE="1")
    before = dict(os.environ)
    runtime = AppRuntime(config)
    identity = runtime.open_identity()
    assert runtime.open_identity() is identity
    secret = runtime.config.environment_snapshot()["QLH_CLUSTER_SECRET"]
    assert secret and runtime.config.legacy_settings()["CLUSTER_SECRET"] == secret
    assert secret not in repr(runtime.config)
    assert identity.node_id == config.node_id and identity.credential_epoch == 1
    with pytest.raises(FrozenInstanceError):
        runtime.config.node_id = "other"
    document = json.loads(Path(config.node_config_path).read_text("utf-8"))
    assert "cluster_secret" not in document["cluster"]
    second = AppRuntime(SettingsStore(config.environment_snapshot(), dotenv_path=tmp_path / "absent").load())
    identity2 = second.open_identity()
    assert identity2.node_id == identity.node_id and identity2.boot_id != identity.boot_id
    assert second.config.environment_snapshot()["QLH_CLUSTER_SECRET"] == secret
    assert dict(os.environ) == before
    runtime.close()
    second.close()


def test_legacy_identity_migration_preserves_metadata_and_scrubs_copy(tmp_path):
    legacy = tmp_path / "installation" / "node_config.json"
    legacy.parent.mkdir()
    document = {"node": {"node_id": "stable", "role": "client", "role_confirmed": True},
                "cluster": {"cluster_secret": "legacy-secret", "cluster_secret_epoch": 4},
                "retained": {"custom": 12}}
    legacy.write_text(json.dumps(document), "utf-8")
    config = replace(settings(tmp_path / "user", QLH_NODE_ID="stable"), legacy_document=str(legacy))
    runtime = AppRuntime(config)
    identity = runtime.open_identity()
    assert identity.node_id == "stable" and identity.credential_epoch == 4
    for candidate in (legacy, Path(config.node_config_path)):
        migrated = json.loads(candidate.read_text("utf-8"))
        assert migrated["retained"] == {"custom": 12}
        assert "cluster_secret" not in migrated["cluster"]
    assert runtime.config.legacy_settings()["CLUSTER_SECRET"] == "legacy-secret"
    runtime.close()


def test_dotenv_snapshot_precedence_and_stored_secret(tmp_path):
    dot = tmp_path / ".env"
    dot.write_text("QLH_API_PORT=8123\nQLH_CLUSTER_SECRET=from-dotenv\n", "utf-8")
    config = settings(tmp_path, QLH_CLUSTER_SECRET="stored-secret")
    IdentityStore(config.node_config_path, config.secret_store_path).open(config)
    environment = config.environment_snapshot()
    environment.update(QLH_CLUSTER_SECRET="stale-environment", QLH_API_PORT="8124")
    store = SettingsStore(environment, dotenv_path=dot)
    environment["QLH_API_PORT"] = "9999"
    resolved = store.load()
    assert resolved.api_port == 8124
    assert resolved.environment_snapshot()["QLH_CLUSTER_SECRET"] == "stored-secret"
    assert resolved.legacy_settings()["CLUSTER_SECRET"] == "stored-secret"


def test_two_runtimes_really_isolate_storage_identity_callbacks_and_readiness(tmp_path):
    import local_store
    one, two = [AppRuntime(settings(tmp_path / name, QLH_CLUSTER_SECRET=name)) for name in ("one", "two")]
    before = dict(os.environ)
    try:
        one.open_identity()
        two.open_identity()
        a, b = one.services, two.services
        a.store.initialize()
        b.store.initialize()
        a.store.invoke(local_store.set_local_user_settings, {"label": "one"})
        b.store.invoke(local_store.set_local_user_settings, {"label": "two"})
        assert a.store.invoke(local_store.get_local_user_settings)["label"] == "one"
        assert b.store.invoke(local_store.get_local_user_settings)["label"] == "two"
        with a.store.scope():
            assert local_store._sqlite_path() == a.store.path
            with b.store.scope():
                assert local_store._sqlite_path() == b.store.path
            assert local_store._sqlite_path() == a.store.path
        assert a.scheduler is not b.scheduler and a.host is not b.host
        calls = []
        for services, label in ((a, "one"), (b, "two")):
            callbacks = SimpleNamespace(active_execution_model_identity=lambda label=label: calls.append(label))
            services.scheduler.configure_callbacks(callbacks)
            services.host.configure_scheduler_callbacks(callbacks)
        a.scheduler.inference_callbacks.active_execution_model_identity()
        b.host.scheduler_callbacks.active_execution_model_identity()
        assert calls == ["one", "two"]
        assert a.scheduler.inference_callbacks is a.host.scheduler_callbacks
        assert b.scheduler.inference_callbacks is b.host.scheduler_callbacks
        assert a.scheduler._worker_prepare._runtime_factory.__self__ is a
        assert b.scheduler._worker_prepare._runtime_factory.__self__ is b
        assert a.scheduler._runtime_setting("CLUSTER_SECRET") == "one"
        assert b.scheduler._runtime_setting("CLUSTER_SECRET") == "two"
        one.mark_component("scheduler", True)
        assert two.readiness()["components"]["scheduler"] is False
        assert one.identity.node_id != two.identity.node_id
        assert one.config.environment_snapshot()["QLH_CLUSTER_SECRET"] == "one"
        assert two.config.environment_snapshot()["QLH_CLUSTER_SECRET"] == "two"
        assert dict(os.environ) == before
    finally:
        one.close()
        two.close()


def test_stop_during_startup_cannot_publish_late_readiness(tmp_path):
    entered, resume = threading.Event(), threading.Event()
    runtime = AppRuntime(settings(tmp_path))
    def startup():
        entered.set()
        assert resume.wait(5)
        for name in ("local_store", "scheduler", "device_profile"):
            runtime.mark_component(name, True)
    runtime.acquire(startup)
    assert entered.wait(5) and runtime.begin_release()
    with pytest.raises(TimeoutError):
        runtime.wait_startup(timeout=0)
    assert not runtime._closed and runtime._closing
    resume.set()
    runtime.close()
    assert runtime.readiness()["status"] == "stopped" and not runtime.readiness()["ready"]


def test_cleanup_failure_keeps_owner_for_retry(tmp_path):
    calls = []
    class Services:
        def close(self):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("injected close failure")
    runtime = AppRuntime(settings(tmp_path), resources_factory=lambda config, host: Services())
    owner = runtime.services
    with pytest.raises(OSError):
        runtime.close()
    assert runtime._services is owner and not runtime._closed
    assert runtime.readiness()["status"] == "cleanup_failed"
    runtime.close()
    assert runtime._closed and len(calls) == 2


def test_borrowed_host_is_not_unloaded_by_services(tmp_path):
    from qlh_core.bootstrap.services import RuntimeServices
    bindings = []
    host = SimpleNamespace(unload_model=lambda: pytest.fail("borrowed host unloaded"),
                           configure_catalog=bindings.append)
    services = RuntimeServices(settings(tmp_path), host=host)
    assert bindings == [services.catalog_prepare]
    services.close()


def test_native_worker_receives_frozen_environment(tmp_path, monkeypatch):
    import llama_keep_head as native
    captured = []
    class Process:
        def __init__(self, command, **kwargs):
            captured.append((command, kwargs))
    monkeypatch.setattr(native.subprocess, "Popen", Process)
    monkeypatch.setattr(native.KeepHeadUpstream, "_read_worker_response", lambda self: {"ok": True, "n_embd": 8, "n_layer": 2})
    upstream = native.KeepHeadUpstream.__new__(native.KeepHeadUpstream)
    upstream.shim_path, upstream.model_path = tmp_path / "shim.dll", tmp_path / "model.gguf"
    upstream.mode, upstream.cut_layer = "nextn", None
    upstream._environment = {"QLH_KEEP_HEAD_WORKER_LOG": str(tmp_path / "worker.log"), "INSTANCE": "one"}
    monkeypatch.setenv("INSTANCE", "other")
    upstream._init_isolated_worker(n_ctx=32, n_threads=1, n_batch=8, n_seq_max=1, extra_dll_dirs=())
    assert captured[0][1]["env"]["INSTANCE"] == "one"
    assert captured[0][1]["stderr"].closed
    assert os.environ["INSTANCE"] == "other"
    upstream._worker = None
    upstream._lib = upstream._handle = None


def test_invalid_configuration_keeps_http_liveness(tmp_path, monkeypatch):
    import asyncio
    import api_server
    from fastapi.testclient import TestClient
    runtime = AppRuntime()
    monkeypatch.setattr(api_server, "_app_runtime", runtime)
    def fail():
        raise ValueError("invalid configured port")
    monkeypatch.setattr(runtime, "resolve_config", fail)
    runtime.acquire(startup=api_server._run_runtime_startup)
    runtime.wait_startup()
    assert runtime.readiness()["status"] == "failed"
    assert "invalid configured port" in runtime.readiness()["error"]
    assert asyncio.run(api_server.health())["status"] == "ok"
    assert TestClient(api_server.app).get("/api/health").status_code == 200
    runtime.close()


def test_logging_setup_and_close_preserve_embedding_handlers(tmp_path, monkeypatch, caplog):
    import logging
    import config
    import api_server
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path / "logs"))
    root = logging.getLogger()
    original = tuple(root.handlers)
    try:
        api_server.setup_logging()
        assert all(handler in root.handlers for handler in original)
        with caplog.at_level(logging.INFO, logger="r1-host"):
            logging.getLogger("r1-host").info("embedding capture survives runtime start")
        assert "embedding capture survives runtime start" in caplog.text
    finally:
        api_server._close_logging_handlers()
    assert all(handler in root.handlers for handler in original)


def test_tui_default_port_resolves_settings_before_backend_bind(tmp_path, monkeypatch):
    import qlh
    settings_config = replace(settings(tmp_path), api_port=18123)
    monkeypatch.setattr(SettingsStore, "load_listener", lambda self: {
        **settings_config.environment_snapshot(), "QLH_API_PORT": str(settings_config.api_port),
    })
    options = qlh._shell_options(["--auto-start"])
    assert options["port"] == 18123
    assert qlh._shell_options(["--auto-start", "--port", "18124"])["port"] == 18124
    assert qlh._shell_options(["--host", "remote.example"])["port"] == 8000


def test_identity_cleanup_failure_remains_retryable(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"cluster": {"cluster_secret": "retain-until-scrub"}}), "utf-8")
    config = replace(settings(tmp_path / "runtime"), legacy_document=str(legacy))
    original = IdentityStore._write
    def fail_legacy(self, document):
        if self.path == legacy:
            raise OSError("legacy write denied")
        return original(self, document)
    with monkeypatch.context() as patch:
        patch.setattr(IdentityStore, "_write", fail_legacy)
        with pytest.raises(OSError):
            IdentityStore(config.node_config_path, config.secret_store_path, legacy_document=str(legacy)).open(config)
    primary = json.loads(Path(config.node_config_path).read_text("utf-8"))
    assert primary["identity_migration"]["legacy_document"] == str(legacy)
    reopened = SettingsStore(config.environment_snapshot(), dotenv_path=tmp_path / "absent").load()
    final, identity = IdentityStore(reopened.node_config_path, reopened.secret_store_path).open(reopened)
    assert final.environment_snapshot()["QLH_CLUSTER_SECRET"] == "retain-until-scrub"
    assert "cluster_secret" not in json.loads(legacy.read_text("utf-8"))["cluster"]
    assert "identity_migration" not in json.loads(Path(config.node_config_path).read_text("utf-8"))


def test_listener_port_is_same_in_runtime_environment_and_projection(tmp_path):
    runtime = AppRuntime(settings(tmp_path))
    config = runtime.configure_listener(18234)
    assert config.api_port == config.legacy_settings()["API_PORT"] == 18234
    assert config.environment_snapshot()["QLH_API_PORT"] == "18234"
    runtime.open_identity()
    assert runtime.config.legacy_settings()["API_PORT"] == 18234
    runtime.close()


def test_dotenv_interpolation_uses_explicit_environment_without_mutating_process(tmp_path):
    dot = tmp_path / ".env"
    dot.write_text("PREFIX=local\nQLH_STATE_DIR=${PREFIX}/state\nQLH_API_PORT=${PORT_VALUE:-18000}\n", "utf-8")
    config = settings(tmp_path)
    environment = {**config.environment_snapshot(), "PREFIX": str(tmp_path / "explicit"), "PORT_VALUE": "18235"}
    environment.pop("QLH_STATE_DIR")
    before = dict(os.environ)
    resolved = SettingsStore(environment, dotenv_path=dot).load()
    assert resolved.state_dir == str((tmp_path / "explicit" / "state").absolute())
    assert resolved.api_port == 18235
    assert dict(os.environ) == before


def test_two_store_scopes_are_isolated_across_threads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import local_store
    from qlh_core.adapters.local_store import RuntimeLocalStore
    first, second = [RuntimeLocalStore(settings(tmp_path / name)) for name in ("a", "b")]
    first.initialize()
    second.initialize()
    barrier = threading.Barrier(2)
    def write_read(store, label):
        with store.scope():
            local_store.set_local_user_settings({"label": label})
            barrier.wait(timeout=5)
            return local_store.get_local_user_settings()["label"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        a = executor.submit(write_read, first, "a")
        b = executor.submit(write_read, second, "b")
        assert a.result(timeout=5) == "a"
        assert b.result(timeout=5) == "b"


def test_tui_console_silencing_preserves_host_log_capture(caplog):
    import logging
    from tui_backend import BackendSupervisor
    previous = {handler: handler.level for handler in logging.getLogger().handlers}
    try:
        with caplog.at_level(logging.WARNING):
            BackendSupervisor._silence_console_logging()
            logging.getLogger("r1-capture").warning("host capture remains active")
        assert "host capture remains active" in caplog.text
    finally:
        for handler, level in previous.items():
            handler.setLevel(level)


def test_first_connect_and_preferred_endpoint_use_the_owning_runtime(tmp_path, monkeypatch):
    import bootstrap
    import config as legacy_config
    from bootstrap_credentials import seal_bootstrap_credential
    global_secret = legacy_config.CLUSTER_SECRET
    before = dict(os.environ)
    runtimes = []
    class Response:
        def __init__(self, body):
            self.body = body
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps(self.body).encode("utf-8")
    def exchange(request, timeout):
        payload = json.loads(request.data)
        label = payload["node_id"]
        envelope = seal_bootstrap_credential(
            public_key=payload["credential_public_key"], request_nonce=payload["credential_request_nonce"],
            requested_at=payload["credential_requested_at"], cluster_id="cluster-" + label,
            node_id=label, cluster_secret="bootstrap-secret-" + label, secret_epoch=3,
        )
        return Response({"status": "ok", "cluster": {
            "cluster_id": "cluster-" + label, "master_api_host": label + ".master", "master_api_port": 18123,
            "master_tcp_host": label + ".master", "master_tcp_port": 18888,
        }, "node": {"node_id": label, "role": "client"}, "credential_envelope": envelope})
    def forbidden(*args, **kwargs):
        pytest.fail("instance join fell back to process-global persistence")
    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", exchange)
    monkeypatch.setattr(bootstrap, "persist_bootstrap_response", forbidden)
    monkeypatch.setattr(bootstrap, "apply_runtime_config", forbidden)
    try:
        for label in ("first", "second"):
            root = tmp_path / label
            root.mkdir()
            dot = root / ".env"
            dot.write_text("\n".join(f"{key}={value}" for key, value in {
                "QLH_STATE_DIR": (root / "state").as_posix(),
                "QLH_NODE_CONFIG_PATH": (root / "node.json").as_posix(),
                "QLH_NODE_SECRET_STORE_PATH": (root / "secrets.json").as_posix(),
                "QLH_NODE_ID": label, "QLH_NODE_ROLE": "client", "QLH_RUN_MODE": "single",
            }.items()), "utf-8")
            runtime = AppRuntime(SettingsStore({}, dotenv_path=dot).load())
            runtimes.append(runtime)
            initial_boot = runtime.open_identity().boot_id
            bootstrap.first_connect(label + ".master", node_id=label,
                                    accept_bootstrap=runtime.services.scheduler._accept_bootstrap)
            assert runtime.identity.boot_id == initial_boot
            assert runtime.identity.credential_epoch == 3
            assert runtime.config.legacy_settings()["CLUSTER_SECRET"] == "bootstrap-secret-" + label
            assert runtime.services.scheduler._runtime_setting("CLUSTER_SECRET") == "bootstrap-secret-" + label
            assert runtime.services.scheduler.discover_master()["master_host"] == label + ".master"
            preferred = runtime.services.scheduler._persist_master_endpoint_preference("fd7a:115c:a1e0::10", 18889)
            assert preferred["endpoint_preference"]["persisted"] is True
            assert runtime.config.legacy_settings()["CLIENT_MASTER_PORT"] == 18889
            assert runtime.services.scheduler.discover_master()["master_host"] == "fd7a:115c:a1e0::10"
            persisted = json.loads((root / "node.json").read_text("utf-8"))
            assert persisted["cluster"]["master_tcp_host"] == label + ".master"
            assert "cluster_secret" not in persisted["cluster"]
            reloaded = SettingsStore({}, dotenv_path=dot).load()
            assert reloaded.environment_snapshot()["QLH_CLUSTER_SECRET"] == "bootstrap-secret-" + label
            assert reloaded.legacy_settings()["CLIENT_MASTER_PORT"] == 18889
        assert runtimes[0].config.environment_snapshot()["QLH_CLUSTER_SECRET"] == "bootstrap-secret-first"
        assert legacy_config.CLUSTER_SECRET == global_secret
        assert dict(os.environ) == before
    finally:
        for runtime in runtimes:
            runtime.close()


def test_bootstrap_credential_epoch_cannot_regress_or_fork(tmp_path):
    runtime = AppRuntime(settings(tmp_path))
    response = {"cluster": {"cluster_id": "stable", "cluster_secret": "current", "cluster_secret_epoch": 4},
                "node": {"node_id": runtime.config.node_id, "role": "client"}}
    try:
        runtime.accept_bootstrap(response)
        original = runtime.config
        for epoch, secret in ((3, "old"), (4, "fork")):
            with pytest.raises(ValueError):
                runtime.accept_bootstrap({**response, "cluster": {
                    **response["cluster"], "cluster_secret_epoch": epoch, "cluster_secret": secret,
                }})
            assert runtime.config is original
            assert runtime._identity_store.credential(runtime._identity_store.read_document(), {}) == ("current", 4)
    finally:
        runtime.close()


def test_api_listener_uses_dotenv_snapshot_at_the_actual_bind(tmp_path, monkeypatch):
    import api_server
    import network_address
    import uvicorn
    dot = tmp_path / ".env"
    dot.write_text("QLH_BACKEND_HOST=127.0.0.1\nQLH_BACKEND_PORT=18236\n", "utf-8")
    runtime = AppRuntime(SettingsStore(settings(tmp_path).environment_snapshot(), dotenv_path=dot).load())
    monkeypatch.setattr(api_server, "_get_app_runtime", lambda: runtime)
    observed = []
    class Socket:
        def getsockname(self):
            return ("127.0.0.1", 18236)
        def close(self):
            observed.append("closed")
    def bind(hosts, port, **kwargs):
        observed.append((hosts, port))
        return [Socket()]
    class Server:
        def __init__(self, config):
            assert config.host == "127.0.0.1" and config.port == 18236
        def run(self, *, sockets):
            assert sockets[0].getsockname() == ("127.0.0.1", 18236)
    monkeypatch.setattr(network_address, "create_listen_sockets", bind)
    monkeypatch.setattr(uvicorn, "Server", Server)
    monkeypatch.setattr(api_server, "register_uvicorn_server", lambda server: None)
    api_server.run_api_servers()
    assert observed == [(["127.0.0.1"], 18236), "closed"]
    assert runtime.config.api_port == runtime.config.legacy_settings()["API_PORT"] == 18236
    runtime.close()


def test_master_bootstrap_credential_uses_explicit_runtime_path(tmp_path, monkeypatch):
    import api_server
    runtime = AppRuntime(settings(tmp_path, QLH_NODE_ROLE="master"))
    monkeypatch.setattr(api_server, "_get_app_runtime", lambda: runtime)
    before = dict(os.environ)
    try:
        secret, epoch = api_server._bootstrap_cluster_credential()
        boot = runtime.identity.boot_id
        assert secret and epoch == 1
        assert api_server._bootstrap_cluster_credential() == (secret, epoch)
        assert runtime.identity.boot_id == boot
        assert runtime.config.legacy_settings()["CLUSTER_SECRET"] == secret
        assert "cluster_secret" not in json.loads(Path(runtime.config.node_config_path).read_text("utf-8"))["cluster"]
        assert runtime._identity_store.credential(runtime._identity_store.read_document(), {}) == (secret, epoch)
        assert dict(os.environ) == before
    finally:
        runtime.close()


def test_credential_rotation_updates_identity_snapshot_and_rolls_back_on_write_error(tmp_path, monkeypatch):
    runtime = AppRuntime(settings(tmp_path, QLH_NODE_ROLE="master", QLH_CLUSTER_SECRET="original-secret"))
    before = dict(os.environ)
    try:
        original_secret, epoch = runtime.cluster_credential()
        boot = runtime.identity.boot_id
        rotated_secret, rotated_epoch = runtime.rotate_cluster_credential()
        assert rotated_secret != original_secret and rotated_epoch == epoch + 1
        assert runtime.identity.boot_id == boot and runtime.identity.credential_epoch == rotated_epoch
        assert runtime.cluster_credential() == (rotated_secret, rotated_epoch)
        assert runtime.config.legacy_settings()["CLUSTER_SECRET"] == rotated_secret
        config = runtime.config
        def disk_full(document):
            raise OSError("injected config replacement failure")
        monkeypatch.setattr(runtime._identity_store, "_write", disk_full)
        with pytest.raises(OSError):
            runtime.rotate_cluster_credential()
        assert runtime.config is config
        assert runtime.cluster_credential() == (rotated_secret, rotated_epoch)
        assert runtime._identity_store.credential(runtime.read_identity_document(), {}) == (rotated_secret, rotated_epoch)
        assert dict(os.environ) == before
    finally:
        runtime.close()


def test_identity_instances_for_one_document_share_the_transaction_lock(tmp_path):
    path = tmp_path / "node.json"
    first = IdentityStore(str(path), str(tmp_path / "secrets.json"))
    second = IdentityStore(str(path.parent / "child" / ".." / path.name), str(tmp_path / "secrets.json"))
    independent = IdentityStore(str(tmp_path / "other.json"), str(tmp_path / "other-secrets.json"))
    assert first._lock is second._lock
    assert independent._lock is not first._lock


@pytest.mark.skipif(sys.platform != "win32", reason="requires a real Windows sharing violation")
def test_identity_reader_and_other_owner_write_are_serialized_without_retry(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import qlh_core.adapters.settings as module

    path = tmp_path / "node.json"
    first = IdentityStore(str(path), str(tmp_path / "secrets.json"))
    second = IdentityStore(str(path), str(tmp_path / "secrets.json"))
    original = {"node": {"node_id": "stable"}, "retained": [1, 2]}
    first._write(original)
    reader_open = threading.Event()
    release_reader = threading.Event()
    reader_ident = []
    contentions = []
    replacements = []
    real_open = Path.open
    real_read = Path.read_text
    replace_file = module.os.replace

    def hold_reader(candidate, *args, **kwargs):
        if candidate == path and threading.get_ident() in reader_ident:
            # Keep a real CRT read handle alive, without FILE_SHARE_DELETE.
            with real_open(candidate, "r", encoding=kwargs.get("encoding", "utf-8")) as stream:
                reader_open.set()
                assert release_reader.wait(5), "writer did not coordinate with the reader"
                return stream.read()
        return real_read(candidate, *args, **kwargs)

    class ObservedLock:
        def __init__(self, lock):
            self.lock = lock
        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                contentions.append(True)
                # Release only after observing real lock contention, with no sleep.
                release_reader.set()
                self.lock.acquire()
            return self
        def __exit__(self, *args):
            self.lock.release()

    def observed_replace(source, target):
        if Path(target) == path:
            replacements.append(Path(source))
        return replace_file(source, target)

    def read():
        reader_ident.append(threading.get_ident())
        return first.read_document()

    second._lock = ObservedLock(second._lock)
    monkeypatch.setattr(Path, "read_text", hold_reader)
    monkeypatch.setattr(module.os, "replace", observed_replace)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(read)
        try:
            assert reader_open.wait(5), "reader never opened the existing document"
            second.persist_participation("local")
        finally:
            release_reader.set()
        assert pending.result(timeout=5) == original
    assert contentions == [True]
    assert len(replacements) == 1
    assert first.read_document() == {**original, "node": {"node_id": "stable", "desired_inference_mode": "local"}}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.skipif(sys.platform != "win32", reason="requires a real Windows sharing violation")
def test_identity_external_reader_denial_preserves_credentials_and_does_not_retry(tmp_path, monkeypatch):
    import ctypes
    from ctypes import wintypes
    import qlh_core.adapters.settings as module

    runtime = AppRuntime(settings(tmp_path))
    response = {"cluster": {"cluster_id": "stable", "cluster_secret": "committed-test-value",
                            "cluster_secret_epoch": 1},
                "node": {"node_id": runtime.config.node_id, "role": "client"}}
    runtime.accept_bootstrap(response)
    config, identity = runtime.config, runtime.identity
    path = Path(config.node_config_path)
    before = path.read_bytes()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
    assert handle != wintypes.HANDLE(-1).value, ctypes.get_last_error()
    replace_file = module.os.replace
    replacements = []

    def observed_replace(source, target):
        if Path(target) == path:
            replacements.append(Path(source))
        return replace_file(source, target)

    monkeypatch.setattr(module.os, "replace", observed_replace)
    try:
        with pytest.raises(PermissionError) as caught:
            runtime.accept_bootstrap({**response, "cluster": {**response["cluster"],
                "cluster_secret": "uncommitted-test-value", "cluster_secret_epoch": 2}})
        assert caught.value.winerror == 5
        assert len(replacements) == 1
        assert path.read_bytes() == before
        assert runtime.config is config and runtime.identity is identity
        assert runtime.cluster_credential() == ("committed-test-value", 1)
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        assert kernel.CloseHandle(handle)
        runtime.close()
