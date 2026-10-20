"""Explicit, instance-owned construction, readiness and resource lifetime."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

from qlh_core.contracts.runtime import RuntimeConfig


class LazyResource:
    """Compatibility reference; construction requires an actual attribute access."""
    def __init__(self, resolve: Callable[[], Any]) -> None:
        object.__setattr__(self, "_resolve", resolve)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_resolve")(), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_resolve")(), name, value)

    def __delattr__(self, name):
        delattr(object.__getattribute__(self, "_resolve")(), name)


class AppRuntime:
    """One process incarnation. Import/construction performs no resource IO.

    A shared HTTP app acquires references; the final release waits for startup
    before closing. Independent instances own independent services and readiness.
    """
    def __init__(self, config: RuntimeConfig | None = None, *, resources_factory=None,
                 host=None, config_observer=None) -> None:
        self.config = config
        self.identity = None
        self._identity_store = None
        self._config_observer = config_observer
        self._resources_factory = resources_factory
        self._host = host
        self._services = None
        self._lock = threading.RLock()
        self._readiness_lock = threading.RLock()
        self._readiness = self._new_readiness()
        self.startup_done = threading.Event()
        self.stop_requested = threading.Event()
        self._thread = None
        self._users = 0
        self._closing = False
        self._closed = False

    @staticmethod
    def _new_readiness():
        return {"process_ready": False, "ready": False, "status": "starting",
                "components": dict.fromkeys(("local_store", "scheduler", "device_profile"), False),
                "error": None, "started_at": None, "ready_at": None}

    def reset_readiness(self):
        with self._readiness_lock:
            self._readiness.clear()
            self._readiness.update(self._new_readiness(), started_at=time.time())

    def mark_process_ready(self):
        with self._readiness_lock:
            self._readiness["process_ready"] = True

    def mark_component(self, name, ready, *, error=None):
        with self._readiness_lock:
            if self.stop_requested.is_set() or name not in self._readiness["components"]:
                return
            self._readiness["components"][name] = bool(ready)
            complete = all(self._readiness["components"].values())
            self._readiness["ready"] = complete
            if complete:
                self._readiness.update(status="ready", error=None, ready_at=time.time())
            elif error:
                self._readiness.update(status="degraded", error=str(error))

    def mark_failure(self, error):
        with self._readiness_lock:
            self._readiness.update(ready=False, status="failed", error=f"{type(error).__name__}: {error}")

    def readiness(self):
        with self._readiness_lock:
            return {**self._readiness, "components": dict(self._readiness["components"])}

    def resolve_config(self):
        with self._lock:
            if self.config is None:
                from qlh_core.adapters.settings import SettingsStore
                self.config = SettingsStore().load()
            return self.config

    def configure_listener(self, port, *, environment=None):
        """Freeze the actual listener port before startup and service advertising."""
        from dataclasses import replace
        import json
        with self._lock:
            if self._thread is not None or self._services is not None:
                raise RuntimeError("listener configuration must precede startup")
            if self.config is None and environment is not None:
                from qlh_core.adapters.settings import SettingsStore
                self.config = SettingsStore(environment).load()
            config = self.resolve_config()
            resolved = int(port)
            if not 1 <= resolved <= 65535:
                raise ValueError("invalid API listener port")
            environment = config.environment_snapshot()
            environment["QLH_API_PORT"] = str(resolved)
            values = config.legacy_settings()
            values["API_PORT"] = resolved
            self.config = replace(config, api_port=resolved, environment=tuple(sorted(environment.items())),
                                  settings_json=json.dumps(values, sort_keys=True))
            return self.config

    @property
    def services(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("runtime is closed")
            if self._services is not None:
                return self._services
            if self._closing:
                raise RuntimeError("runtime is closing or closed")
            config = self.resolve_config()
            factory = self._resources_factory
            if factory is None:
                from qlh_core.bootstrap.services import RuntimeServices
                factory = RuntimeServices
            self._services = factory(config, host=self._host)
            bind = getattr(self._services, "bind_runtime", None)
            if bind is not None:
                bind(self)
            return self._services

    def open_identity(self):
        with self._lock:
            if self.identity is not None:
                return self.identity
            if self._closing or self._closed:
                raise RuntimeError("runtime is closing or closed")
            self.resolve_config()
            self.config, self.identity = self._get_identity_store().open(self.config)
            if self._services is not None:
                self._services.apply_config(self.config)
            return self.identity

    def _get_identity_store(self):
        if self._identity_store is None:
            from qlh_core.adapters.settings import IdentityStore
            config = self.resolve_config()
            self._identity_store = IdentityStore(config.node_config_path, config.secret_store_path,
                                                legacy_document=config.legacy_document)
        return self._identity_store

    def read_identity_document(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("runtime is closed")
            return self._get_identity_store().read_document()

    def _publish_identity_config(self, config):
        from dataclasses import replace
        self.config = config
        self.identity = replace(self.identity, node_id=config.node_id,
                                credential_epoch=int(config.environment_snapshot().get("QLH_CLUSTER_SECRET_EPOCH", 1)))
        if self._services is not None:
            self._services.apply_config(config)
        if self._config_observer is not None:
            self._config_observer(config)

    def accept_bootstrap(self, response):
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("runtime is closing or closed")
            self.open_identity()
            config = self._identity_store.accept_bootstrap(self.config, response)
            self._publish_identity_config(config)

    def persist_participation(self, mode):
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("runtime is closing or closed")
            self._get_identity_store().persist_participation(mode)

    def persist_preferred_master_endpoint(self, host, port):
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("runtime is closing or closed")
            self.open_identity()
            config, endpoint = self._identity_store.persist_preferred_master_endpoint(self.config, host, port)
            self._publish_identity_config(config)
            return endpoint

    def rotate_cluster_credential(self):
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("runtime is closing or closed")
            self.open_identity()
            config, credential = self._identity_store.rotate_cluster_credential(self.config)
            self._publish_identity_config(config)
            return credential

    def cluster_credential(self):
        """Return the startup credential, creating it explicitly for a master join."""
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("runtime is closing or closed")
            self.open_identity()
            secret = self.config.environment_snapshot().get("QLH_CLUSTER_SECRET", "")
            if secret:
                return secret, self.identity.credential_epoch
            config, credential = self._identity_store.ensure_cluster_credential(self.config)
            self._publish_identity_config(config)
            return credential

    def acquire(self, startup=None):
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("runtime is closing or closed")
            self._users += 1
            if self._users > 1:
                return
            self.stop_requested.clear()
            self.startup_done.clear()
            self.reset_readiness()
            self.mark_process_ready()
            def run():
                try:
                    if startup is not None:
                        startup()
                    else:
                        self.open_identity()
                        if not self.stop_requested.is_set():
                            services = self.services
                            with services.store.scope():
                                services.start(self)
                except BaseException as exc:
                    self.mark_failure(exc)
                finally:
                    self.startup_done.set()
            self._thread = threading.Thread(target=run, name="runtime-startup", daemon=True)
            self._thread.start()

    def begin_release(self):
        with self._lock:
            if self._users == 0:
                return False
            self._users -= 1
            if self._users:
                return False
            self._closing = True
            self.stop_requested.set()
            return True

    def wait_startup(self, timeout=30):
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise TimeoutError("runtime startup is still draining")

    def mark_cleanup_failure(self, error):
        with self._readiness_lock:
            self._readiness.update(ready=False, status="cleanup_failed", error=f"{type(error).__name__}: {error}")

    def finish_close(self):
        with self._lock:
            self._closing = False
            self._closed = True
        with self._readiness_lock:
            self._readiness.update(process_ready=False, ready=False, status="stopped")

    def close(self):
        # Explicit close also handles constructed-but-never-started resources.
        with self._lock:
            if self._closed:
                return
            self._closing = True
            self._users = 0
            self.stop_requested.set()
        try:
            self.wait_startup()
            if self._services is not None:
                self._services.close()
        except BaseException as exc:
            self.mark_cleanup_failure(exc)
            raise
        self.finish_close()
