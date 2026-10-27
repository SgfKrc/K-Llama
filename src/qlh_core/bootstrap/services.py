"""Composition adapter for retained services while domain owners migrate."""
from __future__ import annotations

import threading

class RuntimeServices:
    def __init__(self, config, *, host=None, callbacks=None):
        from model_host import ModelHost
        from scheduler import Scheduler
        from qlh_core.adapters.local_store import RuntimeLocalStore
        self.config = config
        self.store = RuntimeLocalStore(config)
        from qlh_core.control.catalog import CatalogService
        self.catalog = CatalogService(self.store.catalog_repository())
        from qlh_core.adapters.artifacts import ArtifactStore, CatalogLoadGuard
        self.artifacts = ArtifactStore()
        self.catalog_prepare = CatalogLoadGuard(self.catalog, self.artifacts)
        self._owns_host = host is None
        self._stage_host = None
        self._stage_host_lock = threading.RLock()
        self.host = host if host is not None else ModelHost(initial_quant=config.quant_type, environment=config.environment_snapshot())
        self.host.configure_catalog(self.catalog_prepare)
        self.scheduler = Scheduler(host=self.host, runtime_config=config)
        self.requests = self.scheduler._execution_app.requests
        self.runtime_controller = getattr(self.host, "runtime_controller", None)
        self.scheduler._catalog = self.catalog
        if callbacks is not None:
            self.scheduler.configure_callbacks(callbacks)
            self.host.configure_scheduler_callbacks(callbacks)
        self.scheduler._worker_prepare.configure_runtime(self.stage_host)
        self.participation = self.scheduler._participation
        self.journal = None
        self._closing = False
        self._closed = False

    def bind_runtime(self, runtime):
        self.participation.bind_persistence(runtime.persist_participation)
        self.scheduler._accept_bootstrap = runtime.accept_bootstrap
        self.scheduler._persist_preferred_endpoint = runtime.persist_preferred_master_endpoint
        self.scheduler._read_identity_document = runtime.read_identity_document

    def stage_host(self):
        with self._stage_host_lock:
            if self._closed or self._closing:
                raise RuntimeError("runtime services are closing or closed")
            if self._stage_host is None:
                from inference_service.engine_host import EngineHost
                self._stage_host = EngineHost(environment=self.config.environment_snapshot())
                self._stage_host._host.configure_catalog(self.catalog_prepare)
            return self._stage_host

    def apply_config(self, config):
        self.config = config
        self.host.configure_runtime_environment(config.environment_snapshot())
        self.scheduler._runtime_config = config
        from scheduler import _bootstrap_api_port
        self.scheduler._runtime_settings.update({
            **config.legacy_settings(), "NODE_ID": config.node_id, "NODE_ROLE": config.node_role,
            "RUN_MODE": config.run_mode, "SERVER_IP": config.server_ip, "SERVER_PORT": config.server_port,
            "_configured_node_id": lambda: config.node_id,
            "_bootstrap_api_port": lambda: _bootstrap_api_port(environment=config.environment_snapshot()),
        })

    def start(self, runtime):
        self.store.initialize()
        runtime.mark_component("local_store", True)
        if runtime.stop_requested.is_set():
            return
        self.scheduler.start(host=self.config.server_ip, port=self.config.server_port)
        runtime.mark_component("scheduler", True)
        from device_profiler import get_profile
        profile = get_profile().to_dict()
        if runtime.stop_requested.is_set():
            return
        self.scheduler.update_local_device_profile(profile)
        runtime.mark_component("device_profile", True)

    def close(self):
        with self._stage_host_lock:
            if self._closed:
                return
            self._closing = True
        try:
            self.scheduler.stop()
        except Exception as exc:
            # A draining request still owns native resources. Preserve the
            # dependency graph for a later close instead of blocking in unload.
            raise ExceptionGroup("runtime cleanup failed", [exc]) from exc
        actions = []
        if self._stage_host is not None:
            actions.append(self._stage_host.close)
        if self.journal is not None:
            actions.append(self.journal.close)
        if self._owns_host:
            actions.append(self.host.unload_model)
        errors = []
        for action in actions:
            try:
                action()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("runtime cleanup failed", errors)
        self._closed = True
