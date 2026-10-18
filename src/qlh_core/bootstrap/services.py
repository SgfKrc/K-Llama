"""Composition adapter for retained services while domain owners migrate."""
from __future__ import annotations

import threading

from qlh_core.contracts.runtime import ParticipationSnapshot


class SchedulerParticipation:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._revision = 0

    def snapshot(self):
        enabled = self.scheduler.get_distributed_inference_enabled()
        return ParticipationSnapshot("distributed" if enabled else "local", self._revision, bool(enabled))

    def set_mode(self, mode):
        # Unload/retirement and API command migration remain R3/R4 responsibilities.
        raise RuntimeError("participation mode changes require the existing retirement command")


class RuntimeServices:
    def __init__(self, config, *, host=None, callbacks=None):
        from cluster_fence import ControlFence
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
        self.fence = ControlFence.from_environment(config.environment_snapshot())
        self.scheduler = Scheduler(host=self.host, runtime_config=config)
        self.scheduler._catalog = self.catalog
        self.scheduler.set_control_fence(self.fence)
        if callbacks is not None:
            self.scheduler.configure_callbacks(callbacks)
            self.host.configure_scheduler_callbacks(callbacks)
        self.scheduler._worker_prepare.configure_runtime(self.stage_host)
        self.participation = SchedulerParticipation(self.scheduler)
        self.auto_role = None
        self.journal = None
        self._closing = False
        self._closed = False

    def bind_runtime(self, runtime):
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
        self.scheduler._runtime_settings = {
            **config.legacy_settings(), "NODE_ID": config.node_id, "NODE_ROLE": config.node_role,
            "RUN_MODE": config.run_mode, "SERVER_IP": config.server_ip, "SERVER_PORT": config.server_port,
            "_configured_node_id": lambda: config.node_id,
            "_bootstrap_api_port": lambda: _bootstrap_api_port(environment=config.environment_snapshot()),
        }

    def configure_authority(self):
        if self.config.node_role == "auto" and self.auto_role is None:
            from cluster_auto_role_runtime import install_auto_role_controller
            self.auto_role = install_auto_role_controller(
                self.scheduler, self.fence, environment=self.config.environment_snapshot())
            self.fence = self.scheduler._control_fence
        return self.auto_role

    def start(self, runtime):
        self.store.initialize()
        runtime.mark_component("local_store", True)
        if runtime.stop_requested.is_set():
            return
        self.configure_authority()
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
        actions = [self.scheduler.stop]
        if self.auto_role is not None:
            actions.append(self.auto_role.close)
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
