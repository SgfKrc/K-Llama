"""Bind scheduler resources to the quorum controller's published runtime role."""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class SchedulerAutoRoleMixin:
    def _on_auto_role_decision(self, decision):
        if not self._running:
            return
        if decision.runtime_role != self._auto_role_bound_role:
            self._auto_role_binding_ready = False
        self._auto_role_binding_event.set()
        with self._auto_role_binding_lock:
            thread = self._auto_role_binding_thread
            if thread is None or not thread.is_alive():
                self._auto_role_binding_thread = threading.Thread(
                    target=self._auto_role_binding_loop, name="auto-role-binding", daemon=True)
                self._auto_role_binding_thread.start()

    def _auto_role_binding_loop(self):
        while self._running and not self._startup_cancel_event.is_set():
            if not self._auto_role_binding_event.wait(.2):
                continue
            self._auto_role_binding_event.clear()
            try:
                self._reconcile_auto_role_resources()
            except Exception:
                self._auto_role_binding_ready = False
                self._auto_role_controller.on_quorum_loss("auto_role_binding_failed")
                self._auto_role_binding_event.set()
                logger.exception("auto-role resource binding failed closed")
                self._startup_cancel_event.wait(.2)

    def _stop_auto_role_binding(self):
        self._auto_role_binding_event.set()
        thread = self._auto_role_binding_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(30)
            if thread.is_alive():
                raise TimeoutError("auto-role resource binding is still draining")
        self._auto_role_binding_thread = None
        self._auto_role_binding_ready = False

    def _reconcile_auto_role_resources(self):
        controller = self._auto_role_controller
        if not self._running or controller is None or controller.state == "stopped":
            return
        role = controller.runtime_role
        if role != self._auto_role_bound_role or not self._auto_role_binding_ready:
            self._auto_role_binding_ready = False
            self._model_supply.cancel_active()
            self._model_supply_worker.cancel()
            requests = self._execution_app.requests
            snapshot = requests.queue_snapshot()
            for item in snapshot["tasks"]:
                requests.cancel(item["request_id"])
            if snapshot["current_task_id"]:
                requests.cancel(snapshot["current_task_id"])
            previous = self._tcp_client
            if previous is not None:
                previous.on_disconnect = None
                previous.disconnect()
            self._on_master_connection_lost(previous, notify_auto_role=False)
            self._worker_prepare.retire_and_join("authority_changed", timeout=30)
            if self.get_distributed_inference_enabled():
                with self._host.full_chat_execution_lock:
                    candidate = self._host.pipeline_candidate_status()
                    if candidate["active"]:
                        self._host.abort_pipeline_candidate(candidate["transaction_id"])
                    if self._host.is_pipeline_prepared:
                        self._host.unload_model()
            with self._task_worker_refresh_lock:
                if self._tcp_client is previous:
                    self._tcp_client = None
            if self._tcp_server is not None:
                self._tcp_server.disconnect_all_clients("authority_changed")
            with self._nodes_lock:
                self.nodes.clear()
            self.init_nodes()
            if role == "master":
                self._load_pipeline_recovery_state()
                self._restore_pipeline_model_for_recovery()
                self._start_deferred_network_identity()
            else:
                self._start_client_health_monitor()
            if not self._running or controller.runtime_role != role:
                self._auto_role_binding_event.set()
                return
            self._auto_role_bound_role = role
            self._auto_role_binding_ready = True
            logger.info("event=auto_role_bound role=%s state=%s", role, controller.state)
        if role != "client" or controller.state != "follower":
            return
        certificate = controller.certificate
        if certificate is None:
            return
        runtime = getattr(self, "_auto_role_runtime", None)
        client = self._tcp_client
        # Only a configured voter's signed active proof chooses an auto endpoint.
        endpoint = runtime.leader_endpoint(certificate.leader_id) if runtime is not None else None
        if endpoint is None or not self._running:
            return
        host, port = endpoint
        if (client is not None and getattr(client, "is_registered", False)
                and client.server_host == host and client.server_port == port):
            return
        self.connect_to_master(host, port, _automatic=True)
