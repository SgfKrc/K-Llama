"""Master's remote native-load barrier; inventory never grants execution."""
from __future__ import annotations
from copy import deepcopy
import logging
from worker_prepare_contract import validate_identity, validate_ack, stage_identity, WorkerPrepareError
from worker_assignment_state import new_assignment_id, PHASE_READY

logger = logging.getLogger("scheduler")


class WorkerPrepareCoordinatorMixin:
    def cancel_worker_prepare_operation(self, operation_id):
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if (not isinstance(transaction, dict)
                    or transaction.get("phase") not in {"preparing", "committing_local", "committing", "committing_global"}
                    or not any(value.get("worker_prepare_identity", {}).get("operation_id") == operation_id
                               for value in transaction.get("expected", {}).values())):
                return False
            transaction["cancel_requested"] = True
            config_id = transaction["config_id"]
        self._abort_pipeline_load_transaction(config_id, "pipeline_prepare_cancelled", "model preparation cancelled by user")
        return True

    def _build_worker_prepare_assignment(self, assignment, *, config_id, generation, model_info, operation_id=""):
        result = deepcopy(assignment)
        node = result["node_id"]
        artifact = result.get("layer_artifact")
        if not isinstance(artifact, dict):
            raise WorkerPrepareError("prepare_artifact_identity_missing")
        model = result.get("stage_model_identity")
        if not isinstance(model, dict):
            value = self._route_a_stage_model_identity(node, result)
            if value is None:
                raise WorkerPrepareError("prepare_stage_model_identity_missing")
            model = {key: getattr(value, key) for key in ("model_id", "engine", "format", "revision", "sha256")}
        manifest_sha = ""
        if operation_id:
            operation = self._model_supply.get(operation_id)
            receipt = operation.get("nodes", {}).get(node, {})
            if (operation.get("status") != "available" or receipt.get("status") != "available"
                    or receipt.get("artifact_sha256") != artifact.get("artifact_sha256")
                    or receipt.get("layer_range") != [result["start_layer"], result["end_layer"]]):
                raise WorkerPrepareError("prepare_supply_identity_changed")
            manifest_sha = receipt.get("manifest_sha256", "")
            if not manifest_sha:
                raise WorkerPrepareError("prepare_supply_manifest_missing")
        identity = validate_identity({
            "operation_id": operation_id or "prepare_" + config_id,
            "config_id": config_id, "generation": generation,
            "assignment_id": new_assignment_id(), "node_id": node,
            "model_id": model_info["model_id"], "model_sha256": model_info["model_sha256"],
            "stage_model_identity": model, "artifact_sha256": artifact["artifact_sha256"],
            "manifest_sha256": manifest_sha, "source_model_id": artifact.get("source_model_id", ""),
            "source_model_sha256": artifact.get("source_model_sha256", ""),
            "tokenizer_sha256": artifact.get("tokenizer_sha256", ""),
            "layer_range": [result["start_layer"], result["end_layer"]],
            "total_layers": model_info["total_layers"], "hidden_size": artifact.get("hidden_size", 0),
            "segment_mode": artifact.get("segment_mode", ""),
        })
        result.update(config_id=config_id, generation=generation, assignment_id=identity["assignment_id"],
                      phase="prepare", worker_prepare_identity=identity, prepare_identity=stage_identity(identity),
                      plan_id=assignment.get("plan_id", ""))
        return result

    def _send_pipeline_worker_config(self, node_id, config):
        """One send path for first delivery, retries, commit and terminal cleanup."""
        identity = config.get("worker_prepare_identity")
        if isinstance(identity, dict) and not config.get("cleanup_action") and not config.get("release"):
            with self._layer_config_lock:
                transaction = self._pipeline_load_transaction
                if (not transaction or transaction.get("config_id") != identity["config_id"]
                        or transaction.get("phase") not in {"preparing", "committing"}
                        or transaction.get("expected", {}).get(node_id, {}).get("phase") != config.get("phase")):
                    return False
        if not isinstance(identity, dict):
            return self._tcp_server.send_layer_config(node_id, config)
        from transport_port import MessageType
        from worker_prepare_contract import validate_command
        action = config.get("cleanup_action") or ("release" if config.get("release") else config.get("phase", "prepare"))
        command = {"action": action, "identity": deepcopy(identity)}
        if action == "prepare":
            command.update(context_size=4096, middle_channel="keep_head_layer_out")
        if action in {"abort", "finalize"} or (action == "release" and config.get("cleanup_action")):
            command["cleanup_id"] = config["config_id"]
        return self._tcp_server.send_to_client(node_id, validate_command(command), MessageType.MODEL_PREPARE)

    def _handle_worker_prepare_ack(self, node_id, payload):
        payload = {key: value for key, value in payload.items() if key != "control_certificate"}
        try:
            ack = validate_ack(payload)
        except WorkerPrepareError as exc:
            logger.warning("event=worker_prepare_ack_rejected node=%s reason=%s", node_id, exc.reason_code)
            return
        identity = ack["identity"]
        if identity["node_id"] != node_id:
            return
        phase = ack["phase"]
        if phase in {"abort", "finalize"} or (phase == "release" and "cleanup_id" in ack):
            cleanup_id = ack["cleanup_id"]
            with self._layer_config_lock:
                record = self._pipeline_model_cleanup_ledger.get(cleanup_id)
                if (not record or record["node_id"] != node_id or record["action"] != phase
                        or record.get("payload", {}).get("worker_prepare_identity") != identity
                        or ack["status"] == "error"):
                    return
                # The original candidate generation stays in the identity while
                # the ledger's generation fences the later cleanup command.
                data = {"node_id": node_id, "config_id": cleanup_id,
                        "generation": record["generation"], "assignment_id": record["assignment_id"],
                        "cleanup_action": phase, "candidate_config_id": identity["config_id"],
                        "status": ack["status"], "release": phase in {"abort", "release"}}
                if phase in {"abort", "finalize"}:
                    data["aborted_config_id" if phase == "abort" else "finalized_config_id"] = identity["config_id"]
            cleaned = self._ack_pipeline_model_cleanup(node_id, data, native_ack=ack)
            if cleaned and phase == "release":
                with self._layer_config_lock:
                    expected = self._worker_assignments.expected(node_id)
                    if expected.get("worker_prepare_identity") == identity:
                        self._worker_assignments.release(node_id, reason_code="worker_runtime_released")
            return
        advance = ""
        abort = None
        retire = False
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if not isinstance(transaction, dict):
                return
            expected = transaction.get("expected", {}).get(node_id, {})
            if (expected.get("worker_prepare_identity") != identity
                    or transaction.get("config_id") != identity["config_id"]
                    or transaction.get("connection_generations", {}).get(node_id) != self._assignment_connection_generation(node_id)):
                logger.warning("event=worker_prepare_ack_stale node=%s config=%s phase=%s", node_id, identity["config_id"], phase)
                return
            if node_id in self._pipeline_worker_opt_out:
                return
            if ack["status"] == "error":
                if phase == expected.get("phase") and transaction.get("phase") in {"preparing", "committing", "committing_global", "ready"}:
                    transaction.setdefault("acks", {})[node_id] = deepcopy(ack)
                    transaction.setdefault("ready_nodes", set()).discard(node_id)
                    retire = transaction.get("phase") == "ready"
                    if not retire:
                        abort = (identity["config_id"], "pipeline_worker_" + phase + "_failed", ack["error"])
                    else:
                        self._worker_assignments.invalidate(node_id, reason_code="worker_native_runtime_lost")
            elif (phase == "prepare" and transaction.get("phase") == "preparing") or (phase == "commit" and transaction.get("phase") == "committing"):
                if expected.get("phase") != phase:
                    return
                data = deepcopy(ack)
                data.update(node_id=node_id, config_id=identity["config_id"], assignment_id=identity["assignment_id"],
                            generation=identity["generation"], layer_range=identity["layer_range"],
                            worker_prepare_identity=identity)
                transaction.setdefault("acks", {})[node_id] = data
                transaction.get("retry_state", {}).pop(node_id, None)
                key = "prepared_nodes" if phase == "prepare" else "ready_nodes"
                transaction.setdefault(key, set()).add(node_id)
                logger.info("event=worker_native_loaded_ready node=%s operation=%s config=%s generation=%s assignment=%s phase=%s range=%s artifact=%s",
                            node_id, identity["operation_id"], identity["config_id"], identity["generation"],
                            identity["assignment_id"], phase, identity["layer_range"], identity["artifact_sha256"])
                if transaction[key] == set(transaction.get("worker_ids", set())):
                    advance = phase
                    if phase == "commit":
                        transaction["phase"] = "committing_global"
                self._persist_pipeline_lifecycle_locked()
        if retire:
            self._invalidate_pipeline_load_transaction("worker_native_runtime_lost")
        elif abort:
            self._abort_pipeline_load_transaction(*abort)
        elif advance == "prepare":
            self._commit_pipeline_load_transaction(identity["config_id"])
        elif advance == "commit":
            self._complete_pipeline_load_transaction(identity["config_id"])

    def _worker_prepare_assignment_loaded(self, node_id, assignment):
        identity = assignment.get("worker_prepare_identity")
        if not isinstance(identity, dict):
            return False, "worker_native_prepare_required"
        with self._layer_config_lock:
            plan = self._active_pipeline_capacity_plan
            if (not isinstance(plan, dict) or plan.get("transaction_phase") != "ready"
                    or plan.get("config_id") != identity["config_id"]
                    or plan.get("generation") != identity["generation"]
                    or not any(item.get("node_id") == node_id
                               and item.get("worker_prepare_identity") == identity
                               for item in plan.get("assignments", []))):
                return False, "worker_native_plan_not_published"
            state = self._worker_assignments.state(node_id)
            if (state is None or state.phase != PHASE_READY
                    or state.expected_config.get("worker_prepare_identity") != identity
                    or state.connection_generation != self._assignment_connection_generation(node_id)):
                return False, "worker_native_ready_identity_mismatch"
            ack = state.acknowledgement
            if ack.get("phase") != "commit" or ack.get("loaded") is not True or ack.get("worker_prepare_identity") != identity:
                return False, "worker_native_not_loaded"
        return True, "ready"
