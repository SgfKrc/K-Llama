"""Authoritative model operation, supply, deployment and cleanup transactions.

Execution adapters submit observations. Only a committed snapshot authorizes a
wire command or request admission. Native work and transport stay outside this
owner and its repository transaction.
"""
from __future__ import annotations

from copy import deepcopy
from qlh_core.contracts.deployment import (
    ModelControlError, OPERATION_TERMINAL, DEPLOYMENT_TERMINAL, DEPLOYMENT_TRANSITIONS, command_key, digest,
    frozen_assignments, require_runtime_receipt,
)


class DeploymentService:
    def __init__(self, repository, *, boot_id, clock):
        self.repository = repository
        self.boot_id = boot_id
        self.clock = clock
        self._started = False
        self._operation_fences = {}

    def snapshot(self):
        state = self.repository.read()
        if state["boot_id"] == self.boot_id:
            for operation_id, value in self._operation_fences.items():
                old = state["operations"].get(operation_id)
                if old and old["status"] not in OPERATION_TERMINAL:
                    state["operations"][operation_id] = {**deepcopy(value), "durable": False,
                                                         "commit_scope": "volatile_failure"}
        return state

    def fence_operation_failure(self, value):
        # A failed disk cannot accept a durable terminal. Keep the admission
        # fence in this owner and label its projection until an explicit retry
        # can commit it; queries never perform the retry themselves.
        if value.get("status") not in {"failed", "cancelled", "superseded"}:
            raise ModelControlError("operation_failure_fence_invalid")
        self._operation_fences[value["operation_id"]] = deepcopy(value)

    def start(self):
        if self._started:
            return self.snapshot()
        existing = self.snapshot()
        if existing["revision"] and not self.repository.recovery_status().get("verified"):
            raise ModelControlError("model_control_journal_unverified")
        def recover(state):
            if state["boot_id"] == self.boot_id:
                return
            state["boot_id"] = self.boot_id
            state["active_deployment"] = ""
            for operation in state["operations"].values():
                if operation["status"] not in OPERATION_TERMINAL:
                    operation.update(status="failed", reason_code="INTENT_MASTER_RESTARTED",
                                     reason="The master restarted during model selection; retry the selection")
            for supply in state["supplies"].values():
                if supply["status"] not in {"available", "failed", "cancelled"}:
                    supply.update(status="failed", reason_code="SUPPLY_MASTER_RESTARTED")
            for deployment in state["deployments"].values():
                if deployment["status"] not in {"aborted", "retired"}:
                    deployment.update(status="invalidated" if deployment.get("published") else "aborting",
                                      validity="unknown", reason_code="deployment_process_restarted")
            # Old commands remain auditable, but a new boot cannot replay loading.
            for command in state["outbox"].values():
                if command["action"] in {"prepare", "commit"}:
                    command["state"] = "superseded"
        result = self.repository.transact(recover)
        self._started = True
        return result

    def save_supply(self, value):
        value = deepcopy(value)
        def save(state):
            self._require_boot(state)
            old = state["supplies"].get(value["operation_id"])
            key = value.get("idempotency_key")
            if old is None and key:
                for other in state["supplies"].values():
                    if other.get("idempotency_key") == key:
                        code = ("operation_idempotency_reused" if other["intent"] == value["intent"]
                                else "operation_idempotency_conflict")
                        raise ModelControlError(code)
            if old and old["status"] in {"failed", "cancelled"} and old["status"] != value["status"]:
                raise ModelControlError("supply_terminal_conflict")
            state["supplies"][value["operation_id"]] = value
            if old is None and value.get("wait_for_nodes"):
                previous = state["operations"].get(state["selected_operation"])
                if previous and previous["status"] not in OPERATION_TERMINAL:
                    previous.update(status="superseded", phase="superseded", reason_code="INTENT_SUPERSEDED")
                state["selected_operation"] = value["operation_id"]
                state["operations"][value["operation_id"]] = {
                    "operation_id": value["operation_id"], "status": "queued", "phase": "accepted",
                    "intent": deepcopy(value["intent"]), "idempotency_key": value.get("idempotency_key", ""),
                    "created_at": value["created_at"], "deadline": value["created_at"] + value["intent"].get("timeout_seconds", 3600),
                    "reason_code": "", "reason": "",
                }
            operation = state["operations"].get(value["operation_id"])
            if operation and operation["status"] not in OPERATION_TERMINAL | {"preparing"}:
                operation["phase"] = {
                    "queued": "accepted", "inspecting": "resolving", "inventory": "resolving",
                    "waiting_nodes": "waiting_capacity", "waiting_capacity": "waiting_capacity",
                    "planning": "planning", "available": "supplying", "failed": "failed", "cancelled": "cancelled",
                }.get(value["status"], "supplying")
        return self.repository.transact(save)

    def save_operation(self, value, *, select=False, expected_selected=None, idempotency_key=""):
        value = deepcopy(value)
        def save(state):
            self._require_boot(state)
            operation_id = value["operation_id"]
            old = state["operations"].get(operation_id)
            if old and old["status"] in OPERATION_TERMINAL:
                if old["status"] != value["status"]:
                    raise ModelControlError("operation_terminal_conflict")
                return
            if select:
                if expected_selected is not None and state["selected_operation"] != expected_selected:
                    raise ModelControlError("operation_selection_conflict")
                if idempotency_key:
                    for other in state["operations"].values():
                        if other.get("idempotency_key") == idempotency_key and other["operation_id"] != operation_id:
                            raise ModelControlError("operation_idempotency_conflict")
                    value["idempotency_key"] = idempotency_key
                previous = state["operations"].get(state["selected_operation"])
                if previous and previous["status"] not in OPERATION_TERMINAL and previous["operation_id"] != operation_id:
                    previous.update(status="superseded", reason_code="INTENT_SUPERSEDED")
                state["selected_operation"] = operation_id
            for key in ("intent", "created_at", "deadline", "idempotency_key"):
                if old is not None and key not in value:
                    value[key] = deepcopy(old.get(key))
            value["updated_at"] = self.clock()
            value["phase"] = "accepted" if value["status"] == "queued" else value["status"]
            state["operations"][operation_id] = value
        result = self.repository.transact(save)
        self._operation_fences.pop(value["operation_id"], None)
        return result

    def _require_boot(self, state):
        if state["boot_id"] != self.boot_id:
            raise ModelControlError("deployment_boot_superseded")

    @staticmethod
    def _wire(node, config):
        identity = config.get("worker_prepare_identity")
        if not isinstance(identity, dict):
            return deepcopy(config)
        action = config.get("cleanup_action") or ("release" if config.get("release") else config.get("phase", "prepare"))
        value = {"action": action, "identity": deepcopy(identity)}
        if action == "prepare":
            value.update(context_size=config.get("context_size", 4096), middle_channel="keep_head_layer_out")
            if "load_spec" in config:
                value["load_spec"] = deepcopy(config["load_spec"])
        if action in {"abort", "finalize"} or (action == "release" and config.get("cleanup_action")):
            value["cleanup_id"] = config["config_id"]
        return value

    def save_execution(self, lifecycle, *, expected, acknowledgements, authority, connections, recovery_input=False):
        lifecycle, expected = deepcopy(lifecycle), deepcopy(expected)
        def save(state):
            if state["boot_id"] != self.boot_id:
                raise ModelControlError("deployment_boot_superseded")
            config_id = lifecycle.get("config_id", "")
            phase = lifecycle.get("phase", "")
            if recovery_input:
                phase = "invalidated" if lifecycle.get("published") else "aborted"
            statuses = {"preparing": "preparing", "committing_local": "prepared", "committing": "committing",
                        "committing_global": "committing", "ready": "active", "aborted": "aborting",
                        "invalidated": "invalidated", "rejected": "aborting"}
            deployment = state["deployments"].get(config_id)
            if config_id:
                if deployment and phase in {"committing_local", "committing", "committing_global", "ready"}:
                    if (deployment["authority"] != authority or deployment["connections"] != connections
                            or deployment.get("input_revisions", {}) != lifecycle.get("input_revisions", {})):
                        raise ModelControlError("deployment_input_changed")
                    if deployment["status"] in {"aborting", "aborted", "invalidated", "retiring", "retired"}:
                        raise ModelControlError("deployment_retired")
                if deployment is None:
                    deployment = {"deployment_id": config_id, "boot_id": self.boot_id,
                                  "authority": deepcopy(authority), "connections": deepcopy(connections),
                                  "created_at": self.clock(), "assignments": deepcopy(expected),
                                  "input_revisions": deepcopy(lifecycle.get("input_revisions", {}))}
                    state["deployments"][config_id] = deployment
                native = any(item.get("worker_prepare_identity", {}).get("load_spec_digest")
                             for item in expected.values())
                target = statuses.get(phase, "candidate")
                previous = deployment.get("status", "candidate")
                if previous in DEPLOYMENT_TERMINAL and target != previous:
                    # settle_cleanup commits before the executor saves its ledger.
                    # Reobserving that same terminal transaction must keep the
                    # settled status; it cannot reopen a deployment or generation.
                    prior = state.get("lifecycle") or {}
                    terminal_observation = (previous, target) in {
                        ("retired", "invalidated"), ("aborted", "aborting")}
                    # Explicit model replacement retires a validated recovery
                    # target by clearing its historical publication flag. That
                    # observation still belongs to the settled retired deployment.
                    if (recovery_input and previous == "retired" and target == "aborting"
                            and lifecycle.get("phase") == "invalidated"
                            and lifecycle.get("published") is False
                            and all(prior.get(key) == lifecycle.get(key) for key in (
                                "model_id", "model_type", "model_sha256", "total_layers",
                                "tokenizer_sha256", "quant_type", "plan_id", "assignments",
                                "local_assignment"))):
                        terminal_observation = True
                    if (not terminal_observation
                            or any(prior.get(key) != lifecycle.get(key)
                                   for key in ("config_id", "generation", "phase"))):
                        raise ModelControlError("deployment_transition_invalid")
                    target = previous
                if native and not recovery_input:
                    if frozen_assignments(deployment["assignments"]) != frozen_assignments(expected):
                        raise ModelControlError("deployment_assignment_changed")
                    if target != previous and target not in DEPLOYMENT_TRANSITIONS[previous]:
                        raise ModelControlError("deployment_transition_invalid")
                if native and phase in {"committing_local", "committing_global", "ready"} and not recovery_input:
                    required_phase = "prepare" if phase == "committing_local" else "commit"
                    for node, item in expected.items():
                        identity = item.get("worker_prepare_identity")
                        if not identity:
                            continue
                        ack = acknowledgements.get(node, {})
                        if (ack.get("phase") != required_phase or ack.get("loaded") is not True
                                or ack.get("status") != ("prepared" if required_phase == "prepare" else "ready")
                                or ack.get("identity", ack.get("worker_prepare_identity")) != identity):
                            raise ModelControlError("deployment_commit_receipts_missing")
                        require_runtime_receipt(item, ack.get("runtime_receipt"))
                local = lifecycle.get("local_assignment")
                if deployment.get("local_assignment") != local and "local_assignment" in deployment:
                    raise ModelControlError("deployment_assignment_changed")
                if local is not None:
                    deployment["local_assignment"] = deepcopy(local)
                    if phase in {"committing", "committing_global", "ready"} and not recovery_input:
                        require_runtime_receipt(local, lifecycle.get("local_receipt"))
                if phase == "ready":
                    if not expected or any(
                            (acknowledgements.get(node, {}).get("loaded") is not True
                             or acknowledgements[node].get("phase") != "commit")
                            if item.get("worker_prepare_identity") else acknowledgements.get(node, {}).get("status") != "ready"
                            for node, item in expected.items()):
                        raise ModelControlError("deployment_commit_receipts_missing")
                    for node, item in expected.items():
                        identity = item.get("worker_prepare_identity")
                        if identity:
                            ack = acknowledgements[node]
                            if (ack.get("identity", ack.get("worker_prepare_identity")) != identity
                                    or ack.get("status") != "ready"):
                                raise ModelControlError("deployment_commit_receipt_identity_mismatch")
                            if identity.get("load_spec_digest"):
                                require_runtime_receipt(item, ack.get("runtime_receipt"))
                    if local is not None:
                        require_runtime_receipt(local, lifecycle.get("local_receipt"))
                    active = state["active_deployment"]
                    if active and active != config_id:
                        raise ModelControlError("deployment_retire_required")
                    operations = {item.get("worker_prepare_identity", {}).get("operation_id") for item in expected.values()}
                    for operation_id in operations - {None, ""}:
                        operation = state["operations"].get(operation_id)
                        if operation is not None:
                            if operation_id in self._operation_fences:
                                raise ModelControlError("deployment_operation_persistence_failed")
                            if operation["status"] in {"cancelled", "failed", "superseded"} or state["selected_operation"] != operation_id:
                                raise ModelControlError("deployment_operation_superseded")
                            if operation.get("deadline", float("inf")) < self.clock():
                                raise ModelControlError("deployment_operation_deadline_exceeded")
                            operation.update(status="ready", phase="ready", config_id=config_id,
                                             generation=lifecycle.get("generation", 0), reason_code="", reason="")
                    state["active_deployment"] = config_id
                elif phase in {"aborted", "invalidated", "rejected"} and state["active_deployment"] == config_id:
                    state["active_deployment"] = ""
                deployment.update(status=target, published=bool(lifecycle.get("published")),
                                  validity="current" if phase == "ready" else "pending", observed_at=self.clock(),
                                  reason_code=lifecycle.get("reason_code", ""), receipts=deepcopy(acknowledgements))
            # The executor snapshot is a recovery input, never an independent DB writer.
            state["lifecycle"] = lifecycle
            commands = [(node, config) for node, config in expected.items()
                        if phase in {"preparing", "committing"}]
            debts = {record["cleanup_id"]: deepcopy(record) for record in lifecycle.get("cleanup_ledger", [])}
            # Removing an obligation requires a matching explicit settle_cleanup command.
            for key, debt in debts.items():
                if key in state["cleanup"] and state["cleanup"][key] != debt:
                    raise ModelControlError("cleanup_identity_conflict")
            state["cleanup"].update(debts)
            commands += [(record["node_id"], record["payload"]) for record in debts.values()]
            for node, config in commands:
                payload = self._wire(node, config)
                key = command_key(node, payload)
                if key not in state["outbox"]:
                    state["outbox"][key] = {"command_id": key, "node_id": node, "payload": payload,
                        "action": payload.get("action", config.get("phase", "cleanup")), "boot_id": self.boot_id,
                        "authority": deepcopy(authority), "connection_generation": connections.get(node, 0),
                        "state": "pending", "attempts": 0, "created_at": self.clock()}
            # ACK completion is durable; retransmission remains safe if it raced.
            for command in state["outbox"].values():
                ack = acknowledgements.get(command["node_id"], {})
                if (command["action"] in {"prepare", "commit"} and ack.get("phase") == command["action"]
                        and ack.get("identity", ack.get("worker_prepare_identity")) == command["payload"].get("identity")):
                    command["state"] = "acknowledged"
        return self.repository.transact(save)

    def authorize_command(self, node, payload, *, authority, connection_generation):
        state = self.snapshot()
        command = state["outbox"].get(command_key(node, payload))
        if not command or command["state"] == "superseded":
            return False
        if command["action"] in {"prepare", "commit"}:
            deployment = state["deployments"].get(payload.get("identity", {}).get("config_id"))
            return (state["boot_id"] == self.boot_id == command["boot_id"]
                    and deployment is not None and deployment["status"] == ("preparing" if command["action"] == "prepare" else "committing")
                    and command["authority"] == authority and command["connection_generation"] == connection_generation)
        # Cleanup may cross a reconnect/authority renewal; its exact old assignment
        # stays fixed and a reply is admitted only on the current connection.
        return True

    def settle_cleanup(self, cleanup_id, *, evidence):
        def settle(state):
            debt = state["cleanup"].get(cleanup_id)
            if debt is None:
                return
            identity = debt.get("payload", {}).get("worker_prepare_identity")
            if evidence.get("kind") == "process_ended":
                if (evidence.get("boot_id") != debt.get("process_boot_id") or not evidence.get("boot_id")
                        or any(evidence.get(key) != debt.get(key) or not evidence.get(key)
                               for key in ("process_id", "process_start_time"))):
                    raise ModelControlError("cleanup_process_evidence_invalid")
            elif (evidence.get("kind") != "receipt" or evidence.get("identity") != identity
                  or evidence.get("action") != debt["action"]):
                raise ModelControlError("cleanup_receipt_mismatch")
            state["cleanup"].pop(cleanup_id)
            for command in state["outbox"].values():
                if command["payload"].get("cleanup_id") == cleanup_id:
                    command["state"] = "acknowledged"
            lifecycle = state.get("lifecycle")
            if lifecycle:
                lifecycle["cleanup_ledger"] = [r for r in lifecycle.get("cleanup_ledger", []) if r["cleanup_id"] != cleanup_id]
            deployment_id = debt.get("candidate_config_id")
            deployment = state["deployments"].get(deployment_id)
            if deployment and not any(d.get("candidate_config_id") == deployment_id for d in state["cleanup"].values()):
                if deployment["status"] == "aborting":
                    deployment["status"] = "aborted"
                elif deployment["status"] in {"invalidated", "retiring"}:
                    deployment["status"] = "retired"
        return self.repository.transact(settle)

    def runnable(self, config_id, *, authority, connections, input_revisions=None):
        state = self.snapshot()
        value = state["deployments"].get(config_id)
        return bool(state["boot_id"] == self.boot_id and state["active_deployment"] == config_id
                    and value and value["boot_id"] == self.boot_id and value["status"] == "active"
                    and value["authority"] == authority and value["connections"] == connections
                    and (input_revisions is None or value.get("input_revisions", {}) == input_revisions))


    def retain_publication(self, location, *, evidence):
        key = digest({"location": location, "evidence": evidence})
        def retain(state):
            state.setdefault("artifact_cleanup", {}).setdefault(key, {
                "cleanup_id": key, "location": location, "evidence": deepcopy(evidence),
                "status": "pending", "reason_code": "publication_registration_failed", "created_at": self.clock(),
            })
        self.repository.transact(retain)
        return key


    def note_delivery(self, node, payload, *, delivered):
        def observe(state):
            command = state["outbox"].get(command_key(node, payload))
            if command is not None:
                command["attempts"] += 1
                command["last_delivery"] = self.clock()
                command["delivered"] = bool(delivered)
        return self.repository.transact(observe)
