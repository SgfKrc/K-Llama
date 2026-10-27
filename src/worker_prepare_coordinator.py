"""Master's remote native-load barrier; inventory never grants execution."""
from __future__ import annotations
from copy import deepcopy
import logging
import threading
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

    @staticmethod
    def _synthesize_self_cut_artifact(assignment, receipt, total_layers, capabilities=None):
        """Build the artifact identity that a self-cutting worker just produced.

        A `local_cut` node has no pre-cut segment at planning time: the supply phase
        cuts exactly the assigned range and reports it in the operation receipt.
        Without this the coordinator raised `prepare_artifact_identity_missing` even
        though the segment existed and had been verified.

        The receipt carries the new segment's digests, but **source identity is a
        property of the model, not of the range** — so `source_model_sha256` /
        `tokenizer_sha256` / `source_model_id` / `hidden_size` are taken from whatever
        artifact the worker advertises (its range may be stale; its source is not).
        """
        if not isinstance(receipt, dict) or receipt.get("status") != "available":
            return None
        artifact_sha = str(receipt.get("artifact_sha256", "") or "")
        manifest_sha = str(receipt.get("manifest_sha256", "") or "")
        start = int(assignment.get("start_layer", 0) or 0)
        end = int(assignment.get("end_layer", 0) or 0)
        if not artifact_sha or not manifest_sha:
            return None
        # ★ 2026-10-24（#125）：**同一供给内存在两次容量求解，结果会漂移**。
        #   实测：供给阶段求得 rog-5060=[8,24) 并真的裁出了 24 层；准备阶段
        #   （worker_prepare）又求解一次，得到 [8,19)，于是下面的区间相等判据失败，
        #   抛 `prepare_artifact_identity_missing`（409），整个 660 s 的供给白费。
        #   **收据是唯一的事实源**：它记录的就是刚裁出并已验证的那个段。因此这里以
        #   收据区间为准，而不是用 assignment 的区间去否决一个已存在的段。
        receipt_range = receipt.get("layer_range")
        if (isinstance(receipt_range, (list, tuple)) and len(receipt_range) == 2
                and all(type(bound) is int for bound in receipt_range)):
            if [int(receipt_range[0]), int(receipt_range[1])] != [start, end]:
                logger.warning(
                    "prepare_assignment_range_drift node=%s assignment=[%s,%s) "
                    "receipt=%s operation=%s —— 同一供给内两次求解漂移，以供给收据为准",
                    assignment.get("node_id"), start, end, list(receipt_range),
                    receipt.get("operation_id") or "<none>",
                )
            start, end = int(receipt_range[0]), int(receipt_range[1])
            # 让调用方拿到的 assignment 与收据一致（下游按它加载段）。
            assignment["start_layer"], assignment["end_layer"] = start, end
        elif receipt.get("layer_range") != [start, end]:
            return None

        def _field(name):
            value = receipt.get(name)
            if value not in (None, "", 0):
                return value
            for item in (capabilities or {}).get("layer_artifacts") or ():
                if isinstance(item, dict) and item.get(name) not in (None, "", 0):
                    return item[name]
            return value

        total = int(total_layers or 0)
        if start <= 0:
            mode = "head" if 0 < end < total else "full"
        elif total > 0 and end >= total:
            mode = "tail"
        else:
            mode = "middle"
        return {
            "model_id": str(_field("model_id") or ""),
            "artifact_sha256": artifact_sha,
            "manifest_sha256": manifest_sha,
            "layer_range": [start, end],
            "source_model_id": str(_field("source_model_id") or ""),
            "source_model_sha256": str(_field("source_model_sha256") or ""),
            "tokenizer_sha256": str(_field("tokenizer_sha256") or ""),
            "hidden_size": int(_field("hidden_size") or 0),
            "segment_mode": mode,
            "layer_weight_bytes": list(receipt.get("layer_weight_bytes") or []),
        }

    def _build_worker_prepare_assignment(self, assignment, *, config_id, generation, model_info, operation_id=""):
        result = deepcopy(assignment)
        node = result["node_id"]
        artifact = result.get("layer_artifact")
        operation = None
        receipt = {}
        if operation_id:
            operation = self._model_supply.get(operation_id)
            receipt = operation.get("nodes", {}).get(node, {})
        if not isinstance(artifact, dict):
            # ★ 2026-10-23（#117）：自裁节点在**分配时**没有现成工件 —— 段是供给阶段
            #   才裁出来的（切点还可能随容量账变化，所以不能指望"分配时就持有"）。
            #   用本次供给的**收据**合成工件身份，而不是直接判身份缺失。
            artifact = self._synthesize_self_cut_artifact(
                result, receipt, (model_info or {}).get("total_layers"),
                capabilities=(self._task_worker_control.worker_snapshot(node) or {}).get("capabilities") or {},
            )
            if artifact is None:
                # ★ 2026-10-23（#117）：合成失败必须留现场 —— 收据里到底是状态、
                #   摘要还是区间对不上，此前只有一个错误码。
                logger.warning(
                    "prepare_artifact_identity_missing node=%s range=[%s,%s) "
                    "operation=%s receipt_status=%s receipt_range=%s "
                    "artifact_sha=%s manifest_sha=%s",
                    node, result.get("start_layer"), result.get("end_layer"),
                    operation_id or "<none>",
                    (receipt or {}).get("status"),
                    (receipt or {}).get("layer_range"),
                    str((receipt or {}).get("artifact_sha256", ""))[:12],
                    str((receipt or {}).get("manifest_sha256", ""))[:12],
                )
                raise WorkerPrepareError("prepare_artifact_identity_missing")
            result["layer_artifact"] = artifact
        model = result.get("stage_model_identity")
        if not isinstance(model, dict):
            value = self._route_a_stage_model_identity(node, result)
            if value is None:
                raise WorkerPrepareError("prepare_stage_model_identity_missing")
            model = {key: getattr(value, key) for key in ("model_id", "engine", "format", "revision", "sha256")}
        manifest_sha = ""
        if operation_id:
            if (operation.get("status") != "available" or receipt.get("status") != "available"
                    or receipt.get("artifact_sha256") != artifact.get("artifact_sha256")
                    or receipt.get("layer_range") != [result["start_layer"], result["end_layer"]]):
                raise WorkerPrepareError("prepare_supply_identity_changed")
            manifest_sha = receipt.get("manifest_sha256", "")
            if not manifest_sha:
                raise WorkerPrepareError("prepare_supply_manifest_missing")
        worker = self._task_worker_control.worker_snapshot(node)
        if worker and (worker.get("capabilities") or {}).get("load_spec_version") != 1:
            raise WorkerPrepareError("prepare_load_spec_not_negotiated")
        from qlh_core.planning.load_spec import assignment_load_spec
        spec = assignment_load_spec(result, artifact_format=model["format"])
        result.update(load_spec=spec.to_dict(), context_size=spec.context_size)
        identity_input = {
            "load_spec_digest": spec.digest,
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
        }
        try:
            identity = validate_identity(identity_input)
        except WorkerPrepareError as _exc:
            # ★ 2026-10-23（#117）：身份被拒时必须能看到**具体哪个字段**不合规 ——
            #   该判据有十余个抛点，此前只有一个错误码，自裁路径上白绕了好几轮。
            logger.warning(
                "prepare identity rejected: code=%s node=%s range=%s "
                "model_id=%r model_sha=%r artifact_sha=%r manifest_sha=%r "
                "source_model_id=%r source_model_sha=%r tokenizer_sha=%r "
                "total_layers=%r hidden_size=%r segment_mode=%r stage_model=%s",
                _exc.reason_code, node, identity_input["layer_range"],
                identity_input["model_id"], str(identity_input["model_sha256"])[:16],
                str(identity_input["artifact_sha256"])[:16],
                str(identity_input["manifest_sha256"])[:16],
                identity_input["source_model_id"],
                str(identity_input["source_model_sha256"])[:16],
                str(identity_input["tokenizer_sha256"])[:16],
                identity_input["total_layers"], identity_input["hidden_size"],
                identity_input["segment_mode"], model,
            )
            raise
        result.update(config_id=config_id, generation=generation, assignment_id=identity["assignment_id"],
                      phase="prepare", worker_prepare_identity=identity, prepare_identity=stage_identity(identity),
                      plan_id=assignment.get("plan_id", ""))
        return result

    def _send_pipeline_worker_config(self, node_id, config):
        """One send path for first delivery, retries, commit and terminal cleanup."""
        send_options = {}
        epoch_getter = getattr(self._tcp_server, "get_authenticated_peer_epoch", None)
        if callable(epoch_getter):
            epoch = epoch_getter(node_id)
            if epoch is None:
                return False
            send_options["registration_epoch"] = epoch
        # All first sends, retries and terminal cleanup share this barrier.
        # Keep cleanup debt durable until the current connection has received
        # HELLO_ACK; a REGISTER ACK alone never authorizes MODEL_PREPARE.
        if node_id in self._task_worker_control.pending_worker_ids():
            return False
        identity = config.get("worker_prepare_identity")
        if isinstance(identity, dict):
            worker = self._task_worker_control.worker_snapshot(node_id)
            if worker and not (worker.get("connected") and worker.get("accepted")):
                return False
        if isinstance(identity, dict) and not config.get("cleanup_action") and not config.get("release"):
            with self._layer_config_lock:
                transaction = self._pipeline_load_transaction
                if (not transaction or transaction.get("config_id") != identity["config_id"]
                        or transaction.get("phase") not in {"preparing", "committing"}
                        or transaction.get("expected", {}).get(node_id, {}).get("phase") != config.get("phase")):
                    return False
        if not isinstance(identity, dict):
            return self._tcp_server.send_layer_config(node_id, config, **send_options)
        from transport_port import MessageType
        from worker_prepare_contract import validate_command
        action = config.get("cleanup_action") or ("release" if config.get("release") else config.get("phase", "prepare"))
        command = {"action": action, "identity": deepcopy(identity)}
        if action == "prepare":
            command.update(context_size=config.get("context_size", 4096), middle_channel="keep_head_layer_out")
            if "load_spec" in config:
                command["load_spec"] = deepcopy(config["load_spec"])
        if action in {"abort", "finalize"} or (action == "release" and config.get("cleanup_action")):
            command["cleanup_id"] = config["config_id"]
        command = validate_command(command)
        owner = getattr(self, "_model_control", None)
        if owner is not None and not owner.authorize_command(node_id, command,
                authority=self._model_control_authority(), connection_generation=self._assignment_connection_generation(node_id)):
            return False
        delivered = self._tcp_server.send_to_client(node_id, command, MessageType.MODEL_PREPARE, **send_options)
        if owner is not None:
            owner.note_delivery(node_id, command, delivered=delivered is not False)
        return delivered

    def _handle_worker_prepare_ack(self, node_id, payload, *, defer_native_work=False):
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
            if ack["status"] != "error" and "load_spec_digest" in identity:
                receipt = ack.get("runtime_receipt")
                if receipt is None:
                    return
                from worker_prepare_contract import validate_runtime_receipt
                try:
                    validate_runtime_receipt(receipt, identity=identity)
                except WorkerPrepareError:
                    return
                if expected.get("load_spec", {}).get("materialization") == "safetensors_layer_offload":
                    from qlh_core.contracts.deployment import ModelControlError, require_runtime_receipt
                    try:
                        require_runtime_receipt(expected, receipt)
                    except ModelControlError as exc:
                        ack.update(status="error", error=exc.code)
                else:
                    # ★ 2026-10-26（LOCAL-EXECUTORS-01）：预算口径**统一到
                    #   `qlh_core.contracts.deployment.materialization_budget`**。
                    #
                    #   这条判据此前有两份独立实现（这里与 `require_runtime_receipt`
                    #   的 else 分支）。我上一轮只改了这里，结果 `require_runtime_receipt`
                    #   仍按 `required_bytes` 拒掉回执 ⇒ 生命周期持久化失败
                    #   （`deployment_materialization_budget_exceeded`）⇒ 事务卡死到超时
                    #   （实测 10:41:18 起完全静默，master/TABLET 永远停在 `loading`）。
                    #   现在两处共用同一个函数，改一处不会再漏另一处。
                    from qlh_core.contracts.deployment import materialization_budget
                    materialization = expected.get("load_spec", {}).get("materialization")
                    budget = materialization_budget(expected, str(materialization or ""))
                    if budget and receipt["materialized_bytes"] + receipt["load_spec"]["kv_cache_bytes"] > budget:
                        # ★ 把**两边各自的口径**打出来：预算来自求解器按源描述符的估算，
                        #   而 `materialized_bytes` 是 worker 按它自己那份工件实际加载的量，
                        #   量化段与源差异很大时两者基准不同 —— 只看 "exceeded" 无法区分
                        #   是预算算小了还是 worker 多加载了。
                        logger.warning(
                            "event=prepare_budget_exceeded node=%s materialized=%s "
                            "kv_cache=%s total=%s budget=%s required=%s "
                            "capacity=%s reserve=%s materialization=%s",
                            node_id, receipt["materialized_bytes"],
                            receipt["load_spec"]["kv_cache_bytes"],
                            receipt["materialized_bytes"] + receipt["load_spec"]["kv_cache_bytes"],
                            budget, expected.get("required_bytes"),
                            expected.get("capacity_bytes"), expected.get("reserve_bytes"),
                            expected.get("load_spec", {}).get("materialization"),
                        )
                        ack.update(status="error", error="prepare_materialization_budget_exceeded")
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
        elif advance:
            self._advance_worker_prepare_transaction(
                identity["config_id"], advance, background=defer_native_work,
            )

    def _advance_worker_prepare_transaction(self, config_id, phase, *, background=False):
        """Keep native loading/publication off the TCP receive thread.

        Claim each barrier once before starting work. Duplicate ACKs can still
        refresh their receipt while the same transaction is waiting to advance.
        The transition methods recheck the captured config and current phase.
        """
        transition = (self._commit_pipeline_load_transaction if phase == "prepare"
                      else self._complete_pipeline_load_transaction)
        if not background:
            return transition(config_id)
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            expected_phase = "preparing" if phase == "prepare" else "committing_global"
            if (not transaction or transaction.get("config_id") != config_id
                    or transaction.get("phase") != expected_phase):
                return
            scheduled = transaction.setdefault("native_advances_scheduled", set())
            if phase in scheduled:
                return
            scheduled.add(phase)

        def advance():
            try:
                transition(config_id)
            except Exception:
                logger.exception("pipeline native transition failed: config=%s phase=%s", config_id, phase)
                self._abort_pipeline_load_transaction(config_id, "pipeline_native_transition_failed")

        threading.Thread(target=advance, name=f"pipeline-{phase}-{config_id}", daemon=True).start()

    def _worker_prepare_assignment_loaded(self, node_id, assignment):
        # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**本机附加执行器不走 prepare/commit 回执**。
        #
        #   同机 CPU 段没有 TCP 连接，因此本方法后面的每一道检查都不适用于它：
        #     * `worker_prepare_identity` —— 本机写的是 `identity`；
        #     * `_worker_assignments.state(node_id)` —— 根本没有它的 worker 状态；
        #     * `ack["phase"] == "commit"` / `ack["loaded"]` —— 没有远端回执。
        #   它的"就绪"由 `freeze_pipeline_candidate_assignment` 成功即代表：工件已在
        #   本机、load_spec 已冻结。此前落到硬取 `worker_prepare_identity` 那条
        #   ⇒ `worker_native_prepare_required` ⇒ 引擎设置阶段直接失败
        #   （实测 11:51:17：CPU 段第一次真正 stage 就被挡）。
        if str(assignment.get("executor_id") or ""):
            return True, "local_executor"
        identity = assignment.get("worker_prepare_identity")
        if not isinstance(identity, dict):
            return False, "worker_native_prepare_required"
        owner = getattr(self, "_model_control", None)
        if owner is not None and not owner.runnable(identity["config_id"],
                authority=self._model_control_authority(), connections=self._model_control_connections(),
                input_revisions=self._model_control_input_revisions(self._pipeline_load_transaction or {})):
            return False, "deployment_not_published"
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
