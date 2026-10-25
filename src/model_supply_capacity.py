"""Fresh operation-scoped budgets over the existing authenticated inventory channel.

The supply coordinator owns query/reply ordering. Discovery profiles are never
mutated, and a budget snapshot grants neither an assignment nor loaded-ready.
A new wire operation id correlates each inventory query even with older Android
workers; inventory does not change their active supplied asset.
"""
from __future__ import annotations

from copy import deepcopy
import logging
import time
import uuid

logger = logging.getLogger(__name__)
# ★ 2026-10-24（#117）：时效改为可调（默认 300 s）。此前硬编码 30 秒，而 `local_cut`
# 节点要在准入前按分配区间把段裁出来（实测 3 GB 级远超 30 秒）⇒ 三段布局已经全部
# `available` 之后仍会以 `SUPPLY_RUNTIME_BUDGET_EXPIRED` 失败。
from config import MODEL_SUPPLY_BUDGET_AGE_S as MAX_BUDGET_AGE_SECONDS


def _error(code, message):
    from model_supply import ModelSupplyError
    return ModelSupplyError(code, message)


def _connection(coordinator, node_id):
    server = coordinator.scheduler._tcp_server
    info = server.get_client_info(node_id) if server else None
    if not coordinator._worker_handshake_ready(node_id, info):
        raise _error("SUPPLY_RUNTIME_WORKER_UNAVAILABLE", f"Worker {node_id} has no current authenticated handshake")
    return (info.get("connected_at"), server.get_authenticated_peer_epoch(node_id))


def _validate(coordinator, query, *, check_age=True):
    coordinator.ensure_prepare_not_cancelled(query["operation_id"])
    if coordinator._preparing != query["operation_id"]:
        raise _error("SUPPLY_OPERATION_STALE", "Runtime budget no longer belongs to the entering model operation")
    coordinator._check_receivers(query["operation_id"])
    if coordinator._operations[query["operation_id"]].get("plan") != query["plan"]:
        raise _error("SUPPLY_OPERATION_STALE", "Supplied assignment changed during runtime admission")
    for node_id, identity in query["connections"].items():
        if _connection(coordinator, node_id) != identity:
            raise _error("SUPPLY_RUNTIME_CONNECTION_CHANGED", f"Worker {node_id} changed registration during runtime admission")
    if check_age and time.monotonic() - query["started_monotonic"] > MAX_BUDGET_AGE_SECONDS:
        raise _error("SUPPLY_RUNTIME_BUDGET_EXPIRED", "Runtime budget expired before model admission")


def _assignment_capability_matches(capabilities, assignment, query):
    """Bind a received hello/projection to the supplied segment, never its alias."""
    node_id, engine = assignment["node_id"], assignment["engine"]
    layer_range = [assignment["start_layer"], assignment["end_layer"]]
    expected_sha = query["artifacts"][node_id]
    artifacts = capabilities.get("layer_artifacts") or []
    models = capabilities.get("models") or []
    # ★ 2026-10-23（#117）：自裁节点声明的是"**能承担**"的整域（如 [0, total)），
    #   而分配给它的是其中一个子区间。精确相等会把它判成"没有广告这个段"，
    #   供给便永远卡在 SUPPLY_RUNTIME_CAPABILITY_TIMEOUT（实测 ROG 声明 [0,32]、
    #   分配 [8,16)，反复超时）。判据应为**区间覆盖**。
    advertised_ranges = capabilities.get("layer_ranges") or []
    if not any(
        isinstance(item, (list, tuple)) and len(item) == 2
        and item[0] <= layer_range[0] and layer_range[1] <= item[1]
        for item in advertised_ranges
    ):
        return False
    for artifact in artifacts:
        if (artifact.get("layer_range") != layer_range
                or artifact.get("artifact_sha256") != expected_sha
                or artifact.get("source_model_sha256") != query["source_model_sha256"]
                or artifact.get("tokenizer_sha256") != query["tokenizer_sha256"]):
            continue
        if any(model.get("model_id") == artifact.get("model_id")
               and model.get("sha256") == expected_sha and model.get("engine") == engine
               and model.get("format") == ("gguf" if engine == "llama_cpp" else "safetensors")
               for model in models):
            return True
    return False


def _supplied_capabilities_ready(coordinator, query):
    control = coordinator.scheduler._task_worker_control
    return all(_assignment_capability_matches(
        control.worker_snapshot(a["node_id"]).get("capabilities") or {}, a, query)
        for a in query["plan"]["assignments"] if a["node_id"] != "master")


def handle_capacity_inventory(coordinator, node_id, payload):
    """Called under the coordinator condition; consume only a pending query id."""
    query_id = payload.get("operation_id")
    if not isinstance(query_id, str):
        return False
    query = coordinator._capacity_queries.get(query_id)
    if query is None:
        return False
    if (payload.get("action") != "inventory_result" or payload.get("node_id") != node_id
            or node_id not in query["connections"]):
        return True
    try:
        _validate(coordinator, query)
    except Exception:
        # Wake the waiter so it reports the authoritative cancellation/epoch
        # error. A stale packet must not replace a current budget.
        coordinator._condition.notify_all()
        return True
    query["replies"].setdefault(node_id, deepcopy(payload))
    query["received_at"].setdefault(node_id, time.time())
    coordinator._condition.notify_all()
    return True



def runtime_capacity_source(runtime, device, *, prefix):
    """Retain the actual CUDA budget authority; older workers use torch alone."""
    if device != "cuda":
        return f"{prefix}.ram.available_bytes_live"
    from cuda_memory_budget import TORCH_BUDGET_SOURCE, UUID_MIN_BUDGET_SOURCE
    source = runtime.get("budget_source") if isinstance(runtime, dict) else None
    if source not in {TORCH_BUDGET_SOURCE, UUID_MIN_BUDGET_SOURCE}:
        source = TORCH_BUDGET_SOURCE
    return f"{prefix}.{source}"


def project_inventory_capacity(record, inventory, engine, *, source_prefix="inventory"):
    """Bind fresh inventory to the selected runtime, without granting readiness.

    Host RAM constrains CUDA source staging independently; it is never added to
    VRAM. Multipliers here apply to unprojected execution descriptors. Supply
    header projections carry their own dtype/materialization byte accounting.
    """
    from qlh_core.contracts.load import RuntimeLoadSpec

    node_id = record["node_id"]
    engines = inventory.get("engines")
    if (inventory.get("distributed") is not True or not isinstance(engines, list)
            or engine not in {"pytorch", "llama_cpp"} or engine not in engines):
        raise _error("SUPPLY_RUNTIME_ENGINE_UNAVAILABLE", f"Selected runtime on {node_id} is no longer available")
    available, device = inventory.get("memory_available_bytes"), "cpu"
    runtime = None
    if engine == "pytorch":
        runtime = inventory.get("pytorch_runtime")
        if (not isinstance(runtime, dict) or runtime.get("execution_device") not in {"cpu", "cuda"}
                or type(runtime.get("target_element_size")) is not int
                or runtime["target_element_size"] not in {2, 4}):
            raise _error("SUPPLY_RUNTIME_BUDGET_MISSING", f"Selected runtime on {node_id} has no current budget")
        available, device = runtime.get("available_bytes"), runtime["execution_device"]
    if type(available) is not int or available <= 0:
        raise _error("SUPPLY_RUNTIME_BUDGET_MISSING", f"Selected runtime on {node_id} has no positive free-memory sample")
    source = runtime_capacity_source(runtime, device, prefix=source_prefix)
    # ★ 2026-10-23（#117）：×2.0 只适用于**未投影**的执行描述符（本函数自己的
    #   约定，见 docstring）—— PyTorch 在 CPU 上把 fp16 权重展开成 fp32，所以按
    #   描述符字节算完必须翻倍。GGUF 段是 `gguf_segment` 物化：按**原生量化宽度**
    #   （Q4_K_M）就地使用、不 upcast，字节账已经是执行宽度，再乘 2 会把 5.78GB 的
    #   整模算成 11.6GB ⇒ 三机容量账凭空减半（实测 rog 只算得出 2 层、android 13
    #   层、master 8 层，合计 23 < 32）⇒ 求解器永远搜不出可行解，`waiting_nodes`
    #   每 20~30 秒重算一次直到超时。
    supplied_spec = record.get("load_spec")
    projected = (
        isinstance(supplied_spec, dict)
        and str(supplied_spec.get("materialization", "") or "") == "gguf_segment"
    )
    # ★ 2026-10-25：`engine == "llama_cpp"` 同样必须算 1.0。
    #   上面 #117 的结论（"GGUF 段按原生量化宽度就地使用、再乘 2 会把整模算成
    #   11.6GB ⇒ 三机容量账凭空减半"）当时只落到了 `projected` 上，而 `projected`
    #   依赖 `record["load_spec"]`；上游 `bind_supplied_runtime_capacity` 对
    #   **非 CUDA/safetensors** 的 assignment 会直接早退、从不注入 `load_spec`
    #   （见该函数 `if mode != "safetensors_layer_offload" and not (cuda_spec or
    #   quoted_cuda): return record`），于是 CPU/GGUF 节点（rog-5060、
    #   android-542cd254）的 `projected` 恒为 False ⇒ multiplier 恒为 2.0
    #   ⇒ **准备阶段的容量账比供给阶段少一半**。
    #   实测后果：供给阶段求解出 rog=[8,24)/android=[24,32)，准备阶段因容量
    #   "不再满足"而按设计解除 `candidate_layer_ranges` 锁定、重排成
    #   [8,19)/[19,32) ⇒ 与已裁好的段对不上 ⇒ `prepare_assignment_range_drift`
    #   与 worker 回执 `prepare_receipt_artifact_resolution_failed`，整次供给作废。
    #   ★ 判据与**物理真相**一致：GGUF 段是原生量化宽度、就地使用不 upcast，故
    #   `engine == "llama_cpp"` 恒取 1.0。这与 `docs/已知问题记录.md` 登记的决定同向
    #   （`runtime_multiplier` 只适用于**未投影**描述符，即 PyTorch CPU 的 fp16→fp32）。
    #   ⚠️ 它与**供给阶段**（`model_supply.py` 的 `1.0 if local_cut else 2.0`）**刻意
    #   不同**：非自裁 llama_cpp 在供给阶段仍保守取 2.0（#117 有意保留）。两阶段在此
    #   不一致是设计，方向是**供给阶段高估一倍**（更保守），不会放过装不下的方案。
    #   （此处原先写作"判据现与供给阶段…一致"，是错误陈述 —— 它会把容量归因排查带偏。）
    runtime_multiplier = 1.0 if (
        device == "cuda" or projected or engine == "llama_cpp"
    ) else 2.0
    result = {**deepcopy(record), "capacity_bytes": available, "capacity_source": source,
              "execution_device": device, "engine": engine,
              "runtime_multiplier": runtime_multiplier}
    if device == "cuda" and isinstance(runtime.get("budget_observations"), dict):
        result["capacity_observations"] = deepcopy(runtime["budget_observations"])
    if "load_spec" in result:
        try:
            value = result["load_spec"]
            spec = value if isinstance(value, RuntimeLoadSpec) else RuntimeLoadSpec.from_dict(value)
        except (TypeError, ValueError) as exc:
            raise _error("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", "Invalid runtime load_spec") from exc
        materializations = {"safetensors_key_filtered"} if engine == "pytorch" else {"gguf_segment"}
        if engine == "pytorch" and spec.materialization == "safetensors_layer_offload":
            supported = runtime.get("materializations")
            if device != "cuda" or not isinstance(supported, list) or spec.materialization not in supported:
                raise _error("SUPPLY_RUNTIME_MATERIALIZATION_UNAVAILABLE", "Selected runtime has no layer offload adapter")
            materializations.add("safetensors_layer_offload")
        if (result.get("load_spec_digest", spec.digest) != spec.digest
                or spec.device.partition(":")[0] != device or spec.materialization not in materializations
                or (runtime is not None and {"float16": 2, "float32": 4}.get(spec.dtype)
                    != runtime["target_element_size"])):
            raise _error("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", "Inventory disagrees with selected engine/device/dtype load_spec")
        result.update(load_spec=spec.to_dict(), load_spec_digest=spec.digest, execution_device=spec.device)
    host = inventory.get("memory_available_bytes")
    if device == "cuda" and type(host) is int and host >= 0:
        result.update(host_capacity_bytes=host, host_capacity_source=f"{source_prefix}.ram.available_bytes_live")
    return result


def bind_supplied_runtime_capacity(record, assignment, inventory):
    """Restore a supplied CUDA policy and its exact target/host range budgets.

    Artifact byte quotes remain source-precision evidence. The runtime quote
    includes all selected boundary weights and replaces their additive GPU cost;
    only the fresh record's configured reserve and solver margin are added.
    """
    from qlh_core.planning.load_spec import assignment_load_spec

    value = assignment.get("load_spec")
    mode = value.get("materialization") if isinstance(value, dict) else None
    cuda_assignment = str(assignment.get("execution_device", "")).startswith("cuda")
    cuda_spec = isinstance(value, dict) and str(value.get("device", "")).startswith("cuda:")
    quoted_cuda = (assignment.get("engine") == "pytorch" and cuda_assignment
                   and any(key in assignment for key in (
                       "device_weight_bytes", "runtime_target_element_size", "runtime_target_dtype")))
    if mode != "safetensors_layer_offload" and not (cuda_spec or quoted_cuda):
        return record
    if not isinstance(value, dict):
        raise _error("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", "Supplied CUDA quote has no frozen load policy")
    try:
        spec = assignment_load_spec(assignment, artifact_format="safetensors")
    except (TypeError, ValueError) as exc:
        raise _error("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", "Supplied CUDA load policy is invalid") from exc
    source, conversion = assignment.get("host_source_bytes"), assignment.get("host_conversion_bytes")
    host_required, weight = assignment.get("host_required_bytes"), assignment.get("device_weight_bytes")
    host_capacity = inventory.get("memory_available_bytes")
    start, end = assignment.get("start_layer"), assignment.get("end_layer")
    embedding, lm_head = assignment.get("has_embedding"), assignment.get("has_lm_head")
    if (not spec.device.startswith("cuda:")
            or spec.materialization not in {"safetensors_key_filtered", "safetensors_layer_offload"}
            or assignment.get("engine", "pytorch") != "pytorch"
            or assignment.get("execution_device", spec.device) not in {spec.device, "cuda"}
            or assignment.get("runtime_materialization", spec.materialization) != spec.materialization
            or spec.quant_type != {"float16": "fp16", "float32": "fp32"}.get(spec.dtype)
            or any(type(size) is not int or size < 0 for size in (source, conversion))
            or type(host_required) is not int or host_required <= 0 or host_required < source + conversion
            or type(weight) is not int or weight <= 0
            or any(type(bound) is not int for bound in (start, end)) or not 0 <= start < end
            or type(embedding) is not bool or type(lm_head) is not bool
            or embedding != ("embedding" in spec.components) or lm_head != ("lm_head" in spec.components)
            or assignment.get("load_spec_digest") != spec.digest):
        raise _error("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", "Supplied CUDA range quote is invalid")
    if (spec.materialization == "safetensors_key_filtered" and "target_weight_bytes" in assignment
            and (type(assignment["target_weight_bytes"]) is not int
                 or assignment["target_weight_bytes"] != weight)):
        raise _error("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", "Resident target weight quote disagrees with CUDA budget")
    if type(host_capacity) is not int or host_capacity < host_required:
        raise _error("SUPPLY_RUNTIME_HOST_CAPACITY_INSUFFICIENT", "Host RAM changed before CUDA admission")
    result = deepcopy(record)
    result.update(load_spec=spec.to_dict(), load_spec_digest=spec.digest,
                  runtime_materialization=spec.materialization,
                  host_capacity_bytes=host_capacity, host_required_bytes=host_required,
                  host_source_bytes=source, host_conversion_bytes=conversion,
                  retained_component_bytes=0,
                  range_weight_bytes=[{"layer_range": [start, end], "weight_bytes": weight,
                                       "has_embedding": embedding, "has_lm_head": lm_head}])
    return result


def bind_supplied_offload_capacity(record, assignment, inventory):
    """Compatibility entry point; CUDA resident and offload quotes share a binder."""
    return bind_supplied_runtime_capacity(record, assignment, inventory)


class SupplyCapacityEvidence:
    """One immutable inventory snapshot, validated again at the solve boundary."""
    def __init__(self, coordinator, query, inventories):
        self._coordinator = coordinator
        self._query = deepcopy(query)
        self._inventories = deepcopy(inventories)

    def touch(self) -> None:
        """★ 2026-10-23（#116）：把准入年龄窗口的基准推到「最后一次实际进展」。

        工件传输（实测 5.4 GB、数十分钟）是供给流程的正常组成。窗口若只从证据
        准备时刻起算，任何大工件的准入都会以 SUPPLY_RUNTIME_BUDGET_EXPIRED 失败；
        改以「最近一次传输进展」起算后，**停滞**仍会被原样检出（窗口长度不变）。
        """
        self._query["started_monotonic"] = time.monotonic()

    def project(self, records):
        c, query = self._coordinator, self._query
        with c._condition:
            # ★ 2026-10-23（已知问题记录 #116）：年龄检查**必须保留** —— 证据是
            #   "一次不可变的 inventory 快照，在 solve 边界再校验一次"（类注释），
            #   放宽会削弱"准入必须基于新鲜容量"的合同（由
            #   test_pipeline_capacity_freshness 与 test_supply_capacity_evidence 断言）。
            #   缺陷从来不是这个检查，而是**证据在工件传输前准备、却在传输后被使用**：
            #   实测把 5.4 GB 层段传给 ROG 需数十分钟，于是 11:12:25 出的预算到
            #   11:16:35 准入时必然超过 MAX_BUDGET_AGE_SECONDS ⇒ 任何大工件的供给
            #   都以 SUPPLY_RUNTIME_BUDGET_EXPIRED 失败。
            #   修法：传输进展即续期（touch()），窗口长度不变、基准跟随最后一次实际
            #   进展；传输停滞超过窗口仍会正常失败。
            _validate(c, query)
            # A selection change must not reuse a budget for another manifest.
            operation = c._operations[query["operation_id"]]
            if not _supplied_capabilities_ready(c, query):
                raise _error("SUPPLY_RUNTIME_CAPABILITY_CHANGED", "Worker segment capabilities changed before runtime admission")
        # ★ 2026-10-25（LOCAL-EXECUTORS-01）：**按 `(node_id, executor_id)` 索引**。
        #   同机两个执行器（GPU 段 + CPU 段）**共用 `node_id`**，用
        #   `{record["node_id"]: record}` 建索引会让后者**静默覆盖**前者。实测：
        #   本机 CPU 段覆盖了主段 ⇒ ① 主记录从结果里消失（求解器看不到 master 的
        #   GPU 段）② CPU 段被当成 master 的段处理，拿到主段的切点
        #   `candidate_layer_ranges=[[0,8]]`，与它自己声明的 `[8,24)` 冲突 ⇒ 无解。
        #   `executor_id` 缺省 "" ⇒ 键退化为 `(node_id, "")` ⇒ 与改动前逐位等价。
        def _executor_key(record):
            return (record["node_id"], str(record.get("executor_id") or ""))

        by_id = {_executor_key(record): record for record in records}
        # ★ 2026-10-25（LOCAL-EXECUTORS-01）：把**供给 plan 的 assignment 身份**打出来。
        #   串味排查的下一步依赖它：CPU 段仍拿到主段的 `candidate_layer_ranges`，而唯一
        #   剩下的注入点就是这个循环（`record["candidate_layer_ranges"]` 由 assignment
        #   的 start/end 决定）。若 plan 里的 `master` assignment 缺 `executor_id`，
        #   它就会匹配到 CPU 段 —— 那意味着**plan 自身也按 node_id 串味**，需要在更上游修。
        logger.info(
            "供给 plan 身份: %s",
            [(item.get("node_id"), item.get("executor_id", ""),
              item.get("start_layer"), item.get("end_layer"))
             for item in (query.get("plan") or {}).get("assignments", [])],
        )
        result, provenance = [], []
        handled: set[tuple[str, str]] = set()
        for assignment in query["plan"]["assignments"]:
            node_id, engine = assignment["node_id"], assignment["engine"]
            key = (node_id, str(assignment.get("executor_id") or ""))
            if key not in by_id:
                raise _error("SUPPLY_RUNTIME_NODE_UNAVAILABLE", f"Selected node {node_id} is no longer eligible")
            if node_id != "master" and not _assignment_capability_matches(by_id[key], assignment, query):
                raise _error("SUPPLY_RUNTIME_CAPABILITY_CHANGED", "Worker capacity projection does not match the supplied segment")
            inventory = self._inventories[node_id]
            selected_record = bind_supplied_runtime_capacity(by_id[key], assignment, inventory)
            record = project_inventory_capacity(selected_record, inventory, engine)
            available, device, source = (record["capacity_bytes"], record["execution_device"],
                                         record["capacity_source"])
            if engine == "llama_cpp" and "retained_component_bytes" in assignment:
                record["retained_component_bytes"] = assignment["retained_component_bytes"]
            # Preserve advertised ranges/artifact contracts and the configured
            # reserve. Only the selected runtime's memory authority changes.
            # ★ 2026-10-23（#117）：把**供给时的切点**转成候选约束，锁定本次布局。
            #   否则 prepare 会重新自由求解：内存波动一次就可能选出不同切点（实测
            #   rog 供给 [8,20)、prepare 却算出 [8,21)）⇒ 与已裁好的段对不上 ⇒
            #   `prepare_artifact_identity_missing`。容量确实不再满足时，求解入口会
            #   去掉这个约束重排（见 scheduler 的容量求解），即"内存增加不动切点、
            #   减少才重切"。
            record["candidate_layer_ranges"] = [
                [int(assignment["start_layer"]), int(assignment["end_layer"])]
            ]
            result.append(record)
            handled.add(key)
            item = {"node_id": node_id, "engine": engine, "execution_device": device,
                    "capacity_bytes": available, "capacity_source": source,
                    "requested_at": query["requested_at"], "received_at": query["received_at"][node_id],
                    "registration_epoch": query["connections"].get(node_id, (None, None))[1],
                    "operation_id": query["operation_id"], "query_id": query["query_id"],
                    "layer_range": [assignment["start_layer"], assignment["end_layer"]],
                    "artifact_sha256": operation["nodes"][node_id].get("artifact_sha256", "")}
            item.update(reserve_bytes=record.get("reserve_bytes", 0),
                        reserve_source=record.get("reserve_source", "explicit"),
                        runtime_multiplier=record["runtime_multiplier"])
            for field in ("host_required_bytes", "host_capacity_bytes", "host_capacity_source",
                          "load_spec_digest", "range_weight_bytes", "capacity_observations"):
                if field in record:
                    item[field] = deepcopy(record[field])
            provenance.append(item)
            logger.info("event=pipeline_runtime_budget operation=%s query=%s node=%s engine=%s device=%s capacity_bytes=%s capacity_source=%s reserve_bytes=%s multiplier=%s registration_epoch=%s layer_ranges=%s",
                        query["operation_id"], query["query_id"], node_id, engine, device, available, source,
                        record.get("reserve_bytes"), record["runtime_multiplier"], item["registration_epoch"], record.get("layer_ranges"))
        # ★ 2026-10-25（LOCAL-EXECUTORS-01）：**未被本次 assignment 覆盖的记录也要保留**。
        #   本机附加执行器（CPU 段）**不在供给 plan 里** —— 它由本进程自己执行、没有
        #   工件传输流程，所以 `query["plan"]["assignments"]` 永远不会提到它。若在此
        #   按旧行为只返回"被处理过的"记录，求解器就永远看不到它，"同机两段"无法成立。
        #   注意用 `deepcopy`：`by_id` 里的是调用方传入的记录对象，直接塞进结果会让
        #   下游的原地修改（如补身份字段）反向污染调用方的列表。
        for leftover_key, leftover in by_id.items():
            if leftover_key not in handled:
                result.append(deepcopy(leftover))
        return result, provenance


def prepare_capacity_evidence(coordinator, operation_id):
    """Refresh each selected worker; old replies cannot satisfy this new query."""
    c = coordinator
    with c._condition:
        c.ensure_prepare_not_cancelled(operation_id)
        if c._preparing != operation_id:
            raise _error("SUPPLY_PREPARE_REQUIRED", "Runtime budget refresh requires the selected prepare owner")
        operation = c.get(operation_id)
        plan = deepcopy(operation["plan"])
        if operation["status"] != "available" or not isinstance(plan, dict) or not plan.get("assignments"):
            raise _error("SUPPLY_ASSETS_NOT_AVAILABLE", "Runtime admission requires supplied assignments")
        worker_ids = [a["node_id"] for a in plan["assignments"] if a["node_id"] != "master"]
        if not worker_ids:
            raise _error("SUPPLY_WORKERS_UNAVAILABLE", "Distributed admission requires supplied workers")
        query_id = "supply_" + uuid.uuid4().hex
        query = {"operation_id": operation_id, "query_id": query_id, "plan": plan,
                 "connections": {node: _connection(c, node) for node in worker_ids},
                 "replies": {}, "received_at": {}, "started_monotonic": time.monotonic(),
                 "requested_at": time.time(),
                 "source_model_sha256": operation.get("source_model_sha256"),
                 "tokenizer_sha256": operation.get("tokenizer_sha256"),
                 "artifacts": {a["node_id"]: operation["nodes"][a["node_id"]]["artifact_sha256"]
                               for a in plan["assignments"]}}
        c._capacity_queries[query_id] = query
    try:
        from transport_port import MessageType
        with c._condition:
            # Registration publishes available before its asynchronous hello
            # refresh necessarily reaches the coordinator. Wait for this exact
            # segment without holding the condition or accepting old ranges.
            handshake_deadline = time.monotonic() + MAX_BUDGET_AGE_SECONDS
            while True:
                _validate(c, query, check_age=False)
                remaining = handshake_deadline - time.monotonic()
                if remaining <= 0:
                    raise _error("SUPPLY_RUNTIME_CAPABILITY_TIMEOUT", "Worker did not advertise the supplied segment before runtime admission")
                if _supplied_capabilities_ready(c, query):
                    break
                c._condition.wait(min(remaining, 0.25))
            logger.info("event=supply_prepare_capabilities_ready operation=%s query=%s elapsed_ms=%d",
                        operation_id, query_id, (time.monotonic() - query["started_monotonic"]) * 1000)
            query["started_monotonic"], query["requested_at"] = time.monotonic(), time.time()
            _validate(c, query)
            for node in worker_ids:
                try:
                    sent = c.scheduler._tcp_server.send_to_client(node, {
                        "action": "inventory", "operation_id": query_id,
                        "model_id": "", "source_model_sha256": "",
                    }, MessageType.MODEL_SUPPLY)
                except OSError as exc:
                    raise _error("SUPPLY_RUNTIME_WORKER_UNAVAILABLE", f"Worker {node} could not be queried") from exc
                if sent is False:
                    raise _error("SUPPLY_RUNTIME_WORKER_UNAVAILABLE", f"Worker {node} could not be queried")
        # An empty selection queries resources without hashing model sources.
        # Existing PC/Android inventory handlers already support this request.
        # The local inventory uses the identical runtime sampling path.
        local = c.scheduler._model_supply_worker.inventory(model_id="")
        with c._condition:
            query["received_at"]["master"] = time.time()
            while True:
                _validate(c, query)
                if all(node in query["replies"] for node in worker_ids):
                    inventories = {"master": local, **query["replies"]}
                    evidence = SupplyCapacityEvidence(c, query, inventories)
                    # ★ 2026-10-23（#116）：登记，以便工件传输期间续期准入窗口。
                    c.register_capacity_evidence(operation_id, evidence)
                    logger.info("event=supply_prepare_resources_ready operation=%s query=%s elapsed_ms=%d",
                                operation_id, query_id, (time.monotonic() - query["started_monotonic"]) * 1000)
                    return evidence
                remaining = MAX_BUDGET_AGE_SECONDS - (time.monotonic() - query["started_monotonic"])
                c._condition.wait(min(remaining, 0.25))
    finally:
        with c._condition:
            c._capacity_queries.pop(query_id, None)
