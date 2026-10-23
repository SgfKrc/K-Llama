"""
调度控制模块 — 节点状态管理、推理任务分发、流水线调度
========================================================
功能职责:
1. 节点状态管理（空闲/忙碌/离线）
2. 推理任务分发、流程启停
3. 异常捕获、错误上报
4. 流水线数据流调度控制
5. TCP 服务端集成 — 接收从节点注册、维护连接

依赖: threading, logging, socket
"""

import collections
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid
from enum import Enum
from typing import Any, Mapping, Optional, Callable, Sequence, TYPE_CHECKING

# ★ #31 M2：层流水线支持的架构走**单一事实来源**（此前这里硬编码 `{"qwen","qwen2"}`）
from pipeline_model_descriptor import PIPELINE_RUNTIME_MODEL_TYPES
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from model_host import InferenceHost, SchedulerCallbacks

from model_host import get_model_host
from koakuma_engine import (
    Capability,
    backend_capabilities,
    backend_id_for,
    runtime_supports,
)
from task_worker_capabilities import node_supports_forward_layers
from network_path import build_client_network_path_view
from pipeline_capacity import PipelineCapacityError, solve_pipeline_capacity
from pipeline_node_contract import (
    PipelineNodeContractError,
    build_aggregate_resource_view,
    pipeline_layout_from_capacity_plan,
)
from pipeline_reshard import PipelineArtifactAvailability, PipelineReshardCoordinator
import scheduler_layer_plan as _layer_plan
from scheduler_types import (
    InferenceTask, NodeInfo, NodeRole, NodeState,
    ANDROID_PRESENCE_HEARTBEAT_INTERVAL_SECONDS,
    TASK_WORKER_HEALTH_TIMEOUT_FLOOR_SECONDS,
    WORKER_HEARTBEAT_MAX_AGE,
    assert_worker_liveness_thresholds,
    task_worker_control_plane_health_timeout_seconds,
)
# ★ 2026-10-07（DIST-NEXT-3）：assignment 的权威视图（单一事实源）。
from worker_assignment_state import WorkerAssignmentRegistry
from scheduler_sidecars import SchedulerSidecarMixin
from llama_rpc_contract import RpcShardLeaseBook
from qwen3_pipeline_transaction import (
    Qwen3PipelineDryRunTransaction,
    Qwen3PipelineProtocolError,
)
from qwen3_pipeline_loopback import (
    Qwen3LoopbackError,
    Qwen3PipelineLoopbackWorker,
    sign_loopback_message,
    validate_loopback_base_url,
    verify_loopback_message,
)
from qwen3_pipeline_sidecar import Qwen3PipelineSidecarSession, Qwen3SidecarError
from qwen3_pipeline_multisidecar import (
    Qwen3MultiSidecarError,
    Qwen3PipelineMultiSidecar,
    cleanup_qwen3_local_artifacts,
)
from gemma4_pipeline_multisidecar import (
    Gemma4MultiSidecarError,
    Gemma4PipelineMultiSidecar,
)
from gemma4_pipeline_sidecar import (
    Gemma4PipelineSidecarSession,
    Gemma4SidecarError,
)
from cluster_fence import ControlFence
from cluster_auto_role import AutoRoleController
from cluster_handoff import HandoffCoordinator

from execution_provider import (
    ModelIdentity as TaskModelIdentity,
    StageRequest as TaskProviderStageRequest,
    sanitize_result_metadata as sanitize_task_result_metadata,
)
from task_worker_adapter import (
    RELEASE_REASON_DISCONNECTED,
    RELEASE_REASON_HEARTBEAT_STALE,
    RemoteFullWorkerProvider,
    TaskWorkerControlPlane,
    remote_provider_id,
)
from task_worker_protocol import (
    PROTOCOL_VERSION as TASK_WORKER_PROTOCOL_VERSION,
    WorkerMessage,
    WorkerProtocolError,
    build_message as build_task_worker_message,
    canonical_message_bytes as task_worker_message_bytes,
    canonical_sha256 as task_worker_sha256,
    decode_message as decode_task_worker_message,
    worker_protocol_status,
)

# PyTorch is optional. Keep it out of the llama.cpp control-plane import path.
from torch_runtime import LazyTorch, require_torch, torch_available

torch = LazyTorch()

from config import (
    RUN_MODE, HEARTBEAT_INTERVAL,
    SERVER_IP, SERVER_PORT,
    NODE_ROLE, NODE_ID, MAX_NODES,
    PIPELINE_TIMEOUT, PIPELINE_MODEL_SYNC_TIMEOUT, PIPELINE_STEP_TIMEOUT,
    PIPELINE_QUEUE_POLL_INTERVAL,
    PIPELINE_QUEUE_MAX_SIZE, PIPELINE_QUEUE_RESULT_TTL,
    TASK_WORKER_EXPERIMENTAL_ENABLED,
)

logger = logging.getLogger(__name__)

# ★ 2026-10-07（DIST-NEXT-4）：Android presence 心跳间隔的唯一来源是
#   `scheduler_types.ANDROID_PRESENCE_HEARTBEAT_INTERVAL_SECONDS`；这里保留既有名字
#   （`scheduler_cluster` 通过 `_runtime_setting` 读它）。
ANDROID_HTTP_CLIENT_HEARTBEAT_INTERVAL_SECONDS = (
    ANDROID_PRESENCE_HEARTBEAT_INTERVAL_SECONDS
)
ANDROID_HTTP_CLIENT_LEASE_SECONDS = 120
ANDROID_HTTP_CLIENT_TIMEOUT_SECONDS = ANDROID_HTTP_CLIENT_LEASE_SECONDS
_LAYER_ASSIGNMENT_CACHE_VERSION = 3

# ★ 2026-10-07（DIST-NEXT-4）：启动期自检 —— 心跳间隔与容忍上限必须自洽，
#   否则直接以清晰原因失败（历史踩过「心跳 45s vs 上限 10s」的静默错配）。
assert_worker_liveness_thresholds(HEARTBEAT_INTERVAL)

from scheduler_task_worker import SchedulerTaskWorkerMixin, _TaskWorkerActiveAttempt
from scheduler_cluster import SchedulerClusterMixin
from scheduler_auto_role import SchedulerAutoRoleMixin
from scheduler_pipeline import SchedulerPipelineMixin
# ★ 2026-10-07（DIST-NEXT-7）：A1 隔离状态自证（独立 namespace，启动即见）。
from relay_a1_legacy import log_isolation_status

# Keep the current scheduler import surface explicit while the implementation
# is split into smaller modules. Private helpers listed here are compatibility
# hooks used by bootstrap and Android capability gates.
__all__ = [
    "Scheduler",

    "NodeInfo",
    "NodeState",
    "NodeRole",
    "_node_supports_forward_layers",
    "_bootstrap_api_port",
]

def _sample_pipeline_token_id(logits, temperature: float, top_p: float) -> int:
    """Sample one token with the same zero-temperature semantics as local inference.

    logits 的载体随主节点引擎而变：PyTorch 给张量，llama.cpp 给 numpy `[1, 1, n_vocab]`。
    这里统一折成 numpy 再走同一条数值路径，避免为两种载体写两套采样逻辑（口径分叉
    比多一次转换危险得多）。
    """
    import numpy as np

    if logits is None or getattr(logits, "ndim", None) != 3 or logits.shape[0] != 1:
        shape = getattr(logits, "shape", None)
        raise ValueError(f"流水线 logits 形状无效: {shape}")

    if hasattr(logits, "detach"):
        values = logits.detach().to(device="cpu", dtype=None).numpy()
    else:
        values = np.asarray(logits)
    # Local ModelManager uses do_sample=False when temperature <= 0. Keeping
    # that exact contract also avoids dividing fp16/bf16 logits by 1e-8,
    # which can create infinities and poison the CUDA context in multinomial.
    next_logits = values[:, -1, :].astype(np.float32)
    if float(temperature) <= 0:
        return int(np.argmax(next_logits, axis=-1)[0])

    scaled_logits = next_logits / max(float(temperature), 1e-5)
    if not bool(np.isfinite(scaled_logits).all()):
        raise RuntimeError("流水线 logits 包含 NaN/Inf，拒绝执行采样")
    shifted = scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    probs = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    if not bool(np.isfinite(probs).all()):
        raise RuntimeError("流水线采样概率包含 NaN/Inf")

    order = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, order, axis=-1)
    nucleus = min(1.0, max(0.0, float(top_p)))
    cumsum = np.cumsum(sorted_probs, axis=-1)
    cutoff = cumsum > nucleus
    cutoff[..., 1:] = cutoff[..., :-1].copy()
    cutoff[..., 0] = False
    filtered_probs = np.where(cutoff, 0.0, sorted_probs)
    probability_sum = np.sum(filtered_probs, axis=-1, keepdims=True)
    if not bool(np.isfinite(probability_sum).all()) or bool((probability_sum <= 0).any()):
        raise RuntimeError("流水线采样概率无有效候选 token")
    filtered_probs = filtered_probs / probability_sum
    # numpy 没有 multinomial：用逆变换采样，与 torch.multinomial 同分布。
    uniform = np.random.random(size=(filtered_probs.shape[0], 1))
    cumulative = np.cumsum(filtered_probs, axis=-1)
    picked = np.argmax(cumulative >= uniform, axis=-1)
    return int(order[0, picked[0]])

def _bootstrap_api_port(default: int = 8000, *, environment=None) -> int:
    """Return the master API port used for first-connect bootstrap."""
    environment = os.environ if environment is None else environment
    for name in ("QLH_BOOTSTRAP_API_PORT", "QLH_MASTER_API_PORT"):
        raw = (environment.get(name) or "").strip()
        if not raw:
            continue
        try:
            port = int(raw)
        except ValueError:
            logger.warning("%s=%r 不是有效端口，回退到 %s", name, raw, default)
            continue
        if 1 <= port <= 65535:
            return port
        logger.warning("%s=%r 超出端口范围，回退到 %s", name, raw, default)
    return default

def _is_auth_register_failure(reason: str) -> bool:
    text = reason or ""
    return any(marker in text for marker in (
        "认证失败", "HMAC", "签名不匹配", "集群密钥"
    ))

def _configured_node_id() -> str:
    try:
        from node_runtime import node_runtime
        return node_runtime.get_node_id()
    except Exception:
        return NODE_ID

def _sync_runtime_node_config(node_id: str = None, node_role: str = None) -> None:
    """Keep runtime node identity aligned with runtime config mutations.

    阶段 0.3：运行时状态由 node_runtime 单例持有（config.py 不再被写回）；
    模块级 NODE_ID/NODE_ROLE 同步更新以保持本模块内既有读取行为不变。
    """
    global NODE_ID, NODE_ROLE
    try:
        from node_runtime import node_runtime
    except Exception:
        node_runtime = None

    if node_id:
        NODE_ID = str(node_id)
        if node_runtime is not None:
            node_runtime.set_node_id(NODE_ID)
    if node_role:
        NODE_ROLE = str(node_role)
        if node_runtime is not None:
            node_runtime.set_node_role(NODE_ROLE)

def _node_supports_forward_layers(node: "NodeInfo") -> bool:
    """Admit layer work through the same concrete engine schema as hello."""
    return node_supports_forward_layers(node)



class Scheduler(
    SchedulerSidecarMixin, SchedulerTaskWorkerMixin,
    SchedulerAutoRoleMixin, SchedulerClusterMixin, SchedulerPipelineMixin,
):
    """
    主节点调度器

    负责:
    - 管理所有从节点状态
    - 接收前端推理请求，分发给流水线
    - 监控任务执行，处理异常
    - 控制流水线启停
    - 集成 TCP 服务端，接收从节点注册
    - 流水线请求队列（PipelineQueue）
    """

    def set_distributed_inference_enabled(self, enabled: bool) -> dict:
        if not enabled:
            self._cleanup_task_worker_prepare("local_inference_mode")
        return super().set_distributed_inference_enabled(enabled)

    def _runtime_setting(self, name: str):
        return self._runtime_settings[name]

    def _qwen3_multisidecar_factory(self):
        return Qwen3PipelineMultiSidecar

    def __init__(
        self,
        host: Optional["InferenceHost"] = None,
        callbacks: Optional["SchedulerCallbacks"] = None,
        *, runtime_config=None,
    ):
        self._runtime_config = runtime_config
        import config as configured
        self._runtime_settings = {**vars(configured), **globals()}
        if runtime_config is not None:
            self._runtime_settings.update({
            **runtime_config.legacy_settings(),
            "NODE_ID": runtime_config.node_id, "NODE_ROLE": runtime_config.node_role,
            "RUN_MODE": runtime_config.run_mode, "SERVER_IP": runtime_config.server_ip,
            "SERVER_PORT": runtime_config.server_port,
            "_configured_node_id": lambda: runtime_config.node_id,
            "_bootstrap_api_port": lambda: _bootstrap_api_port(environment=runtime_config.environment_snapshot()),
        })
        # 推理宿主默认使用全局单例；回调由 API composition root 以具名
        # Protocol bundle 注入，避免 scheduler 反向依赖 api_server。
        self._host = host if host is not None else get_model_host()
        self._control_fence: ControlFence | None = None
        self._auto_role_controller: AutoRoleController | None = None
        self._auto_role_binding_ready = False
        self._auto_role_bound_role = None
        self._auto_role_binding_event = threading.Event()
        self._auto_role_binding_lock = threading.Lock()
        self._auto_role_binding_thread = None
        self._handoff_coordinator: HandoffCoordinator | None = None
        self._callbacks = callbacks if callbacks is not None else getattr(
            self._host, "scheduler_callbacks", None,
        )
        self.nodes: dict[str, NodeInfo] = {}
        self._current_task: Optional[InferenceTask] = None
        self._infer_tasks: dict[str, InferenceTask] = {}
        self._task_lock = threading.Lock()
        self._running = False
        # ★ 2026-10-05：`start()` 的幂等守卫锁。双栈（`0.0.0.0` + `::`）下
        #   `uvicorn.Server` 的 lifespan 会**并发**调两次 `start()`，非原子的
        #   「检查后设置」两边都会通过 ⇒ 必须在锁内占位。
        self._start_lock = threading.Lock()
        # 启动期后台发现线程必须能被 stop() 立即唤醒，避免停止后仍发起连接。
        self._startup_cancel_event = threading.Event()
        self._network_identity_thread: Optional[threading.Thread] = None
        self._network_identity_lock = threading.Lock()
        self.on_task_complete: Optional[Callable] = None

        # TCP 服务端（分布式模式下启动）。回调线程通过
        # _tcp_callback_context 绑定触发事件的具体 server；普通控制面
        # 线程则读取 _tcp_server_default。
        self._tcp_server_default = None  # 延迟导入，避免循环依赖
        self._tcp_callback_context = threading.local()
        # TCP 客户端（从节点连接主节点后创建）。必须在此默认初始化，
        # 否则从未连接过主节点的实例在错误路径直接访问
        # self._tcp_client 会抛 AttributeError 而不是优雅返回 False
        self._tcp_client = None
        from qlh_core.bootstrap.model_control import build_model_control
        from qlh_core.adapters.local_store import RuntimeLocalStore
        self._model_control = build_model_control(RuntimeLocalStore(runtime_config) if runtime_config is not None else None)
        from qlh_core.bootstrap.execution import build_requests
        self._execution_app = build_requests(self, RuntimeLocalStore(runtime_config) if runtime_config is not None else None)
        from model_supply import ModelSupplyCoordinator
        from model_supply_worker import ModelSupplyWorker
        self._model_supply_worker = ModelSupplyWorker(self, root=(Path(runtime_config.state_dir) / "model-supply/received") if runtime_config is not None else None)
        from worker_prepare import WorkerPrepareController
        self._worker_prepare = WorkerPrepareController(self)
        self._model_supply = ModelSupplyCoordinator(self, root=(Path(runtime_config.state_dir) / "model-supply/operations") if runtime_config is not None else None)

        # 从节点：等待主节点推理结果
        self._client_pending_results: dict = {}
        self._client_pending_events: dict[str, threading.Event] = {}
        self._client_pending_lock = threading.Lock()

        # 主节点：转发请求取消和并发准入。队列仍负责实际推理串行化，
        # 信号量只限制等待队列结果的包装线程数量。
        self._forward_cancel_events: dict[tuple[str, str], threading.Event] = {}
        self._forward_cancel_lock = threading.Lock()
        self._forward_infer_slots = threading.BoundedSemaphore(
            max(1, min(PIPELINE_QUEUE_MAX_SIZE + 1, 32))
        )

        # 流水线推理状态（主节点侧）
        self._pipeline_results: dict = {}       # key → result data
        self._pipeline_events: dict = {}        # key → threading.Event
        self._pipeline_active_tasks: set[str] = set()
        self._pipeline_task_contracts: dict[str, dict] = {}
        self._chain_ack_state: dict = {}        # task_id → step → node_id → ack/error
        self._pipeline_lock = threading.Lock()
        self._nodes_lock = threading.RLock()    # Phase 2.1: 保护 self.nodes 并发读写（可重入）
        self._kv_cache_lock = threading.Lock()   # Phase 2.2: 保护 _kv_cache 并发读写
        self._kv_cache: dict = {}               # task_id → past_key_values（本节点层范围的 KV cache）
        self._layer_config_retry_state: dict[str, dict] = {}
        # ASSIGN-SOURCE-01: active wire contract, receipt and lifecycle live in
        # one registry. `_layer_config_pushed` remains a derived property.
        self._worker_assignments = WorkerAssignmentRegistry()
        self._layer_config_lock = threading.RLock()
        self._layer_config_push_lock = threading.Lock()
        # Model transitions temporarily invalidate the local descriptor. Do
        # not publish a transient release from concurrent status callbacks.
        self._layer_config_model_change_depth = 0
        self._layer_config_push_deferred = False
        self._layer_config_push_deferred_authoritative = False
        self._layer_config_push_deferred_require_distributed = False
        self._layer_execution_lock = threading.RLock()
        self._active_pipeline_task_ids: set[str] = set()
        self._pending_layer_config: Optional[tuple[str, dict]] = None
        self._layer_config_inflight: set[str] = set()
        self._last_layer_config_ack_payload: Optional[dict] = None
        self._active_layer_config: Optional[dict] = None
        self._layer_config_generation = 0
        self._layer_config_receive_sequence = 0
        self._latest_layer_config_receive_sequence = 0
        self._latest_layer_config_generation = 0
        self._latest_layer_assignment_id = ""
        self._pipeline_worker_reserved = False
        self._pipeline_worker_opted_out = False
        self._pipeline_worker_opt_out: set[str] = set()
        self._pipeline_load_transaction: Optional[dict] = None
        # Terminal model-runtime cleanup is independent from the one mutable
        # load transaction.  A disconnected worker may keep an old candidate
        # alive while later transactions continue on the remaining topology;
        # retain that debt until the exact abort/finalize receipt arrives.
        self._pipeline_model_cleanup_ledger: dict[str, dict] = {}
        # A persisted active transaction must be republished after a master
        # restart before distributed requests are admitted.
        self._pipeline_recovery_pending = False
        self._pipeline_recovery_failure = ""
        self._pipeline_recovery_state: Optional[dict] = None
        # Validation of restored obligations is independent of later write I/O.
        self._pipeline_recovery_validated = True
        self._pipeline_lifecycle_persist_ok = True
        # Qwen3 remains outside production runtime admission.  This isolated
        # state machine exercises the C2 lifecycle without network dispatch,
        # weight materialization, or full-model fallback.
        self._qwen3_pipeline_dry_run: Optional[Qwen3PipelineDryRunTransaction] = None
        self._qwen3_loopback_base_url = ""
        self._qwen3_loopback_workers: dict[str, Qwen3PipelineLoopbackWorker] = {}
        self._qwen3_loopback_ack_nonces: dict[str, str] = {}
        self._qwen3_local_chain: Optional[Qwen3PipelineMultiSidecar] = None
        self._qwen3_local_contract: Optional[dict] = None
        self._qwen3_local_parity: dict = {}
        self._qwen3_local_chain_lock = threading.RLock()
        self._qwen3_local_artifact_root_override: Optional[str] = None
        self._qwen3_network_handoff_transport = None
        self._qwen3_network_transfer_coordinator = None
        self._qwen3_artifact_transfer_runtime = None
        self._qwen3_peer_request_verifier = None
        # Gemma 4 Unified uses a distinct Transformers 5.17.0 sidecar.  This
        # local chain is an explicit development route and never changes the
        # production pipeline runtime allow-list.
        self._gemma4_local_chain: Optional[Gemma4PipelineMultiSidecar] = None
        self._gemma4_local_contract: Optional[dict] = None
        self._gemma4_local_chain_lock = threading.RLock()
        self._gemma4_local_artifact_root_override: Optional[str] = None
        self._gemma4_assignment_paths: dict[str, str] = {}
        self._gemma4_sidecar_python_override: Optional[str] = None
        self._model_runtime_contract_lock = threading.RLock()
        self._active_pipeline_capacity_plan: Optional[dict] = None
        self._pipeline_assignment_rollbacks: dict[str, dict] = {}
        # Recovery is control-plane state. A candidate is never published to
        # an executor before its capacity, artifact, and epoch gates pass.
        self._pipeline_reshard_coordinator: Optional[PipelineReshardCoordinator] = None
        self._pipeline_reshard_last_decision: Optional[dict] = None
        self._prepared_layer_configs: dict[str, dict] = {}
        # 同时到达的分布式请求都可能要求权威同步，必须用计数而非
        # 布尔值，避免前一个请求结束时把后一个请求降级为普通推送。
        self._authoritative_layer_sync_requests = 0
        self._local_pipeline_cancelled: set[str] = set()
        self._local_pipeline_cancelled_order: collections.deque = collections.deque()
        self._local_pipeline_steps: dict[str, int] = {}
        # Optional Transport v2 runtime factory.  It is unset by default so
        # all existing TCP clients keep the Legacy wire path unchanged.
        self._transport_runtime_factory: Optional[Callable[[str], object]] = None
        self._pipeline_accounted_tasks: set = set()  # 主节点侧：已完成记账的流水线任务
        self._pipeline_accounted_order: collections.deque = collections.deque()
        self._local_pipeline_counted_tasks: set = set()  # 从节点侧：已本地计数的流水线任务
        self._local_pipeline_error_tasks: set = set()    # 从节点侧：已本地计错的流水线任务
        self._local_pipeline_accounted_order: collections.deque = collections.deque()
        self._inference_lock = threading.Lock()  # GPU 推理互斥锁（防止并发执行）

        # ---- L5: 多节点日志聚合状态 ----
        self._pending_log_responses: dict = {}    # node_id → response data
        self._pending_log_events: dict = {}       # node_id → threading.Event
        self._pending_log_lock = threading.Lock()

        # ---- 当前执行请求上下文 ----
        self._pipeline_context = threading.local()

        # 最大节点数（可动态调整）
        self._max_nodes: int = MAX_NODES
        self._master_connect_lock = threading.Lock()
        self._role_transition_lock = threading.Lock()
        self._client_health_thread: Optional[threading.Thread] = None
        self._client_health_start_lock = threading.Lock()
        self._layer_config_retry_thread: Optional[threading.Thread] = None
        self._distributed_inference_enabled: Optional[bool] = None
        from qlh_core.control.participation import ParticipationService
        self._participation = ParticipationService(enabled=self._runtime_settings.get("DISTRIBUTED_INFERENCE_ENABLED", True))
        self._local_device_profile: Optional[dict] = {}
        self._runtime_layer_override: Optional[list] = None
        self._ha_state_lock = threading.RLock()
        self._ha_state_loaded = False
        self._ha_state: dict = {
            "spare_master": None,
            "spare_master_active": False,
            "pending_new_master_id": "",
            "transfer_logs": [],
            "spare_master_logs": [],
        }
        self._task_worker_control = TaskWorkerControlPlane(
            health_timeout_seconds=task_worker_control_plane_health_timeout_seconds(
                HEARTBEAT_INTERVAL,
            ),
        )
        # ★ 2026-10-07（DIST-NEXT-7）：启动即自证 A1 隔离状态（`event=relay_a1_isolation`）——
        #   「A1 会不会被调度选中」不再需要逐个开关推断。
        log_isolation_status(self.get_relay_a1_status())
        self._task_worker_refresh_lock = threading.Lock()
        self._task_worker_refresh_requested = False
        self._task_worker_refresh_generation = 0
        self._remote_task_worker_providers: dict[
            str, RemoteFullWorkerProvider
        ] = {}
        self._task_worker_stage_lock = threading.RLock()
        self._task_worker_active_attempts: dict[
            str, _TaskWorkerActiveAttempt
        ] = {}
        # Short-lived terminal identities let a cancel that crossed a final
        # result receive an idempotent acknowledgement instead of an
        # unknown_attempt rejection.
        self._task_worker_completed_attempts: dict[
            str, tuple[_TaskWorkerActiveAttempt, float]
        ] = {}
        self._task_worker_seen_messages: dict[
            str, tuple[str, list[dict]]
        ] = {}
        self._task_worker_seen_order: collections.deque[str] = (
            collections.deque()
        )

        # 有界 FIFO 请求队列

        self._execution_app.requests.capacity = PIPELINE_QUEUE_MAX_SIZE

    def set_transport_runtime_factory(self, factory: Optional[Callable[[str], object]]) -> None:
        """Inject a per-peer Transport v2 bridge for local/production rollout."""

        if factory is not None and not callable(factory):
            raise TypeError("transport runtime factory must be callable or None")
        self._transport_runtime_factory = factory

    def _new_transport_runtime(self, node_id: str) -> object | None:
        factory = self._transport_runtime_factory
        if not callable(factory):
            return None
        try:
            return factory(str(node_id))
        except Exception:
            logger.warning("transport runtime factory failed for %s", node_id, exc_info=True)
            return None

    def _transport_runtime_kwargs(self, node_id: str) -> dict[str, object]:
        """Return an opt-in constructor kwarg without breaking test doubles."""

        runtime = self._new_transport_runtime(node_id)
        return {"transport_runtime": runtime} if runtime is not None else {}

    # ================================================================
    # 启动 / 停止
    # ================================================================

    def start(self, host: str = None, port: int = None) -> None:
        """Start once and release the reservation if initialization fails."""
        with self._start_lock:
            if self._running:
                logger.info("scheduler already running; ignoring duplicate start")
                return
            self._running = True
        try:
            self._worker_prepare.reopen()
            with self._task_worker_stage_lock:
                self._task_worker_io_stopped = False
            self._start_impl(host=host, port=port)
        except Exception:
            try:
                self.stop()
            except Exception:
                logger.warning("scheduler startup rollback failed", exc_info=True)
                self._running = False
            raise

    def _start_impl(self, host: str = None, port: int = None) -> None:
        """
        启动调度器。

        初始化节点状态；若为分布式模式，启动 TCP 服务端监听。

        主节点启动后自动:
        - 检测当前可广告地址（已登录时优先 Tailscale，否则可达 LAN）
        - 将物理 MAC 身份保存到用户自持的主节点 SQLite
        - 通过本机 bootstrap 配置和 Tailnet 提供发现，不依赖远端数据库

        Args:
            host: TCP 监听地址（默认 0.0.0.0，接受所有接口连接）
            port: TCP 监听端口（默认 config.SERVER_PORT）
        """
        # ★ 2026-10-05（两段实测根因）：`run_api_servers()` 在通配地址下会为
        #   `0.0.0.0` 与 `::` **各起一个 `uvicorn.Server`**（共享同一 `app`），
        #   于是 `_lifespan` 执行两次、本方法被调两次。此前没有任何幂等保护 ⇒
        #   进程内出现**两个 `TCPServer` 实例**同时监听 8888：`_tcp_server_default`
        #   被第二次覆盖（setter 见本文件 `_tcp_server` 属性），而 worker 的连接
        #   只被其中一个实例 accept。请求线程读的是 `_tcp_server_default`，
        #   因此 `clients` 为空 ⇒ 一连串下游症状：
        #     `_get_pipeline_readiness` 的 `connected` 为空 ⇒
        #     `worker_tcp_disconnected`；assignment 拿不到
        #     `execution="stage_offer_v3"`；capacity 求解器从未被调用而退化成
        #     `simple_weight` 分配（`Layer 13-20` / `15-24` 即由此而来）。
        #   日志判据：10-05 每次启动都有**同毫秒两条** `TCP服务端启动`，
        #   而 10-03 三机验收时只有一条。
        #   已在运行时直接返回，避免重复 init_nodes / create_server。
        #   ★ 注意：两次 `start()` 是**并发**的（日志时间戳同毫秒），非原子的
        #     「检查后设置」两边都会通过 ⇒ 必须在 `_start_lock` 内占位。
        self._startup_cancel_event.clear()

        # P4.5 HA-ROLE-AUTO-01: construct and start the quorum-backed role
        # controller at the composition root. Missing key/peer configuration
        # is reduced to read-only instead of falling back to a static master.
        if self._runtime_setting("NODE_ROLE") == "auto":
            runtime = getattr(self, "_auto_role_runtime", None)
            if getattr(self, "_auto_role_controller", None) is None:
                try:
                    from cluster_auto_role_runtime import install_auto_role_controller

                    runtime = install_auto_role_controller(self, self._control_fence, **({"environment": self._runtime_config.environment_snapshot()} if self._runtime_config is not None else {}))
                except Exception as exc:
                    from cluster_auto_role import AutoRoleController

                    authority = self._control_fence.authority if self._control_fence is not None else None
                    self.set_auto_role_controller(
                        AutoRoleController(
                            str(self._runtime_setting("NODE_ID") or "auto-node"),
                            mode="auto",
                            authority=authority,
                            fence=self._control_fence,
                        )
                    )
                    runtime = None
                    logger.error(
                        "auto role configuration unavailable; entering read-only: %s",
                        getattr(exc, "code", type(exc).__name__),
                    )
            available = runtime.available_voter_ids if runtime is not None else None
            decision = self.start_auto_role(available_voter_ids=available)
            logger.info(
                "event=auto_role_start accepted=%s state=%s runtime_role=%s reason=%s",
                decision.get("accepted"), decision.get("state"),
                decision.get("runtime_role"), decision.get("reason"),
            )

        self._model_control.start()
        self._execution_app.requests.start()
        self.init_nodes()
        if self._effective_role() == "master":
            self._load_pipeline_recovery_state()
            self._restore_pipeline_model_for_recovery()

        # Reconcile only the active model's assignment cache.  This is a
        # local, bounded cleanup and never touches the user's full model tree.
        try:
            active_model_id = str(
                getattr(self._host, "active_model_id", "")
                or getattr(self._host, "_active_model_id", "")
                or ""
            )
            if active_model_id:
                from model_sync import reconcile_pipeline_assignment_cache

                reconcile_pipeline_assignment_cache(active_model_id)
        except Exception:
            logger.warning("pipeline assignment cache reconcile failed", exc_info=True)

        # 存储检测到的局域网 IP 和 MAC 地址
        self._lan_ip: str = ""
        self._mac_addresses: list[str] = []
        self._master_identity_verified: bool = False
        self._master_identity_reason: str = ""

        if self._runtime_setting("RUN_MODE") == "distributed":
            from transport_port import create_server

            # 绑定到 0.0.0.0 接受所有接口连接（而非占位符 192.168.x.x）
            bind_host = host or "0.0.0.0"
            actual_port = self._runtime_setting("SERVER_PORT") if port is None else port

            try:
                server = create_server(bind_host, actual_port)
                self._tcp_server = server
                if self._control_fence is not None:
                    server.set_control_fence(self._control_fence)
                server.start(
                    on_message=self._bind_tcp_server_callback(
                        server, self._on_tcp_message,
                    ),
                    on_disconnect=self._bind_tcp_server_callback(
                        server, self._on_tcp_disconnect,
                    ),
                    on_registration_confirmed=self._bind_tcp_server_callback(
                        server, self._on_tcp_registration_confirmed,
                    ),
                )
            except Exception as e:
                self._tcp_server = None
                logger.error(
                    "分布式 TCP 监听启动失败 (%s:%s): %s；"
                    "继续提供本地主节点全模型推理",
                    bind_host, actual_port, e, exc_info=True,
                )

            # 检测实际局域网 IP 和 MAC 地址
            if self._effective_role() == "master":
                # Network address and MAC identity are populated by the
                # post-startup worker below.
                logger.info(f"调度器已启动（分布式模式），监听 {bind_host}:{actual_port}，局域网 IP: {self._lan_ip}，MAC: {self._mac_addresses}")

                # 验证主节点身份（MAC 匹配）
                self._master_identity_reason = "startup_deferred"

                # ★ MAC 不匹配时的处理策略
                if self._master_identity_reason == "mac_mismatch":
                    # 尝试在本机 bootstrap 配置或 Tailnet 中发现已确认的主节点。
                    discovery = self.discover_master()
                    if discovery.get("found"):
                        # bootstrap/Tailnet 发现已确认主节点 → 自动切换为从节点
                        stale_note = "（心跳过期，IP 可能已变更）" if discovery.get("stale") else ""
                        logger.warning(
                            f"⛔ 主节点身份验证失败！本机 MAC 与本地 SQLite 记录不匹配。\n"
                            f"   已发现主节点 ({discovery['master_host']}:{discovery['master_port']}){stale_note}，\n"
                            f"   自动切换为从节点模式并尝试连接..."
                        )
                        # 启动后台线程处理切换（避免阻塞 start()）
                        threading.Thread(
                            target=self._auto_switch_to_client,
                            args=(discovery["master_host"], discovery["master_port"]),
                            name="auto-switch-client",
                            daemon=True,
                        ).start()
                    else:
                        # 未发现其他主节点：保留本机服务，但不覆盖已有身份记录。
                        logger.error(
                            f"⛔ 主节点身份验证失败 — 拒绝覆盖本机 SQLite 身份记录！\n"
                            f"   本机 MAC 与已记录身份不匹配，且未发现其他主节点。\n"
                            f"   如需更换主节点机器，请先在原主节点的后台管理中"
                            f"使用「重置主节点身份」功能。\n"
                            f"   当前将以单机模式运行（不覆盖本机 SQLite 身份）。"
                        )
                else:
                    if self._tcp_server and self._tcp_server._running:
                        port_text = self._tcp_server.port
                        if "master" in self.nodes:
                            self.nodes["master"].address = f"{self._lan_ip}:{port_text}"
                        logger.info(
                            "主节点本地身份已就绪，广告地址 %s:%s；"
                            "从节点通过 bootstrap 配置和 Tailnet 发现",
                            self._lan_ip,
                            port_text,
                        )
                    else:
                        logger.warning(
                            "主节点 TCP 未监听，跳过主节点地址广告；"
                            "本地推理仍可用"
                        )
                # 检查是否需要向备用主节点发送接管通知（转让后新主节点启动）
                threading.Thread(
                    target=self.deactivate_spare_master_on_startup,
                    name="spare-deactivate",
                    daemon=True,
                ).start()
            else:
                logger.info(f"调度器已启动（分布式模式，从节点），监听 {bind_host}:{actual_port}")
                # 从节点启动后台线程：监控主节点健康状态 + 自动重连
                self._start_client_health_monitor()
                # 启动后尝试自动发现并连接主节点
                threading.Thread(
                    target=self._auto_connect_on_startup,
                    name="auto-connect-startup",
                    daemon=True,
                ).start()
        else:
            logger.info("调度器已启动（单机模式）")

        if self._runtime_setting("RUN_MODE") == "distributed" and self._effective_role() == "master":
            self._start_deferred_network_identity()

        # 启动流水线请求队列（仅主节点，FIFO 串行）
        if self._effective_role() == "master":
            self._execution_app.requests.start()
            logger.info("流水线请求队列已就绪")

        # 已经是 client 的节点由上面的 auto-connect 唯一路径处理；这里只处理
        # 尚未确认身份、可能需要从 provisional master 切换的节点。
        if (self._effective_role() == "master"
                and self._master_identity_reason != "startup_deferred"
                and self.can_join_existing_master()):
            threading.Thread(
                target=self._auto_join_tailnet_master_on_startup,
                name="tailnet-master-discovery",
                daemon=True,
            ).start()

        self._auto_role_bound_role = self._effective_role()
        self._auto_role_binding_ready = True
        runtime = getattr(self, "_auto_role_runtime", None)
        if runtime is not None:
            runtime.tcp_port = self._tcp_server.port if self._tcp_server is not None else 0
            runtime.start_lease_renewal(self._auto_role_controller, on_decision=self._on_auto_role_decision)

    def _start_deferred_network_identity(self) -> None:
        """Schedule network address, identity, and discovery work after start."""
        with self._network_identity_lock:
            thread = self._network_identity_thread
            if thread is not None and thread.is_alive():
                return
            self._network_identity_thread = threading.Thread(
                target=self._initialize_network_identity,
                name="network-identity-startup",
                daemon=True,
            )
            self._network_identity_thread.start()

    def _initialize_network_identity(self) -> None:
        """Complete distributed identity without delaying local service start."""
        if self._startup_cancel_event.is_set() or not self._running:
            return
        try:
            from transport_port import detect_lan_ip, get_mac_addresses

            self._lan_ip = detect_lan_ip()
            self._mac_addresses = get_mac_addresses()
            if self._startup_cancel_event.is_set() or not self._running:
                return

            if self._effective_role() != "master":
                return
            if self._auto_role_controller is not None and self._auto_role_controller.mode == "auto":
                if self._tcp_server is not None:
                    with self._nodes_lock:
                        if "master" in self.nodes:
                            self.nodes["master"].address = f"{self._lan_ip}:{self._tcp_server.port}"
                return
            self._verify_master_identity()
            if self._master_identity_reason == "mac_mismatch":
                discovery = self.discover_master()
                if discovery.get("found"):
                    logger.warning(
                        "master identity mismatch; switching to discovered master "
                        "%s:%s",
                        discovery["master_host"], discovery["master_port"],
                    )
                    threading.Thread(
                        target=self._auto_switch_to_client,
                        args=(discovery["master_host"], discovery["master_port"]),
                        name="auto-switch-client",
                        daemon=True,
                    ).start()
                else:
                    logger.error(
                        "master identity mismatch; no confirmed master discovered",
                    )
            elif self._tcp_server and self._tcp_server._running:
                if "master" in self.nodes:
                    self.nodes["master"].address = (
                        f"{self._lan_ip}:{self._tcp_server.port}"
                    )

            if (self._master_identity_reason != "mac_mismatch"
                    and self._effective_role() == "master"
                    and self.can_join_existing_master()):
                threading.Thread(
                    target=self._auto_join_tailnet_master_on_startup,
                    name="tailnet-master-discovery",
                    daemon=True,
                ).start()
        except Exception as exc:
            self._master_identity_reason = "network_probe_failed"
            logger.warning("deferred network identity initialization failed: %s", exc)

    def stop(self) -> None:
        """停止调度器"""
        self._running = False
        self._auto_role_binding_ready = False
        self._startup_cancel_event.set()
        self._execution_app.requests.close()
        self._cleanup_task_worker_prepare("worker_prepare_stopped", close=True)
        runtime = getattr(self, "_auto_role_runtime", None)
        if runtime is not None:
            runtime.stop_lease_renewal()
        self._stop_auto_role_binding()
        controller = getattr(self, "_auto_role_controller", None)
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                logger.debug("停止自动主节点控制器失败", exc_info=True)
        tcp_client = getattr(self, "_tcp_client", None)
        if tcp_client is not None:
            tcp_client.on_disconnect = None
            try:
                tcp_client.disconnect()
            except Exception:
                logger.debug("停止主节点连接失败", exc_info=True)
        if self._tcp_server:
            self._tcp_server.stop()
        logger.info("调度器已停止")

    @property
    def _tcp_server(self):
        """Return the server bound to the current TCP callback thread.

        The default is used by ordinary scheduler work.  Receive callbacks
        use a thread-local binding supplied when each server is registered,
        so a second server cannot redirect an existing scheduler instance.
        """
        callback_server = getattr(self._tcp_callback_context, "server", None)
        return callback_server or self._tcp_server_default

    @_tcp_server.setter
    def _tcp_server(self, value):
        self._tcp_server_default = value

    def _bind_tcp_server_callback(self, server, callback):
        """Bind one control-plane event to its explicit TCPServer source."""
        def bound_callback(*args, **kwargs):
            previous = getattr(self._tcp_callback_context, "server", None)
            self._tcp_callback_context.server = server
            try:
                return callback(*args, **kwargs)
            finally:
                if previous is None:
                    try:
                        del self._tcp_callback_context.server
                    except AttributeError:
                        pass
                else:
                    self._tcp_callback_context.server = previous

        return bound_callback

    @property
    def inference_host(self) -> "InferenceHost":
        """Return the host selected for this scheduler instance."""
        return self._host

    @property
    def inference_callbacks(self) -> Optional["SchedulerCallbacks"]:
        """Return the explicit callback bundle selected for this scheduler."""
        return self._callbacks

    def configure_callbacks(self, callbacks: "SchedulerCallbacks") -> None:
        """Set one complete callback bundle; individual callback ordering is irrelevant."""
        if callbacks is None:
            raise TypeError("scheduler callbacks are required")
        self._callbacks = callbacks

    def _require_callbacks(self) -> "SchedulerCallbacks":
        callbacks = self._callbacks
        if callbacks is None:
            callbacks = getattr(self._host, "scheduler_callbacks", None)
            self._callbacks = callbacks
        if callbacks is None:
            raise RuntimeError("scheduler callbacks are not configured")
        return callbacks

    # ================================================================
    # 节点管理
    # ================================================================

    def _effective_role(self) -> str:
        """
        返回当前节点的有效角色。

        正常情况返回 scheduler 模块持有的运行时 NODE_ROLE；若 MAC 不匹配
        时自动切换到 client 模式，则返回 "client"（通过 _role_override 覆盖）。
        node_config/bootstrap 会同步这个模块级值，不能回读启动时已经过期的
        config.NODE_ROLE。
        """
        configured_role = self._runtime_setting("NODE_ROLE")
        controller = getattr(self, "_auto_role_controller", None)
        if configured_role == "auto":
            return controller.runtime_role if controller is not None else "client"
        override = getattr(self, '_role_override', None)
        if override:
            return override
        if controller is not None:
            return controller.runtime_role
        return configured_role

    # ================================================================
    # 动态模型分层
    # ================================================================

    @staticmethod
    def _gpu_is_integrated(gpu: dict) -> bool:
        """Compatibility facade for GPU classification."""
        return _layer_plan.gpu_is_integrated(gpu)

    @classmethod
    def _select_scoring_gpu(cls, device_info: dict) -> dict:
        """Compatibility facade for scoring GPU selection."""
        return _layer_plan.select_scoring_gpu(
            device_info, gpu_is_integrated_fn=cls._gpu_is_integrated,
        )

    @staticmethod
    def _node_is_island_gateway(device_info: dict) -> bool:
        """Compatibility facade for island-gateway profile detection."""
        return _layer_plan.node_is_island_gateway(device_info)

    def _compute_node_weight(self, device_info: dict, *, execution_device: str | None = None) -> float:
        """Compatibility facade for device-profile scoring."""
        return _layer_plan.compute_node_weight(
            device_info,
            select_gpu=self._select_scoring_gpu,
            gpu_is_integrated_fn=self._gpu_is_integrated,
            execution_device=execution_device,
            debug=logger.debug,
        )

    def _sync_node_rtt(self, node_id: str, tcp_client) -> None:
        """
        从 TCP 客户端同步 RTT 到 NodeInfo。

        每次心跳后调用，将 TCP 层测量的 RTT 同步到节点信息中，
        供分层算法和前端展示使用。
        """
        with self._nodes_lock:
            node = self.nodes.get(node_id)
            if node is None:
                return
            node.last_heartbeat = time.time()
            avg_rtt = getattr(tcp_client, 'avg_rtt_ms', 0.0)
            if avg_rtt > 0:
                node.last_rtt_ms = avg_rtt  # 当前 EWMA 值
                node.avg_rtt_ms = avg_rtt

    def _get_node_vram_mb(self, node_id: str) -> float:
        """
        从节点设备画像中提取可用显存 (MB)。

        优先级: GPU 专用显存 > 系统可用 RAM（CPU/集显模式）
        """
        with self._nodes_lock:
            node = self.nodes.get(node_id)
        if not node or not node.device_info:
            return 0.0
        gpu = self._select_scoring_gpu(node.device_info)
        ram = node.device_info.get("ram", {})
        uses_dedicated_vram = bool(
            isinstance(gpu, dict)
            and gpu.get("cuda_available", False)
            and not self._gpu_is_integrated(gpu)
            and gpu.get("vram_total_gb", 0) > 0
        )
        if uses_dedicated_vram:
            return gpu["vram_total_gb"] * 1024  # GB → MB
        if isinstance(ram, dict):
            return ram.get("available_gb", 0) * 1024
        return 0.0

    def _check_vram_constraint(self, node_id: str, layers_count: int,
                                has_embedding: bool = False,
                                has_lm_head: bool = False) -> tuple:
        """
        检查节点是否有足够显存承载分配的层范围。

        Args:
            node_id: 节点 ID
            layers_count: 分配的 Transformer 层数
            has_embedding: 是否包含 Embedding 层
            has_lm_head: 是否包含 LM Head

        Returns:
            (ok: bool, needed_mb: float, available_mb: float)
        """
        vram_available = self._get_node_vram_mb(node_id)
        if vram_available <= 0:
            return (True, 0, 0)  # 无法判断 → 放行

        # 根据当前模型结构和分层加载的真实精度估算内存。CUDA 分层使用
        # FP16，CPU/集显分层使用 FP32，都不能套用全模型 int4/int8 缩放系数。
        from config import (
            MIN_VRAM_PER_LAYER_MB, EMBEDDING_VRAM_MB, LM_HEAD_VRAM_MB,
            SAFE_VRAM_MARGIN, LAYER_VRAM_FACTOR, QUANT_TYPE,
        )
        layer_mb, embedding_mb, lm_head_mb = self._get_layer_memory_estimate_mb(
            node_id=node_id,
            fallback=(MIN_VRAM_PER_LAYER_MB, EMBEDDING_VRAM_MB, LM_HEAD_VRAM_MB),
            quant_factors=LAYER_VRAM_FACTOR,
            configured_quant=QUANT_TYPE,
        )
        vram_needed = layers_count * layer_mb
        if has_embedding:
            vram_needed += embedding_mb
        if has_lm_head:
            vram_needed += lm_head_mb
        vram_needed *= SAFE_VRAM_MARGIN

        ok = vram_available >= vram_needed
        return (ok, round(vram_needed, 1), round(vram_available, 1))

    def _get_layer_memory_estimate_mb(self, node_id: str,
                                      fallback: tuple,
                                      quant_factors: dict,
                                      configured_quant: str) -> tuple:
        """Return per-layer, embedding and LM-head memory in MiB."""
        manager = self._host
        model_config = getattr(getattr(manager, "model", None), "config", None)

        with self._nodes_lock:
            node = self.nodes.get(node_id)
        gpu = self._select_scoring_gpu(node.device_info if node else {})
        uses_cuda = bool(
            gpu.get("cuda_available", False)
            and not self._gpu_is_integrated(gpu)
        ) if isinstance(gpu, dict) else False
        bytes_per_parameter = 2 if uses_cuda else 4

        descriptor = {}
        get_descriptor = getattr(manager, "get_pipeline_descriptor", None)
        if callable(get_descriptor):
            try:
                descriptor = get_descriptor() or {}
            except Exception:
                logger.debug("读取流水线模型描述器失败，继续使用结构估算", exc_info=True)
        layer_bytes = descriptor.get("layer_weight_bytes") or []
        component_bytes = descriptor.get("component_weight_bytes") or {}
        if layer_bytes:
            # The artifact byte counts are exact for its source dtype. CPU layer
            # loading widens FP16/BF16 to FP32; CUDA preserves the source width.
            source_bits_factor = 1.0
            if not uses_cuda:
                source_bits_factor = 2.0
            mib = 1024.0 * 1024.0
            average_layer_mb = (
                sum(int(value) for value in layer_bytes) / len(layer_bytes)
                * source_bits_factor / mib
            )
            embedding_mb = (
                int(component_bytes.get("embedding", 0) or 0)
                * source_bits_factor / mib
            )
            lm_head_bytes = int(component_bytes.get("lm_head", 0) or 0)
            if not lm_head_bytes and descriptor.get("tie_word_embeddings", False):
                lm_head_bytes = int(component_bytes.get("embedding", 0) or 0)
            lm_head_mb = lm_head_bytes * source_bits_factor / mib
            if average_layer_mb > 0:
                return average_layer_mb, embedding_mb, lm_head_mb

        if model_config is None and manager is not None:
            model_path = getattr(manager, "_model_path", "") or ""
            config_path = os.path.join(model_path, "config.json")
            if os.path.isfile(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as handle:
                        model_config = type("ConfigView", (), json.load(handle))()
                except (OSError, TypeError, ValueError):
                    model_config = None

        if getattr(model_config, "model_type", "") == "qwen2":
            try:
                hidden = int(getattr(model_config, "hidden_size"))
                intermediate = int(getattr(model_config, "intermediate_size"))
                attention_heads = int(getattr(model_config, "num_attention_heads"))
                kv_heads = int(getattr(model_config, "num_key_value_heads", attention_heads))
                vocab = int(getattr(model_config, "vocab_size"))
                head_dim = hidden // attention_heads
                attention_params = (
                    hidden * hidden * 2
                    + hidden * kv_heads * head_dim * 2
                )
                mlp_params = hidden * intermediate * 3
                norm_params = hidden * 2
                mib = 1024.0 * 1024.0
                layer_mb = (
                    attention_params + mlp_params + norm_params
                ) * bytes_per_parameter / mib
                io_mb = vocab * hidden * bytes_per_parameter / mib
                return layer_mb, io_mb, io_mb
            except (TypeError, ValueError, ZeroDivisionError, AttributeError):
                logger.debug("读取 Qwen2 模型结构内存参数失败，使用回退估算", exc_info=True)

        quant = getattr(manager, "quant_type", None) or configured_quant
        # CPU workers materialize FP32 weights; fallback constants use FP16 as baseline.
        factor = quant_factors.get(quant, 1.0) if uses_cuda else 2.0
        return tuple(float(value) * factor for value in fallback)

    def _layer_assignment_cache_key(self, total_layers: int) -> str:
        """Bind dynamic cache to executable nodes, profiles, model and algorithm."""
        with self._layer_config_lock:
            opted_out = set(self._pipeline_worker_opt_out)
        with self._nodes_lock:
            nodes = []
            for node_id, info in self.nodes.items():
                if not _node_supports_forward_layers(info):
                    continue
                relay_for_worker = getattr(self, "_relay_segment_for_worker", None)
                is_relay_worker = (
                    callable(relay_for_worker)
                    and relay_for_worker(node_id) is not None
                )
                if node_id in opted_out and not is_relay_worker:
                    continue
                # 孤岛网关不参与层拆分，缓存键同步排除保持一致
                if self._node_is_island_gateway(info.device_info):
                    continue
                if info.role != NodeRole.MASTER and not info.is_available():
                    continue
                nodes.append({
                    "node_id": node_id,
                    "role": str(info.role),
                    "device_info": info.device_info or {},
                })
        manager = self._host
        # ★ 空路径必须**保持为空**：`os.path.abspath("")` 返回当前工作目录，
        #   会让下面的 `os.walk` 递归整个工作树（实测单次 200s+，把
        #   `/api/cluster/config` 拖成分钟级；见 `已知问题记录.md` #111）。
        raw_model_path = (
            (getattr(manager, "_model_path", "") or "") if manager else ""
        )
        model_path = os.path.abspath(raw_model_path) if raw_model_path else ""
        weight_fingerprint = []
        if model_path and os.path.isdir(model_path):
            try:
                for root, _dirs, files in os.walk(model_path):
                    for name in sorted(files):
                        if not name.lower().endswith((".safetensors", ".bin")):
                            continue
                        path = os.path.join(root, name)
                        stat = os.stat(path)
                        weight_fingerprint.append((
                            os.path.relpath(path, model_path).replace(os.sep, "/"),
                            stat.st_size,
                            stat.st_mtime_ns,
                        ))
            except OSError:
                weight_fingerprint = []
        payload = {
            "version": _LAYER_ASSIGNMENT_CACHE_VERSION,
            "total_layers": int(total_layers),
            "model_id": getattr(manager, "active_model_id", "") if manager else "",
            "model_path": model_path,
            "quant_type": getattr(manager, "quant_type", "") if manager else "",
            "weight_fingerprint": weight_fingerprint,
            "nodes": sorted(nodes, key=lambda item: item["node_id"]),
        }
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _get_total_model_layers(self) -> int:
        """Return the active PyTorch model's real decoder-layer count."""
        from config import TOTAL_MODEL_LAYERS

        manager = self._host
        if manager is not None:
            get_descriptor = getattr(manager, "get_pipeline_descriptor", None)
            if callable(get_descriptor):
                try:
                    descriptor_layers = int(
                        (get_descriptor() or {}).get("total_layers", 0) or 0
                    )
                    if descriptor_layers > 0:
                        return descriptor_layers
                except (TypeError, ValueError):
                    pass
                except Exception:
                    logger.debug("读取流水线描述器层数失败", exc_info=True)
            for value in (
                getattr(manager, "_total_model_layers", 0),
                getattr(getattr(getattr(manager, "model", None), "config", None),
                        "num_hidden_layers", 0),
            ):
                try:
                    if int(value) > 0:
                        return int(value)
                except (TypeError, ValueError):
                    pass

            model_path = getattr(manager, "_model_path", "") or ""
            config_path = os.path.join(model_path, "config.json")
            if os.path.isfile(config_path):
                try:
                    import json

                    with open(config_path, "r", encoding="utf-8") as handle:
                        value = int(json.load(handle).get("num_hidden_layers", 0))
                    if value > 0:
                        return value
                except (OSError, TypeError, ValueError):
                    logger.debug("读取当前模型层数失败: %s", config_path, exc_info=True)
        return TOTAL_MODEL_LAYERS

    def _get_active_pipeline_model_info(
        self, *, descriptor: Optional[dict] = None,
        model_path: str = "", quant_type: str = "",
    ) -> dict:
        """Describe the master's pipeline artifact — PyTorch **or** llama.cpp.

        不需要完整加载模型：两条引擎路径各自给出描述器，本方法只做形状归一与
        `PIPELINE_RUNTIME_MODEL_TYPES` 判定。GGUF 侧由
        `LlamaCppEngine.get_pipeline_descriptor()` 读文件头提供。
        """
        manager = self._host
        if descriptor is None:
            if not manager or not runtime_supports(manager, Capability.FORWARD_LAYERS):
                return {}
            get_descriptor = getattr(manager, "get_pipeline_descriptor", None)
            if not callable(get_descriptor):
                return {}
            try:
                descriptor = get_descriptor() or {}
            except Exception:
                logger.warning("读取主节点流水线模型描述器失败", exc_info=True)
                return {}
        else:
            descriptor = dict(descriptor)
        if not descriptor.get("pipeline_runtime_supported", False):
            return {}
        model_path = os.path.abspath(
            model_path
            or getattr(manager, "_full_model_path", "")
            or getattr(manager, "_model_path", "")
            or ""
        )
        # 目录（PyTorch artifact）或**单文件**（GGUF）都接受。llama.cpp 路径用「裁层
        # GGUF」表达层段，不是一个装 safetensors 的目录 —— 只认目录会把去 torch 化的
        # 边缘主节点挡在流水线之外。
        if not model_path or not (
            os.path.isdir(model_path) or os.path.isfile(model_path)
        ):
            return {}
        model_type = str(descriptor.get("model_type", "") or "").lower()
        # ★ #31 M2：走单一事实来源（硬编码会让 hybrid 在这里返回 `{}` ⇒ master 静默不推层配置）
        if model_type not in PIPELINE_RUNTIME_MODEL_TYPES:
            return {}
        return {
            "model_id": descriptor.get("model_id")
            or getattr(manager, "active_model_id", "") or "",
            "model_path": model_path,
            "model_sha256": descriptor.get("model_sha256")
            or self._get_master_model_sha256(),
            "model_type": model_type,
            "total_layers": int(descriptor["total_layers"]),
            "quant_type": quant_type or getattr(manager, "quant_type", "") or "",
            "inspection_mode": descriptor.get("inspection_mode", ""),
            "weight_bytes": int(descriptor.get("weight_bytes", 0) or 0),
        }

    def compute_layer_assignment(self, nodes: list = None) -> list:
        """
        根据节点硬件配置动态计算模型分层方案。

        按已声明的执行能力和容量分配连续层段。

        Args:
            nodes: 可选，指定节点列表；若为 None 则使用 self.nodes 中所有节点

        Returns:
            [{node_id, role, start_layer, end_layer, layers_count,
              has_embedding, has_lm_head, score}]
        """
        from config import (
            QUANT_TYPE, MIN_VRAM_PER_LAYER_MB, EMBEDDING_VRAM_MB,
            LM_HEAD_VRAM_MB, LAYER_VRAM_FACTOR, SAFE_VRAM_MARGIN,
        )

        total_layers = self._get_total_model_layers()

        # 收集明确声明 forward_layers 能力的节点；未声明能力的 Android
        # 继续按旧行为排除，避免把 HTTP 客户端误当成层工作器。
        with self._layer_config_lock:
            opted_out = set(self._pipeline_worker_opt_out)
        relay_for_worker = getattr(self, "_relay_segment_for_worker", None)

        def _is_relay_worker(node_id: str) -> bool:
            return (
                callable(relay_for_worker)
                and relay_for_worker(node_id) is not None
            )

        if nodes is None:
            # Phase 2.1: 快照 self.nodes 后解锁迭代，防止 TCP 回调并发修改 dict
            with self._nodes_lock:
                nodes_snapshot = list(self.nodes.items())
            node_list = [
                {"node_id": nid, "role": info.role,
                 "node_type": info.node_type,
                 "device_info": info.device_info}
                for nid, info in nodes_snapshot
                if _node_supports_forward_layers(info)
                and (nid not in opted_out or _is_relay_worker(nid))
                and not self._node_is_island_gateway(info.device_info)
                and (
                    info.role == NodeRole.MASTER
                    or not hasattr(info, "is_available")
                    or info.is_available()
                )
            ]
        else:
            node_list = [
                n for n in nodes
                if _node_supports_forward_layers(n)
                and (
                    n.get("node_id") not in opted_out
                    or _is_relay_worker(str(n.get("node_id", "")))
                )
                and not self._node_is_island_gateway(n.get("device_info", {}))
            ]

        # ★ A1 / X 档（Y 档第二条缺口 1）：把「由远端 relay 段代跑本段」的 worker 摘出来。
        #   这类节点**不参与层分配**（`scheduler_pipeline.py:2256` 的注释即：它不需要本地
        #   ModelHost / 模型），但**必须留在流水线里** —— 它承载 `engine="relay_middle"` 的端点段。
        #   此前只有 `scheduler_pipeline` 的 assignment 构造路径认 relay 段，这条**动态分层**路
        #   不认 ⇒ 它被当普通层节点分层（实测日志 `client_TABLET-2TLUCNU8: Layer 18-24 (6层)`）
        #   ⇒ 而 worker 没有本地模型、无法确认分层释放 ACK
        #   ⇒ `pipeline_distributed_workers_unavailable` ⇒ `/api/chat` 503。
        node_list, relay_nodes = self._split_relay_nodes(node_list)

        # ★ Y 档第二条（层区间认领 + 扣除）：relay 段认领的层由**远端段服务**执行，
        #   不属于任何本机层节点。这里先校验认领（重叠 / 角色 / 连续性，任一不成立即
        #   fail-closed），再把「本机层节点可用的层预算」`layer_budget` 作为分配上界 ——
        #   这样 master 只会拿到 `[0, k)`，不再出现「master 跑满全部层、远端段在**已过
        #   全部层**的 hidden 上重算」那个根因形态。
        layer_budget = self._relay_claimed_layer_prefix(total_layers)

        if layer_budget == 0:
            # 全部层都被 relay 段认领 ⇒ 本机层节点无层可跑（主节点只做聚合，
            # `master_participates` 会因此为 False，`_run_pipeline` 已有该分支）。
            logger.info(
                "全部层由 relay 段认领（本机层节点无层可跑）: %s",
                [n["node_id"] for n in relay_nodes],
            )
            return self._append_relay_assignments([], relay_nodes, total_layers)

        if not node_list:
            if relay_nodes:
                logger.info(
                    "仅 relay 段节点参与（不占层）: %s",
                    [n["node_id"] for n in relay_nodes],
                )
                return self._append_relay_assignments([], relay_nodes, total_layers)
            logger.warning("没有可用的层前向节点参与流水线层拆分")
            return []

        # 单节点：全部（**未被 relay 段认领**的）层给该节点。多节点时 master 也参与首段层计算，
        # run_pipeline() 会先在 master 本地执行其层范围，再把 hidden_states
        # 交给第一个 worker，避免浪费主节点 CUDA 独显算力。
        if len(node_list) == 1:
            n = node_list[0]
            return [{
                "node_id": n["node_id"],
                "role": n["role"],
                "start_layer": 0,
                "end_layer": layer_budget,
                "layers_count": layer_budget,
                "has_embedding": True,
                "has_lm_head": True,
                "score": 50.0,
            }]

        return self._simple_weight_assignment(
            node_list, layer_budget, relay_nodes, relay_total_layers=total_layers,
        )

    def _simple_weight_assignment(self, node_list: list,
                                   total_layers: int,
                                   relay_nodes: list | None = None,
                                   relay_total_layers: int | None = None) -> list:
        """
        按资源权重分配连续层段。

        `total_layers` 是**本机层节点可用的层预算**（已扣除 relay 段认领层，即 `k`）；
        `relay_total_layers` 是**模型总层数**，仅用于生成 relay 段条目（零层占位）。
        两者**不可混用** —— 混用会让 relay 条目落在错误的层号上。

        算法:
          1. 计算各节点算力权重
          2. 按权重比例分配 Transformer 层
          3. 按 master 优先排序
          4. 首节点含 Embedding，末节点含 LM Head
          5. 显存约束校验
        """
        # Step 1: 计算权重
        for n in node_list:
            n["score"] = self._compute_node_weight(n.get("device_info", {}))

        total_weight = sum(n["score"] for n in node_list)

        # 权重全为 0 → 均分
        if total_weight <= 0:
            base = total_layers // len(node_list)
            remainder = total_layers % len(node_list)
            node_list.sort(key=lambda n: (n["role"] != "master", -n.get("score", 0)))
            assignments = []
            cursor = 0
            for i, n in enumerate(node_list):
                count = base + (1 if i < remainder else 0)
                assignments.append({
                    "node_id": n["node_id"],
                    "role": n["role"],
                    "start_layer": cursor,
                    "end_layer": cursor + count,
                    "layers_count": count,
                    "has_embedding": (i == 0),
                    "has_lm_head": (i == len(node_list) - 1),
                    "score": 0.0,
                })
                cursor += count
            assignments = self._normalize_master_anchor(
                assignments, node_list, total_layers
            )
            assignments = self._apply_vram_constraints(assignments)
            return self._append_relay_assignments(
                self._normalize_master_anchor(assignments, node_list, total_layers),
                relay_nodes, relay_total_layers or total_layers,
            )

        # Step 2: 按比例分配全部 Transformer 层
        distributable = total_layers
        raw_layers = []
        for n in node_list:
            proportion = n["score"] / total_weight
            raw = max(1, round(proportion * distributable))
            raw_layers.append(raw)

        # 修正 rounding 误差
        diff = distributable - sum(raw_layers)
        if diff > 0:
            for i in range(diff):
                idx = i % len(raw_layers)
                raw_layers[idx] += 1
        elif diff < 0:
            # 从低分节点优先削减，保留高分节点层数
            sorted_indices = sorted(range(len(raw_layers)), key=lambda i: node_list[i]["score"])
            for _ in range(-diff):
                reduced = False
                for idx in sorted_indices:
                    if raw_layers[idx] > 1:
                        raw_layers[idx] -= 1
                        reduced = True
                        break
                # 所有剩余节点都已降至 1 层 → 削去最低分节点（将被过滤移除）
                if not reduced:
                    for idx in sorted_indices:
                        if raw_layers[idx] >= 1:
                            raw_layers[idx] -= 1
                            break

        # ★ 清理分配层数 ≤ 0 的节点（极端情况：节点数远超层数）
        valid_pairs = [(idx, layers) for idx, layers in enumerate(raw_layers) if layers > 0]
        if len(valid_pairs) < len(raw_layers):
            logger.warning(
                f"节点数 ({len(raw_layers)}) 超过可分配层数 ({distributable})，"
                f"{len(raw_layers) - len(valid_pairs)} 个低分节点将被排除"
            )

        # Step 3: 排序（master 优先，同角色按权重降序），跳过已移除节点
        sorted_pairs = sorted(
            enumerate(node_list),
            key=lambda x: (x[1]["role"] != "master", -x[1]["score"])
        )
        sorted_indices = [i for i, _ in sorted_pairs if raw_layers[i] > 0]

        # Step 4: 构建区间
        assignments = []
        cursor = 0
        for order, idx in enumerate(sorted_indices):
            n = node_list[idx]
            count = raw_layers[idx]
            assignments.append({
                "node_id": n["node_id"],
                "role": n["role"],
                "start_layer": cursor,
                "end_layer": cursor + count,
                "layers_count": count,
                "has_embedding": (order == 0),
                "has_lm_head": (order == len(sorted_indices) - 1),
                "score": round(n["score"], 1),
            })
            cursor += count

        # Step 5: 先决定 I/O 头归属，再按真实内存负担校验。
        assignments = self._normalize_master_anchor(
            assignments, node_list, total_layers
        )
        assignments = self._apply_vram_constraints(assignments)
        assignments = self._normalize_master_anchor(
            assignments, node_list, total_layers
        )

        logger.info(
            f"动态分层计算完成: {len(assignments)} 节点, "
            f"本机 {total_layers} 层, 策略=simple_weight"
        )
        for a in assignments:
            logger.info(
                f"  {a['node_id']}: Layer {a['start_layer']}-{a['end_layer']} "
                f"({a['layers_count']}层) embed={a['has_embedding']} "
                f"lm_head={a['has_lm_head']} score={a['score']}"
            )

        return self._append_relay_assignments(
            assignments, relay_nodes or [], relay_total_layers or total_layers,
        )

    def _relay_claimed_layer_prefix(self, total_layers: int) -> int:
        """★ Y 档第二条：校验 relay 段的**层区间认领**，返回"本机层节点可用的前缀长度" `k`。

        语义：relay 段认领的层由**远端段服务**执行，不属于任何本机层节点。
        在「链式单段」架构下（`_run_pipeline` 的 `full_chain = [master] + pipeline`，
        每个层节点只能表达**一个连续区间**），唯一可表达的合法形态是：

        **本机层节点跑 `[0, k)`，relay 段连续覆盖 `[k, total_layers)`。**

        （`k == 0` 合法：全部层都由远端段跑，主节点只做聚合 —— `master_participates`
        会因此为 False，`_run_pipeline` 已有该分支。）

        校验（任一不成立即 **fail-closed**，抛 `ValueError`，绝不静默放过）：

        1. 各段认领区间**互不重叠**；
        2. **段角色与位置匹配**：当前 hidden 输入路径拒绝 `head`；`tail` ⇒
           `end == total_layers`；`middle` ⇒ `start > 0`（允许贴顶并返回 hidden）；
        3. 认领区间**连续覆盖**到 `total_layers`（中间不留空洞）—— 留空洞意味着那段层
           无人执行，而那正是本线要修的根因形态（主节点"顺手"把别人的层也跑了）。
        """
        if total_layers <= 0:
            raise ValueError(f"total_layers 非法: {total_layers}")

        claims: list[tuple[int, int, str, str]] = []
        with self._layer_config_lock:
            relay_map = dict(getattr(self, "_relay_segment_map_cache", None) or {})
        for node_id, spec in relay_map.items():
            if not isinstance(spec, dict):
                continue
            try:
                start = int(spec.get("layer_start", -1))
                end = int(spec.get("layer_end", -1))
            except (TypeError, ValueError):
                raise ValueError(
                    f"relay 段 {node_id} 的层区间不可解析: {spec!r}") from None
            role = str(spec.get("role", "") or "")
            if start < 0 or end <= start or end > total_layers:
                raise ValueError(
                    f"relay 段 {node_id} 的层区间越界: [{start}, {end}) / total={total_layers}")
            claims.append((start, end, role, node_id))

        if not claims:
            return total_layers

        for start, end, role, node_id in claims:
            if role == "head":
                raise ValueError(
                    f"relay head segment is unsupported for hidden input: {node_id} [{start}, {end})"
                )
            # `tail` 段自己吐 **token**（远端做完 argmax）⇒ 必须覆盖到顶，否则它拿不到末层。
            if role == "tail" and end != total_layers:
                raise ValueError(
                    f"tail 段 {node_id} 必须覆盖到顶（{total_layers}），实际 [{start}, {end})")
            # `middle` 段**吃 hidden** ⇒ 不能从 0 起（0 处还没有 hidden，只有 token 输入）；
            #   但它**允许覆盖到顶** —— 它只输出 hidden，`lm_head` 由主节点在收到末节点回的
            #   hidden 后自己跑（`scheduler_pipeline.py:4594-4595` 的推荐拓扑）
            #   ⇒ `master[0,8) + middle[8,24)` 是**完全合法**的产品形态，不得误判为 tail。
            if role == "middle" and start == 0:
                raise ValueError(
                    f"middle 段 {node_id} 不能从 0 起（它吃 hidden；0 处应为本机节点），"
                    f"实际 [{start}, {end})")

        ordered = sorted(claims, key=lambda item: item[0])
        # The current production execution path has one relay worker per
        # pipeline (master -> relay).  A second relay claim is not executable
        # yet: relay workers reject chain_next and PeerClient does not forward
        # relay output to another relay.  Reject it at admission instead of
        # reporting an admitted plan that will fail on the first token.
        if len(ordered) > 1:
            raise ValueError(
                "当前 relay 执行路径只支持单段 relay；多段 relay chain 尚未接线"
            )
        for left, right in zip(ordered, ordered[1:]):
            if right[0] < left[1]:
                raise ValueError(
                    f"relay 段层区间重叠: [{left[0]}, {left[1]}) 与 [{right[0]}, {right[1]})")

        k = ordered[0][0]
        cursor = k
        for start, end, _role, node_id in ordered:
            if start != cursor:
                raise ValueError(
                    f"relay 段层区间不连续（{cursor} 处断裂，下一段为 {node_id} [{start}, {end})）"
                    f" —— 空洞内的层将无人执行")
            cursor = end
        if cursor != total_layers:
            raise ValueError(
                f"relay 段层区间未覆盖到顶（止于 {cursor} / {total_layers}）"
                f" —— 其上的层将无人执行")
        return k

    def _relay_layer_claim_ranges(self, total_layers: int) -> dict[str, tuple[int, int]]:
        """★ Y 档第二条：`{node_id: (start, end)}` —— 各 relay 段认领的层区间。

        校验与 `_relay_claimed_layer_prefix` **同源**：先调它把规则跑一遍（任一不成立即抛
        `ValueError`，fail-closed），再取区间 —— 避免两套判据漂移。
        用途：capacity plan 里 relay 零层条目要带上认领区间，供下游层覆盖校验
        （`pipeline_node_contract.validate_pipeline_nodes`）识别"这些层由远端段执行"。
        """
        self._relay_claimed_layer_prefix(total_layers)   # 校验（fail-closed）
        with self._layer_config_lock:
            relay_map = dict(getattr(self, "_relay_segment_map_cache", None) or {})
        ranges: dict[str, tuple[int, int]] = {}
        for node_id, spec in relay_map.items():
            if not isinstance(spec, dict):
                continue
            try:
                ranges[node_id] = (
                    int(spec.get("layer_start", -1)), int(spec.get("layer_end", -1)),
                )
            except (TypeError, ValueError):
                continue
        return ranges

    def _split_relay_nodes(self, node_list: list) -> tuple[list, list]:
        """★ A1 / X 档（Y 档第二条缺口 1）：把 relay 段节点从**层分配**里摘出来。

        返回 `(参与层分配的节点, relay 段节点)`。判据是 `_relay_segment_for_worker()` 非空
        （它已内含 `QLH_RELAY_ENABLED` 开关与请求级闸门）—— 开关默认关时**恒返回空**，
        对既有路径零影响。

        为什么必须摘：relay 段节点**没有本地模型**（`scheduler_pipeline.py:2256` 注释即
        "this scheduler host does not need a local ModelHost/model loaded"）⇒ 若把它当普通层
        节点分配层，worker 侧无法确认分层释放 ACK ⇒ `pipeline_distributed_workers_unavailable`。

        两条动态分层路（`compute_layer_assignment` 与 `_simple_weight_assignment`）**都**要用它，
        所以抽成方法而不是内联局部变量（首版内联就踩了 `NameError`）。
        """
        if not node_list:
            return node_list, []
        relay_for = getattr(self, "_relay_segment_for_worker", None)
        if not callable(relay_for):
            return node_list, []
        relay_nodes = [n for n in node_list if relay_for(n["node_id"]) is not None]
        if not relay_nodes:
            return node_list, []
        relay_ids = {n["node_id"] for n in relay_nodes}
        return [n for n in node_list if n["node_id"] not in relay_ids], relay_nodes

    def _append_relay_assignments(self, assignments: list, relay_nodes: list,
                                  total_layers: int) -> list:
        """★ A1 / X 档（Y 档第二条缺口 1）：把 relay 段节点以 `engine="relay_middle"` 追加进分层结果。

        它们**不占层**（`layers_count=0`、`start_layer == end_layer == total_layers`），
        `has_embedding` / `has_lm_head` 都为 False —— 层的 I/O 头仍归真正跑层的节点。
        worker 侧据 `engine == "relay_middle"` 直接走 relay 分支
        （`scheduler_pipeline.py:2255`），**不需要本地模型**。

        刻意在**最后**追加（即 `_normalize_master_anchor` / `_apply_vram_constraints` 之后）：
        那两个函数按"层数"推理，0 层条目不该参与它们的锚定与显存校验。
        """
        if not relay_nodes:
            return assignments
        resolved = getattr(self, "_relay_segment_for_worker", None)
        for n in relay_nodes:
            spec = resolved(n["node_id"]) if callable(resolved) else None
            assignments.append({
                "node_id": n["node_id"],
                "role": n["role"],
                "start_layer": total_layers,
                "end_layer": total_layers,
                "layers_count": 0,
                "has_embedding": False,
                "has_lm_head": False,
                "score": round(float(n.get("score", 0.0) or 0.0), 1),
                "engine": "relay_middle",
                "relay_segment": spec,
            })
        return assignments

    def _normalize_master_anchor(self, assignments: list, node_list: list,
                                 total_layers: int) -> list:
        """Compatibility facade for master-anchored assignment normalization."""
        return _layer_plan.normalize_master_anchor(
            assignments,
            node_list,
            total_layers,
            compute_weight=self._compute_node_weight,
            resequence=self._resequence_assignments,
            warning=logger.warning,
        )

    @staticmethod
    def _resequence_assignments(assignments: list) -> list:
        """Compatibility facade for contiguous assignment ranges."""
        return _layer_plan.resequence_assignments(assignments)

    def _apply_vram_constraints(self, assignments: list) -> list:
        """
        显存约束校验：检查每节点是否有足够显存承载分配的层。

        若不足，将超额层转移给 VRAM 最充裕的节点；层数为 0 的节点从列表中移除。
        转移后重新计算所有节点的 layer 区间以保持连续性。

        ★ 保护规则（方案 A）:
          - 首节点（has_embedding）至少保留 1 层 Transformer，确保 Embedding 权重有
            同节点层可锚定，避免出现「纯 Embedding 节点」。
          - 末节点（has_lm_head）同样至少保留 1 层。
          - 若连 1 层都保不住，打 ERROR 日志并跳过转移（降级为本地推理兜底）。
        """
        pending = []
        for a in assignments:
            ok, needed, available = self._check_vram_constraint(
                a["node_id"], a["layers_count"],
                a["has_embedding"], a["has_lm_head"],
            )
            if not ok and available > 0:
                max_fit = 0
                for count in range(int(a["layers_count"]), -1, -1):
                    fits, _needed, _available = self._check_vram_constraint(
                        a["node_id"], count, a["has_embedding"], a["has_lm_head"]
                    )
                    if fits:
                        max_fit = count
                        break
                minimum = 1 if (a["has_embedding"] or a["has_lm_head"]) else 0
                if max_fit < minimum:
                    logger.error(
                        f"❌ 节点 {a['node_id']} 连 I/O 头和 1 层都无法容纳"
                        f"（需 {needed}MB / 可用 {available}MB），保留原分配并由就绪检查回退。"
                    )
                    continue
                overflow = max(0, int(a["layers_count"]) - max_fit)
                if overflow:
                    a["layers_count"] = max_fit
                    pending.append((a["node_id"], overflow))

        for source_id, overflow in pending:
            while overflow > 0:
                candidates = []
                for other in assignments:
                    if other["node_id"] == source_id:
                        continue
                    fits, _, _ = self._check_vram_constraint(
                        other["node_id"], other["layers_count"] + 1,
                        other["has_embedding"], other["has_lm_head"],
                    )
                    if fits:
                        candidates.append(other)
                if not candidates:
                    logger.error("❌ %s 的 %s 个超额层没有节点具备剩余容量", source_id, overflow)
                    # 保持完整覆盖，让 worker 加载失败 ACK 触发全模型回退，而不是静默丢层。
                    source = next(a for a in assignments if a["node_id"] == source_id)
                    source["layers_count"] += overflow
                    break
                target = max(
                    candidates,
                    key=lambda item: (
                        self._get_node_vram_mb(item["node_id"]),
                        float(item.get("score", 0) or 0),
                    ),
                )
                target["layers_count"] += 1
                overflow -= 1

        # 清除层数为 0 的节点
        assignments[:] = [a for a in assignments if a["layers_count"] > 0]

        # ★ 重新计算所有节点的 layer 区间，确保 start_layer/end_layer 连续。
        # LM Head 是输出投影，不属于 Transformer 层区间；主节点可在 worker
        # 返回尾层 hidden_states 后执行它，避免弱 worker 计算/回传全词表 logits。
        master_index = next((
            i for i, item in enumerate(assignments)
            if item.get("node_id") == "master" or item.get("role") == "master"
        ), None)
        lm_head_index = len(assignments) - 1
        if master_index is not None and assignments:
            try:
                master_score = float(assignments[master_index].get("score", 0) or 0)
                tail_score = float(assignments[-1].get("score", 0) or 0)
                if master_score >= tail_score:
                    lm_head_index = master_index
            except (TypeError, ValueError):
                pass
        cursor = 0
        for i, a in enumerate(assignments):
            a["start_layer"] = cursor
            a["end_layer"] = cursor + a["layers_count"]
            a["has_embedding"] = (i == 0)
            a["has_lm_head"] = (i == lm_head_index)
            cursor += a["layers_count"]

        return assignments

    def get_layer_assignments(self) -> dict:
        """
        获取当前分层配置。

        优先返回当前进程的手动覆盖，否则根据实时节点画像动态计算。

        Returns:
            {
                "total": 24,
                "strategy": "dynamic" | "capacity" | "manual",
                "assignments": [...],
                "computed_at": timestamp | null,
            }
        """

        total_layers = self._get_total_model_layers()
        with self._layer_config_lock:
            opted_out = set(self._pipeline_worker_opt_out)
            capacity_plan = (
                dict(self._active_pipeline_capacity_plan)
                if self._active_pipeline_capacity_plan else None
            )

        if capacity_plan and capacity_plan.get("admitted"):
            return {
                "total": int(capacity_plan.get("total_layers", total_layers)),
                "strategy": "capacity",
                "assignments": [
                    dict(item) for item in capacity_plan.get("assignments", [])
                ],
                "computed_at": capacity_plan.get("computed_at"),
                "plan_id": capacity_plan.get("plan_id", ""),
            }

        if self._runtime_layer_override:
            relay_for_worker = getattr(self, "_relay_segment_for_worker", None)
            overrides = self._normalize_manual_assignments(
                [
                    item for item in self._runtime_layer_override
                    if (
                        item.get("node_id") not in opted_out
                        or (
                            callable(relay_for_worker)
                            and relay_for_worker(str(item.get("node_id", "")))
                            is not None
                        )
                    )
                ]
            )
            if (
                overrides
                and self._manual_assignments_are_executable(overrides)
                and max(int(item.get("end_layer", 0)) for item in overrides)
                == total_layers
            ):
                return {
                    "total": total_layers,
                    "strategy": "manual",
                    "assignments": overrides,
                    "computed_at": None,
                }

        cache_key = self._layer_assignment_cache_key(total_layers)

        # 动态计算
        assignments = self.compute_layer_assignment()

        actual_strategy = "dynamic"

        result = {
            "total": total_layers,
            "strategy": actual_strategy,
            "assignments": assignments,
            "computed_at": time.time(),
            "cache_key": cache_key,
        }

        return result

    def _reap_stale_worker_reservations(
        self, now: Optional[float] = None,
    ) -> dict[str, list[str]]:
        """★ 2026-10-07（DIST-NEXT-4b）：回收「心跳过期且无在跑 attempt」的 reservation。

        审计 P0-3 要求「断线、心跳过期、服务重建都必须**原子撤销** assignment 和
        reservation，并记录**单一 reason code**」。断线已有专门路径（`notify_disconnect`
        全量回收）；这里补的是**心跳过期但 TCP 仍在线**的窗口：

        * 只回收**已终结**的条目（`release_stale_reservations`），不打断 in-flight 执行；
        * 逐节点记一条 `event=task_worker_reservations_released`，`reason` 固定为
          `worker_heartbeat_stale`（单一 reason code，便于 grep 与对账）。

        返回 `{node_id: [reservation_id...]}`；幂等，可安全地周期性调用。
        """
        current = time.time() if now is None else float(now)
        with self._task_worker_stage_lock:
            providers = dict(self._remote_task_worker_providers)
        if not providers:
            return {}
        released: dict[str, list[str]] = {}
        for node_id, provider in providers.items():
            with self._nodes_lock:
                node = self.nodes.get(node_id)
            if node is None:
                continue
            try:
                fresh = node.is_heartbeat_fresh(current, WORKER_HEARTBEAT_MAX_AGE)
            except Exception:       # 嵌入/测试构造的节点可能没有该方法
                continue
            if fresh:
                continue
            freed = provider.release_stale_reservations(RELEASE_REASON_HEARTBEAT_STALE)
            if freed:
                released[node_id] = freed
                logger.warning(
                    "event=task_worker_reservations_released node_id=%s reason=%s "
                    "released=%s",
                    node_id, RELEASE_REASON_HEARTBEAT_STALE, freed,
                )
        return released

    def _get_pipeline_capacity_nodes(
        self, eligible_node_ids: Optional[set[str]] = None,
    ) -> list[dict]:
        """Project live layer-worker profiles into explicit free-memory budgets."""
        from contextlib import nullcontext
        from config import PIPELINE_CAPACITY_RESERVE_MB
        from koakuma_engine import canonical_backend

        # ★ 2026-10-07（DIST-NEXT-4b）：规划前先回收「心跳已过期且没有在跑 attempt」的
        #   reservation。TCP 半开/心跳线程失效的节点在巡检窗口（约 131s）里 TCP 仍算在线
        #   ⇒ 不走断线回收路径，`max_concurrency=1` 的槽位会被「已预留未执行」永久占住。
        self._reap_stale_worker_reservations()

        reserve_bytes = int(PIPELINE_CAPACITY_RESERVE_MB * 1024 * 1024)
        with self._layer_config_lock:
            opted_out = set(self._pipeline_worker_opt_out)
        with self._nodes_lock:
            snapshot = list(self.nodes.items())

        # Task-worker v3 ranges are an execution contract, not merely a
        # readiness hint. Feed admitted Android ranges into the capacity
        # solver so it cannot produce an assignment the worker must reject.
        layer_ranges_by_node: dict[str, list[list[int]]] = {}
        layer_artifacts_by_node: dict[str, list[dict]] = {}
        models_by_node: dict[str, list[dict]] = {}
        worker_engines: dict[str, str] = {}
        # ★ 2026-10-03：设备自荐的层容量（本地裁层后可承载的层数上限）。与
        #   `layer_ranges` 同源（v3 hello capabilities）但语义不同：ranges 是
        #   "当前已就绪、马上能跑的区间"，budget 是"能自裁并承载的上限" ⇒ 有了它，
        #   求解器才能给该节点分配任意连续区间，而不是被它预置的那一段钉死。
        layer_budget_by_node: dict[str, dict] = {}
        # ★ 2026-10-05（DIST-3 三机实测）：工件段类型（`head`/`middle`/`tail`）。
        #   与 `layer_ranges` / `layer_budget` 同源（v3 hello capabilities），但要
        #   单独透传给容量求解器：区间只说"覆盖哪些层"，不区分工件含不含
        #   embedding / lm_head，于是中间段会被分到末段（实测失败原因）。
        segment_mode_by_node: dict[str, str] = {}
        if self._effective_role() == "master" and TASK_WORKER_EXPERIMENTAL_ENABLED:
            try:
                worker_status = self._task_worker_control.status(role="master")
            except Exception:
                worker_status = {}
            for worker in worker_status.get("workers", []):
                if not isinstance(worker, dict) or not worker.get("layer_stage_dispatch_enabled"):
                    continue
                capabilities = worker.get("capabilities")
                if not isinstance(capabilities, dict):
                    continue
                node_id = str(worker.get("node_id", "") or "")
                if not node_id:
                    continue
                ranges = capabilities.get("layer_ranges")
                if isinstance(ranges, list) and ranges:
                    layer_ranges_by_node[node_id] = ranges
                artifacts = capabilities.get("layer_artifacts")
                if isinstance(artifacts, list) and artifacts:
                    layer_artifacts_by_node[node_id] = artifacts
                models = capabilities.get("models")
                if isinstance(models, list) and models:
                    models_by_node[node_id] = models
                model_engines = {
                    canonical_backend(item.get("engine"))
                    for item in (models or []) if isinstance(item, dict)
                } if isinstance(models, list) else set()
                model_engines.discard("")
                # Selected model identity takes precedence over installed engines.
                if len(model_engines) == 1:
                    worker_engines[node_id] = next(iter(model_engines))
                else:
                    engines = capabilities.get("engines")
                    if isinstance(engines, list) and len(engines) == 1:
                        worker_engines[node_id] = canonical_backend(engines[0])
                budget = capabilities.get("layer_budget")
                if isinstance(budget, dict):
                    layer_budget_by_node[node_id] = budget
                segment_mode = capabilities.get("segment_mode")
                if isinstance(segment_mode, str) and segment_mode.strip():
                    segment_mode_by_node[node_id] = segment_mode.strip().lower()
        if (
            layer_ranges_by_node
            or layer_artifacts_by_node
            or layer_budget_by_node
            or segment_mode_by_node
        ):
            logger.info(
                "容量节点层段投影: ranges=%s budget=%s segment_mode=%s",
                {k: v for k, v in layer_ranges_by_node.items()},
                # ★ 2026-10-23：打印**完整** budget，而不是只看 max_layers。
                #   `local_cut` 决定 `layer_ranges` 是硬约束还是仅"当前已就绪"的信息
                #   （capacity.py:918）；只打 max_layers 时，它到底是 false 还是缺失
                #   完全不可见，排查"预置段钉死切点"时白走了一大圈。
                layer_budget_by_node,
                segment_mode_by_node,
            )

        records = []
        effective_id = self.get_effective_node_id()
        local_node_id = (
            "master" if self._effective_role() == "master" else effective_id
        )
        local_engine = ""
        local_device = ""
        local_node = next((
            node for node_id, node in snapshot if node_id == local_node_id
        ), None)
        if local_node is not None:
            # Preparation keeps the active manager intact until commit. Read the
            # actual candidate under its existing fence rather than the active
            # host engine or a hardware-only CUDA guess. Never warm a lazy manager.
            try:
                candidate_lock = object.__getattribute__(
                    self._host, "_pipeline_candidate_lock",
                )
            except AttributeError:
                candidate_lock = nullcontext()
            with candidate_lock:
                try:
                    candidate = object.__getattribute__(
                        self._host, "_pipeline_candidate_transaction",
                    )
                except AttributeError:
                    candidate = None
                candidate_selected = candidate is not None and candidate.phase in {
                    "prepared", "materialized", "committed",
                }
                runtime = candidate.manager if candidate_selected else None
            # Runtime accessors may take the full-chat lock. Keep all calls out
            # of the candidate fence so this reader adds no candidate -> full edge.
            if not candidate_selected:
                peek_manager = getattr(self._host, "peek_manager", None)
                runtime = peek_manager() if callable(peek_manager) else self._host
            local_engine = backend_id_for(runtime, default="")
            if local_engine == "llama_cpp":
                # ModelManager's legacy llama.cpp wrapper has a PyTorch-only
                # get_device() that may still say CUDA; llama.cpp executes on CPU.
                local_device = "cpu"
            elif runtime is not None:
                get_device = getattr(runtime, "get_device", None)
                try:
                    # Before materialization, ModelManager.get_device() falls
                    # back to DEVICE. Share the filtered layer loader's selector.
                    runtime_module = sys.modules.get("model_module")
                    manager_type = getattr(runtime_module, "ModelManager", ())
                    if (
                        local_engine == "pytorch"
                        and isinstance(runtime, manager_type)
                        and runtime.is_pipeline_prepared
                        and runtime.model is None
                    ):
                        device, _dtype = runtime_module._select_layer_runtime(
                            runtime.quant_type,
                        )
                        local_device = str(device).split(":", 1)[0].lower()
                    elif callable(get_device):
                        local_device = str(get_device()).split(":", 1)[0].lower()
                except Exception:
                    local_device = "unknown"
                    logger.warning(
                        "runtime capacity device unavailable: engine=%s",
                        local_engine, exc_info=True,
                    )
            if candidate_selected:
                with candidate_lock:
                    current = object.__getattribute__(
                        self._host, "_pipeline_candidate_transaction",
                    )
                    if (
                        current is not candidate or current.manager is not runtime
                        or current.phase not in {
                            "prepared", "materialized", "committed", "finalized",
                        }
                    ):
                        # Abort/replacement during device lookup invalidates this
                        # snapshot. Reject it instead of budgeting a stale engine.
                        local_device = "unknown"
            if runtime is None:
                select_engine = getattr(self._host, "select_engine", None)
                if callable(select_engine):
                    local_engine = canonical_backend(select_engine(local_node.device_info))
        now = time.time()
        for node_id, node in snapshot:
            # ★ 逐条件诊断：把「为什么某个在线 PC 没进容量候选」直接打出来。
            #   此前是一个 5 条件的 `or` 短路，出问题时只能靠猜 —— 排查 relay 跨机拓扑时
            #   正是卡在这里（`candidate_node_count: 1` 却看不出被哪条排除）。
            skip_reason = ""
            relay_for_node = getattr(self, "_relay_segment_for_worker", None)
            is_relay_worker = (
                callable(relay_for_node)
                and relay_for_node(node_id) is not None
            )
            if not _node_supports_forward_layers(node):
                skip_reason = "no_forward_layers"
            elif eligible_node_ids is not None and node_id not in eligible_node_ids:
                skip_reason = "not_eligible"
            elif node_id in opted_out and not is_relay_worker:
                skip_reason = "opted_out"
            elif self._node_is_island_gateway(node.device_info):
                skip_reason = "island_gateway"
            elif (
                node.role != NodeRole.MASTER
                and node_id != effective_id
                and not node.is_available()
            ):
                skip_reason = "not_available"
            elif (
                node.role != NodeRole.MASTER
                and node_id != effective_id
                and not node.is_heartbeat_fresh(now, WORKER_HEARTBEAT_MAX_AGE)
            ):
                # ★ 2026-10-05（DIST-2）：容量候选此前只看 `NodeState`，而 TCP 半开
                #   （对端进程已死、不发 FIN）要等巡检约 129s 才置 OFFLINE ⇒ 静默节点
                #   在这段窗口里**一直占容量**，会分配出既非声明区间、层数也不对的
                #   结果（实测：Y700 被带走后 master 仍把它算进规划）。
                #   这里与 `_get_pipeline_readiness` 共用同一阈值 `WORKER_HEARTBEAT_MAX_AGE`。
                skip_reason = "heartbeat_stale"
            if skip_reason:
                logger.info(
                    "容量候选跳过 %s: reason=%s node_type=%s role=%s eligible=%s opted_out=%d",
                    node_id, skip_reason, getattr(node, "node_type", "?"),
                    node.role,
                    (node_id in eligible_node_ids) if eligible_node_ids is not None else "n/a",
                    len(opted_out),
                )
                continue
            # ★ A1 / X 档（Y 档第二条缺口 4）：relay 段节点**不能**被剔除出候选集 ——
            #   求解器的 `require_distributed and len(usable) < 2` 闸门（`pipeline_capacity.py:221`）
            #   会因为"只剩 master 一个可用节点"而拒掉整个 plan
            #   （实测 `reason=distributed placement requires at least two usable PC nodes`）。
            #   它**也不需要**本地容量预算：段工件由**远端** relay_mid_service 持有，本节点只转发。
            #   ⇒ 收录它，但容量取"占位值"（`relay_exempt`），**不参与真实的内存准入**。
            #   它实际跑不跑层由下发 assignment 里的 `engine="relay_middle"` 决定
            #   （见 `compute_layer_assignment` 的 `_append_relay_assignments`）。
            if is_relay_worker:
                device_info = dict(node.device_info or {})
                ram = device_info.get("ram", {})
                total_gb = 0.0
                if isinstance(ram, dict):
                    try:
                        total_gb = float(ram.get("total_gb", 0) or 0)
                    except (TypeError, ValueError):
                        total_gb = 0.0
                records.append({
                    "node_id": node_id,
                    "role": node.role,
                    "capacity_bytes": max(1024 ** 3, int(total_gb * 1024 ** 3)),
                    "reserve_bytes": 0,
                    "runtime_multiplier": 1.0,
                    "execution_device": "cpu",
                    "capacity_source": "relay_exempt",
                    "score": 0.0,
                })
                continue
            device_info = dict(node.device_info or {})
            gpu = self._select_scoring_gpu(device_info)
            cuda_discrete = bool(
                isinstance(gpu, dict)
                and gpu.get("cuda_available", False)
                and not self._gpu_is_integrated(gpu)
            )
            engine = worker_engines.get(node_id) or canonical_backend(
                device_info.get("backend_id") or device_info.get("engine"),
                default="pytorch",
            )
            if node_id == local_node_id:
                engine = local_engine or engine
            execution_device = (
                "cuda" if engine == "pytorch" and cuda_discrete else "cpu"
            )
            if node_id == local_node_id and local_device:
                execution_device = local_device
            capacity_gb = 0.0
            capacity_source = ""
            runtime_multiplier = 2.0
            if execution_device == "cuda":
                if cuda_discrete:
                    try:
                        capacity_gb = float(gpu.get("vram_free_gb", 0) or 0)
                    except (TypeError, ValueError):
                        capacity_gb = 0.0
                capacity_source = "gpu.vram_free_gb" if capacity_gb > 0 else ""
                runtime_multiplier = 1.0
            elif execution_device == "cpu":
                ram = device_info.get("ram", {})
                if isinstance(ram, dict):
                    try:
                        capacity_gb = float(ram.get("available_gb", 0) or 0)
                    except (TypeError, ValueError):
                        capacity_gb = 0.0
                capacity_source = "ram.available_gb" if capacity_gb > 0 else ""
                # Android presence reports the same free-memory evidence as
                # an absolute byte count.  Do not discard an admitted layer
                # worker merely because it does not use the PC `ram` shape.
                if capacity_gb <= 0:
                    memory = device_info.get("memory", {})
                    if isinstance(memory, dict):
                        try:
                            available_bytes = float(
                                memory.get("available_bytes", 0) or 0
                            )
                        except (TypeError, ValueError):
                            available_bytes = 0.0
                        if available_bytes > 0:
                            capacity_gb = available_bytes / (1024 ** 3)
                            capacity_source = "memory.available_bytes"
                if (
                    capacity_source == "ram.available_gb"
                    and (node_id == effective_id or (
                        node.role == NodeRole.MASTER
                        and self._effective_role() == "master"
                    ))
                ):
                    try:
                        import psutil

                        available_bytes = int(psutil.virtual_memory().available)
                    except (ImportError, OSError, ValueError):
                        available_bytes = 0
                    capacity_gb = max(0, available_bytes) / (1024 ** 3)
                    capacity_source = "ram.available_bytes_live" if available_bytes > 0 else ""
            record = {
                "node_id": node_id,
                "role": node.role,
                "capacity_bytes": max(0, int(capacity_gb * 1024 ** 3)),
                "reserve_bytes": reserve_bytes,
                "reserve_source": "config.PIPELINE_CAPACITY_RESERVE_MB",
                "runtime_multiplier": runtime_multiplier,
                "execution_device": execution_device,
                "capacity_source": capacity_source,
                "engine": engine,
                "score": self._compute_node_weight(device_info, execution_device=execution_device),
            }
            if node_id in layer_ranges_by_node:
                record["layer_ranges"] = layer_ranges_by_node[node_id]
            if node_id in layer_artifacts_by_node:
                record["layer_artifacts"] = layer_artifacts_by_node[node_id]
            if node_id in models_by_node:
                record["models"] = models_by_node[node_id]
            if node_id in layer_budget_by_node:
                record["layer_budget"] = layer_budget_by_node[node_id]
            # ★ 2026-10-05（DIST-3）：段类型透传给求解器，供其拒绝
            #   「中间段接末段 / 末段接中间段」这类分配。
            if node_id in segment_mode_by_node:
                record["segment_mode"] = segment_mode_by_node[node_id]
            # ★ 2026-10-07：**本地节点**也要声明自己的层段工件区间。
            #   背景：`pipeline_capacity.py` 的 `range_constrained = any(
            #   node.get("layer_ranges") is not None ...)` —— 只要**任一**远端节点
            #   声明了区间，全链就必须由区间拼满 `[0, total)`（契约也要求
            #   `pipeline_node_contract.py` 的 "cover every layer exactly once"）。
            #   本地 master 此前**从不**带 `layer_ranges`（`layer_ranges_by_node`
            #   只从 task worker hello 取）⇒ 一旦远端声明区间，本地这段就成了
            #   覆盖缺口，实测 `status=rejected
            #   reason=pipeline_layer_range_coverage_insufficient`。
            #   本机若是**裁层 GGUF 上游**（`QLH_LAYER_GGUF`），用与 PC worker
            #   同一份推导（`_configured_layer_artifact`）声明自己的区间。
            #   ⚠️ 判据必须与 `_pipeline_node_metadata` 一致：主节点的本地 ID 是
            #   字面量 `"master"`，而 `get_effective_node_id()` 返回的是
            #   `_configured_node_id()`（可能是主机名）—— 两者**不等**，
            #   用后者会让本地节点永远匹配不上（实测投影里只有远端两个节点）。
            if (
                node_id == local_node_id
                and "layer_ranges" not in record
                and node_id not in layer_ranges_by_node
            ):
                local_artifact = self._configured_layer_artifact()
                if local_artifact:
                    record["layer_ranges"] = [
                        [int(local_artifact["start"]), int(local_artifact["end"])],
                    ]
                    local_mode = str(local_artifact.get("segment_mode") or local_artifact.get("mode", "") or "").lower()
                    if local_mode in {"head", "middle", "tail"}:
                        record["segment_mode"] = local_mode
            records.append(record)
        return records

    def _pipeline_node_metadata(self) -> dict[str, dict]:
        """Project live scheduler nodes into opaque layout-location metadata."""
        local_node_id = (
            "master" if self._effective_role() == "master"
            else self.get_effective_node_id()
        )
        with self._nodes_lock:
            snapshot = list(self.nodes.items())
        # Public host identity is lazy-safe. A prepared candidate must not
        # relabel a still-serving active plan; fresh plans bind their engine
        # from the same candidate-aware capacity records used for admission.
        local_engine = backend_id_for(self._host, default="")
        worker_engines: dict[str, str] = {}
        if self._effective_role() == "master" and TASK_WORKER_EXPERIMENTAL_ENABLED:
            try:
                worker_status = self._task_worker_control.status(role="master")
            except Exception:
                worker_status = {}
            for worker in worker_status.get("workers", []):
                if not isinstance(worker, dict):
                    continue
                capabilities = worker.get("capabilities")
                if not isinstance(capabilities, dict):
                    continue
                models = capabilities.get("models")
                model_engines = {
                    str(item.get("engine", "")).strip()
                    for item in models
                    if isinstance(item, dict)
                    and str(item.get("engine", "")).strip()
                } if isinstance(models, list) else set()
                # A PC can have both runtimes installed while its selected
                # layer artifact is GGUF.  The executable model identity is
                # more specific than the installation-wide engine list.
                if len(model_engines) == 1:
                    worker_engines[str(worker.get("node_id", ""))] = next(
                        iter(model_engines)
                    )
                    continue
                engines = capabilities.get("engines")
                if not isinstance(engines, list):
                    continue
                normalized = [
                    str(value).strip() for value in engines
                    if str(value).strip()
                ]
                if len(normalized) == 1:
                    worker_engines[str(worker.get("node_id", ""))] = normalized[0]
        return {
            node_id: {
                "is_local": node_id == local_node_id,
                # Public plans distinguish local/remote without exporting an
                # address that belongs to the transport control plane.
                "location": "local" if node_id == local_node_id else f"node:{node_id}",
                "kind": "local" if node_id == local_node_id else "remote_pipeline",
                "federated": node_id != local_node_id,
                "engine": (
                    "relay_middle"
                    if self._is_relay_host(node_id)
                    else (
                        local_engine if node_id == local_node_id and local_engine
                        else worker_engines.get(
                            node_id,
                            str((_node.device_info or {}).get("backend_id") or "pytorch"),
                        )
                    )
                ),
            }
            for node_id, _node in snapshot
        }

    def _attach_pipeline_node_contract(
        self, plan: dict, *, capacity_nodes: Optional[list[dict]] = None,
    ) -> dict:
        """Attach the canonical node layout or reject a malformed admission."""
        if not isinstance(plan, dict) or plan.get("admitted") is not True:
            return plan
        result = dict(plan)
        if capacity_nodes is not None:
            engines = {
                item["node_id"]: item["engine"] for item in capacity_nodes
                if item.get("engine")
            }
            result["assignments"] = [
                {**item, "engine": engines[item["node_id"]]}
                if item.get("node_id") in engines and not item.get("engine")
                else dict(item)
                for item in (result.get("assignments") or [])
            ]
        # ★ A1 / X 档（Y 档第二条缺口 6 / 2b）：带**认领区间**的 relay 零层条目**不再剔除**。
        #   它们现在是 `start_layer/end_layer = 认领区间`（如 `(8, 24)`）+ `claimed_layers`，
        #   正好补上"`[k, total)` 由远端段执行"这一块覆盖，让 layout 校验
        #   （**恰好连续覆盖全部层**）通过；而主节点自己只拿 `[0, k)`。
        #   真正**无区间**的零层条目（异常形态）仍然剔除，避免污染 layout。
        layout_assignments = [
            item for item in (result.get("assignments") or [])
            if int(item.get("layers_count", 1) or 0) > 0
            or (
                str(item.get("capacity_source", "") or "") == "relay_exempt"
                and isinstance(item.get("claimed_layers"), (list, tuple))
                and len(item.get("claimed_layers") or ()) == 2
            )
        ]
        try:
            layout = pipeline_layout_from_capacity_plan(
                {**result, "assignments": layout_assignments},
                node_metadata=self._pipeline_node_metadata(),
            )
        except PipelineNodeContractError as exc:
            logger.warning(
                "布局契约校验失败: %s | layout_assignments=%s",
                exc,
                [
                    {k: item.get(k) for k in
                     ("node_id", "start_layer", "end_layer", "layers_count",
                      "has_embedding", "has_lm_head", "capacity_source")}
                    for item in layout_assignments
                ],
            )
            result.update({
                "status": "rejected",
                "admitted": False,
                "reason_code": "pipeline_node_contract_invalid",
                "reason": str(exc),
                "assignments": [],
                "pipeline_layout": None,
            })
            return result
        result["pipeline_layout"] = layout.to_dict()
        return result

    def _activate_pipeline_reshard_coordinator(self, plan: Mapping[str, Any]) -> None:
        """Bind a ready capacity plan to an epoch-fenced recovery topology."""
        try:
            layout = pipeline_layout_from_capacity_plan(
                plan, node_metadata=self._pipeline_node_metadata(),
            )
            coordinator = PipelineReshardCoordinator(
                layout,
                lease_book=RpcShardLeaseBook(control_fence=self._control_fence),
            )
        except (PipelineNodeContractError, ValueError) as exc:
            logger.warning("未启用自动重分片合同: %s", exc)
            with self._layer_config_lock:
                self._pipeline_reshard_coordinator = None
                self._pipeline_reshard_last_decision = {
                    "status": "unavailable",
                    "reason_code": "pipeline_reshard_layout_invalid",
                    "reason": str(exc),
                }
            return
        with self._layer_config_lock:
            self._pipeline_reshard_coordinator = coordinator
            self._pipeline_reshard_last_decision = {
                "status": "active",
                "reason_code": "",
                "epoch": coordinator.epoch,
            }

    def _stage_pipeline_reshard_after_disconnect(self, node_id: str) -> Optional[dict]:
        """Re-solve an active topology after a worker loss without writeback."""
        if self._effective_role() != "master":
            return None
        with self._layer_config_lock:
            coordinator = self._pipeline_reshard_coordinator
        if coordinator is None or node_id not in {
            item.node_id for item in coordinator.layout.nodes
        }:
            return None
        get_descriptor = getattr(self._host, "get_pipeline_descriptor", None)
        descriptor = get_descriptor() if callable(get_descriptor) else {}
        if not isinstance(descriptor, dict) or not descriptor:
            report = {
                "status": "rejected",
                "accepted": False,
                "reason_code": "pipeline_reshard_descriptor_unavailable",
            }
        else:
            from config import PIPELINE_CAPACITY_SAFETY_MARGIN

            server = self._tcp_server
            get_client_ids = getattr(server, "get_client_ids", None)
            connected_ids = (
                set(get_client_ids()) if callable(get_client_ids)
                else set(getattr(server, "clients", {}).keys()) if server else set()
            )
            connected_ids.update({"master", self.get_effective_node_id()})
            connected_ids.difference_update(self._task_worker_full_model_ids())

            failed_node_ids = {node_id}
            snapshot = getattr(coordinator, "snapshot", None)
            if callable(snapshot):
                for staged in snapshot().get("staged", []):
                    failed_node_ids.update(
                        str(item) for item in staged.get("failed_node_ids", [])
                        if str(item)
                    )

            decision = coordinator.stage_failure(
                failed_node_ids,
                descriptor=descriptor,
                capacity_nodes=self._get_pipeline_capacity_nodes(connected_ids),
                node_metadata=self._pipeline_node_metadata(),
                safety_margin=PIPELINE_CAPACITY_SAFETY_MARGIN,
            )
            report = decision.to_dict()
        with self._layer_config_lock:
            self._pipeline_reshard_last_decision = report
        logger.warning(
            "节点断线自动重分片计划: node=%s status=%s reason=%s",
            node_id, report.get("status", ""), report.get("reason_code", ""),
        )
        return report

    def _commit_ready_pipeline_reshard(self, plan: Mapping[str, Any]) -> Optional[bool]:
        """Commit the staged epoch after the matching load transaction is ready."""
        with self._layer_config_lock:
            coordinator = self._pipeline_reshard_coordinator
        if coordinator is None:
            return None
        staged = tuple(coordinator.snapshot().get("staged", []))
        if not staged:
            return None
        plan_id = str(plan.get("plan_id", "") or "")
        matching = next((
            item for item in staged if item.get("capacity_plan_id") == plan_id
        ), None)
        if matching is None:
            with self._layer_config_lock:
                self._pipeline_reshard_last_decision = {
                    "status": "rejected",
                    "reason_code": "pipeline_reshard_plan_mismatch",
                }
            logger.error(
                "重分片事务计划不匹配，拒绝发布: active=%s staged=%s",
                plan_id, [item.get("capacity_plan_id", "") for item in staged],
            )
            return False
        staged_plan = coordinator.staged_plan(str(matching.get("plan_id", "")))
        if staged_plan is None:
            return False
        try:
            ready_layout = pipeline_layout_from_capacity_plan(
                plan, node_metadata=self._pipeline_node_metadata(),
            )
        except PipelineNodeContractError:
            return False
        if ready_layout.contract_sha256 != staged_plan.candidate_layout.contract_sha256:
            return False
        # Reaching ready proves the master materialized its segment and every
        # worker ACKed its own model identity, range, and boundary duties.
        for requirement in staged_plan.requirements:
            coordinator.record_artifact(PipelineArtifactAvailability(
                node_id=requirement.node_id,
                model_sha256=requirement.model_sha256,
                artifact_kind=requirement.artifact_kind,
                layer_range=requirement.layer_range,
                artifact_sha256=requirement.artifact_sha256,
                verified=True,
                has_embedding=requirement.has_embedding,
                has_lm_head=requirement.has_lm_head,
            ))
        committed = coordinator.commit(
            staged_plan.plan_id, expected_epoch=staged_plan.base_epoch,
        )
        with self._layer_config_lock:
            self._pipeline_reshard_last_decision = committed.to_dict()
        if not committed.accepted:
            logger.error("重分片 epoch 提交被拒绝: %s", committed.reason_code)
        return committed.accepted

    def get_pipeline_reshard_status(self) -> dict:
        """Return an address-free recovery projection for the API and TUI."""
        with self._layer_config_lock:
            coordinator = self._pipeline_reshard_coordinator
            last = (
                dict(self._pipeline_reshard_last_decision)
                if self._pipeline_reshard_last_decision else None
            )
        if coordinator is None:
            return {"status": "inactive", "epoch": 0, "last_decision": last}
        snapshot = coordinator.snapshot()
        return {
            "status": snapshot["status"],
            "epoch": snapshot["epoch"],
            "active_contract_sha256": snapshot["layout"]["contract_sha256"],
            "staged": snapshot["staged"],
            "last_decision": last,
        }

    def get_pipeline_capacity_plan(
        self, eligible_node_ids: Optional[set[str]] = None,
        *, descriptor: Optional[dict] = None,
        require_distributed: bool = False,
        capacity_evidence=None,
    ) -> dict:
        """Compute an all-or-nothing metadata-only cluster capacity plan.

        ``require_distributed`` is request-scoped. It bypasses a cached
        single-node plan and asks the solver for at least two participating
        nodes, so an explicit distributed request cannot be satisfied by a
        master-only placement.
        """
        if (
            not require_distributed
            and eligible_node_ids is None
            and descriptor is None
        ):
            with self._layer_config_lock:
                active = (
                    dict(self._active_pipeline_capacity_plan)
                    if self._active_pipeline_capacity_plan else None
                )
                transaction = self._pipeline_load_transaction
                if transaction:
                    transaction_snapshot = {
                        "config_id": transaction.get("config_id", ""),
                        "generation": transaction.get("generation", 0),
                        "transaction_phase": transaction.get("phase", ""),
                        "prepared_node_count": len(
                            transaction.get("prepared_nodes", set())
                        ),
                        "ready_node_count": len(
                            transaction.get("ready_nodes", set())
                        ),
                        "worker_count": len(transaction.get("worker_ids", set())),
                        "reason_code": transaction.get("reason_code", ""),
                        "reason": transaction.get("reason", ""),
                    }
                    transaction_plan = dict(transaction.get("plan", {}))
                else:
                    transaction_snapshot = None
                    transaction_plan = None
            if active:
                if active.get("reason_code") == "pipeline_layer_range_coverage_insufficient":
                    active["reason"] = (
                        "advertised layer_ranges cannot cover the requested contiguous layer interval"
                    )
                return self._attach_pipeline_node_contract(active)
            if transaction_snapshot and transaction_plan:
                transaction_plan.update(transaction_snapshot)
                if transaction_snapshot.get("transaction_phase") in {
                    "rejected", "aborted", "invalidated",
                }:
                    # Retain terminal transaction diagnostics without
                    # exposing the stale plan as executable capacity.
                    transaction_plan["admitted"] = False
                if transaction_plan.get("reason_code") == "pipeline_layer_range_coverage_insufficient":
                    transaction_plan["reason"] = (
                        "advertised layer_ranges cannot cover the requested contiguous layer interval"
                    )
                return self._attach_pipeline_node_contract(transaction_plan)

        if descriptor is None:
            get_descriptor = getattr(self._host, "get_pipeline_descriptor", None)
            descriptor = get_descriptor() if callable(get_descriptor) else {}
        if not isinstance(descriptor, dict) or not descriptor:
            return {
                "status": "unavailable",
                "admitted": False,
                "reason_code": "pipeline_descriptor_unavailable",
                "assignments": [],
            }
        if not descriptor.get("pipeline_runtime_supported", False):
            return {
                "status": "rejected",
                "admitted": False,
                "reason_code": "pipeline_runtime_unsupported",
                "model_id": str(descriptor.get("model_id", "") or ""),
                "model_type": str(descriptor.get("model_type", "") or ""),
                "assignments": [],
            }
        from config import PIPELINE_CAPACITY_SAFETY_MARGIN, PIPELINE_PREFER_ALL_WORKERS

        # ★ Y 档第二条：relay 段认领的层由远端段执行 ⇒ 本机层节点的容量上界是 `k`
        #   （未被认领的前缀）。这里复用调度层的**同一套校验**（重叠 / 角色 / 连续性）；
        #   校验失败即拒绝（fail-closed），绝不静默按全量分层——那正是本线修的根因形态。
        try:
            total_layers_for_claim = int(descriptor.get("total_layers", 0) or 0)
            local_layer_budget = (
                self._relay_claimed_layer_prefix(total_layers_for_claim)
                if total_layers_for_claim > 0 else None
            )
            relay_claims = (
                self._relay_layer_claim_ranges(total_layers_for_claim)
                if total_layers_for_claim > 0 else None
            )
        except ValueError as exc:
            return {
                "status": "rejected",
                "admitted": False,
                "reason_code": "relay_layer_claim_invalid",
                "reason": str(exc),
                "assignments": [],
            }

        try:
            logger.info(
                "容量求解输入: local_layer_budget=%s relay_claims=%s eligible=%s",
                local_layer_budget, relay_claims,
                sorted(eligible_node_ids) if eligible_node_ids else None,
            )
            capacity_nodes = self._get_pipeline_capacity_nodes(eligible_node_ids)
            evidence_provenance = []
            if capacity_evidence is not None:
                capacity_nodes, evidence_provenance = capacity_evidence.project(capacity_nodes)
            # ★ 2026-10-23（#117）：**跨格式**主节点（PyTorch 描述器 + GGUF 层段）没有
            #   `source_model_sha256`。求解器靠它把 worker 段的 per-layer 字节补进
            #   部分为零的 `layer_weight_bytes`，缺了就 fail-closed 判描述器非法
            #   （`pipeline_capacity_descriptor_invalid`，实测供给卡在 preparing）。
            #   段的身份本来就在 `capacity_nodes` 的 `layer_artifacts` 里，在这里补上；
            #   求解器自身的 fail-closed 判据保持不变。
            if isinstance(descriptor, dict) and not descriptor.get("source_model_sha256"):
                for _node in capacity_nodes or []:
                    _sha = ""
                    for _artifact in _node.get("layer_artifacts") or ():
                        _sha = str(
                            _artifact.get("source_model_sha256", "") or ""
                        ).strip().lower()
                        if _sha:
                            break
                    if _sha:
                        descriptor = dict(descriptor)
                        descriptor["source_model_sha256"] = _sha
                        logger.info(
                            "容量描述器补齐源身份: source_model_sha256=%s（取自 %s 的层段工件）",
                            _sha[:16], _node.get("node_id"),
                        )
                        break
            result = solve_pipeline_capacity(
                descriptor,
                capacity_nodes,
                safety_margin=PIPELINE_CAPACITY_SAFETY_MARGIN,
                require_distributed=require_distributed,
                local_layer_budget=local_layer_budget,
                relay_claims=relay_claims,
                prefer_all_workers=PIPELINE_PREFER_ALL_WORKERS,
            )
            if (not result.get("admitted")
                    and any("candidate_layer_ranges" in node for node in capacity_nodes or [])):
                # ★ 2026-10-23（#117）：候选区间来自**供给时的切点**（用于锁定复用，
                #   见 model_supply_capacity 的证据投影）。容量确实不再满足时才放开，
                #   允许重新切分布局 —— 即"可用内存增加就不动切点、减少才重切"。
                released_nodes = [
                    {key: value for key, value in node.items()
                     if key != "candidate_layer_ranges"}
                    for node in capacity_nodes
                ]
                logger.info(
                    "锁定的切点不再满足容量，放开候选约束重排: locked=%s",
                    [node.get("candidate_layer_ranges") for node in capacity_nodes or []],
                )
                result = solve_pipeline_capacity(
                    descriptor,
                    released_nodes,
                    safety_margin=PIPELINE_CAPACITY_SAFETY_MARGIN,
                    require_distributed=require_distributed,
                    local_layer_budget=local_layer_budget,
                    relay_claims=relay_claims,
                    prefer_all_workers=PIPELINE_PREFER_ALL_WORKERS,
                )
        except PipelineCapacityError as exc:
            # ★ 2026-10-23（#117）：把**具体**原因打出来。此前调用方只记
            #   reason_code，"描述器非法"到底是缺 total_layers、缺
            #   source_model_sha256 还是别的，排查时完全不可见。
            logger.warning(
                "容量求解拒绝: reason_code=pipeline_capacity_descriptor_invalid detail=%s",
                exc, exc_info=True,
            )
            return {
                "status": "rejected",
                "admitted": False,
                "reason_code": "pipeline_capacity_descriptor_invalid",
                "reason": str(exc),
                "assignments": [],
            }
        if result.get("admitted"):
            from model_preflight import bind_model_preflight

            result = bind_model_preflight(
                result,
                descriptor=descriptor,
                nodes=capacity_nodes,
                require_distributed=require_distributed,
            )
        if evidence_provenance:
            result["capacity_evidence"] = evidence_provenance
        result["computed_at"] = time.time()
        result["transaction_phase"] = "planned" if result.get("admitted") else "rejected"
        result.setdefault("require_distributed", bool(require_distributed))
        logger.info(
            "容量求解结果: admitted=%s reason=%s assignments=%s",
            result.get("admitted"), result.get("reason_code"),
            [(a.get("node_id"), a.get("start_layer"), a.get("end_layer"))
             for a in result.get("assignments", [])],
        )
        if result.get("reason_code") == "pipeline_layer_range_coverage_insufficient":
            result["reason"] = (
                "advertised layer_ranges cannot cover the requested contiguous layer interval"
            )
        return self._attach_pipeline_node_contract(result, capacity_nodes=capacity_nodes)

    def _build_manual_pipeline_capacity_plan(
        self, assignments: list[dict], *, descriptor: Optional[dict] = None,
    ) -> dict:
        """Validate a user-forced split without replacing it with full-model fit.

        The normal solver intentionally minimizes participating nodes.  That is
        correct for automatic admission, but it would make a deliberate 21/24
        split impossible whenever the master can still hold the whole model.
        Manual plans retain the same byte and reserve checks while preserving
        the requested contiguous ranges.
        """
        from config import PIPELINE_CAPACITY_SAFETY_MARGIN
        from pipeline_capacity import _descriptor_costs, _required_bytes

        if descriptor is None:
            getter = getattr(self._host, "get_pipeline_descriptor", None)
            descriptor = getter() if callable(getter) else {}
        if not isinstance(descriptor, dict) or not descriptor:
            return {
                "status": "rejected", "admitted": False,
                "reason_code": "pipeline_descriptor_unavailable",
                "assignments": [],
            }

        try:
            layer_bytes, embedding_bytes, per_node_bytes, output_bytes = (
                _descriptor_costs(descriptor)
            )
        except PipelineCapacityError as exc:
            return {
                "status": "rejected", "admitted": False,
                "reason_code": "pipeline_capacity_descriptor_invalid",
                "reason": str(exc), "assignments": [],
            }

        total_layers = len(layer_bytes)
        node_ids = {str(item.get("node_id", "")) for item in assignments}
        capacity_records = self._get_pipeline_capacity_nodes(node_ids)
        records = {
            item["node_id"]: item
            for item in capacity_records
        }
        planned = []
        artifact_source_sha256 = ""
        for item in assignments:
            node_id = str(item.get("node_id", ""))
            start = int(item.get("start_layer", 0) or 0)
            end = int(item.get("end_layer", 0) or 0)
            if start < 0 or end <= start or end > total_layers:
                return {
                    "status": "rejected", "admitted": False,
                    "reason_code": "pipeline_capacity_manual_range_invalid",
                    "reason": f"node {node_id} has invalid range [{start},{end})",
                    "assignments": [],
                }
            record = records.get(node_id)
            if record is None:
                return {
                    "status": "rejected", "admitted": False,
                    "reason_code": "pipeline_capacity_manual_node_unavailable",
                    "reason": f"node {node_id} has no usable capacity",
                    "assignments": [],
                }
            local_cut = bool(
                isinstance(record.get("layer_budget"), dict)
                and record["layer_budget"].get("local_cut")
            )
            selected_artifact = None
            segment_mode = str(record.get("segment_mode", "") or "").lower()
            if not local_cut:
                artifacts = record.get("layer_artifacts")
                if isinstance(artifacts, list) and artifacts:
                    selected_artifact = next((
                        dict(artifact) for artifact in artifacts
                        if isinstance(artifact, dict)
                        and list(artifact.get("layer_range") or []) == [start, end]
                    ), None)
                    if selected_artifact is None:
                        return {
                            "status": "rejected", "admitted": False,
                            "reason_code": "pipeline_segment_contract_unsatisfied",
                            "reason": (
                                f"node {node_id} has no exact artifact for "
                                f"[{start},{end})"
                            ),
                            "assignments": [],
                        }
                    segment_mode = str(
                        selected_artifact.get("segment_mode", "") or ""
                    ).lower()
                    source_sha256 = str(
                        selected_artifact.get("source_model_sha256", "") or ""
                    ).lower()
                    if (
                        artifact_source_sha256 and source_sha256
                        and artifact_source_sha256 != source_sha256
                    ):
                        return {
                            "status": "rejected", "admitted": False,
                            "reason_code": "pipeline_segment_contract_unsatisfied",
                            "reason": "manual artifacts have different source identities",
                            "assignments": [],
                        }
                    artifact_source_sha256 = artifact_source_sha256 or source_sha256
                allowed_ranges = record.get("layer_ranges")
                if isinstance(allowed_ranges, list) and allowed_ranges and not any(
                    list(layer_range) == [start, end]
                    for layer_range in allowed_ranges
                    if isinstance(layer_range, (list, tuple))
                ):
                    return {
                        "status": "rejected", "admitted": False,
                        "reason_code": "pipeline_layer_range_coverage_insufficient",
                        "reason": f"node {node_id} does not advertise [{start},{end})",
                        "assignments": [],
                    }
                mode_valid = (
                    (segment_mode == "head" and start == 0)
                    or (segment_mode == "middle" and start > 0 and end < total_layers)
                    or (segment_mode == "tail" and end == total_layers)
                    or (not segment_mode and selected_artifact is None)
                )
                if not mode_valid:
                    return {
                        "status": "rejected", "admitted": False,
                        "reason_code": "pipeline_segment_contract_unsatisfied",
                        "reason": (
                            f"node {node_id} artifact mode {segment_mode!r} "
                            f"cannot execute [{start},{end})"
                        ),
                        "assignments": [],
                    }
            raw_bytes = sum(layer_bytes[start:end]) + per_node_bytes
            if bool(item.get("has_embedding")):
                raw_bytes += embedding_bytes
            if bool(item.get("has_lm_head")):
                raw_bytes += output_bytes
            required = _required_bytes(
                raw_bytes, record, PIPELINE_CAPACITY_SAFETY_MARGIN,
            )
            if required > int(record["capacity_bytes"]):
                return {
                    "status": "rejected", "admitted": False,
                    "reason_code": "pipeline_capacity_manual_insufficient",
                    "reason": (
                        f"node {node_id} requires {required} bytes, "
                        f"has {record['capacity_bytes']} bytes"
                    ),
                    "assignments": [],
                }
            planned_item = {
                **dict(item),
                "raw_weight_bytes": raw_bytes,
                "required_bytes": required,
                "capacity_bytes": record["capacity_bytes"],
                "headroom_bytes": record["capacity_bytes"] - required,
                "reserve_bytes": record["reserve_bytes"],
                "runtime_multiplier": record["runtime_multiplier"],
                "execution_device": record["execution_device"],
                "capacity_source": record["capacity_source"],
            }
            if selected_artifact is not None:
                planned_item["layer_artifact"] = selected_artifact
            planned.append(planned_item)

        plan_identity = {
            "model_id": descriptor.get("model_id", ""),
            "model_sha256": descriptor.get("model_sha256", ""),
            "assignments": [
                {
                    key: item.get(key)
                    for key in (
                        "node_id", "start_layer", "end_layer",
                        "required_bytes", "capacity_bytes",
                        "layer_artifact",
                    )
                }
                for item in planned
            ],
        }
        plan = {
            "schema_version": 1,
            "status": "admitted",
            "admitted": True,
            "reason_code": "manual_override",
            "plan_id": hashlib.sha256(
                json.dumps(plan_identity, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "model_id": str(descriptor.get("model_id", "") or ""),
            "model_type": str(descriptor.get("model_type", "") or ""),
            "model_sha256": str(descriptor.get("model_sha256", "") or ""),
            "total_layers": total_layers,
            "raw_model_bytes": sum(layer_bytes) + embedding_bytes
            + per_node_bytes + output_bytes,
            "safety_margin": PIPELINE_CAPACITY_SAFETY_MARGIN,
            "candidate_node_count": len(records),
            "excluded_nodes": [],
            "assignments": planned,
            "control_only_nodes": [
                node_id for node_id in records if node_id not in node_ids
            ],
            "participating_node_count": len(planned),
            "single_node_full_model_candidates": [],
            "aggregate_only": len(planned) > 1,
            "computed_at": time.time(),
        }
        from model_preflight import bind_model_preflight

        plan = bind_model_preflight(
            plan,
            descriptor=descriptor,
            nodes=capacity_records,
            require_distributed=len(planned) > 1,
        )
        return self._attach_pipeline_node_contract(plan, capacity_nodes=capacity_records)

    def _normalize_manual_assignments(self, assignments: list) -> list:
        """补齐手动区间的运行字段，并按节点能力放置 Embedding/LM Head。"""
        normalized = []
        for raw in sorted(assignments, key=lambda item: int(item.get("start_layer", 0))):
            item = dict(raw)
            start = int(item.get("start_layer", 0))
            end = int(item.get("end_layer", 0))
            node = self.nodes.get(item.get("node_id", ""))
            item["start_layer"] = start
            item["end_layer"] = end
            item["layers_count"] = end - start
            item["role"] = item.get("role") or (node.role if node else "client")
            item["score"] = self._compute_node_weight(
                node.device_info if node else {}
            )
            normalized.append(item)
        return self._resequence_assignments(normalized)

    @staticmethod
    def _manual_assignments_are_executable(assignments: list) -> bool:
        """The current runner always executes the master Embedding segment first."""
        masters = [
            item for item in assignments
            if item.get("node_id") == "master" or item.get("role") == "master"
        ]
        return bool(
            len(masters) == 1
            and int(masters[0].get("start_layer", -1)) == 0
            and masters[0].get("has_embedding", False)
        )

    def reset_layer_assignments(self) -> dict:
        """
        清除手动分层覆盖，恢复自动（dynamic）策略。

        仅主节点可调用。

        Returns:
            {"status": "ok", "strategy": "dynamic"}
        """
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可重置分层配置"}

        self._runtime_layer_override = None

        # 强制重新计算
        assignments = self.compute_layer_assignment()
        result = {
            "total": self._get_total_model_layers(),
            "strategy": "dynamic",
            "assignments": assignments,
            "computed_at": time.time(),
        }

        logger.info("分层配置已重置为自动策略")
        self.push_layer_config_to_clients()
        return {"status": "ok", "strategy": "dynamic", "assignments": result["assignments"]}

    def override_layer_assignments(self, assignments: list) -> dict:
        """
        手动覆盖分层配置（仅主节点）。

        验证:
          - 所有区间必须连续且完整覆盖 0-24
          - node_id 必须是已注册节点
          - 区间不能重叠

        Args:
            assignments: [{node_id, start_layer, end_layer}]

        Returns:
            {status, message, current_assignments}
        """
        self._require_control_write("cluster.layers.override")
        total_layers = self._get_total_model_layers()

        def invalid(reason_code: str, reason: str) -> dict:
            return {
                "status": "invalid",
                "reason_code": reason_code,
                "reason": reason,
            }

        if self._effective_role() != "master":
            return {
                "status": "denied",
                "reason_code": "layer_override_master_required",
                "reason": "仅主节点可覆盖分层配置",
            }

        # 基本验证
        if not assignments or not isinstance(assignments, list):
            return invalid("layer_assignments_empty", "分层配置不能为空")

        # 收集所有区间，排序验证连续性
        intervals = []
        for a in assignments:
            node_id = a.get("node_id", "")
            start = a.get("start_layer", 0)
            end = a.get("end_layer", 0)

            if node_id not in self.nodes:
                return invalid("layer_assignment_node_unknown", f"未知节点: {node_id}")
            # ★ 2026-09-19：判据由「平台」改为「**能力**」。原实现（Phase 4.1）按
            #   `node_type == "android"` 一律拒绝，理由写「Android 无 PyTorch 推理能力」——
            #   但 **Android 可以跑 llama.cpp/GGUF 引擎，而 llama.cpp 现在也能做层前向**
            #   （`forward_layers_from_hidden` / `forward_layers_to_hidden`）⇒ 原判据**过宽**，
            #   把「有 GGUF 引擎的 Android」也一并挡掉了。详见 `_node_supports_forward_layers`。
            node = self.nodes[node_id]
            if not _node_supports_forward_layers(node):
                return invalid(
                    "layer_assignment_node_unsupported",
                    f"节点 {node_id}（{node.node_type}）不支持层前向传播",
                )
            if start < 0 or end > total_layers or start >= end:
                return invalid(
                    "layer_assignment_range_invalid",
                    f"节点 {node_id} 区间 [{start}, {end}) 无效（范围 0-{total_layers}）",
                )
            intervals.append((start, end, node_id))

        # 排序后验证连续性
        intervals.sort(key=lambda x: x[0])

        covered = 0
        for start, end, node_id in intervals:
            if start != covered:
                return invalid(
                    "layer_assignment_range_discontinuous",
                    f"节点 {node_id} 区间 [{start}, {end}) 不连续（期望从 {covered} 开始）",
                )
            covered = end

        if covered != total_layers:
            return invalid(
                "layer_assignment_coverage_incomplete",
                f"总覆盖范围 {covered} ≠ {total_layers}，分层未完整覆盖",
            )

        master_intervals = [item for item in intervals if item[2] == "master"]
        if len(master_intervals) != 1 or master_intervals[0][0] != 0:
            return invalid(
                "layer_assignment_master_must_be_first",
                "主节点必须且只能承担从 Layer 0 开始的首段",
            )

        normalized_assignments = self._normalize_manual_assignments([
            {"node_id": node_id, "start_layer": start, "end_layer": end}
            for start, end, node_id in intervals
        ])
        self._runtime_layer_override = [dict(item) for item in normalized_assignments]

        # 推送到已连接从节点
        self.push_layer_config_to_clients()

        logger.info(f"分层配置已手动覆盖: {len(assignments)} 个节点")
        return {
            "status": "ok",
            "message": "分层配置已更新（手动模式），已推送至从节点",
            "current_assignments": {
                "total": total_layers,
                "strategy": "manual",
                "assignments": normalized_assignments,
                "computed_at": None,
            },
        }

    def push_layer_config_to_clients(self) -> None:
        with self._layer_config_push_lock:
            if not self.get_distributed_inference_enabled():
                return
            # Keep the transition check under the same lock used by the
            # publisher.  Otherwise a transition can begin after the check
            # but before the publish and still leak a transient release.
            with self._layer_config_lock:
                if self._layer_config_model_change_depth:
                    self._layer_config_push_deferred = True
                    logger.info(
                        "defer layer config push during model transition depth=%d",
                        self._layer_config_model_change_depth,
                    )
                    return
                retry_authoritative = bool(
                    self._layer_config_push_deferred_authoritative
                )
                retry_require_distributed = bool(
                    self._layer_config_push_deferred_require_distributed
                )
                if retry_authoritative:
                    self._authoritative_layer_sync_requests += 1
            try:
                self._push_layer_config_to_clients_locked(
                    require_distributed=retry_require_distributed,
                )
            except Exception:
                # Keep the deferred request armed so a later status/hello
                # callback can retry it after a transient publish failure.
                raise
            else:
                with self._layer_config_lock:
                    self._layer_config_push_deferred = False
                    self._layer_config_push_deferred_authoritative = False
                    self._layer_config_push_deferred_require_distributed = False
            finally:
                if retry_authoritative:
                    with self._layer_config_lock:
                        self._authoritative_layer_sync_requests = max(
                            0,
                            self._authoritative_layer_sync_requests - 1,
                        )

    def _begin_layer_config_model_change(self) -> None:
        """Fence transient pushes while the local model is being replaced."""
        # Serialize with an in-flight publish before marking the transition.
        # This gives the fence a linearization point: either a push completes
        # before the transition, or it is deferred after the transition starts.
        with self._layer_config_push_lock:
            with self._layer_config_lock:
                self._layer_config_model_change_depth += 1

    def _end_layer_config_model_change(self) -> None:
        """End a model transition and flush one deferred push."""
        with self._layer_config_push_lock:
            with self._layer_config_lock:
                self._layer_config_model_change_depth = max(
                    0, self._layer_config_model_change_depth - 1,
                )
                flush = (
                    self._layer_config_model_change_depth == 0
                    and self._layer_config_push_deferred
                )
                authoritative = bool(
                    self._layer_config_push_deferred_authoritative
                )
                require_distributed = bool(
                    self._layer_config_push_deferred_require_distributed
                )
                if flush:
                    self._layer_config_push_deferred = False
                    self._layer_config_push_deferred_authoritative = False
                    self._layer_config_push_deferred_require_distributed = False
            if flush:
                if authoritative:
                    with self._layer_config_lock:
                        self._authoritative_layer_sync_requests += 1
                try:
                    self._push_layer_config_to_clients_locked(
                        require_distributed=require_distributed,
                    )
                except Exception:
                    with self._layer_config_lock:
                        self._layer_config_push_deferred = True
                        self._layer_config_push_deferred_authoritative = (
                            self._layer_config_push_deferred_authoritative
                            or authoritative
                        )
                        self._layer_config_push_deferred_require_distributed = (
                            self._layer_config_push_deferred_require_distributed
                            or require_distributed
                        )
                    logger.warning(
                        "deferred layer config flush failed after model transition",
                        exc_info=True,
                    )
                finally:
                    if authoritative:
                        with self._layer_config_lock:
                            self._authoritative_layer_sync_requests = max(
                                0,
                                self._authoritative_layer_sync_requests - 1,
                            )

    def _task_worker_full_model_ids(self) -> set[str]:
        """Return healthy Task Worker peers that advertise a full model."""
        if (
            self._effective_role() != "master"
            or not TASK_WORKER_EXPERIMENTAL_ENABLED
        ):
            return set()
        try:
            status = self._task_worker_control.status(role="master")
        except Exception:
            return set()
        worker_ids: set[str] = set()
        for worker in status.get("workers", []):
            if not isinstance(worker, dict):
                continue
            if not (
                worker.get("healthy")
                and worker.get("accepted")
                and worker.get("manual_stage_dispatch_enabled")
            ):
                continue
            capabilities = worker.get("capabilities") or {}
            if not isinstance(capabilities, dict):
                continue
            # A worker that advertises an exact layer range is eligible for the
            # layer pipeline even when it also exposes a full-model identity.
            # Only full-model-only workers keep the legacy opt-out priority.
            stage_types = capabilities.get("stage_types", [])
            has_layer_stage = (
                isinstance(stage_types, list)
                and "layer_forward" in stage_types
                and bool(capabilities.get("layer_ranges"))
            )
            if capabilities.get("models") and not has_layer_stage:
                node_id = str(worker.get("node_id", "") or "")
                # ★ relay 宿主不是 Full Worker（判据唯一的出处在 `_is_relay_host()`）：
                #   它不能声明 `forward_layers`（声明了就会拒绝 legacy 层配置，而 relay
                #   委派正是走那条通道），所以这里的 `has_layer_stage` 恒为假；但它要的
                #   恰恰是那份 legacy 层配置，不该被当 Full Worker 释放预留。
                if node_id and not self._is_relay_host(node_id):
                    worker_ids.add(node_id)
        return worker_ids

    # ================================================================
    # TCP 消息处理
    # ================================================================

    def _on_tcp_registration_confirmed(self, client_id: str) -> None:
        """Push scheduling state only after the REGISTER ACK is on the wire."""
        if self._effective_role() != "master":
            return
        with self._nodes_lock:
            node = self.nodes.get(client_id)
            if node is None or node.role == NodeRole.MASTER:
                return
        epoch_getter = getattr(self._tcp_server, "get_authenticated_peer_epoch", None)
        if callable(epoch_getter) and epoch_getter(client_id) is None:
            # A late callback for a replaced socket cannot mark the successor
            # pending, replay its cleanup or broadcast a new scheduling view.
            return

        # Fence the socket before any recovery/cleanup delivery. REGISTER ACK
        # is not a task-worker HELLO_ACK; Android cannot accept native commands
        # while it is still negotiating that acknowledgement.
        task_worker_handshake_pending = (
            TASK_WORKER_EXPERIMENTAL_ENABLED
            and getattr(node, "node_type", "pc") in {"pc", "android"}
        )
        if task_worker_handshake_pending:
            self._task_worker_control.mark_worker_connection_pending(
                client_id,
                connection_generation=self._assignment_connection_generation(client_id),
            )

        qwen3_release = []
        with self._layer_config_lock:
            qwen3_transaction = self._qwen3_pipeline_dry_run
            if (
                qwen3_transaction is not None
                and qwen3_transaction.network_dispatch
                and qwen3_transaction.phase in {"aborted", "releasing"}
                and client_id in qwen3_transaction.worker_ids
            ):
                qwen3_release = [
                    item for item in qwen3_transaction.release_messages()
                    if item.get("node_id") == client_id
                ]
        if qwen3_release:
            self._dispatch_qwen3_loopback_messages(
                qwen3_release, best_effort=True,
            )

        cleanup_replayed = self._replay_pipeline_model_cleanup_for_node(
            client_id,
        )

        # REGISTER ACK precedes the client's task-worker hello. Fence this
        # connection from legacy layer assignment until that hello arrives.
        # The hello handler resolves the fence and triggers a fresh push.
        if cleanup_replayed:
            logger.info(
                "legacy_layer_config skipped reason=model_cleanup_pending node=%s",
                client_id,
            )
        elif task_worker_handshake_pending:
            logger.info(
                "legacy_layer_config skipped reason=task_worker_handshake_pending node=%s",
                client_id,
            )
        else:
            self.push_layer_config_to_clients()
        self._push_node_list_to_client(client_id)
        self._push_node_update_to_all_clients(
            client_id, "add", self.nodes.get(client_id)
        )

    def _on_tcp_message(self, client_id: str, msg: dict) -> None:
        """
        TCP 消息回调（由 TCPServer 调用）。

        根据消息类型更新节点状态、心跳等。
        """
        msg_type = msg.get("type", "")

        if msg_type == "register":
            data = msg.get("data", {})
            if data.get("node_type") == "pipeline_peer":
                reason = "worker peer 数据连接已禁用；层段数据必须经 master 转发"
                if self._tcp_server:
                    self._tcp_server.reject_client(client_id, reason)
                logger.warning(
                    "拒绝 legacy pipeline_peer 注册: client=%s reason=%s",
                    client_id, reason,
                )
                return
            # get_client_info 对未知 client_id 返回 None（如从节点收到主节点的
            # register 拒绝回执、或注册与断开竞态），必须兜底为空 dict，
            # 否则回调崩溃会杀死接收循环、触发无限重连
            client_info = (
                self._tcp_server.get_client_info(client_id)
                if self._tcp_server else {}
            ) or {}
            advertised_addr = (
                client_info.get("advertised_addr")
                or data.get("advertised_address")
                or client_info.get("addr", "")
            )
            device_info = dict(data.get("device_info", {}) or {})
            peer_addr = client_info.get("peer_addr", "")
            if peer_addr:
                device_info["tcp_peer_addr"] = peer_addr
            device_info["tcp_advertised_addr"] = advertised_addr
            registered = self.register_node(
                node_id=client_id,
                role=data.get("role", ""),
                address=advertised_addr,
                hostname=data.get("hostname", ""),
                device_info=device_info,
                network_type=data.get("network_type", client_info.get("network_type", "unknown")),
                node_type=data.get("node_type", "pc"),
                model_sha256=data.get("model_sha256", ""),
            )
            if not registered:
                reason = "节点注册被调度器拒绝：容量已满或角色无效"
                logger.warning("event=tcp_register_rejected client_id=%s reason=%s", client_id, reason)
                if self._tcp_server:
                    self._tcp_server.reject_client(client_id, reason)
                return

            # Arm the fence before TCPServer confirms REGISTER. Unconfirmed
            # sockets are also excluded by the generation-bound send path.
            task_worker_handshake_pending = (
                self._effective_role() == "master"
                and TASK_WORKER_EXPERIMENTAL_ENABLED
                and data.get("node_type", "pc") in {"pc", "android"}
            )
            # ★ #106/#114：fence **只能**由 task-worker HELLO 解除，所以同一个
            #   边沿必须把"本集群要求协商"写进 REGISTER ACK。否则没开本地实验
            #   开关的节点永远不发 hello ⇒ 它的 release 债务永远投递不出去，
            #   并经 _retirement_state 把整集群 /api/models/load 卡成 503。
            #   声明与加 fence 共用同一条件，两者恒等。
            declare_handshake = getattr(self._tcp_server, "set_task_worker_hello_required", None)
            if callable(declare_handshake):
                declare_handshake(client_id, task_worker_handshake_pending)
            if task_worker_handshake_pending:
                # This callback precedes REGISTER ACK. Use the exact inbound
                # registration epoch even while that socket is unconfirmed.
                epoch_getter = getattr(self._tcp_server, "get_registration_peer_epoch", None)
                epoch = (epoch_getter(client_id) if callable(epoch_getter)
                         else self._assignment_connection_generation(client_id))
                if epoch is not None:
                    self._task_worker_control.mark_worker_connection_pending(
                        client_id, connection_generation=epoch,
                    )

            # TCPServer sends REGISTER ACK first, then invokes
            # _on_tcp_registration_confirmed.  Keeping all business pushes in
            # that post-ACK callback prevents layer_config from racing the
            # client's registration handshake.

        elif msg_type == "heartbeat":
            # ★ 2026-10-08：诊断期已过（定位结论：心跳确实到达 master，且 `tcp_comm` 层
            #   已回 ACK），降级为 debug 以免每 15 秒×N worker 刷屏。
            logger.debug("event=tcp_heartbeat_received client_id=%s", client_id)
            if self._effective_role() == "master":
                self._task_worker_control.mark_worker_heartbeat(client_id)
            client_info = (
                self._tcp_server.get_client_info(client_id)
                if self._tcp_server else None
            ) or {}
            with self._nodes_lock:
                if client_id in self.nodes:
                    node = self.nodes[client_id]
                    node.last_heartbeat = time.time()
                    if client_info.get("avg_rtt_ms"):
                        node.avg_rtt_ms = client_info["avg_rtt_ms"]
                        node.last_rtt_ms = client_info.get(
                            "last_rtt_ms", node.last_rtt_ms,
                        )

            # ★ 2026-10-08（真机复测根因）：**必须回心跳应答**。
            #
            #   Android task worker 的 `SocketTaskWorkerTransport` 设了 `soTimeout = 45_000`
            #   （`TaskWorkerClient.kt:472`/`:480`），而它每 15 秒发一次 `heartbeat`。此前
            #   master 只 `mark_worker_heartbeat` + 刷新 `last_heartbeat`、**从不回包** ⇒ 连接
            #   在 45 秒后必然 `SocketTimeoutException` ⇒ 断开重连。真机表现正是「Y700 每 ~37
            #   秒客户端主动断开一次」（`tcp_comm: 客户端 android-21af7c52 已断开`，间隔 37/36
            #   秒），落在断开窗口里的请求就报
            #   `pipeline_layer_range_coverage_insufficient` ⇒ "分布式时好时坏"。
            #
            #   PC 从节点那条路径早在 `tcp_comm.py:1711` 回 `HEARTBEAT_ACK`，task worker 这条
            #   一直缺失；Android 侧 `receiveEvent()` 已显式把 `heartbeat_ack` 映射为
            #   `TaskWorkerInboundEvent.HeartbeatAck`（吞掉、不抛），因此回包是安全的。
            tcp_server = self._tcp_server
            if tcp_server is not None:
                from transport_port import MessageType

                payload = msg.get("data") if isinstance(msg, dict) else None
                t_send = payload.get("t_send", 0) if isinstance(payload, dict) else 0
                try:
                    tcp_server.send_to_client(
                        client_id, {"t_send": t_send}, MessageType.HEARTBEAT_ACK,
                    )
                except Exception as exc:  # noqa: BLE001 - 应答失败不影响主流程
                    logger.debug(
                        "event=tcp_heartbeat_ack_failed client_id=%s error=%s",
                        client_id, exc,
                    )

        elif msg_type == "model_prepare":
            payload = msg.get("data")
            if isinstance(payload, dict):
                if self._effective_role() == "master":
                    self._handle_worker_prepare_ack(client_id, payload, defer_native_work=True)
                elif client_id == "master":
                    self._worker_prepare.handle(payload, self._tcp_client)

        elif msg_type == "model_supply":
            payload = msg.get("data")
            if isinstance(payload, dict):
                if self._effective_role() == "master":
                    self._model_supply.handle(client_id, payload)
                elif client_id == "master":
                    self._model_supply_worker.handle(payload, self._tcp_client)

        elif msg_type == "task_worker":
            self._handle_task_worker_message(client_id, msg)

        elif msg_type == "status_res":
            # 从节点状态上报
            data = msg.get("data", {})
            profile_changed = False
            with self._nodes_lock:
                if client_id in self.nodes:
                    if "state" in data:
                        try:
                            self.nodes[client_id].state = NodeState(data["state"])
                        except ValueError:
                            pass
                    device_info = data.get("device_info")
                    if isinstance(device_info, dict) and device_info:
                        updated_info = dict(device_info)
                        previous_info = dict(self.nodes[client_id].device_info or {})
                        # Keep the accepted hello contract across hardware-only
                        # STATUS_RES refreshes; hello owns runtime capability changes.
                        for key in ("tcp_peer_addr", "tcp_advertised_addr", "engine_capabilities"):
                            if key in previous_info and key not in updated_info:
                                updated_info[key] = previous_info[key]
                        profile_changed = previous_info != updated_info
                        self.nodes[client_id].device_info = updated_info
            if profile_changed and self._effective_role() == "master":
                logger.info("从节点设备画像已更新，重新计算分层: node=%s", client_id)
                try:
                    self.push_layer_config_to_clients()
                except Exception as e:
                    logger.warning("设备画像更新后重新下发分层失败: %s", e, exc_info=True)

        elif msg_type == "error":
            data = msg.get("data", {})
            with self._nodes_lock:
                if client_id in self.nodes:
                    self.nodes[client_id].error_count += 1
            logger.error(f"节点 {client_id} 上报错误: {data.get('message', 'unknown')}")

        elif msg_type == "infer_forward":
            # 从节点转发推理请求给主节点
            self.handle_infer_forward(client_id, msg)

        elif msg_type == "infer_cancel":
            data = msg.get("data", {})
            forward_request_id = str(data.get("forward_request_id", ""))
            with self._forward_cancel_lock:
                cancel_event = self._forward_cancel_events.get(
                    (client_id, forward_request_id)
                )
            if cancel_event is not None:
                cancel_event.set()
                logger.info(
                    "收到转发推理取消: client=%s request=%s",
                    client_id, forward_request_id,
                )

        elif msg_type == "infer_result":
            # 主节点返回推理结果给从节点
            data = msg.get("data", {})
            forward_request_id = str(data.get("forward_request_id", ""))
            result_entry = {
                "task_id": data.get("task_id", ""),
                "forward_request_id": forward_request_id,
                "status": data.get("status", "ok"),
                "content": data.get("content", ""),
                "metrics": data.get("metrics", {}),
                "error": data.get("error", ""),
                "thinking_content": data.get("thinking_content"),
                "followups": data.get("followups", []),
            }
            with self._client_pending_lock:
                if not forward_request_id and len(self._client_pending_events) == 1:
                    # 兼容尚未升级的主节点；并发时绝不猜测结果归属。
                    forward_request_id = next(iter(self._client_pending_events))
                    result_entry["forward_request_id"] = forward_request_id
                event = self._client_pending_events.get(forward_request_id)
                if event is not None:
                    self._client_pending_results[forward_request_id] = result_entry
                    event.set()
            if event is None:
                logger.warning(
                    "丢弃无等待者的迟到推理结果: task=%s request=%s",
                    data.get("task_id", ""), forward_request_id or "-",
                )
            else:
                logger.info(
                    "收到推理结果: task=%s request=%s len=%d",
                    data.get("task_id", ""), forward_request_id,
                    len(data.get("content", "")),
                )

        elif msg_type == "role_transfer":
            # 从节点收到主节点转让通知
            self._handle_role_transfer(client_id, msg)

        elif msg_type == "role_transfer_ack":
            # 主节点收到从节点的转让确认
            self._handle_role_transfer_ack(client_id, msg)

        elif msg_type == "spare_master_designate":
            # 从节点收到备用主节点指定通知
            self._handle_spare_master_designate(client_id, msg)

        elif msg_type == "spare_master_designate_ack":
            # 主节点收到从节点的备用指定确认
            self._handle_spare_master_designate_ack(client_id, msg)

        elif msg_type == "spare_master_activate":
            # 备用主节点收到激活（暂代主节点职责）通知
            self._handle_spare_master_activate(client_id, msg)

        elif msg_type == "spare_master_activate_ack":
            # 主节点收到备用主节点的激活确认
            self._handle_spare_master_activate_ack(client_id, msg)

        elif msg_type == "spare_master_deactivate":
            # 备用主节点收到新主节点的接管通知
            self._handle_spare_master_deactivate(client_id, msg)

        elif msg_type == "node_list_sync":
            # ---- 从节点收到主节点推送的全量节点列表 ----
            data = msg.get("data", {})
            if data.get("request") == "node_list":
                # 从节点请求节点列表 → 主节点响应
                self._push_node_list_to_client(client_id)
            else:
                # 从节点接收全量节点列表
                nodes_data = data.get("nodes", [])
                if nodes_data:
                    self._apply_node_list_sync(nodes_data)
                    logger.info(
                        f"📋 收到主节点推送的全量节点列表: {len(nodes_data)} 个节点"
                    )

        elif msg_type == "node_update":
            # ---- 从节点收到单节点变更通知 ----
            data = msg.get("data", {})
            action = data.get("action", "")
            node_data = data.get("node", {})
            if action and node_data:
                self._apply_node_update(action, node_data)
                logger.info(
                    f"📋 节点变更: {action} {node_data.get('node_id', '?')}"
                )

        elif msg_type == "layer_forward":
            # ---- 从节点：收到主节点的层前向传播指令 ----
            threading.Thread(
                target=self._handle_layer_forward,
                args=(client_id, msg),
                name=f"layer-forward-{msg.get('data', {}).get('task_id', 'unknown')}",
                daemon=True,
            ).start()

        elif msg_type == "layer_result":
            # ---- 主节点：收到从节点的层前向传播结果 ----
            self._handle_layer_result(client_id, msg)

        elif msg_type == "chain_forward":
            # ---- 从节点：收到 master 转发的下一逻辑段 hidden ----
            threading.Thread(
                target=self._handle_chain_forward,
                args=(client_id, msg),
                name=f"chain-forward-{msg.get('data', {}).get('task_id', 'unknown')}",
                daemon=True,
            ).start()

        elif msg_type == "chain_forward_ack":
            # ---- 主节点：收到链式转发每跳 ACK / 错误回报 ----
            self._handle_chain_forward_ack(client_id, msg)

        elif msg_type == "pipeline_done":
            # ---- 从节点：流水线任务完成，清理 KV 缓存 ----
            data = msg.get("data", {})
            task_id = data.get("task_id", "")
            if task_id:
                with self._layer_config_lock:
                    self._local_pipeline_cancelled.discard(task_id)
                    self._local_pipeline_steps.pop(task_id, None)
                with self._kv_cache_lock:
                    if task_id in self._kv_cache:
                        del self._kv_cache[task_id]
                        logger.info(f"🧹 流水线任务 {task_id} KV 缓存已清理")
                self._record_local_pipeline_participation(task_id, success=True)
                self._finish_local_pipeline_task(task_id)

        elif msg_type == "pipeline_abort":
            # ---- 从节点/主节点：流水线任务取消 ----
            data = msg.get("data", {})
            task_id = data.get("task_id", "")
            if task_id:
                with self._layer_config_lock:
                    participated = task_id in self._active_pipeline_task_ids
                    self._local_pipeline_steps.pop(task_id, None)
                self._mark_local_pipeline_cancelled(task_id)
                with self._kv_cache_lock:
                    if task_id in self._kv_cache:
                        del self._kv_cache[task_id]
                if participated and data.get("count_error", True):
                    self._record_local_pipeline_participation(task_id, success=False)
                self._finish_local_pipeline_task(task_id)
            logger.warning(f"⚠️ 流水线任务 {task_id} 已取消")

        elif msg_type == "pipeline_pause":
            # ---- 从节点/主节点：流水线暂停（二期协同抢占，协议预留）----
            logger.info(f"⏸️ 收到 PIPELINE_PAUSE: client={client_id}")

        elif msg_type == "pipeline_resume":
            # ---- 从节点/主节点：流水线恢复（二期协同抢占，协议预留）----
            logger.info(f"▶️ 收到 PIPELINE_RESUME: client={client_id}")

        elif msg_type == "layer_config":
            # ---- 从节点：收到主节点推送的分层配置 ----
            data = msg.get("data", {})
            self._schedule_layer_config(client_id, data)

        elif msg_type == "layer_config_ack":
            # ---- 主节点：从节点完成模型层加载后的确认 ----
            self._handle_layer_config_ack(client_id, msg)

        elif msg_type == "qwen3_pipeline_dry_run":
            self._handle_qwen3_loopback_request(client_id, msg)

        elif msg_type == "qwen3_pipeline_dry_run_ack":
            self._handle_qwen3_loopback_ack(client_id, msg)

        elif msg_type == "layer_worker_opt_out":
            # ---- 主节点：从节点明确切换为本地推理，不再分配模型层 ----
            if self._effective_role() == "master":
                self._handle_layer_worker_opt_out(client_id, msg)
            else:
                logger.warning("非主节点忽略分层 worker 退出请求: %s", client_id)

        elif msg_type == "layer_worker_opt_in":
            # ---- 主节点：从节点重新允许接收 PyTorch 分层配置 ----
            if self._effective_role() == "master":
                self._handle_layer_worker_opt_in(client_id, msg)
            else:
                logger.warning("非主节点忽略分层 worker 加入请求: %s", client_id)

        elif msg_type == "log_request":
            # ---- L5: 主节点拉取从节点最近日志 ----
            data = msg.get("data", {})
            limit = data.get("limit", 100)
            try:
                # Keep this branch self-contained: log aggregation can be
                # invoked independently of the inference message handlers.
                from transport_port import MessageType

                callbacks = self._require_callbacks()
                entries, _ = callbacks.snapshot_recent_logs()
                filtered = callbacks.filter_recent_logs(
                    entries,
                    level=data.get("level", ""),
                    name=data.get("name", ""),
                    node_id=data.get("node_id", ""),
                    request_id=data.get("request_id", ""),
                )
                result_entries = filtered[-limit:]
                response = {
                    "node_id": self.get_effective_node_id(),
                    "logs": result_entries,
                    "count": len(result_entries),
                    "matched": len(filtered),
                    "buffer_size": len(entries),
                }
                # 从节点使用 _tcp_client.send_data 回复主节点
                tcp_client = getattr(self, '_tcp_client', None)
                if tcp_client:
                    tcp_client.send_data(response, MessageType.LOG_RESPONSE)
                    logger.debug(
                        "event=log_aggregation_sent node_id=%s count=%d requester=%s",
                        self.get_effective_node_id(), len(result_entries), client_id,
                    )
                else:
                    logger.warning(
                        "event=log_aggregation_send_failed node_id=%s reason=tcp_client_none",
                        self.get_effective_node_id(),
                    )
            except Exception as e:
                logger.warning(
                    "event=log_aggregation_error node_id=%s error=%s",
                    self.get_effective_node_id(), str(e)[:200],
                )

        elif msg_type == "log_response":
            # ---- L5: 主节点收到从节点日志响应 ----
            data = msg.get("data", {})
            worker_node_id = data.get("node_id", client_id)
            with self._pending_log_lock:
                self._pending_log_responses[worker_node_id] = data
                _evt = self._pending_log_events.get(worker_node_id)
                if _evt is not None:
                    _evt.set()

        else:
            logger.debug(f"未知消息类型: {msg_type}, client={client_id}")

    def _set_pipeline_result_error(self, task_id: str, node_id: str,
                                   error: str, step: int = -1) -> None:
        """主节点侧：写入节点错误并唤醒等待该节点结果的流水线线程。"""
        if not task_id or not node_id:
            return
        key = f"{task_id}:{node_id}"
        with self._pipeline_lock:
            if task_id not in self._pipeline_active_tasks:
                return
            self._pipeline_results[key] = {
                "task_id": task_id,
                "node_id": node_id,
                "error": error,
                "step": step,
            }
            event = self._pipeline_events.get(key)
            if event is not None:
                event.set()

    def _on_tcp_disconnect(self, client_id: str) -> None:
        """TCP 断连回调（由 TCPServer 调用）"""
        abort_details = None
        qwen3_disconnect = None
        retirement_records = []
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            cleanup_changed = False
            for record in self._pipeline_model_cleanup_ledger.values():
                if record.get("node_id") == client_id:
                    record["state"] = "await_reconnect"
                    cleanup_changed = True
            if cleanup_changed:
                if isinstance(transaction, dict):
                    pending = set(transaction.get("finalize_pending", set()))
                    pending.discard(client_id)
                    transaction["finalize_pending"] = pending
                    transaction.get("finalize_retry_state", {}).pop(
                        client_id, None,
                    )
                self._persist_pipeline_lifecycle_locked()
            if (
                transaction
                and transaction.get("phase") in {
                    "preparing", "committing_local", "committing",
                    "committing_global",
                }
                and client_id in set(transaction.get("worker_ids", set()))
            ):
                abort_details = (
                    str(transaction.get("config_id", "")),
                    "pipeline_worker_disconnected",
                    f"worker {client_id} disconnected during transaction",
                )
            active_plan = self._active_pipeline_capacity_plan
            transaction_plan = (
                transaction.get("plan") if isinstance(transaction, dict) else None
            )

            def _plan_uses_node(plan: object) -> bool:
                if not isinstance(plan, dict):
                    return False
                return any(
                    str(item.get("node_id", "")) == client_id
                    for item in plan.get("assignments", [])
                    if isinstance(item, dict)
                )

            uses_active_plan = _plan_uses_node(active_plan)
            uses_transaction_plan = _plan_uses_node(transaction_plan)
            if uses_active_plan or (
                transaction
                and transaction.get("phase") == "ready"
                and (
                    client_id in set(transaction.get("worker_ids", set()))
                    or uses_transaction_plan
                )
            ):
                self._active_pipeline_capacity_plan = None
                if transaction and abort_details is None:
                    transaction["published"] = bool(transaction.get("published") or transaction.get("phase") == "ready")
                    if transaction["published"]:
                        for node, expected in transaction.get("expected", {}).items():
                            identity = expected.get("worker_prepare_identity")
                            if isinstance(identity, dict):
                                retirement_records.append(self._record_pipeline_model_cleanup_locked(
                                    node_id=node, candidate_config_id=identity["config_id"],
                                    generation=self._layer_config_generation, action="release",
                                    assignment_id=identity["assignment_id"],
                                    payload={"worker_prepare_identity": identity, "release": True},
                                ))
                    transaction["phase"] = "invalidated"
                    transaction["reason_code"] = "pipeline_worker_disconnected"
                    transaction["reason"] = (
                        f"worker {client_id} disconnected after plan activation"
                    )
            qwen3_transaction = self._qwen3_pipeline_dry_run
            if qwen3_transaction is not None:
                qwen3_disconnect = qwen3_transaction.disconnect(client_id)
                if qwen3_disconnect is not None and qwen3_transaction.network_dispatch:
                    worker = self._qwen3_loopback_workers.get(
                        qwen3_transaction.contract["contract_sha256"], None,
                    )
                    if worker is not None:
                        worker.abort_sidecar_sessions()
                    self._qwen3_loopback_workers.pop(
                        qwen3_transaction.contract["contract_sha256"], None,
                    )
        if abort_details is not None:
            self._abort_pipeline_load_transaction(*abort_details)
        if retirement_records:
            with self._layer_config_lock:
                self._persist_pipeline_lifecycle_locked()
            for record in retirement_records:
                self._dispatch_pipeline_model_cleanup(record)
        # A disconnected socket can no longer satisfy the assignment that was
        # registered for it.  Drop that per-node expectation before the
        # reconnect callback publishes a fresh generation; otherwise an old
        # ACK/retry entry can participate in readiness during the reconnect
        # window and race the new configuration.
        self._clear_layer_config_state(client_id)
        if qwen3_disconnect is not None:
            logger.warning(
                "Qwen3 dry-run 因节点断线中止: node=%s contract=%s",
                client_id,
                self.get_qwen3_pipeline_dry_run_status().get(
                    "contract_sha256", ""
                ),
            )
            outbound = qwen3_disconnect.get("outbound", [])
            if outbound:
                self._dispatch_qwen3_loopback_messages(
                    outbound, best_effort=True,
                )
        self._task_worker_control.disconnect_worker(client_id)
        with self._task_worker_stage_lock:
            remote_provider = self._remote_task_worker_providers.get(client_id)
        if remote_provider is not None:
            # ★ 2026-10-07（DIST-NEXT-4b）：断线走的 reason code 显式给定（单一来源），
            #   provider 侧会记 `event=task_worker_reservations_released`。
            remote_provider.notify_disconnect(
                reason_code=RELEASE_REASON_DISCONNECTED,
            )
        with self._forward_cancel_lock:
            client_cancellations = [
                event for (owner_id, _), event
                in self._forward_cancel_events.items()
                if owner_id == client_id
            ]
        for event in client_cancellations:
            event.set()
        self._fail_pending_pipeline_results_for_node(
            client_id, f"节点 {client_id} TCP 连接已断开"
        )
        with self._nodes_lock:
            old_state = None
            if client_id in self.nodes:
                old_state = self.nodes[client_id].state.value
                self.nodes[client_id].state = NodeState.OFFLINE
                logger.info(
                    f"🔌 节点 {client_id} TCP 断开，已标记 offline "
                    f"(old_state={old_state}, role={self._effective_role()})"
                )
            else:
                logger.debug(f"未知节点 {client_id} TCP 断开，跳过状态标记")
            # 快照数据后释放锁，避免持锁进行 TCP 发送
            need_push = (self._effective_role() == "master")
            node_info = self.nodes.get(client_id)

        # L5: 清理断连节点的待处理日志聚合状态（避免残留 Event 导致超时等待）
        with self._pending_log_lock:
            _evt = self._pending_log_events.pop(client_id, None)
            if _evt is not None:
                _evt.set()  # 唤醒等待线程（将收到空结果）
            self._pending_log_responses.pop(client_id, None)

        if need_push:
            self._push_node_update_to_all_clients(
                client_id, "update", node_info
            )
        self._observe_node_disconnected(client_id)
        reshard = self._stage_pipeline_reshard_after_disconnect(client_id) if need_push else None
        if need_push:
            if reshard and reshard.get("accepted"):
                self.request_authoritative_layer_sync(require_distributed=True)
            else:
                self.push_layer_config_to_clients()

    def _on_master_connection_lost(self, source_client=None, *, notify_auto_role=True) -> None:
        """Worker-side cleanup when PIPELINE_DONE/ABORT can no longer arrive."""
        if (source_client is not None
                and getattr(self, "_tcp_client", None) is not source_client):
            logger.debug("忽略旧主节点连接的迟到断连回调")
            return

        if notify_auto_role:
            self.auto_role_on_disconnect()
        self._cleanup_task_worker_prepare("coordinator_disconnected", client=source_client)
        self._task_worker_control.disconnect_coordinator()
        with self._task_worker_stage_lock:
            for active in self._task_worker_active_attempts.values():
                active.cancel_reason = "coordinator_disconnected"
                active.cancel_event.set()

        with self._client_pending_lock:
            for request_id, event in self._client_pending_events.items():
                self._client_pending_results[request_id] = {
                    "forward_request_id": request_id,
                    "status": "error",
                    "content": "",
                    "metrics": {},
                    "error": "与主节点的连接已断开",
                }
                event.set()
        with self._kv_cache_lock:
            active_tasks = list(self._kv_cache)
            self._kv_cache.clear()
        with self._layer_config_lock:
            active_tasks.extend(self._active_pipeline_task_ids)
            active_tasks = list(dict.fromkeys(active_tasks))
            self._active_pipeline_task_ids.clear()
            self._local_pipeline_steps.clear()
            self._pending_layer_config = None
            self._active_layer_config = None
            self._last_layer_config_ack_payload = None
            self._pipeline_worker_reserved = False
            self._layer_config_inflight.clear()
            self._layer_config_receive_sequence += 1
            self._latest_layer_config_receive_sequence = (
                self._layer_config_receive_sequence
            )
            self._latest_layer_config_generation = 0
            self._latest_layer_assignment_id = ""
        for task_id in active_tasks:
            self._record_local_pipeline_participation(task_id, success=False)
        if active_tasks:
            logger.warning(
                "主节点连接中断，已清理 %d 个本地流水线任务",
                len(active_tasks),
            )

    # ================================================================
    # 节点列表同步（主 → 从）
    # ================================================================

    # ================================================================
    # 角色转让 — 主节点身份转移
    # ================================================================

    # ================================================================
    # 备用主节点管理
    # ================================================================

    # ---- 备用主节点：激活（暂代） / 接管（退出暂代） ----

    # ---- 新主节点启动：向备用主节点发送接管通知 ----

    # ================================================================
    # 任务调度
    # ================================================================

    def start_infer_task(self, prompt: str, request_id: str = None) -> str:
        """
        启动一轮完整推理任务。

        Args:
            prompt: 用户输入文本
            request_id: API 请求 ID（L5: 链路追踪）

        Returns:
            task_id: 任务唯一标识

        """
        # 节点就绪与降级由 run_pipeline_safe() 统一决定。这里不能先把
        # worker 标成 BUSY，否则 readiness 会把 BUSY 误判为离线并强制回退。
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task = InferenceTask(task_id=task_id, prompt=prompt)
        task.state = "running"
        task.start_time = time.time()

        with self._task_lock:
            self._current_task = task
            self._infer_tasks[task_id] = task

        logger.info(
            "event=infer_task_start task_id=%s request_id=%s prompt_len=%d",
            task_id, request_id or "-", len(prompt),
        )

        # ★ 2026-10-08（遗留清理）：此处原为「TODO: 触发流水线推理流程」+ 三步流程占位注释。
        #   该流程**早已实现**，且不在本类内 —— 产品入口是 `api_server` 的分布式链路：
        #   Route-A（`stage_offer_v3`）由 master 做首段 prefill → 层段 worker 依次转发 →
        #   尾段 decode 回传，见 `scheduler_pipeline._run_route_a_stage_pipeline`。
        #   本方法只负责登记任务并把 `task_id` 交给调用方；占位注释移除，避免"待实现"的误导。
        return task_id

    def complete_infer_task(self, task_id: str, result: str,
                            metrics: dict = None) -> None:
        """完成指定的转发任务；并发任务不能清除彼此的状态。"""
        completed = None
        with self._task_lock:
            current = self._infer_tasks.pop(task_id, None)
            if current is None:
                return
            current.state = "done"
            current.end_time = time.time()
            current.result = result
            current.metrics = metrics or {}
            completed = current
            if self._current_task is current:
                self._current_task = None

        logger.info(
            "推理任务完成: %s, 耗时=%.2fs, 结果长度=%d",
            task_id, completed.end_time - completed.start_time, len(result),
        )
        if self.on_task_complete:
            self.on_task_complete(completed)

    def fail_infer_task(self, task_id: str, error_msg: str) -> None:
        """失败指定的转发任务，不影响稍后启动的其他任务。"""
        with self._task_lock:
            current = self._infer_tasks.pop(task_id, None)
            if current is None:
                return
            current.state = "error"
            current.end_time = time.time()
            current.error_msg = error_msg
            if self._current_task is current:
                self._current_task = None
        logger.error("推理任务异常: %s - %s", task_id, error_msg)

    def stop_infer_task(self) -> None:
        """强制停止推理、重置流水线"""
        with self._task_lock:
            if self._current_task:
                self._current_task.state = "done"
                self._current_task.end_time = time.time()
                logger.info(f"推理任务已停止: {self._current_task.task_id}")
                self._current_task = None

        # ★ 2026-10-05（DIST-2 收拢现状）：此处原有「恢复所有节点为空闲」——
        #   `if self.nodes[nid].state == NodeState.BUSY: update_node_state(ONLINE)`。
        #   但 `BUSY` **从未被任何代码置位** ⇒ 该分支恒假，整段是死逻辑，已移除
        #   连同 `NodeState.BUSY` / `NodeState.ERROR` 两个枚举值。节点可用性现在由
        #   「`NodeState`（ONLINE/OFFLINE）+ 心跳新鲜度」两维表达，
        #   见 `scheduler_types.NodeState` 的说明。

        # ★ 2026-10-08（遗留清理）：此处原有两条 TODO（「发送 TASK_STOP 指令给所有从节点」
        #   与「清空所有节点 KV 缓存」）—— 均已不再需要，且**本方法本身无调用方**：
        #   ① 分布式取消走 v3 的 `stage_cancel`（worker 侧收敛并释放槽位，见
        #      `task_worker_adapter` 的 `stage_cancel_acknowledged` 通路）；
        #   ② "清空 KV"由**层配置换代**完成（`_publish_layer_configs` + assignment 状态机，
        #      worker 在重配置时丢弃旧 KV），不需要逐节点下发清理指令。
        #   ⇒ TODO 移除，避免误以为存在尚未实现的 STOP/KV 通路。

    def on_task_finished(self, result: str, metrics: dict = None) -> None:
        """
        任务完成回调。

        NOTE: 此方法当前无调用方，为死代码（dead code），保留供未来流水线任务完成通知使用。
        """
        with self._task_lock:
            task_id = self._current_task.task_id if self._current_task else ""
        if task_id:
            self.complete_infer_task(task_id, result, metrics)

    def on_task_error(self, error_msg: str) -> None:
        """
        任务异常回调。

        Args:
            error_msg: 错误描述
        """
        with self._task_lock:
            task_id = self._current_task.task_id if self._current_task else ""
        if task_id:
            self.fail_infer_task(task_id, error_msg)

    # ================================================================
    # 状态查询
    # ================================================================

    # L5: 多节点日志聚合

            # 保持本机服务可用，但绝不覆盖已有身份记录。

    # ================================================================
    # 分布式推理开关
    # ================================================================

    # ================================================================
    # 任务转发（从节点 → 主节点）
    # ================================================================

    # ================================================================
    # 分布式流水线推理（阶段 3：流水线调度引擎）
    # ================================================================

    # ================================================================
    # 协同抢占辅助方法 (Phase 2)
    # ================================================================

    # ================================================================
    # 流水线请求队列集成（Phase 4 — 多请求排队）
    # ================================================================

    # ================================================================
    # 从节点：主节点健康监控 + 自动重连
    # ================================================================
