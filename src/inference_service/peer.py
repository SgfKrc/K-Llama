"""从节点客户端（微服务架构改造计划 §1.5）。

复制自 scheduler.py 的 client 角色分支（_connect_to_master_locked /
_handle_layer_config_locked / _handle_layer_forward_locked /
_send_layer_config_ack / _send_layer_result / pipeline_done|abort），
**scheduler.py 源文件不动**；复用 tcp_comm.TCPClient 既有协议
（帧格式 / MessageType / HMAC 认证 / 心跳语义不变）。

宿主适配（scheduler 内部状态 → 实例属性）：
  _layer_execution_lock / _layer_config_lock / _kv_cache_lock → 本类 RLock
  _active_layer_config / _local_pipeline_steps / _active_pipeline_task_ids
    / _local_pipeline_cancelled / _kv_cache → 本类实例属性
  model_manager / model_host → EngineHost（本进程数据面宿主）
  get_effective_node_id() → self._node_id

未复制（开发期从节点不承担，归 scheduler-svc 控制面）：
  legacy 多段 CHAIN_FORWARD（由 master 向下一逻辑段转发，worker 不互连）、
  bootstrap 首次连接部署、备用主节点/角色转让、DB 主节点发现。
"""
import base64
import logging
import os
import socket
import threading
import time
from typing import Any, Dict, Optional

# ★ #31 M2：层流水线支持的架构走**单一事实来源**（此前硬编码 `{"qwen","qwen2"}`）
from pipeline_model_descriptor import PIPELINE_RUNTIME_MODEL_TYPES
from model_load_resolver import (
    ModelLoadFacts,
    OP_DYNAMIC_LAYER_RANGE,
    effective_pytorch_cuda_available,
    resolve_model_load,
)
from torch_runtime import cuda_available

logger = logging.getLogger("inference_service.peer")

# Keep this lightweight peer import independent from the full scheduler module.
RELAY_HIDDEN_WIRE_FORMAT = "qlh.relay_hidden.f32.v1"


def _generation_ack_fields(config: object, fallback: object = None) -> dict[str, Any]:
    for source in (config, fallback):
        if isinstance(source, dict) and "generation" in source:
            return {"generation": source["generation"]}
    return {}


def _assignment_ack_fields(config: object, fallback: object = None) -> dict[str, Any]:
    """Echo assignment identity without normalizing its wire representation."""
    for source in (config, fallback):
        if isinstance(source, dict) and "assignment_id" in source:
            return {"assignment_id": source["assignment_id"]}
    return {}


class PeerClient:
    """从节点客户端：连接主节点 + 层段加载 + 层前向执行闭环。"""

    def __init__(
        self,
        master_host: Optional[str] = None,
        master_port: Optional[int] = None,
        node_id: Optional[str] = None,
        device_info: Optional[dict] = None,
    ):
        # 层段执行宿主（EngineHost 构造轻量：model_host + config，不触发
        # model_module；首次 load 才加载）
        from inference_service.engine_host import EngineHost

        self._host = EngineHost()
        self._host.role = "client"

        import config as cfg

        self._master_host = master_host or os.environ.get(
            "QLH_CLIENT_MASTER_HOST", "127.0.0.1"
        )
        self._master_port = master_port or int(os.environ.get(
            "QLH_CLIENT_MASTER_PORT", getattr(cfg, "SERVER_PORT", 8888)
        ))
        if node_id:
            self._node_id = node_id
        else:
            configured_node_id = os.environ.get("QLH_NODE_ID", "") or ""
            if not configured_node_id or configured_node_id == "master":
                self._node_id = f"client_{socket.gethostname()}"
            else:
                self._node_id = configured_node_id
        self._device_info = dict(device_info or {})

        # ---- 层配置/流水线状态（scheduler 内部状态 → 实例属性）----
        self._layer_execution_lock = threading.RLock()
        self._layer_config_lock = threading.RLock()
        self._kv_cache_lock = threading.RLock()
        self._active_layer_config: Optional[dict] = None
        self._local_pipeline_steps: Dict[str, int] = {}
        self._active_pipeline_task_ids: set = set()
        self._local_pipeline_cancelled: set = set()
        self._kv_cache: Dict[str, Any] = {}
        # Relay sessions are task-scoped. Closing a client sends CLOSE and
        # resets the remote runner's KV state, so it must not happen per step.
        self._relay_sessions: Dict[str, Any] = {}
        self._pending_layer_config: Optional[tuple] = None
        # Receiver callbacks are dispatched on independent threads. Keep a
        # monotonic fence so a delayed release cannot erase a newer relay
        # assignment after a master restart.
        self._latest_layer_config_generation = 0
        self._latest_layer_config_id = ""
        self._latest_layer_assignment_id = ""
        # Capability is sticky for the process lifetime. Once the coordinator
        # has used assignment identity, accepting an identity-less retry would
        # reopen the stale-message hole that the assignment fence closes.
        self._layer_assignment_identity_required = False
        self._running = False
        self._client: Optional[Any] = None  # tcp_comm.TCPClient
        self._reconnect_delay = 5.0

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------
    def connect(self) -> dict:
        """连接主节点（TCPClient 注册 + 心跳由 tcp_comm 内部处理）。"""
        import config as cfg
        from tcp_comm import TCPClient

        advertise_port = getattr(cfg, "SERVER_PORT", 8888)
        client = TCPClient(
            server_host=self._master_host,
            server_port=self._master_port,
            client_id=self._node_id,
            role="client",
            node_type=os.environ.get("QLH_NODE_TYPE", "pc"),
            advertise_port=advertise_port,
            device_info=self._device_info,
        )
        self._client = client

        def _on_heartbeat() -> None:
            self._report_device_profile()

        def _on_disconnect() -> None:
            self._handle_disconnect()

        client.on_heartbeat = _on_heartbeat
        client.on_disconnect = _on_disconnect

        ok = client.connect(on_message=self._on_message)
        if ok:
            self._running = True
            self._report_device_profile()
            logger.info(
                "✅ 从节点已连接主节点: %s:%s (node_id=%s)",
                self._master_host, self._master_port, self._node_id,
            )
            return {
                "status": "connected",
                "node_id": self._node_id,
                "master": f"{self._master_host}:{self._master_port}",
            }
        return {
            "status": "failed",
            "reason": f"连接主节点 {self._master_host}:{self._master_port} 失败",
        }

    def _handle_disconnect(self) -> None:
        """Drop local pipeline state and let ``run_forever`` reconnect."""
        # The receive loop owns the transport failure notification.  Mark the
        # peer inactive so run_forever() drives a fresh registration; otherwise
        # a reset socket leaves the process alive but orphaned.
        self._running = False
        with self._layer_execution_lock:
            self._close_all_relay_sessions()
        logger.warning("与主节点连接断开: %s:%s", self._master_host, self._master_port)
        with self._layer_config_lock:
            self._active_layer_config = None
            self._pending_layer_config = None
            self._local_pipeline_steps.clear()
            self._active_pipeline_task_ids.clear()
            self._local_pipeline_cancelled.clear()
        # A reconnect must not reuse KV state from a socket that has already
        # lost its coordinator. Keep the generation fence intact so delayed
        # messages from the old connection cannot overwrite a new assignment.
        with self._kv_cache_lock:
            self._kv_cache.clear()

    def run_forever(self) -> None:
        """阻塞运行：连接失败/断开后自动重连（简单退避）。"""
        while True:
            if self._client is None or not self._running:
                result = self.connect()
                if result.get("status") != "connected":
                    logger.info(
                        "连接失败: %s，%.0fs 后重试",
                        result.get("reason", "unknown"), self._reconnect_delay,
                    )
            time.sleep(self._reconnect_delay)

    # ------------------------------------------------------------------
    # TCP 消息分发（复制 scheduler._on_tcp_message 的 client 相关分支）
    # ------------------------------------------------------------------
    def _on_message(self, msg: dict) -> None:
        msg_type = msg.get("type", "")
        data = msg.get("data", {})
        if msg_type == "layer_config":
            threading.Thread(
                target=self._handle_layer_config,
                args=(data,),
                name="peer-layer-config",
                daemon=True,
            ).start()
        elif msg_type == "layer_forward":
            threading.Thread(
                target=self._handle_layer_forward,
                args=(data,),
                name=f"peer-layer-forward-{data.get('task_id', 'unknown')}",
                daemon=True,
            ).start()
        elif msg_type == "pipeline_done":
            self._handle_pipeline_done(data)
        elif msg_type == "pipeline_abort":
            self._handle_pipeline_abort(data)
        elif msg_type in ("heartbeat_ack", "register_ack", "status_res"):
            pass  # tcp_comm 内部处理
        else:
            logger.debug("从节点忽略消息类型: %s", msg_type)

    # ------------------------------------------------------------------
    # 层配置（复制 scheduler._handle_layer_config_locked 核心语义）
    # ------------------------------------------------------------------
    def _handle_layer_config(self, data: dict) -> None:
        with self._layer_execution_lock:
            self._handle_layer_config_locked(data)

    def _handle_layer_config_locked(self, data: dict) -> None:
        node_id = self._node_id

        # Validate an explicitly targeted message before touching the local
        # generation watermark.  A release for another peer must not be able
        # to advance this client's fence and reject a later valid config.
        if isinstance(data, dict) and data.get("release"):
            target_node_id = str(data.get("node_id", node_id))
            if target_node_id != node_id:
                logger.warning(
                    "忽略目标不匹配的分层释放: target=%s local=%s",
                    target_node_id, node_id,
                )
                return

        # Config callbacks run on independent receiver threads. Reject an
        # older generation before it can clear a newer relay assignment.
        candidate = data
        if (
            isinstance(data, dict)
            and node_id in data
            and isinstance(data.get(node_id), dict)
        ):
            candidate = data[node_id]
        if (
            isinstance(candidate, dict)
            and candidate.get("node_id") is not None
            and str(candidate.get("node_id")) != node_id
        ):
            logger.warning(
                "忽略目标不匹配的分层配置: target=%s local=%s",
                candidate.get("node_id"), node_id,
            )
            return
        incoming_generation = None
        if isinstance(candidate, dict) and "generation" in candidate:
            try:
                incoming_generation = int(candidate.get("generation"))
            except (TypeError, ValueError):
                incoming_generation = None
        incoming_config_id = (
            str(candidate.get("config_id", "") or "")
            if isinstance(candidate, dict) else ""
        )
        incoming_assignment_present = (
            isinstance(candidate, dict) and "assignment_id" in candidate
        )
        incoming_assignment_id = (
            str(candidate.get("assignment_id", "") or "")
            if incoming_assignment_present else ""
        )
        with self._layer_config_lock:
            latest_generation = int(
                getattr(self, "_latest_layer_config_generation", 0) or 0
            )
            latest_config_id = str(
                getattr(self, "_latest_layer_config_id", "") or ""
            )
            latest_assignment_id = str(
                getattr(self, "_latest_layer_assignment_id", "") or ""
            )
            assignment_identity_required = bool(
                getattr(self, "_layer_assignment_identity_required", False)
                or latest_assignment_id
            )
            # An explicitly present but empty identity is malformed, not a
            # legacy message. Only true field absence may use the old path.
            if incoming_assignment_present and not incoming_assignment_id:
                logger.warning(
                    "ignore layer config with empty assignment_id "
                    "node=%s config=%s generation=%s",
                    node_id,
                    incoming_config_id or "legacy",
                    incoming_generation,
                )
                return
            if assignment_identity_required and not incoming_assignment_id:
                logger.info(
                    "ignore identity-less layer config after assignment fencing "
                    "node=%s config=%s generation=%s latest=%s/%s/%s",
                    node_id,
                    incoming_config_id or "legacy",
                    incoming_generation,
                    latest_generation,
                    latest_config_id or "legacy",
                    latest_assignment_id,
                )
                return
            # Once a versioned assignment has been accepted, an unversioned
            # legacy message is necessarily older than the active lifecycle.
            # Accepting it would let a delayed release clear a newer relay
            # config during a master reconnect.
            if incoming_generation is None and latest_generation > 0:
                logger.info(
                    "ignore unversioned stale layer config node=%s config=%s latest=%s/%s",
                    node_id,
                    incoming_config_id or "legacy",
                    latest_generation,
                    latest_config_id or "legacy",
                )
                return
            if incoming_generation is not None:
                # Once a lifecycle carries a generation, config_id is its
                # identity fence.  A missing id must not be treated as an
                # idempotent duplicate: a delayed release with the same
                # generation could otherwise clear the active assignment.
                if not incoming_config_id:
                    logger.warning(
                        "ignore versioned layer config without config_id "
                        "node=%s generation=%s latest=%s/%s",
                        node_id,
                        incoming_generation,
                        latest_generation,
                        latest_config_id or "legacy",
                    )
                    return
                stale = (
                    incoming_generation < latest_generation
                    or (
                        incoming_generation == latest_generation
                        and latest_config_id
                        and incoming_config_id != latest_config_id
                    )
                    or (
                        incoming_generation == latest_generation
                        and latest_config_id == incoming_config_id
                        and latest_assignment_id
                        and incoming_assignment_id != latest_assignment_id
                    )
                )
                if stale:
                    logger.info(
                        "ignore stale layer config node=%s config=%s generation=%s "
                        "assignment=%s latest=%s/%s/%s",
                        node_id,
                        incoming_config_id or "legacy",
                        incoming_generation,
                        incoming_assignment_id or "legacy",
                        latest_generation,
                        latest_config_id or "legacy",
                        latest_assignment_id or "legacy",
                    )
                    return
                if (
                    incoming_generation > latest_generation
                    or (
                        incoming_generation == latest_generation
                        and not latest_config_id
                    )
                ):
                    self._latest_layer_config_generation = incoming_generation
                    self._latest_layer_config_id = incoming_config_id
            if incoming_assignment_id:
                self._latest_layer_assignment_id = incoming_assignment_id
                self._layer_assignment_identity_required = True

        # 兼容两种格式：新版直接是 assignment；旧版 {node_id: assignment}
        if isinstance(data, dict) and data.get("release"):
            self._close_all_relay_sessions()
            with self._layer_config_lock:
                self._active_layer_config = None
                self._local_pipeline_steps.clear()
            if data.get("abort"):
                try:
                    from model_sync import remove_pipeline_assignment_cache

                    model_id = str(data.get("model_id", "") or "")
                    aborted_config_id = str(data.get("aborted_config_id", "") or "")
                    if model_id and aborted_config_id:
                        remove_pipeline_assignment_cache(
                            model_id, aborted_config_id, node_id,
                        )
                except Exception:
                    logger.warning("清理已中止的 assignment 缓存失败", exc_info=True)
            # ★ A1 / X 档（Y 档第二条缺口 2）：release ACK **必须**回 `release` 与 `generation`
            #   两个字段 —— 主节点 `scheduler_pipeline.py:1930-1934` 的 `released` 判据是
            #   `status == "released"` **且** `release is True` **且** `generation` 相等。
            #   此前只回 status ⇒ 主节点恒判「从节点分层释放 ACK 未通过」⇒ 每 5 秒重发
            #   （实测刷了 99 次 attempt），进而 `pipeline_distributed_workers_unavailable`。
            self._send_layer_config_ack({
                "node_id": node_id,
                "config_id": str(data.get("config_id", "")),
                "status": "released",
                "release": True,
                "generation": int(data.get("generation", 0) or 0),
                **_assignment_ack_fields(data),
            })
            logger.info("分层配置已释放: config_id=%s", data.get("config_id", ""))
            return

        if isinstance(data, dict) and "start_layer" in data and "end_layer" in data:
            cfg = dict(data)
        elif isinstance(data, dict) and node_id in data:
            cfg = dict(data[node_id] or {})
        else:
            error = f"分层配置中未找到本节点 {node_id} 的有效 assignment"
            logger.warning(error)
            self._send_layer_config_ack({
                "node_id": node_id,
                "config_id": data.get("config_id", "") if isinstance(data, dict) else "",
                "status": "error",
                "error": error,
                **_generation_ack_fields(data),
                **_assignment_ack_fields(data),
            })
            return

        config_id = str(cfg.get("config_id", ""))
        target_node_id = str(cfg.get("node_id", node_id))
        start = cfg.get("start_layer", 0)
        end = cfg.get("end_layer", 24)
        has_embed = cfg.get("has_embedding", False)
        has_lm = cfg.get("has_lm_head", False)
        model_id = str(cfg.get("model_id", ""))
        expected_sha256 = str(cfg.get("model_sha256", ""))
        expected_model_type = str(cfg.get("model_type", "")).lower()
        try:
            start = int(start)
            end = int(end)
            total_layers = int(cfg.get("total_layers", 0) or 0)
        except (TypeError, ValueError) as exc:
            error = f"分层配置数字字段无效: {exc}"
            logger.warning(error)
            self._send_layer_config_ack({
                "node_id": node_id, "config_id": config_id,
                "status": "error", "error": error,
                **_generation_ack_fields(cfg, data),
                **_assignment_ack_fields(cfg, data),
            })
            return

        logger.info(
            f"🔧 收到分层配置: 节点={node_id}, Layer {start}-{end}, "
            f"embed={has_embed}, lm_head={has_lm}, config_id={config_id or 'legacy'}"
        )

        # ★ A1 / X 档（Y 档第二条缺口 5）：`engine == "relay_middle"` ⇒ **不加载任何层**。
        #   段工件由远端 relay_mid_service 持有（正是 `scheduler_pipeline.py:2255` 分支的语义：
        #   "this scheduler host does not need a local ModelHost/model loaded"）。
        #   此前这里无条件 `load_model` + `load_layer_range` ⇒ relay worker 白加载一遍 23-24 层
        #   （实测日志反复出现 `✅ 层段加载完成: Layer 23-24`），既浪费又会让它按普通层节点
        #   去响应 LAYER_FORWARD（`a bytes-like object is required, not 'str'`）。
        #   ⇒ 直接回 ready；`layer_range` 用 **list**（与主节点 `expected_range` 同型）。
        if str(cfg.get("engine", "pytorch") or "pytorch").lower() == "relay_middle":
            phase = str(cfg.get("phase", "commit") or "commit")
            if phase not in {"prepare", "commit"}:
                error = f"relay_middle 配置阶段无效: {phase}"
                self._send_layer_config_ack({
                    "node_id": node_id,
                    "config_id": config_id,
                    "status": "error",
                    "error": error,
                    **_generation_ack_fields(cfg, data),
                    **_assignment_ack_fields(cfg, data),
                })
                return
            with self._layer_config_lock:
                self._active_layer_config = dict(cfg)
            # ⚠️ 主节点**两阶段**下发同一个 config：
            #   ① `phase="prepare"` ⇒ ACK 走 `prepared` 判据（`scheduler_pipeline.py:1972-1983`：
            #      `status=="prepared"` 且 `phase=="prepare"` 且 `plan_id`/`layer_range`/
            #      `model_sha256`/`model_type`/`engine` 全等，且 `available_bytes >= required_bytes`）；
            #   ② `phase="commit"` ⇒ ACK 走 `ready` 判据（`:1997-2017`：`status=="ready"`、
            #      `layer_range` 为 **list**、`model_sha256`/`model_type`/`engine` 全等）。
            #   ★ 2026-09-30：此前本分支**不区分 phase、永远回 `prepared`** ⇒ commit 阶段
            #   永远命中 `prepared_late`（"忽略已进入 commit 的迟到 prepare ACK"），
            #   `_layer_config_pushed` 永不加入 ⇒ 主节点持续 `重发分层配置`（实测每 **1s**
            #   一次），worker 也随之反复重设段。relay 节点不落任何层 ⇒ `available_bytes`
            #   给足够大的占位值（`required_bytes` 本就是 0，见 `pipeline_capacity` 零层条目）。
            common = {
                "node_id": node_id,
                "config_id": config_id,
                "plan_id": str(cfg.get("plan_id", "")),
                "layer_range": [int(start), int(end)],
                "model_sha256": expected_sha256,
                "model_type": expected_model_type,
                "engine": "relay_middle",
                "has_embedding": bool(has_embed),
                "has_lm_head": bool(has_lm),
                # ★ 2026-09-30：ACK **必须**回显 `generation` —— 主节点
                #   `scheduler_pipeline.py:1926` 对**非 release** 的期望同样做 generation 门闩
                #   （`if not expected.get("release") and "generation" in expected`），
                #   而 assignment registry 的 canonical expected contract
                #   本就带 generation。缺它 ⇒ 被 generation 门闩拒绝。
                "generation": cfg.get("generation", data.get("generation", 0)),
                **_assignment_ack_fields(cfg, data),
            }
            if phase == "commit":
                self._send_layer_config_ack({**common, "status": "ready"})
                logger.info(
                    "relay_middle 段 commit 就绪（本节点不加载层，段由远端服务持有）: "
                    "Layer %s-%s, config_id=%s",
                    start, end, config_id,
                )
            else:
                self._send_layer_config_ack({
                    **common, "status": "prepared", "phase": "prepare",
                    "available_bytes": 1 << 40,
                })
                logger.info(
                    "relay_middle 段配置就绪（本节点不加载层，段由远端服务持有）: "
                    "Layer %s-%s, config_id=%s",
                    start, end, config_id,
                )
            return

        try:
            if target_node_id != node_id:
                raise ValueError(f"层配置目标节点 {target_node_id} 与本节点 {node_id} 不一致")
            # ★ #31 M2：同上，走单一事实来源
            if expected_model_type not in PIPELINE_RUNTIME_MODEL_TYPES:
                raise ValueError(f"不支持的流水线模型架构: {expected_model_type or 'unknown'}")
            missing_contract = [
                name for name, value in (
                    ("config_id", config_id), ("model_id", model_id),
                    ("model_sha256", expected_sha256), ("total_layers", total_layers),
                )
                if not value
            ]
            if missing_contract:
                raise ValueError("分层配置执行契约不完整: " + ", ".join(missing_contract))

            with self._layer_config_lock:
                self._active_layer_config = None
                self._local_pipeline_steps.clear()
            self._host._host.model_loaded = False

            # 模型同步：确保 worker 模型文件就绪后加载层段
            from model_sync import ensure_model_available, resolve_worker_model_path

            local_model_path = resolve_worker_model_path(model_id)
            if not local_model_path:
                logger.info(f"模型 {model_id} 尚未同步，开始拉取...")
                ensure_model_available(model_id)
                local_model_path = resolve_worker_model_path(model_id)
            if not local_model_path:
                raise RuntimeError(f"模型同步后仍无本地路径: {model_id}")

            requested_quant = str(
                cfg.get("master_quant_type") or cfg.get("quant_type") or "int4"
            )
            profile = self._host._ensure_device_profile()

            resolution = resolve_model_load(
                ModelLoadFacts(
                    model_id=model_id,
                    model_name=model_id,
                    registered=True,
                    has_safetensors=True,
                    safetensors_path=local_model_path,
                    preferred_engine="pytorch",
                    cuda_available=effective_pytorch_cuda_available(
                        system_cuda_available=cuda_available(load=True),
                        profile=profile,
                    ),
                ),
                requested_engine="pytorch",
                requested_quant=requested_quant,
                operation=OP_DYNAMIC_LAYER_RANGE,
            )

            self._host._host.load_layer_range(
                start_layer=start, end_layer=end,
                has_embedding=has_embed, has_lm_head=has_lm,
                model_path=resolution.model_path,
                quant_type=resolution.quant_type,
                profile=profile,
                total_layers=total_layers,
                model_id=resolution.model_id,
            )

            with self._layer_config_lock:
                self._active_layer_config = dict(cfg)
            self._send_layer_config_ack({
                "node_id": node_id,
                "config_id": config_id,
                "status": "ready",
                # Keep the ready ACK shape identical to the scheduler's
                # versioned assignment contract. The old string form made
                # ordinary workers fail the same list comparison that relay
                # workers already satisfy.
                "layer_range": [int(start), int(end)],
                # ★ 2026-09-30：同 relay 分支 —— 非 release 的层配置 ACK 也要回显
                #   `generation`，否则主节点的 generation 门闩会把它整条忽略（见上）。
                "generation": int(cfg.get("generation", 0) or 0),
                **_assignment_ack_fields(cfg, data),
            })
            logger.info(
                f"✅ 层段加载完成: Layer {start}-{end}, config_id={config_id}"
            )
        except Exception as e:
            error = f"分层配置执行失败: {e}"
            logger.error(error, exc_info=True)
            self._send_layer_config_ack({
                "node_id": node_id, "config_id": config_id,
                "status": "error", "error": error,
                **_generation_ack_fields(cfg, data),
                **_assignment_ack_fields(cfg, data),
            })

    def _send_layer_config_ack(self, payload: dict) -> bool:
        from tcp_comm import MessageType

        client = self._client
        if client is None:
            logger.warning("TCP 客户端未连接，无法发送层配置 ACK")
            return False
        try:
            client.send_data(payload, MessageType.LAYER_CONFIG_ACK)
            return True
        except Exception as e:
            logger.error(f"发送层配置 ACK 失败: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # 层前向（复制 scheduler._handle_layer_forward_locked 核心语义）
    # ------------------------------------------------------------------
    def _handle_layer_forward(self, data: dict) -> None:
        with self._layer_execution_lock:
            self._handle_layer_forward_locked(data)

    def _handle_layer_forward_via_relay(self, data: dict) -> bool:
        """★ A1 / X 档（Y 档第二条缺口 8）：委托远端 relay **middle** 段执行本步，再回主节点。

        与 `scheduler_pipeline._handle_layer_forward_via_relay` **同语义**（吃 hidden、吐
        hidden），但跑在**从节点进程**里。此前本文件只实现了「用本地模型跑层」⇒ relay worker
        一收到 `LAYER_FORWARD` 就报「模型未加载」（实测 `层前向失败: step=0: 模型未加载`）。

        ⚠️ 线上格式差异：主节点发来的 `hidden_states` 是 `serialize_tensor_fast` 的 base64
        （它按**首个 worker** 决定格式，看不到末节点是不是 relay）⇒ 这里先反序列化成 tensor、
        取 **raw f32** 交给 relay 段；回来时反向转回主节点期望的格式。
        """
        import base64 as _b64

        import numpy as np
        import torch
        # Product relay frames are explicit raw-f32 plus shape metadata. The
        # tensor-fast fallback below is retained only for older non-relay peers.
        # Relay product frames use explicit raw-f32 metadata; the legacy
        # tensor-fast decoder below remains for mixed-version non-relay frames.
        from tcp_comm import deserialize_tensor_fast

        from relay_segment_client import (
            REQUEST_DEADLINE_EXCEEDED,
            RelaySegmentClient,
        )

        task_id = str(data.get("task_id", "unknown") or "unknown")
        try:
            step = int(data.get("step", 0))
        except (TypeError, ValueError):
            step = -1

        request_deadline_ms = int(data.get("request_deadline_ms", 0) or 0)
        deadline_monotonic = None
        if request_deadline_ms:
            remaining = request_deadline_ms / 1000.0 - time.time()
            if remaining <= 0:
                self._close_relay_session(task_id, abort=True)
                self._send_layer_result(
                    task_id, {}, error=REQUEST_DEADLINE_EXCEEDED
                )
                return False
            deadline_monotonic = time.monotonic() + remaining
            with self._layer_config_lock:
                existing = getattr(self, "_relay_sessions", {}).get(task_id)
                existing_deadline = getattr(
                    existing, "deadline_monotonic", None
                )
            if (
                existing_deadline is not None
                and existing_deadline < deadline_monotonic
            ):
                deadline_monotonic = existing_deadline
            if deadline_monotonic <= time.monotonic():
                self._close_relay_session(task_id, abort=True)
                self._send_layer_result(
                    task_id, {}, error=REQUEST_DEADLINE_EXCEEDED
                )
                return False

        cfg = dict(self._active_layer_config or {})
        spec = cfg.get("relay_segment") or {}
        if not isinstance(spec, dict) or not spec:
            self._send_layer_result(task_id, {}, error="relay_middle 缺少 relay_segment 规格")
            return False
        width = int(spec.get("n_embd", 0) or 0)
        # ★ Y-(b)：按**角色**分流。`middle` = hidden → hidden（把 hidden 转给远端段，再把
        #   hidden 回给主节点）；`tail` = hidden → token（远端段跑完本段并回 **argmax**，
        #   本节点把 token 回给主节点）。`head` 在**从节点这一侧**不支持 —— head 段要拿
        #   token 序列，而从节点从主节点收到的是 hidden，语义不成立 ⇒ 明确拒绝，
        #   绝不"当成 middle 硬算"（那会静默产出错误数值）。
        role = str(spec.get("role", "middle") or "middle").strip().lower()
        if role not in {"middle", "tail"}:
            self._send_layer_result(task_id, {}, error=f"relay_role_unsupported:{role}")
            return False

        raw = data.get("hidden_states")
        if isinstance(raw, str):
            raw_bytes = _b64.b64decode(raw)
        elif isinstance(raw, (bytes, bytearray)):
            raw_bytes = bytes(raw)
        else:
            self._send_layer_result(task_id, {}, error="relay 段委托缺 hidden_states")
            return False

        wire_format = str(data.get("hidden_wire_format", "") or "")
        hidden_shape = data.get("hidden_shape")
        try:
            if wire_format == RELAY_HIDDEN_WIRE_FORMAT:
                # Product relay handoff is raw contiguous f32. Preserve the
                # original shape for the return trip to the master's LM head.
                if not isinstance(hidden_shape, list) or not hidden_shape:
                    raise ValueError("relay hidden_shape is required for raw f32 payload")
                expected_items = 1
                for size in hidden_shape:
                    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                        raise ValueError("relay hidden_shape must contain positive integers")
                    expected_items *= size
                if hidden_shape[-1] != width:
                    raise ValueError(
                        f"relay hidden_shape last dimension {hidden_shape[-1]} != n_embd {width}"
                    )
                if len(raw_bytes) != expected_items * 4:
                    raise ValueError(
                        f"relay raw f32 length mismatch: bytes={len(raw_bytes)} "
                        f"expected={expected_items * 4}"
                    )
                hidden_bytes = raw_bytes
            else:
                # Legacy mixed-version workers still use tensor-fast.
                tensor = deserialize_tensor_fast(raw_bytes)
                hidden_bytes = tensor.detach().cpu().float().contiguous().numpy().tobytes()
                hidden_shape = list(tensor.shape)
        except Exception as exc:
            logger.error("relay hidden wire decode failed: %s", exc, exc_info=True)
            self._send_layer_result(task_id, {}, error=f"relay hidden wire decode failed: {exc}")
            return False

        if width <= 0 or len(hidden_bytes) % (width * 4):
            self._send_layer_result(
                task_id, {}, error="relay 段委托的 hidden 长度与 n_embd 不匹配（需 f32 且整除）"
            )
            return False
        n_tokens = len(hidden_bytes) // (width * 4)

        seq_ids = data.get("seq_ids")
        positions = data.get("positions")
        seq_meta = None
        if seq_ids is not None or positions is not None:
            if not (isinstance(seq_ids, list) and isinstance(positions, list)
                    and len(seq_ids) == n_tokens and len(positions) == n_tokens):
                self._send_layer_result(
                    task_id, {},
                    error="relay seq_ids/positions must match hidden token count",
                )
                return False
            try:
                seq_values = [int(value) for value in seq_ids]
                pos_values = [int(value) for value in positions]
                if any(value < 0 for value in seq_values + pos_values):
                    raise ValueError
                raw_n_seq_id = data.get("n_seq_id")
                if raw_n_seq_id is None:
                    n_seq_values = [1] * n_tokens
                elif isinstance(raw_n_seq_id, list) and len(raw_n_seq_id) == n_tokens:
                    n_seq_values = [int(value) for value in raw_n_seq_id]
                else:
                    raise ValueError
                if any(value < 1 for value in n_seq_values):
                    raise ValueError
                if any(value != 1 for value in n_seq_values):
                    self._send_layer_result(
                        task_id, {},
                        error="relay supports one sequence membership per token",
                    )
                    return False
            except (TypeError, ValueError, OverflowError):
                self._send_layer_result(task_id, {}, error="relay seq_ids/positions are invalid")
                return False
            seq_meta = {
                "n_seq_id": n_seq_values,
                "seq_ids": seq_values,
                "positions": pos_values,
            }

        started = time.time()
        try:
            endpoint_key = (
                str(spec.get("host", "")), int(spec.get("port", 0)), width, role,
            )
            configured_timeout = float(spec.get("timeout", 60.0) or 60.0)
            with self._layer_config_lock:
                sessions = getattr(self, "_relay_sessions", None)
                if sessions is None:
                    sessions = self._relay_sessions = {}
                client = sessions.get(task_id)
            if client is not None and getattr(client, "_relay_endpoint_key", None) != endpoint_key:
                self._close_relay_session(task_id, abort=True)
                client = None
            if client is None:
                client = RelaySegmentClient(
                    endpoint_key[0], endpoint_key[1], n_embd=width,
                    role=role, timeout=configured_timeout,
                    deadline_monotonic=deadline_monotonic,
                )
                client._relay_endpoint_key = endpoint_key
                with self._layer_config_lock:
                    if task_id in self._local_pipeline_cancelled:
                        client.close(grace_timeout=0.0)
                        return False
                    sessions[task_id] = client
            elif deadline_monotonic is not None:
                tighten_deadline = getattr(client, "tighten_deadline", None)
                if callable(tighten_deadline):
                    tighten_deadline(deadline_monotonic)
            deadline_kwargs = (
                {"deadline_monotonic": deadline_monotonic}
                if deadline_monotonic is not None
                else {}
            )
            if role == "tail":
                if seq_meta is None:
                    outcome = client.forward_hidden_to_token(
                        hidden_bytes,
                        n_tokens=n_tokens,
                        **deadline_kwargs,
                    )
                else:
                    outcome = client.forward_hidden_to_token(
                        hidden_bytes,
                        n_tokens=n_tokens,
                        seq_meta=seq_meta,
                        **deadline_kwargs,
                    )
            else:
                if seq_meta is None:
                    outcome = client.forward_hidden(
                        hidden_bytes,
                        n_tokens=n_tokens,
                        **deadline_kwargs,
                    )
                else:
                    outcome = client.forward_hidden(
                        hidden_bytes,
                        n_tokens=n_tokens,
                        seq_meta=seq_meta,
                        **deadline_kwargs,
                    )
        except Exception as exc:
            logger.error("relay 段委托失败: %s", exc, exc_info=True)
            self._close_relay_session(task_id, abort=True)
            code = getattr(exc, "code", "") or str(exc)
            error = (
                REQUEST_DEADLINE_EXCEEDED
                if code == REQUEST_DEADLINE_EXCEEDED
                else f"relay_segment_failed:{code}"
            )
            self._send_layer_result(task_id, {}, error=error)
            return False
        elapsed_ms = (time.time() - started) * 1000

        if not outcome.ok:
            code = getattr(outcome, "error", "") or "relay_internal_error"
            logger.warning(
                "relay 段委托未成功: task=%s step=%s code=%s", task_id, step, code
            )
            self._close_relay_session(task_id, abort=True)
            error = (
                REQUEST_DEADLINE_EXCEEDED
                if code == REQUEST_DEADLINE_EXCEEDED
                else f"relay_segment_failed:{code}"
            )
            self._send_layer_result(task_id, {}, error=error)
            return False

        if (
            deadline_monotonic is not None
            and deadline_monotonic <= time.monotonic()
        ):
            self._close_relay_session(task_id, abort=True)
            self._send_layer_result(
                task_id, {}, error=REQUEST_DEADLINE_EXCEEDED
            )
            return False
        with self._layer_config_lock:
            cancelled = task_id in self._local_pipeline_cancelled
        if cancelled:
            self._close_relay_session(task_id, abort=True)
            return False

        common_response = {
            "task_id": task_id,
            "node_id": self._node_id,
            "step": step,
            "config_id": str(data.get("config_id", "")),
            "model_sha256": str(data.get("model_sha256", "")),
            "model_type": str(data.get("model_type", "")),
            "chain_path": [*[str(x) for x in (data.get("chain_path") or [])], self._node_id],
            "metrics": {
                "time_ms": round(elapsed_ms, 1),
                "kv_cache": False,
                "kv_seq_len": 0,
                "relay_executed": True,
                **outcome.to_metrics(),
            },
        }

        if role == "tail":
            # ★ Y-(b)：末段回 **token**（远端已跑完本段并做完 argmax）。
            token = getattr(outcome, "token", None)
            if token is None:
                self._send_layer_result(task_id, {}, error="relay tail 段未返回 token")
                self._close_relay_session(task_id, abort=True)
                return False
            response = {**common_response, "token": int(token)}
        else:
            try:
                shape = tuple(int(size) for size in hidden_shape)
                out_array = np.frombuffer(bytes(outcome.hidden), dtype=np.float32)
                expected_items = int(np.prod(shape))
                if out_array.size != expected_items:
                    raise ValueError(
                        f"relay output length mismatch: items={out_array.size} "
                        f"expected={expected_items}"
                    )
                out_tensor = torch.from_numpy(out_array.reshape(shape).copy())
            except Exception as exc:
                logger.error("relay 段返回的 hidden 无法还原: %s", exc, exc_info=True)
                self._send_layer_result(task_id, {}, error=f"relay hidden 还原失败: {exc}")
                self._close_relay_session(task_id, abort=True)
                return False
            # Keep the relay contract symmetric with scheduler_pipeline:
            # raw f32 in both directions, with an explicit discriminator.
            response = {
                **common_response,
                "hidden_states": out_tensor.detach().cpu().float().contiguous().numpy().tobytes(),
                "hidden_wire_format": RELAY_HIDDEN_WIRE_FORMAT,
                "hidden_shape": list(out_tensor.shape),
            }
        logger.info(
            "🔁 relay 段委托完成（从节点）: task=%s, step=%s, 段=%s@%s:%s, tokens=%s, time=%.0fms",
            task_id, step, spec.get("role"), spec.get("host"), spec.get("port"),
            n_tokens, elapsed_ms,
        )
        return self._send_layer_result(task_id, response)

    def _close_relay_session(self, task_id: str, *, abort: bool = False) -> None:
        """Close one task-scoped relay connection and forget it."""
        with self._layer_config_lock:
            sessions = getattr(self, "_relay_sessions", None)
            client = sessions.pop(str(task_id), None) if sessions else None
        if client is None:
            return
        try:
            if abort:
                try:
                    client.close(grace_timeout=0.0)
                except TypeError:
                    client.close()
            else:
                client.close()
        except Exception:  # noqa: BLE001 - cleanup must not mask pipeline state
            logger.warning("relay session close failed: task=%s", task_id, exc_info=True)

    def _close_all_relay_sessions(self) -> None:
        with self._layer_config_lock:
            task_ids = list(getattr(self, "_relay_sessions", {}) or {})
        for task_id in task_ids:
            self._close_relay_session(task_id)

    def _handle_layer_forward_locked(self, data: dict) -> None:
        # ★ A1 / X 档（Y 档第二条缺口 8）：`engine == "relay_middle"` ⇒ 本节点**不跑层**，
        #   把 hidden 委托给远端 relay 段（段工件由外边监督的 relay_mid_service 持有）。
        #   此前本方法只会"用本地模型跑层" ⇒ relay worker 一收到 LAYER_FORWARD 就报
        #   「模型未加载」（实测 `层前向失败: step=0: 模型未加载`）。
        from tcp_comm import deserialize_tensor_fast, serialize_tensor_fast

        task_id = str(data.get("task_id", "unknown") or "unknown")
        try:
            step = int(data.get("step", 0))
        except (TypeError, ValueError):
            step = -1
        use_kv_cache = data.get("use_kv_cache", False)
        config_id = str(data.get("config_id", ""))
        model_sha256 = str(data.get("model_sha256", ""))
        model_type = str(data.get("model_type", "")).lower()

        logger.info(
            f"🔬 收到层前向指令: task={task_id}, step={step}, "
            f"kv_cache={'on' if use_kv_cache else 'off'}"
        )

        try:
            with self._layer_config_lock:
                if task_id in self._local_pipeline_cancelled:
                    logger.info("忽略已取消任务的迟到层前向: task=%s", task_id)
                    return
                active_config = dict(self._active_layer_config or {})
                last_step = self._local_pipeline_steps.get(task_id)
                task_active = task_id in self._active_pipeline_task_ids
            if not active_config:
                raise RuntimeError("本节点没有已确认的活动层配置")
            for field, actual in (
                ("config_id", config_id),
                ("model_sha256", model_sha256),
                ("model_type", model_type),
            ):
                if not actual or actual != str(active_config.get(field, "")):
                    raise RuntimeError(
                        f"流水线执行契约不一致: {field}={actual or '-'}, "
                        f"expected={active_config.get(field, '-')}"
                    )
            if step < 0:
                raise RuntimeError(f"无效流水线 step: {step}")
            if step == 0:
                if use_kv_cache:
                    raise RuntimeError("prefill step 0 不得声明使用既有 KV cache")
                if task_active or last_step is not None:
                    raise RuntimeError(f"重复 prefill: task={task_id}")
            else:
                if not use_kv_cache:
                    raise RuntimeError(f"decode step {step} 必须使用 KV cache")
                if not task_active or last_step != step - 1:
                    raise RuntimeError(
                        f"流水线 step 越序: task={task_id}, step={step}, "
                        f"last_step={last_step}"
                    )

            if str(active_config.get("engine", "") or "").lower() == "relay_middle":
                if self._handle_layer_forward_via_relay(data):
                    with self._layer_config_lock:
                        self._local_pipeline_steps[task_id] = step
                        self._active_pipeline_task_ids.add(task_id)
                else:
                    with self._layer_config_lock:
                        self._local_pipeline_cancelled.add(task_id)
                return

            import torch

            mgr = self._host._host
            if not mgr or not getattr(mgr, "is_loaded", False):
                raise RuntimeError("模型未加载")
            loaded_config = getattr(getattr(mgr, "model", None), "config", None)
            actual_model_type = str(getattr(loaded_config, "model_type", "") or "").lower()
            if getattr(mgr, "_engine_type", "") != "pytorch":
                raise RuntimeError(f"worker 引擎已变化: {getattr(mgr, '_engine_type', '')}")
            if actual_model_type != model_type:
                raise RuntimeError(
                    f"worker 模型架构已变化: actual={actual_model_type}, expected={model_type}"
                )

            t_start = time.time()
            input_ids = None
            hidden_states = None
            if "input_ids" in data and data["input_ids"] is not None:
                input_ids = torch.tensor(data["input_ids"], dtype=torch.long)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
            if "hidden_states" in data and data["hidden_states"] is not None:
                hidden_states = deserialize_tensor_fast(data["hidden_states"])

            past_kv = None
            if use_kv_cache:
                with self._kv_cache_lock:
                    past_kv = self._kv_cache.get(task_id)
                if past_kv is None:
                    raise RuntimeError(
                        f"decode step {step} 缺少本地 KV cache: task={task_id}"
                    )

            result = mgr.forward_layers(
                input_ids=input_ids,
                hidden_states=hidden_states,
                attention_mask=(
                    torch.tensor(data["attention_mask"], dtype=torch.long)
                    if data.get("attention_mask") is not None else None
                ),
                position_ids=(
                    torch.tensor(data["position_ids"], dtype=torch.long)
                    if data.get("position_ids") is not None else None
                ),
                past_key_values=past_kv,
                use_cache=True,
                apply_lm_head=bool(data.get("apply_lm_head", False)),
            )
            if task_id in self._local_pipeline_cancelled:
                with self._layer_config_lock:
                    self._local_pipeline_cancelled.discard(task_id)
                logger.info("丢弃已取消任务的迟到计算结果: task=%s", task_id)
                return
            elapsed_ms = (time.time() - t_start) * 1000

            # ★ #31 M4：**优先持有 `result["cache"]`** —— hybrid 的 tuple 会丢 recurrent state
            #   （`linear_attention` 层在 tuple 里是 `None` 占位）。
            #   本文件是「复制自 `scheduler.py` 的 client 角色分支」的既有模式 ⇒ 同样内联，
            #   与 `src/scheduler_pipeline.py` 的 `_prefer_cache_state` **同源（改一处要同步另一处）**。
            if result.get("cache") is not None or result.get("past_key_values"):
                with self._kv_cache_lock:
                    self._kv_cache[task_id] = (
                        result["cache"] if result.get("cache") is not None
                        else result["past_key_values"]
                    )
            else:
                raise RuntimeError("分层前向未返回 KV cache")
            with self._layer_config_lock:
                self._local_pipeline_steps[task_id] = step
                self._active_pipeline_task_ids.add(task_id)

            response = {
                "task_id": task_id,
                "node_id": self._node_id,
                "step": step,
                "config_id": config_id,
                "model_sha256": model_sha256,
                "model_type": model_type,
                "metrics": {
                    "time_ms": round(elapsed_ms, 1),
                    "kv_cache": use_kv_cache,
                    "memory_allocated_gb": (
                        round(torch.cuda.memory_allocated() / (1024**3), 2)
                        if torch.cuda.is_available() else 0
                    ),
                },
            }
            if "hidden_states" in result:
                hs_cpu = result["hidden_states"].detach().cpu()
                response["hidden_states"] = serialize_tensor_fast(hs_cpu)
                response["hidden_shape"] = list(hs_cpu.shape)
            if "logits" in result:
                logits_cpu = result["logits"].detach().cpu()
                response["logits"] = serialize_tensor_fast(logits_cpu)
                response["logits_shape"] = list(logits_cpu.shape)

            self._send_layer_result(task_id, response)
            logger.info(
                f"✅ 层前向完成: task={task_id}, step={step}, "
                f"time={elapsed_ms:.0f}ms"
            )
        except Exception as e:
            error = str(e)
            logger.error(f"层前向失败: task={task_id}, step={step}: {error}")
            self._send_layer_result(task_id, {}, error=error)
            with self._layer_config_lock:
                self._local_pipeline_cancelled.add(task_id)

    def _send_layer_result(self, task_id: str, result_data: dict,
                           error: str = None) -> bool:
        from tcp_comm import MessageType

        if self._client is None or not getattr(self._client, "_running", False):
            logger.error("TCP 客户端未连接，无法发送层前向结果")
            return False

        payload = dict(result_data)
        payload["task_id"] = task_id
        if error:
            payload["error"] = error
        safe_payload = {}
        for k, v in payload.items():
            if isinstance(v, bytes):
                safe_payload[k] = base64.b64encode(v).decode("ascii")
            else:
                safe_payload[k] = v
        try:
            self._client.send_data(safe_payload, MessageType.LAYER_RESULT)
            return True
        except Exception as e:
            logger.error(f"发送层前向结果失败: {e}")
            try:
                self._client.disconnect()
            except Exception:
                pass
            return False

    # ------------------------------------------------------------------
    # 流水线任务清理（复制 scheduler._on_tcp_message 的 pipeline_* 分支）
    # ------------------------------------------------------------------
    def _handle_pipeline_done(self, data: dict) -> None:
        task_id = data.get("task_id", "")
        if task_id:
            with self._layer_execution_lock:
                self._close_relay_session(task_id)
                with self._layer_config_lock:
                    self._local_pipeline_cancelled.discard(task_id)
                    self._local_pipeline_steps.pop(task_id, None)
                    self._active_pipeline_task_ids.discard(task_id)
                with self._kv_cache_lock:
                    self._kv_cache.pop(task_id, None)
                logger.info(f"🧹 流水线任务 {task_id} KV 缓存已清理")

    def _handle_pipeline_abort(self, data: dict) -> None:
        task_id = data.get("task_id", "")
        if task_id:
            # Publish cancellation and break a blocking relay I/O before
            # waiting for the serialized layer execution section.
            with self._layer_config_lock:
                self._local_pipeline_cancelled.add(task_id)
            self._close_relay_session(task_id, abort=True)
            with self._layer_execution_lock:
                with self._layer_config_lock:
                    self._local_pipeline_steps.pop(task_id, None)
                    self._active_pipeline_task_ids.discard(task_id)
                with self._kv_cache_lock:
                    self._kv_cache.pop(task_id, None)
                logger.info(f"流水线任务取消: {task_id}")

    # ------------------------------------------------------------------
    # 设备画像上报（心跳时附带）
    # ------------------------------------------------------------------
    def _report_device_profile(self) -> None:
        try:
            from device_profiler import get_profile

            profiler = get_profile()
            profile_dict = profiler.to_dict()
            self._device_info = profile_dict
            if self._client is not None:
                self._client.device_info = profile_dict
        except Exception as e:
            logger.warning(f"设备画像上报失败: {e}")


def run_peer() -> None:
    """从节点入口（inference_svc_main client 角色调用，不 import fastapi）。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info(
        "从节点启动: master=%s:%s node_id=%s",
        os.environ.get("QLH_CLIENT_MASTER_HOST", "127.0.0.1"),
        os.environ.get("QLH_CLIENT_MASTER_PORT", "8888"),
        os.environ.get("QLH_NODE_ID", "") or f"client_{socket.gethostname()}",
    )
    peer = PeerClient()
    peer.run_forever()
