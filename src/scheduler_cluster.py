"""Cluster membership and node operations mixed into Scheduler."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid

from network_path import build_client_network_path_view
from pipeline_node_contract import build_aggregate_resource_view
from scheduler_types import NodeInfo, NodeRole, NodeState

logger = logging.getLogger("scheduler")


class SchedulerClusterMixin:
    def init_nodes(self) -> None:
        """
        初始化节点状态。
        - 主节点：仅创建 master 自身，从节点通过 TCP 注册动态加入（不再预创建空槽位）
        - 从节点：创建自身记录，等待用户操作连接主节点

        节点在线状态、能力画像和层配置均属于当前 TCP 会话的运行时
        事实；服务重启后由 bootstrap/Tailnet 与重新注册恢复，不能从
        已退场的远端数据库复活陈旧节点。
        """
        effective_role = self._effective_role()

        # ---- 主节点模式：仅创建 master，不预创建 client 空位 ----
        if effective_role == "master":
            now_ts = time.time()
            self.nodes["master"] = NodeInfo(
                node_id="master", role=NodeRole.MASTER,
                state=NodeState.ONLINE,
                hostname="localhost",
                network_type="localhost",
                connected_at=now_ts,
                last_heartbeat=now_ts,
            )

        # ---- 从节点模式：仅创建自身 ----
        else:
            # ★ 安全：从节点绝不能使用 "master" 作为 node_id
            # 否则会和本机 master 运行时记录冲突。
            configured_node_id = self._runtime_setting('_configured_node_id')()
            if not configured_node_id or configured_node_id == "master":
                node_id = f"client_{__import__('socket').gethostname()}"
                if configured_node_id == "master":
                    logger.warning(
                        f"⚠️ 从节点 NODE_ID 配置错误（仍为 \"master\"），"
                        f"已自动生成: {node_id}"
                    )
            else:
                node_id = configured_node_id
            self.nodes[node_id] = NodeInfo(
                node_id=node_id, role=NodeRole.CLIENT,
                state=NodeState.ONLINE,
                hostname="localhost",
            )

        # 设备检测与调度器启动并行执行。若画像先完成，在节点创建后立即补入；
        # 若画像稍后完成，则由 update_local_device_profile() 写回。
        if self._local_device_profile:
            local_node_id = "master" if effective_role == "master" else self.get_effective_node_id()
            with self._nodes_lock:
                local_node = self.nodes.get(local_node_id)
                if local_node is not None:
                    local_node.device_info = dict(self._local_device_profile)

        logger.info(
            f"节点初始化完成: {len(self.nodes)} 个节点 "
            f"(mode={self._runtime_setting('RUN_MODE')}, max_nodes={self._runtime_setting('MAX_NODES')}, my_role={effective_role})"
        )
        for nid, info in self.nodes.items():
            logger.info(f"  {nid}: role={info.role}, state={info.state.value}")


    def _report_local_device_profile(self, tcp_client=None,
                                     node_id: str = None) -> bool:
        """将后台完成的完整设备画像补报给主节点。"""
        profile = dict(self._local_device_profile or {})
        client = tcp_client or getattr(self, "_tcp_client", None)
        if not profile or not client or not getattr(client, "is_registered", False):
            return False
        if getattr(client, "device_info", None) == profile:
            return False

        local_node_id = node_id or self.get_effective_node_id()
        with self._nodes_lock:
            node = self.nodes.get(local_node_id)
            state = node.state.value if node is not None else "online"

        try:
            from transport_port import MessageType
            client.send_data(
                {"state": state, "device_info": profile},
                MessageType.STATUS_RES,
            )
            client.device_info = dict(profile)
            logger.info("完整设备画像已补报主节点: node=%s", local_node_id)
            return True
        except Exception as e:
            logger.warning("完整设备画像补报失败: %s", e, exc_info=True)
            return False


    def update_local_device_profile(self, profile: dict) -> None:
        """将异步硬件检测结果写回本地节点，并使旧动态分层失效。"""
        if not isinstance(profile, dict) or not profile:
            return

        # Hardware facts do not imply an execution path. Project concrete
        # installed/current runtimes without constructing a lazy model manager.
        from task_worker_capabilities import engine_profile
        profile = engine_profile(profile, self._task_worker_engine_schema())
        self._local_device_profile = dict(profile)
        effective_role = self._effective_role()
        local_node_id = "master" if effective_role == "master" else self.get_effective_node_id()
        node_snapshot = None
        with self._nodes_lock:
            node = self.nodes.get(local_node_id)
            if node is not None:
                node.device_info = dict(profile)
                node_snapshot = node

        if node_snapshot is None:
            logger.debug("本地设备画像已缓存，等待节点初始化后写回: %s", local_node_id)
        elif effective_role == "client":
            self._report_local_device_profile(node_id=local_node_id)

        if node_snapshot is None:
            return

        score = self._compute_node_weight(profile)
        gpu = self._select_scoring_gpu(profile)
        logger.info(
            "本地节点设备画像已写入调度器: node=%s gpu=%s score=%.1f",
            local_node_id,
            gpu.get("name", "unknown") if isinstance(gpu, dict) else "unknown",
            score,
        )

        if effective_role == "master":
            try:
                self.push_layer_config_to_clients()
            except Exception as e:
                logger.warning("设备画像更新后重新下发分层失败: %s", e, exc_info=True)


    def register_node(self, node_id: str, role: str, address: str = "",
                      hostname: str = "", device_info: dict = None,
                      network_type: str = "unknown",
                      node_type: str = "pc",
                      model_sha256: str = "") -> bool:
        """
        注册一个从节点。

        调用时机：TCP 服务端收到 REGISTER 消息时。
        支持动态节点数量，允许 MAX_NODES 范围内的任意 client ID 注册。

        Args:
            node_id: 节点标识（"client1" / "client2" / ...）
            role: 节点角色 ("master" | "client")
            address: 客户端地址 "ip:port"
            hostname: 客户端主机名
            device_info: 客户端设备信息
            network_type: 网络连接类型 "wifi" | "ethernet" | "unknown"
            node_type: 设备平台 "pc" | "android"（默认 "pc"）
            model_sha256: 模型 SHA256 校验值（阶段 7，用于跨节点模型一致性验证）

        Returns:
            注册是否成功
        """
        # Android 节点只能作为 client
        if node_type == "android" and role == "master":
            logger.error(f"注册失败: Android 节点不能担任 master 角色")
            return False

        # 动态添加新节点（需检查容量限制）
        # Phase 2.1+: 所有 self.nodes 读写均在 _nodes_lock 保护下
        # Phase 5 review H4: 属性写入也纳入锁内，防止与 deregister_node TOCTOU
        with self._nodes_lock:
            if node_id not in self.nodes:
                if role == NodeRole.MASTER:
                    logger.warning(f"注册失败: 不能动态注册 master 节点")
                    return False
                # 容量检查：只统计在线节点（离线/幽灵不占位）
                online_non_master = [
                    n for n in self.nodes.values()
                    if n.role != "master" and n.is_available()
                ]
                if len(online_non_master) >= self._max_nodes - 1:
                    logger.warning(
                        f"注册失败: 已达到最大在线从节点数量 "
                        f"({len(online_non_master)}/{self._max_nodes - 1})"
                    )
                    return False
                logger.info(f"动态添加节点: {node_id} (type={node_type})")
                self.nodes[node_id] = NodeInfo(
                    node_id=node_id, role=NodeRole.CLIENT,
                    node_type=node_type,
                    state=NodeState.OFFLINE,
                )

            node = self.nodes[node_id]

            if node.role == NodeRole.MASTER:
                logger.warning(f"注册失败: {node_id} 角色为 master，不可被注册覆盖")
                return False

            # NodeInfo 字段更新
            node.state = NodeState.ONLINE
            node.node_type = node_type
            node.model_sha256 = model_sha256
            node.address = address
            node.hostname = hostname
            node.device_info = device_info or {}
            node.network_type = network_type
            node.connected_at = time.time()
            node.last_heartbeat = time.time()

        logger.info(
            f"✅ 节点注册: {node_id} role={role} type={node_type} "
            f"hostname={hostname} addr={address} net={network_type}"
        )
        return True


    def deregister_node(self, node_id: str) -> bool:
        """
        注销一个从节点（断连或主动离线）。

        Args:
            node_id: 节点标识

        Returns:
            注销是否成功
        """
        return self._observe_node_disconnected(node_id)

    def _observe_node_disconnected(self, node_id: str) -> bool:
        """Record a local TCP loss even after this node's lease has been fenced."""
        with self._nodes_lock:
            if node_id not in self.nodes:
                return False

            node = self.nodes[node_id]
            if node.role == NodeRole.MASTER:
                return False  # master 不可注销

            old_state = node.state
            node.state = NodeState.OFFLINE
            node.address = ""
            node.connected_at = 0.0

        # 主节点：推送节点离线更新给所有已连接从节点
        if self._effective_role() == "master":
            self._push_node_update_to_all_clients(
                node_id, "update", node
            )
            # ★ 清除层配置推送记录（节点离线后需重新推送）
            self._clear_layer_config_state(node_id)
            with self._layer_config_lock:
                self._pipeline_worker_opt_out.discard(node_id)

        logger.info(f"节点注销: {node_id} ({old_state.value} → offline)")
        return True


    def update_node_state(self, node_id: str, state: NodeState) -> None:
        """更新节点状态"""
        with self._nodes_lock:
            if node_id in self.nodes:
                old_state = self.nodes[node_id].state
                self.nodes[node_id].state = state
                self.nodes[node_id].last_heartbeat = time.time()
                logger.info(f"节点 {node_id} 状态变更: {old_state.value} -> {state.value}")
            else:
                logger.warning(f"未知节点: {node_id}")


    def record_task_complete(self, node_id: str = None, success: bool = True) -> bool:
        """
        记录一次推理任务完成。

        Args:
            node_id: 执行推理的节点（默认本节点）
            success: 是否成功

        Returns:
            True = 成功更新了已知节点；False = 节点未知或更新失败。
        """
        nid = node_id or self.get_effective_node_id()
        with self._nodes_lock:
            if nid not in self.nodes:
                logger.warning(
                    f"任务计数跳过：未知节点 {nid}，"
                    f"当前已知节点={list(self.nodes.keys())}"
                )
                return False

            if success:
                self.nodes[nid].task_count += 1
            else:
                self.nodes[nid].error_count += 1
            self.nodes[nid].last_heartbeat = time.time()
            node_snapshot = self.nodes[nid]

        if self._effective_role() == "master":
            try:
                self._push_node_update_to_all_clients(nid, "update", node_snapshot)
            except Exception:
                pass
        return True


    def record_task_error(self, node_id: str = None) -> None:
        """记录一次推理任务失败（便捷方法）"""
        self.record_task_complete(node_id, success=False)


    def _record_local_pipeline_participation(self, task_id: str,
                                             success: bool = True) -> bool:
        """从节点在任务终态记账；成功/失败不能对同一 task 重复记录。"""
        if not task_id:
            return False
        if (task_id in self._local_pipeline_counted_tasks
                or task_id in self._local_pipeline_error_tasks):
            return False
        task_set = (self._local_pipeline_counted_tasks if success
                    else self._local_pipeline_error_tasks)
        task_set.add(task_id)
        self._local_pipeline_accounted_order.append((task_id, success))
        while len(self._local_pipeline_accounted_order) > 4096:
            old_task_id, old_success = self._local_pipeline_accounted_order.popleft()
            old_set = (self._local_pipeline_counted_tasks if old_success
                       else self._local_pipeline_error_tasks)
            old_set.discard(old_task_id)
        return self.record_task_complete(success=success)


    def _record_pipeline_task_accounting(self, task_id: str,
                                         pipeline_nodes: list,
                                         success: bool = True) -> dict:
        """
        主节点侧记录一次分布式流水线任务参与情况。

        语义：每个完成的用户请求，master 计 1 次服务请求，每个实际参与的
        PC worker 计 1 次参与请求。按 task_id 幂等，避免 streaming / retry 重复计数。
        """
        if not task_id:
            task_id = f"anonymous_{time.time()}"
        if task_id in self._pipeline_accounted_tasks:
            return {
                "task_id": task_id,
                "deduplicated": True,
                "counted_nodes": [],
                "skipped_unknown_nodes": [],
                "accounting_errors": [],
            }
        self._pipeline_accounted_tasks.add(task_id)
        self._pipeline_accounted_order.append(task_id)
        while len(self._pipeline_accounted_order) > 4096:
            self._pipeline_accounted_tasks.discard(
                self._pipeline_accounted_order.popleft()
            )

        ordered_nodes = [self.get_effective_node_id()]
        for node in pipeline_nodes or []:
            nid = node.get("node_id") if isinstance(node, dict) else str(node)
            if nid and nid not in ordered_nodes:
                ordered_nodes.append(nid)

        counted = []
        skipped = []
        errors = []
        for nid in ordered_nodes:
            try:
                if self.record_task_complete(nid, success=success):
                    counted.append(nid)
                else:
                    skipped.append(nid)
            except Exception as e:
                errors.append({"node_id": nid, "error": str(e)})

        return {
            "task_id": task_id,
            "deduplicated": False,
            "success": success,
            "counted_nodes": counted,
            "workers_counted": [nid for nid in counted if nid != self.get_effective_node_id()],
            "skipped_unknown_nodes": skipped,
            "accounting_errors": errors,
        }


    def register_android_client(self, node_id: str, hostname: str = "",
                                address: str = "", network_type: str = "unknown",
                                device_info: dict = None,
                                client_mode: str = "thin",
                                app_variant: str = "full",
                                app_version: str = "",
                                http_peer: str = "") -> dict:
        """登记 Android HTTP 薄客户端在线状态（不是 TCP worker 注册）。"""
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可登记 Android 客户端"}
        if not node_id or node_id == "master":
            return {"status": "invalid", "reason": "Android node_id 无效"}

        now = time.time()
        info = dict(device_info or {})
        info.update({
            "connection_type": "http_thin",
            "pipeline_worker": False,
            "client_mode": client_mode or "thin",
            "app_variant": app_variant or "full",
            "app_version": app_version or "",
        })
        if http_peer:
            info["http_peer"] = http_peer

        with self._nodes_lock:
            existing = self.nodes.get(node_id)
            is_new = existing is None
            if is_new:
                existing = NodeInfo(
                    node_id=node_id,
                    role=NodeRole.CLIENT,
                    node_type="android",
                    state=NodeState.ONLINE,
                    connected_at=now,
                )
                self.nodes[node_id] = existing
            elif existing.role == NodeRole.MASTER:
                return {"status": "invalid", "reason": f"'{node_id}' 是主节点，不可覆盖"}

            existing.node_type = "android"
            existing.role = NodeRole.CLIENT
            existing.state = NodeState.ONLINE
            existing.hostname = hostname or existing.hostname or node_id
            existing.address = address or existing.address or ""
            existing.network_type = network_type or existing.network_type or "unknown"
            existing.device_info = info
            if not existing.connected_at:
                existing.connected_at = now
            existing.last_heartbeat = now
            existing.presence_generation = max(1, existing.presence_generation + 1)
            existing.presence_lease_id = uuid.uuid4().hex
            existing.presence_expires_at = now + self._runtime_setting('ANDROID_HTTP_CLIENT_LEASE_SECONDS')

        if self._effective_role() == "master":
            self._push_node_update_to_all_clients(
                node_id, "add" if is_new else "update", existing
            )

        logger.info(
            f"📱 Android HTTP 客户端在线: {node_id} host={existing.hostname} "
            f"net={existing.network_type} peer={http_peer}"
        )
        return {
            "status": "registered" if is_new else "updated",
            "node_id": node_id,
            "state": existing.state.value,
            "message": "Android HTTP thin client online",
            "server_time_ms": int(now * 1000),
            "presence_generation": existing.presence_generation,
            "presence_lease_id": existing.presence_lease_id,
            "lease_expires_at_ms": int(existing.presence_expires_at * 1000),
            "heartbeat_interval_seconds": self._runtime_setting('ANDROID_HTTP_CLIENT_HEARTBEAT_INTERVAL_SECONDS'),
        }


    def heartbeat_android_client(self, node_id: str, presence_generation: int = 0,
                                 presence_lease_id: str = "", http_peer: str = "") -> dict:
        """Refresh one Android presence lease without re-registering the node.

        A lease is deliberately fenced by both a monotonically increasing generation
        and an opaque id. This prevents a delayed request from an older app process
        from reviving a newer registration.
        """
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可接收 Android 心跳", "error_code": "not_master"}
        if not node_id or node_id == "master":
            return {"status": "invalid", "reason": "Android node_id 无效", "error_code": "invalid_node_id"}

        now = time.time()
        expired_node = None
        with self._nodes_lock:
            existing = self.nodes.get(node_id)
            if existing is None or existing.node_type != "android":
                return {
                    "status": "rejected", "reason": "Android presence 尚未注册",
                    "error_code": "presence_not_registered",
                }
            if existing.role == NodeRole.MASTER:
                return {"status": "invalid", "reason": f"'{node_id}' 是主节点，不可心跳", "error_code": "invalid_node_id"}
            if not presence_generation or not presence_lease_id:
                return {
                    "status": "rejected", "reason": "缺少 Android presence lease",
                    "error_code": "presence_lease_required",
                }
            if presence_generation != existing.presence_generation:
                return {
                    "status": "rejected", "reason": "Android presence generation 已过期",
                    "error_code": "stale_generation",
                }
            if presence_lease_id != existing.presence_lease_id:
                return {
                    "status": "rejected", "reason": "Android presence lease 已过期",
                    "error_code": "stale_lease",
                }
            if existing.presence_expires_at and now >= existing.presence_expires_at:
                existing.state = NodeState.OFFLINE
                expired_node = existing
            else:
                existing.state = NodeState.ONLINE
                existing.last_heartbeat = now
                existing.presence_expires_at = now + self._runtime_setting('ANDROID_HTTP_CLIENT_LEASE_SECONDS')
                if http_peer:
                    existing.device_info = dict(existing.device_info or {})
                    existing.device_info["http_peer"] = http_peer

        if expired_node is not None:
            if self._effective_role() == "master":
                self._push_node_update_to_all_clients(node_id, "update", expired_node)
            return {
                "status": "rejected", "reason": "Android presence lease 已过期，请重新注册",
                "error_code": "lease_expired",
            }

        if self._effective_role() == "master":
            self._push_node_update_to_all_clients(node_id, "update", existing)
        return {
            "status": "heartbeat",
            "node_id": node_id,
            "state": existing.state.value,
            "server_time_ms": int(now * 1000),
            "presence_generation": existing.presence_generation,
            "presence_lease_id": existing.presence_lease_id,
            "lease_expires_at_ms": int(existing.presence_expires_at * 1000),
            "heartbeat_interval_seconds": self._runtime_setting('ANDROID_HTTP_CLIENT_HEARTBEAT_INTERVAL_SECONDS'),
        }


    def _refresh_http_client_states(self, now: float = None) -> None:
        """按 last_heartbeat 将过期的 Android HTTP 薄客户端标记为 offline。"""
        now = now or time.time()
        changed = []
        with self._nodes_lock:
            for node in self.nodes.values():
                if node.node_type != "android":
                    continue
                info = node.device_info or {}
                if info.get("connection_type") != "http_thin":
                    continue
                if node.state == NodeState.ONLINE and node.last_heartbeat:
                    lease_expired = node.presence_expires_at and now >= node.presence_expires_at
                    if lease_expired or now - node.last_heartbeat > self._runtime_setting('ANDROID_HTTP_CLIENT_TIMEOUT_SECONDS'):
                        node.state = NodeState.OFFLINE
                        changed.append(node)

        if not changed:
            return

        for node in changed:
            if self._effective_role() == "master":
                self._push_node_update_to_all_clients(node.node_id, "update", node)
            logger.info(f"📱 Android HTTP 客户端心跳过期: {node.node_id} → offline")


    def get_available_nodes(self) -> list:
        """获取所有可用节点（含状态字典）"""
        with self._nodes_lock:
            return [n.to_dict() for n in self.nodes.values() if n.is_available()]


    def check_nodes_ready(self) -> bool:
        """检查旧版 PyTorch LAYER_CONFIG worker 是否就绪。"""
        if self._runtime_setting('RUN_MODE') == "single":
            return True
        self._refresh_http_client_states()
        with self._nodes_lock:
            clients = [
                n for n in self.nodes.values()
                if n.role != NodeRole.MASTER and n.node_type == "pc"
            ]
            if not clients:
                return True  # 没有 PC 从节点也算就绪（单节点集群）
            for n in clients:
                if not n.is_available():
                    return False
        return True


    def _push_node_list_to_client(self, client_id: str) -> None:
        """
        向指定从节点推送全量节点列表。

        调用时机:
        - 从节点注册成功后
        - 从节点主动请求 (node_list_sync with request="node_list")
        """
        if not self._tcp_server or not self._tcp_server._running:
            return
        try:
            from transport_port import MessageType
            # Phase 2.1+: 快照后解锁，避免持锁进行 TCP 发送（防止锁排序问题）
            with self._nodes_lock:
                nodes_data = [info.to_dict() for info in self.nodes.values()]
            self._tcp_server.send_to_client(
                client_id,
                {"nodes": nodes_data},
                MessageType.NODE_LIST_SYNC,
            )
            logger.debug(f"已向 {client_id} 推送全量节点列表 ({len(nodes_data)} 个)")
        except Exception as e:
            logger.warning(f"推送节点列表到 {client_id} 失败: {e}")


    def _push_node_update_to_all_clients(self, changed_id: str,
                                         action: str, node_info) -> None:
        """
        向所有已连接从节点推送单节点变更通知。

        Args:
            changed_id: 变更的节点 ID
            action: "add" | "update" | "remove"
            node_info: NodeInfo 对象或 None (remove 时)
        """
        if not self._tcp_server or not self._tcp_server._running:
            return
        try:
            from transport_port import MessageType
            node_data = node_info.to_dict() if node_info else {"node_id": changed_id}
            payload = {
                "action": action,
                "node": node_data,
            }
            for cid in self._tcp_server.get_client_ids():
                if cid == changed_id:
                    continue  # 不推送给变更节点自身
                try:
                    self._tcp_server.send_to_client(
                        cid, payload, MessageType.NODE_UPDATE
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"推送节点更新失败: {e}")


    def _apply_node_list_sync(self, nodes_data: list) -> None:
        """
        从节点：应用主节点推送的全量节点列表。

        保留本地节点信息，用主节点的数据补充/覆盖其他节点。
        """
        # Phase 2.1+: 全量同步需要原子修改 self.nodes，防止与心跳/消息回调并发
        with self._nodes_lock:
            local_id = self.get_effective_node_id()
            now_ts = time.time()
            for nd in nodes_data:
                nid = nd.get("node_id", "")
                if nid == local_id:
                    # ★ 更新自身节点信息（主节点视角更准确: network_type, last_heartbeat 等）
                    local = self.nodes.get(local_id)
                    if local:
                        local.network_type = nd.get("network_type", local.network_type)
                        local.last_heartbeat = nd.get("last_heartbeat", local.last_heartbeat)
                        local.connected_at = nd.get("connected_at", local.connected_at)
                        local.avg_rtt_ms = nd.get("avg_rtt_ms", local.avg_rtt_ms)
                        local.last_rtt_ms = nd.get("last_rtt_ms", local.last_rtt_ms)
                        local.address = nd.get("address", local.address)
                        local.hostname = nd.get("hostname", local.hostname)
                        local.device_info = nd.get("device_info", local.device_info)
                        local.task_count = nd.get("task_count", local.task_count)
                        local.error_count = nd.get("error_count", local.error_count)
                        try:
                            local.state = NodeState(nd.get("state", local.state.value))
                        except ValueError:
                            pass
                    continue
                if nid in self.nodes:
                    # 更新已有节点
                    existing = self.nodes[nid]
                    existing.role = nd.get("role", existing.role)
                    existing.node_type = nd.get("node_type", existing.node_type)
                    existing.hostname = nd.get("hostname", existing.hostname)
                    existing.address = nd.get("address", existing.address)
                    existing.device_info = nd.get("device_info", existing.device_info)
                    existing.network_type = nd.get("network_type", existing.network_type)
                    existing.avg_rtt_ms = nd.get("avg_rtt_ms", existing.avg_rtt_ms)
                    existing.last_rtt_ms = nd.get("last_rtt_ms", existing.last_rtt_ms)
                    existing.task_count = nd.get("task_count", existing.task_count)
                    existing.error_count = nd.get("error_count", existing.error_count)
                    try:
                        existing.state = NodeState(nd.get("state", "offline"))
                    except ValueError:
                        pass
                else:
                    # 新增节点
                    try:
                        state = NodeState(nd.get("state", "offline"))
                    except ValueError:
                        state = NodeState.OFFLINE
                    self.nodes[nid] = NodeInfo(
                        node_id=nid,
                        role=nd.get("role", "client"),
                        node_type=nd.get("node_type", "pc"),
                        state=state,
                        address=nd.get("address", ""),
                        hostname=nd.get("hostname", ""),
                        device_info=nd.get("device_info", {}),
                        network_type=nd.get("network_type", "unknown"),
                        connected_at=nd.get("connected_at", now_ts),
                        last_heartbeat=nd.get("last_heartbeat", now_ts),
                        avg_rtt_ms=nd.get("avg_rtt_ms", 0.0),
                        last_rtt_ms=nd.get("last_rtt_ms", 0.0),
                        task_count=nd.get("task_count", 0),
                        error_count=nd.get("error_count", 0),
                    )


    def _apply_node_update(self, action: str, node_data: dict) -> None:
        """
        从节点：应用单节点变更通知。
        """
        nid = node_data.get("node_id", "")
        if not nid:
            return
        local_id = self.get_effective_node_id()

        # Phase 2.1+: 原子修改 self.nodes，防止与心跳/消息回调并发
        with self._nodes_lock:
            if nid == local_id:
                # ★ 更新自身节点信息（来自主节点的状态更新）
                if action in ("add", "update"):
                    local = self.nodes.get(local_id)
                    if local:
                        local.last_heartbeat = node_data.get("last_heartbeat", local.last_heartbeat)
                        local.network_type = node_data.get("network_type", local.network_type)
                        local.connected_at = node_data.get("connected_at", local.connected_at)
                        local.avg_rtt_ms = node_data.get("avg_rtt_ms", local.avg_rtt_ms)
                        local.last_rtt_ms = node_data.get("last_rtt_ms", local.last_rtt_ms)
                        local.task_count = node_data.get("task_count", local.task_count)
                        local.error_count = node_data.get("error_count", local.error_count)
                        try:
                            local.state = NodeState(node_data.get("state", local.state.value))
                        except ValueError:
                            pass
                return
            if action == "remove":
                self.nodes.pop(nid, None)
            elif action in ("add", "update"):
                now_ts = time.time()
                try:
                    state = NodeState(node_data.get("state", "offline"))
                except ValueError:
                    state = NodeState.OFFLINE
                if nid in self.nodes:
                    existing = self.nodes[nid]
                    existing.state = state
                    existing.role = node_data.get("role", existing.role)
                    existing.node_type = node_data.get("node_type", existing.node_type)
                    existing.hostname = node_data.get("hostname", existing.hostname)
                    existing.address = node_data.get("address", existing.address)
                    existing.device_info = node_data.get("device_info", existing.device_info)
                    existing.network_type = node_data.get("network_type", existing.network_type)
                    existing.avg_rtt_ms = node_data.get("avg_rtt_ms", existing.avg_rtt_ms)
                    existing.last_rtt_ms = node_data.get("last_rtt_ms", existing.last_rtt_ms)
                    existing.task_count = node_data.get("task_count", existing.task_count)
                    existing.error_count = node_data.get("error_count", existing.error_count)
                else:
                    self.nodes[nid] = NodeInfo(
                        node_id=nid,
                        role=node_data.get("role", "client"),
                    node_type=node_data.get("node_type", "pc"),
                    state=state,
                    address=node_data.get("address", ""),
                    hostname=node_data.get("hostname", ""),
                    device_info=node_data.get("device_info", {}),
                    network_type=node_data.get("network_type", "unknown"),
                    connected_at=node_data.get("connected_at", now_ts),
                    last_heartbeat=node_data.get("last_heartbeat", now_ts),
                    avg_rtt_ms=node_data.get("avg_rtt_ms", 0.0),
                    last_rtt_ms=node_data.get("last_rtt_ms", 0.0),
                    task_count=node_data.get("task_count", 0),
                    error_count=node_data.get("error_count", 0),
                )


    def _get_local_network_path_view(self) -> dict | None:
        """Return the client-to-master observation without external I/O."""
        if self._effective_role() != "client":
            return None
        try:
            return build_client_network_path_view(
                getattr(self, "_tcp_client", None)
            )
        except Exception as exc:
            logger.debug(
                "本地网络路径投影失败: error_type=%s",
                type(exc).__name__,
            )
            return None


    def _snapshot_nodes(self, network_path: dict | None = None, *,
                        capacity: Optional[dict] = None) -> dict:
        with self._nodes_lock:
            snapshot = {
                node_id: info.to_dict()
                for node_id, info in self.nodes.items()
            }
            if network_path is not None:
                target_id = next(
                    (
                        node_id
                        for node_id, info in self.nodes.items()
                        if info.role in (NodeRole.MASTER, NodeRole.MASTER.value)
                    ),
                    None,
                )
                if target_id is not None:
                    snapshot[target_id]["network_path"] = network_path

        # ★ 2026-10-08 第 4 条静默路径：`state=online` 只说明心跳在发，不说明该节点
        #   在流水线里干活（Y700 重装清空 DataStore 后即如此：online 但 capacity 的
        #   `worker_count=0`、它只在 `control_only_nodes` 里）。这里把"是否参与"与
        #   "被排除的原因"一并投影到节点上，避免必须交叉 /cluster/nodes 与
        #   /cluster/pipeline-capacity 两个端点才能看出它其实没在干活。
        capacity = capacity if capacity is not None else self._pipeline_capacity_status_snapshot()
        participating = set(capacity.get("participating_node_ids") or ())
        control_only = set(capacity.get("control_only_nodes") or ())
        admitted = capacity.get("admitted")
        plan_reason = str(capacity.get("reason_code", "") or "")
        for node_id, item in snapshot.items():
            is_participating = node_id in participating
            item["pipeline_participating"] = is_participating
            if is_participating:
                exclusion_reason = ""
            elif node_id in control_only:
                exclusion_reason = "capacity_plan_control_only"
            elif admitted is False:
                exclusion_reason = plan_reason or "capacity_plan_rejected"
            elif admitted is None:
                exclusion_reason = "capacity_plan_unknown"
            else:
                exclusion_reason = "capacity_plan_unassigned"
            item["pipeline_exclusion_reason"] = exclusion_reason
        return snapshot

    def _pipeline_capacity_status_snapshot(self) -> dict:
        """Project the current pipeline capacity decision onto the status API.

        The cached decision (active plan, else the in-flight load transaction)
        wins, so a poller never disturbs admission state. When nothing is
        cached it asks the solver once — the same call behind
        ``/api/cluster/pipeline-capacity``, measured side-effect-free at
        ~20 ms — because reporting ``unknown`` whenever the cache is cold would
        hide the very "why" this projection exists to expose.
        """
        with self._layer_config_lock:
            active = (
                dict(self._active_pipeline_capacity_plan)
                if self._active_pipeline_capacity_plan else None
            )
            transaction = self._pipeline_load_transaction
            transaction_plan: Optional[dict] = None
            transaction_snapshot: Optional[dict] = None
            if transaction:
                transaction_plan = dict(transaction.get("plan", {}) or {})
                transaction_snapshot = {
                    "transaction_phase": str(transaction.get("phase", "") or ""),
                    "prepared_node_count": len(transaction.get("prepared_nodes", set()) or ()),
                    "ready_node_count": len(transaction.get("ready_nodes", set()) or ()),
                    "worker_count": len(transaction.get("worker_ids", set()) or ()),
                }

        plan: Optional[dict] = None
        if active:
            plan = active
        elif transaction_plan:
            transaction_plan.update(transaction_snapshot or {})
            plan = transaction_plan
        if not plan:
            try:
                solved = self.get_pipeline_capacity_plan()
            except Exception:
                logger.warning(
                    "event=pipeline_capacity_status_projection_failed", exc_info=True
                )
                solved = None
            if isinstance(solved, dict) and solved:
                plan = dict(solved)

        if not plan:
            return {
                "status": "unknown",
                "admitted": None,
                "reason_code": "pipeline_capacity_not_computed",
                "reason": "",
                "participating_node_ids": [],
                "control_only_nodes": [],
                "worker_count": 0,
                "prepared_node_count": 0,
                "ready_node_count": 0,
                "transaction_phase": "",
                "require_distributed": False,
            }

        assignments = plan.get("assignments") or []
        participating = [
            str(item.get("node_id", ""))
            for item in assignments
            if isinstance(item, dict) and item.get("node_id")
        ]
        return {
            "status": str(plan.get("status", "") or ""),
            "admitted": plan.get("admitted") is True,
            "reason_code": str(plan.get("reason_code", "") or ""),
            "reason": str(plan.get("reason", "") or ""),
            "plan_id": str(plan.get("plan_id", "") or ""),
            "participating_node_ids": participating,
            "participating_node_count": int(
                plan.get("participating_node_count", len(participating)) or 0
            ),
            "control_only_nodes": [
                str(item) for item in (plan.get("control_only_nodes") or ())
            ],
            "worker_count": int(plan.get("worker_count", 0) or 0),
            "prepared_node_count": int(plan.get("prepared_node_count", 0) or 0),
            "ready_node_count": int(plan.get("ready_node_count", 0) or 0),
            "transaction_phase": str(plan.get("transaction_phase", "") or ""),
            "require_distributed": bool(plan.get("require_distributed", False)),
        }


    def get_status(self) -> dict:
        """获取系统整体状态（含节点详情和 TCP 连接信息）"""
        self._refresh_http_client_states()
        network_path = self._get_local_network_path_view()
        capacity_snapshot = self._pipeline_capacity_status_snapshot()
        node_status = self._snapshot_nodes(network_path, capacity=capacity_snapshot)

        current_task = None
        if self._current_task:
            current_task = {
                "task_id": self._current_task.task_id,
                "state": self._current_task.state,
                "elapsed": time.time() - self._current_task.start_time,
            }

        # TCP 服务端状态
        tcp_info = None
        if self._tcp_server:
            tcp_info = {
                "host": self._tcp_server.host,
                "port": self._tcp_server.port,
                "connected_clients": self._tcp_server.get_client_ids(),
                "client_details": {
                    cid: self._tcp_server.get_client_info(cid)
                    for cid in self._tcp_server.get_client_ids()
                },
            }

        # 流水线状态
        pipeline_info = self._get_pipeline_status()

        # 请求队列状态
        queue_info = self._execution_app.requests.queue_snapshot()

        # TCP 客户端状态（从节点视角：到主节点的连接状态）
        tcp_client_info = None
        if self._effective_role() == "client":
            tcp_client = getattr(self, '_tcp_client', None)
            if tcp_client:
                tcp_client_info = {
                    "connected": getattr(tcp_client, 'is_registered', False),
                    "running": getattr(tcp_client, '_running', False),
                    "server_host": getattr(tcp_client, 'server_host', ''),
                    "server_port": getattr(tcp_client, 'server_port', 0),
                    "avg_rtt_ms": round(getattr(tcp_client, 'avg_rtt_ms', 0.0), 1),
                }

        status = {
            "running": self._running,
            "run_mode": self._runtime_setting('RUN_MODE'),
            "nodes": node_status,
            "current_task": current_task,
            "tcp_server": tcp_info,
            "tcp_client": tcp_client_info,
            "nodes_ready": self.check_nodes_ready(),
            "pipeline": pipeline_info,
            "pipeline_capacity": capacity_snapshot,
            "qwen3_pipeline_dry_run": self.get_qwen3_pipeline_dry_run_status(),
            "pipeline_queue": queue_info,
        }
        if network_path is not None:
            status["network_path"] = network_path
        return status


    def get_nodes(self) -> list:
        """获取所有节点详情列表"""
        self._refresh_http_client_states()
        return list(self._snapshot_nodes(self._get_local_network_path_view()).values())


    def get_aggregate_resource_view(self) -> dict:
        """Return the read-only aggregate resource view used by the top layer."""
        self._refresh_http_client_states()
        local_node_id = (
            "master" if self._effective_role() == "master"
            else self.get_effective_node_id()
        )
        with self._nodes_lock:
            nodes = [info.to_dict() for info in self.nodes.values()]
        return build_aggregate_resource_view(nodes, local_node_id=local_node_id)


    def request_node_logs(self, node_id: str, limit: int = 100,
                          level: str = "", name: str = "",
                          timeout: float = 5.0) -> dict | None:
        """
        通过 TCP 向指定从节点拉取最近日志。

        Args:
            node_id: 目标节点 ID
            limit: 返回条数上限
            level: 日志级别过滤 (ERROR/WARNING/INFO/DEBUG)
            name: logger 名称过滤
            timeout: 等待超时秒数

        Returns:
            {node_id, logs, count, matched, buffer_size} 或 None（超时/错误）
        """
        import threading as _thr

        from transport_port import MessageType

        if not self._tcp_server:
            return None

        # 检查节点是否存在且在线
        with self._nodes_lock:
            if node_id not in self.nodes:
                return None
            if self.nodes[node_id].state != NodeState.ONLINE:
                return None

        # 准备信号（加锁保护，防止并发请求覆盖 Event）
        event = _thr.Event()
        with self._pending_log_lock:
            if node_id in self._pending_log_events:
                # 已有等待中的请求，避免 Event 被覆盖导致前一个请求永远超时
                logger.debug(
                    "event=log_aggregation_busy node_id=%s reason=pending_request",
                    node_id,
                )
                return None
            self._pending_log_events[node_id] = event
            self._pending_log_responses.pop(node_id, None)

        try:
            # 发送 LOG_REQUEST
            request_data = {
                "limit": limit,
                "level": level,
                "name": name,
                "node_id": node_id,
            }
            self._tcp_server.send_to_client(node_id, request_data, MessageType.LOG_REQUEST)

            # 等待响应
            signaled = event.wait(timeout)
            if signaled:
                with self._pending_log_lock:
                    result = self._pending_log_responses.pop(node_id, {})
                logger.info(
                    "event=log_aggregation_recv node_id=%s count=%d",
                    node_id, result.get("count", 0),
                )
                return result if result else None

            logger.warning(
                "event=log_aggregation_timeout node_id=%s timeout=%.1fs",
                node_id, timeout,
            )
            return None
        except Exception as e:
            logger.warning(
                "event=log_aggregation_failed node_id=%s error=%s",
                node_id, str(e)[:200],
            )
            return None
        finally:
            with self._pending_log_lock:
                self._pending_log_events.pop(node_id, None)
                # H1: 超时/异常时清理可能已到达的残留响应数据
                self._pending_log_responses.pop(node_id, None)


    def get_config(self) -> dict:
        """获取分布式配置信息（含当前节点角色、动态分层和实际局域网 IP）"""
        from config import (
            SERVER_IP, SERVER_PORT, HEARTBEAT_INTERVAL,
            TOTAL_MODEL_LAYERS,
            QUANT_TYPE, PAGE_SIZE, MAX_PAGE_NUM, MAX_SEQ_LEN,
        )
        # 优先使用运行时检测到的局域网 IP，回退到配置值
        server_ip = getattr(self, '_lan_ip', '') or self._runtime_setting('SERVER_IP')

        # 动态分层配置
        layers_info = self.get_layer_assignments()

        return {
            "run_mode": self._runtime_setting('RUN_MODE'),
            "node_role": self._effective_role(),
            "node_id": self.get_effective_node_id(),
            "max_nodes": self._max_nodes,
            "network": {
                "server_ip": server_ip,
                "server_port": self._runtime_setting('SERVER_PORT'),
                "heartbeat_interval_s": self._runtime_setting('HEARTBEAT_INTERVAL'),
            },
            "layers": layers_info,
            "distributed_inference": {
                "enabled": self.get_distributed_inference_enabled(),
            },
            "model": {
                "quant_type": QUANT_TYPE,
                "page_size": PAGE_SIZE,
                "max_page_num": MAX_PAGE_NUM,
                "max_seq_len": MAX_SEQ_LEN,
            },
            "task_stats": {
                node_id: {
                    "task_count": info.task_count,
                    "error_count": info.error_count,
                }
                for node_id, info in self.nodes.items()
            },
        }


    def connect_to_master(
        self,
        master_host: str,
        master_port: int,
        *,
        force_bootstrap: bool = False,
        persist_preference: bool = False,
        _automatic: bool = False,
    ) -> dict:
        """串行建立唯一的主节点连接。"""
        from network_address import canonical_host

        master_host = canonical_host(master_host)
        master_port = int(master_port)
        with self._master_connect_lock:
            if not self.get_distributed_inference_enabled():
                return {"status": "denied", "reason": "local_inference_mode"}
            current = getattr(self, "_tcp_client", None)
            current_connected = bool(
                current
                and getattr(current, "_running", False)
                and getattr(current, "is_registered", False)
                and getattr(current, "sock", None) is not None
            )
            if (current_connected
                    and str(getattr(current, "server_host", "")) == str(master_host)
                    and int(getattr(current, "server_port", 0)) == int(master_port)):
                preference = self._persist_master_endpoint_preference(
                    master_host, master_port,
                ) if persist_preference else {}
                return {
                    "status": "connected",
                    "node_id": self.get_effective_node_id(),
                    "master": f"{master_host}:{master_port}",
                    "message": "已连接到该主节点",
                    "reused": True,
                    **preference,
                }
            # TCP owns its entire automatic retry intent, including backoff,
            # the EOF-to-retry handoff and finite exhaustion. A periodic health
            # observer must not create another TCPClient for the same target.
            if current is not None and getattr(current, "_reconnect_enabled", False):
                same_target = (str(getattr(current, "server_host", "")) == str(master_host)
                               and int(getattr(current, "server_port", 0)) == master_port)
                if _automatic and same_target:
                    state = getattr(current, "reconnect_state", "connecting")
                    return {"status": "retry_exhausted" if state == "retry_exhausted" else "reconnecting",
                            "node_id": self.get_effective_node_id(), "reused": True,
                            "reason_code": getattr(current, "last_reconnect_error", ""),
                            "master": f"{master_host}:{master_port}"}
                if not current_connected:
                    # An explicit retry or a newly discovered endpoint starts
                    # a fresh intent only after the previous one is stopped.
                    current.disconnect()
            return self._connect_to_master_locked(
                master_host,
                master_port,
                force_bootstrap=force_bootstrap,
                persist_preference=persist_preference,
            )


    def _persist_master_endpoint_preference(self, master_host: str, master_port: int) -> dict:
        """Expose a failed local preference write instead of silently losing it."""
        try:
            from node_config import persist_preferred_master_endpoint

            persist = getattr(self, "_persist_preferred_endpoint", None) or persist_preferred_master_endpoint
            endpoint = persist(master_host, master_port)
            return {
                "endpoint_preference": {
                    "persisted": True,
                    "address_family": endpoint["address_family"],
                },
            }
        except Exception as exc:
            logger.warning("主节点首选地址未能持久化: %s", exc, exc_info=True)
            return {
                "endpoint_preference": {
                    "persisted": False,
                    "reason": "local_config_write_failed",
                },
            }


    def _connect_to_master_locked(
        self,
        master_host: str,
        master_port: int,
        *,
        force_bootstrap: bool = False,
        persist_preference: bool = False,
    ) -> dict:
        """
        从节点主动连接主节点（由前端「连接主节点」按钮触发）。

        仅在 NODE_ROLE="client" 且有 TCP 客户端模块时可用。

        Args:
            master_host: 主节点 IP
            master_port: 主节点端口

        Returns:
            { status, node_id, master, message }
        """
        effective_role = self._effective_role()
        if effective_role != "client":
            return {"status": "denied", "reason": "仅从节点可以连接主节点"}
        connecting_role = "client"

        try:
            import config as cfg

            # ★ 安全：从节点绝不能使用 "master" 作为 client_id
            configured_node_id = self._runtime_setting('_configured_node_id')()
            if not configured_node_id or configured_node_id == "master":
                node_id = f"client_{__import__('socket').gethostname()}"
            else:
                node_id = configured_node_id

            def _run_first_connect_bootstrap(reason: str) -> None:
                nonlocal node_id, master_host, master_port
                from bootstrap import first_connect

                api_port = self._runtime_setting('_bootstrap_api_port')()
                logger.info(
                    "开始首次连接自动部署: reason=%s master_api=%s:%s",
                    reason, master_host, api_port,
                )
                bootstrap_result = first_connect(
                    master_api_host=master_host,
                    master_api_port=api_port,
                    node_id=node_id,
                    node_type=(self._runtime_config.environment_snapshot() if self._runtime_config is not None else os.environ).get("QLH_NODE_TYPE", "pc"),
                    **({"accept_bootstrap": self._accept_bootstrap} if getattr(self, "_accept_bootstrap", None) is not None else {}),
                )
                cluster = bootstrap_result.get("cluster", {})
                node = bootstrap_result.get("node", {})
                node_id = node.get("node_id") or node_id
                master_host = cluster.get("master_tcp_host") or master_host
                master_port = int(cluster.get("master_tcp_port") or master_port)
                logger.info(
                    "首次连接自动部署完成: node_id=%s master=%s:%s",
                    node_id, master_host, master_port,
                )

            if force_bootstrap or not (self._runtime_settings.get("CLUSTER_SECRET", "") if self._runtime_settings is not None else getattr(cfg, "CLUSTER_SECRET", "")):
                try:
                    _run_first_connect_bootstrap(
                        "explicit_join" if force_bootstrap else "missing_secret"
                    )
                except Exception as e:
                    logger.error("首次连接自动部署失败: %s", e, exc_info=True)
                    return {
                        "status": "bootstrap_failed",
                        "reason": f"首次连接自动部署失败: {e}",
                    }

            from transport_port import create_client

            def _publish_client(target_client) -> None:
                # Same short boundary used by hello creation and disconnect
                # cleanup. Never hold it across connect(), hashing or sending.
                with self._task_worker_refresh_lock:
                    self._tcp_client = target_client

            previous_client = getattr(self, "_tcp_client", None)
            previous_callback = (
                getattr(previous_client, "on_disconnect", None)
                if previous_client is not None else None
            )

            def _retire_previous_client() -> None:
                if previous_client is None or previous_client is client:
                    return
                previous_client.on_disconnect = None
                try:
                    previous_client.disconnect()
                except Exception:
                    logger.debug("关闭旧主节点连接失败", exc_info=True)

            def _discard_candidate_client(candidate) -> None:
                candidate.on_disconnect = None
                try:
                    candidate.disconnect()
                except Exception:
                    logger.debug("关闭失败的候选主节点连接失败", exc_info=True)

            def _restore_previous_client() -> None:
                if previous_client is None:
                    _publish_client(None)
                    return
                if not (
                    getattr(previous_client, "_running", False)
                    and getattr(previous_client, "is_registered", False)
                    and getattr(previous_client, "sock", None) is not None
                ):
                    _publish_client(None)
                    return
                previous_client.on_disconnect = previous_callback
                _publish_client(previous_client)
                previous_node_id = getattr(previous_client, "client_id", "")
                if previous_node_id:
                    self._runtime_setting('_sync_runtime_node_config')(
                        node_id=previous_node_id,
                        node_role=connecting_role,
                    )

            advertise_port = self._tcp_server.port if self._tcp_server else self._runtime_setting('SERVER_PORT')
            client = create_client(
                server_host=master_host,
                server_port=master_port,
                client_id=node_id,
                role="client",
                advertise_port=advertise_port,
                device_info=self._local_device_profile,
                **self._transport_runtime_kwargs(node_id),
            )
            # REGISTER_ACK 后主节点会立即下发层配置，接收线程可能在
            # connect() 返回前进入回调。提前绑定连接和最终 node_id，保证
            # 模型同步能取得主节点地址，且 ready/error ACK 能正常发回。
            _publish_client(client)
            self._runtime_setting('_sync_runtime_node_config')(node_id=node_id, node_role=connecting_role)
            # ★ 心跳回调：更新自身节点的心跳时间 + 同步 RTT 测量值
            # _sync_node_rtt 内部已有 _nodes_lock 保护
            def _bind_client_callbacks(target_client) -> None:
                def _on_client_heartbeat() -> None:
                    self._sync_node_rtt(node_id, target_client)
                    self._report_local_device_profile(target_client, node_id)
                    self._task_worker_control.mark_coordinator_heartbeat()

                def _on_client_reconnect() -> None:
                    if (self._tcp_client is target_client
                            and getattr(target_client, "is_registered", False)):
                        self.refresh_task_worker_capabilities()

                target_client.on_heartbeat = _on_client_heartbeat
                target_client.on_reconnect = _on_client_reconnect
                def _on_client_disconnect() -> None:
                    # TCP notifies outside its registration lock. Serialize
                    # cleanup with hello creation, and skip a delayed callback
                    # once this client has already registered again.
                    with self._task_worker_refresh_lock:
                        if (self._tcp_client is not target_client
                                or getattr(target_client, "is_registered", False)):
                            logger.debug("event=task_worker_disconnect_callback_stale")
                            return
                        self._on_master_connection_lost(target_client)

                target_client.on_disconnect = _on_client_disconnect

            _bind_client_callbacks(client)

            def _mark_local_node_online() -> None:
                # NodeInfo.address 表示本节点可被其他节点连接的服务端点，
                # 不能写成主节点地址；主节点地址由 tcp_client.server_host/server_port 表示。
                with self._nodes_lock:
                    if node_id in self.nodes:
                        self.nodes[node_id].state = NodeState.ONLINE
                        self.nodes[node_id].last_heartbeat = time.time()

            ok = client.connect(
                on_message=lambda msg: self._on_tcp_message("master", msg)
            )
            if ok:
                _retire_previous_client()
                _mark_local_node_online()
                self._report_local_device_profile(client, node_id)

                # 更新运行时 node_id，避免 scheduler 模块导入常量滞后。
                self._runtime_setting('_sync_runtime_node_config')(node_id=node_id)
                # 节点在线状态以主节点收到的 TCP REGISTER 为事实源。
                # 从节点只保存 bootstrap 目标，不写共享状态或远端节点表。

                # 主节点注册成功后，请求全量节点列表以同步管理面板
                try:
                    from transport_port import MessageType
                    client.send_data({"request": "node_list"}, MessageType.NODE_LIST_SYNC)
                except Exception:
                    pass
                self._send_task_worker_hello(client)

                logger.info(f"✅ 从节点 {node_id} 已连接到主节点 {master_host}:{master_port}")
                preference = self._persist_master_endpoint_preference(
                    master_host, master_port,
                ) if persist_preference else {}
                return {
                    "status": "connected",
                    "node_id": node_id,
                    "master": f"{master_host}:{master_port}",
                    "message": f"已成功注册到主节点 {master_host}:{master_port}",
                    **preference,
                }
            if getattr(self, '_tcp_client', None) is client:
                _discard_candidate_client(client)
                _restore_previous_client()
            reason = getattr(client, "last_register_error", "") or (
                f"TCP 连接失败 ({master_host}:{master_port})，请检查主节点地址和端口是否正确"
            )
            if self._runtime_setting('_is_auth_register_failure')(reason):
                try:
                    logger.info("TCP 注册认证失败，尝试刷新首次连接配置后重试: %s", reason)
                    _run_first_connect_bootstrap("auth_failed")
                    client = create_client(
                        server_host=master_host,
                        server_port=master_port,
                        client_id=node_id,
                        role="client",
                        advertise_port=advertise_port,
                        device_info=self._local_device_profile,
                        **self._transport_runtime_kwargs(node_id),
                    )
                    _publish_client(client)
                    self._runtime_setting('_sync_runtime_node_config')(node_id=node_id, node_role=connecting_role)
                    _bind_client_callbacks(client)
                    ok = client.connect(
                        on_message=lambda msg: self._on_tcp_message("master", msg)
                    )
                    if ok:
                        _retire_previous_client()
                        _mark_local_node_online()
                        self._report_local_device_profile(client, node_id)
                        self._runtime_setting('_sync_runtime_node_config')(node_id=node_id)
                        try:
                            from transport_port import MessageType
                            client.send_data({"request": "node_list"}, MessageType.NODE_LIST_SYNC)
                        except Exception:
                            pass
                        self._send_task_worker_hello(client)
                        logger.info(
                            "✅ 从节点 %s 刷新配置后已连接到主节点 %s:%s",
                            node_id, master_host, master_port,
                        )
                        preference = self._persist_master_endpoint_preference(
                            master_host, master_port,
                        ) if persist_preference else {}
                        return {
                            "status": "connected",
                            "node_id": node_id,
                            "master": f"{master_host}:{master_port}",
                            "message": f"已刷新自动部署配置并注册到主节点 {master_host}:{master_port}",
                            **preference,
                        }
                    if getattr(self, '_tcp_client', None) is client:
                        _discard_candidate_client(client)
                        _restore_previous_client()
                    reason = getattr(client, "last_register_error", "") or reason
                except Exception as e:
                    logger.error("刷新首次连接配置后重试失败: %s", e, exc_info=True)
                    reason = f"{reason}; 刷新自动部署配置失败: {e}"

            return {
                "status": "failed",
                "reason": reason,
            }
        except Exception as e:
            if 'client' in locals() and getattr(self, '_tcp_client', None) is client:
                try:
                    _discard_candidate_client(client)
                except Exception:
                    pass
                if 'previous_client' in locals() and previous_client is not None:
                    _restore_previous_client()
                else:
                    _publish_client(None)
            logger.error(f"连接主节点 {master_host}:{master_port} 失败: {e}")
            return {"status": "error", "reason": f"{master_host}:{master_port} - {e}"}


    def forward_inference_to_master(self, message: str,
                                     max_new_tokens: int = 512,
                                     temperature: float = 0.7,
                                     top_p: float = 0.9,
                                     show_thinking: bool = False,
                                     enable_thinking: Optional[bool] = None,
                                     session_id: Optional[str] = None,
                                     messages: list = None,
                                     request_id: str = None,   # L5: 链路追踪
                                     routing_preference: str = "auto",
                                     _cancel_event: Optional[threading.Event] = None,
                                     _request_deadline=None,
                                     timeout: float = 120.0,
                                     speculative_gamma: Optional[int] = None) -> dict:
        """
        从节点将推理请求转发给主节点，并等待结果。

        仅从节点可调用，需要已通过 connect_to_master() 建立 TCP 连接。

        Args:
            message: 用户输入
            max_new_tokens: 最大新 token 数
            temperature: 温度
            top_p: top_p
            show_thinking: 是否启用深度思考展示
            session_id: 会话 ID（多会话支持）
            timeout: 等待结果超时秒数

        Returns:
            {status, content, metrics, error}
        """
        # Transit boundaries validate explicit values without resolving an env default.
        speculation = {}
        if speculative_gamma is not None:
            from qlh_core.contracts.speculation import resolve_speculative_gamma
            from qlh_core.contracts.execution import ExecutionError
            try:
                gamma = resolve_speculative_gamma(speculative_gamma)
            except ExecutionError as exc:
                return {"status": "error", "error": str(exc), "reason_code": exc.code}
            if gamma > 1 and routing_preference == "local_only":
                return {"status": "error", "error": "speculative_route_unsupported",
                        "reason_code": "speculative_route_unsupported"}
            speculation["speculative_gamma"] = gamma
        if self._effective_role() != "client":
            return {"status": "denied", "error": "仅从节点可转发推理请求"}

        tcp_client = getattr(self, '_tcp_client', None)
        if not (tcp_client
                and getattr(tcp_client, "_running", False)
                and getattr(tcp_client, "is_registered", False)
                and getattr(tcp_client, "sock", None) is not None):
            return {"status": "disconnected", "error": "未连接到主节点，请先建立连接"}

        forward_request_id = uuid.uuid4().hex
        result_event = threading.Event()
        with self._client_pending_lock:
            self._client_pending_events[forward_request_id] = result_event
            self._client_pending_results.pop(forward_request_id, None)

        try:
            from transport_port import MessageType

            # 发送推理请求
            infer_data = {
                "prompt": message,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "show_thinking": show_thinking,
                "enable_thinking": enable_thinking,
                "session_id": session_id,
                "messages": messages or [{"role": "user", "content": message}],
                "request_id": request_id,   # L5: 链路追踪
                "routing_preference": routing_preference,
                "forward_request_id": forward_request_id,
                **speculation,
            }
            if _request_deadline is not None:
                infer_data["request_deadline_ms"] = (
                    _request_deadline.expires_at_epoch_ms
                )
            tcp_client.send_data(infer_data, MessageType.INFER_FORWARD)
            logger.info(
                "event=infer_forward task_id=n/a request_id=%s forward_request_id=%s "
                "prompt_len=%d",
                request_id or "-", forward_request_id, len(message),
            )

            deadline = time.monotonic() + max(0.0, timeout)
            if _request_deadline is not None:
                deadline = min(
                    deadline,
                    _request_deadline.expires_at_monotonic,
                )
            result_ready = False
            while True:
                if _cancel_event is not None and _cancel_event.is_set():
                    try:
                        tcp_client.send_data(
                            {"forward_request_id": forward_request_id},
                            MessageType.INFER_CANCEL,
                        )
                    except Exception:
                        logger.debug("发送转发推理取消失败", exc_info=True)
                    return {"status": "cancelled", "error": "推理已取消"}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if result_event.wait(timeout=min(0.1, remaining)):
                    result_ready = True
                    break

            if result_ready:
                with self._client_pending_lock:
                    result = self._client_pending_results.pop(
                        forward_request_id, None
                    )
                if result is None:
                    return {"status": "error", "error": "主节点结果状态丢失"}
                logger.info(
                    "收到主节点推理结果: task=%s request=%s len=%d",
                    result.get("task_id", ""), forward_request_id,
                    len(result.get("content", "")),
                )
                if result.get("status") != "ok" or result.get("error"):
                    metrics = result.get("metrics", {})
                    return {
                        "status": "error",
                        "error": result.get("error") or "主节点推理失败",
                        "metrics": metrics,
                        "reason_code": str(
                            result.get("reason_code")
                            or metrics.get("reason_code")
                            or ""
                        ),
                    }
                return {
                    "status": "ok",
                    "content": result.get("content", ""),
                    "metrics": result.get("metrics", {}),
                    "thinking_content": result.get("thinking_content"),
                    "followups": result.get("followups", []),
                }

            # 超时
            request_deadline_exceeded = bool(
                _request_deadline is not None
                and _request_deadline.expired()
            )
            logger.warning(f"推理请求超时 ({timeout}s)")
            try:
                tcp_client.send_data(
                    {"forward_request_id": forward_request_id},
                    MessageType.INFER_CANCEL,
                )
            except Exception:
                logger.debug("发送转发推理取消失败", exc_info=True)
            return {
                "status": "timeout",
                "error": (
                    "request deadline exceeded"
                    if request_deadline_exceeded
                    else f"等待主节点响应超时 ({timeout}s)"
                ),
                "reason_code": (
                    "request_deadline_exceeded"
                    if request_deadline_exceeded else "forward_timeout"
                ),
            }

        except Exception as e:
            logger.error(f"转发推理请求失败: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            with self._client_pending_lock:
                self._client_pending_events.pop(forward_request_id, None)
                self._client_pending_results.pop(forward_request_id, None)


    def get_invite_info(self) -> dict:
        """
        主节点获取邀请信息（供从节点连接使用）。

        优先使用运行时检测到的 Tailscale 地址；未登录 Tailnet 时使用
        当前可达的 LAN 地址，回退到 SERVER_IP。地址不会写入 SQL：LAN
        地址随网络变化在每次启动时重新探测，Tailnet 地址由 Tailscale 管理。

        Returns:
            { master_host, master_port, node_count, connected_clients,
              discovery_methods, mac_addresses, identity_verified, identity_reason }
        """
        # 使用运行时检测到的可广告地址或已有的配置值。
        lan_ip = getattr(self, '_lan_ip', '') or self._runtime_setting('SERVER_IP')
        try:
            from network_address import is_tailscale_ip
            host_source = "tailnet" if is_tailscale_ip(lan_ip) else "lan"
        except Exception:
            host_source = "lan"
        port = self._tcp_server.port if self._tcp_server else self._runtime_setting('SERVER_PORT')
        macs = getattr(self, '_mac_addresses', [])

        # Phase 2.1+: 锁保护迭代计数，防止并发修改
        with self._nodes_lock:
            online_count = sum(1 for n in self.nodes.values() if n.is_available())
            active_non_master = [
                n for n in self.nodes.values()
                if n.role != "master" and (n.is_available() or bool(n.address))
            ]
            capacity_used = 1 + len(active_non_master)
            total_records = len(self.nodes)

        return {
            "master_host": lan_ip,
            "master_host_source": host_source,
            "master_port": port,
            "node_count": capacity_used,
            "total_node_records": total_records,
            "online_count": online_count,
            "max_nodes": self._max_nodes,
            "has_capacity": capacity_used < self._max_nodes,
            "connected_clients": (
                self._tcp_server.get_client_ids() if self._tcp_server else []
            ),
            "discovery_methods": ["local_config", "tailnet"],
            "mac_addresses": macs,
            "identity_verified": getattr(self, '_master_identity_verified', False),
            "identity_reason": getattr(self, '_master_identity_reason', ''),
        }


    def can_join_existing_master(self) -> bool:
        """A configured master cannot change roles without reconfiguration."""
        return False


    def _auto_connect_on_startup(self) -> None:
        """
        后台线程：从节点启动后自动发现并连接主节点。

        启动后最多等待 5 秒（给本机网络与 Tailnet 状态足够时间就绪），
        然后尝试发现主节点并连接。停止时立即取消等待。
        """
        if self._startup_cancel_event.wait(timeout=5) or not self._running:
            return
        # 如果已经连接，跳过。
        tcp_client = getattr(self, '_tcp_client', None)
        if tcp_client and (tcp_client._running or getattr(tcp_client, "_reconnect_enabled", False)):
            logger.info("已有活跃 TCP 连接，跳过启动自动连接")
            return

        try:
            discovery = self.discover_master()
            if discovery.get("found"):
                stale_note = "（心跳过期）" if discovery.get("stale") else ""
                host = discovery["master_host"]
                port = discovery["master_port"]
                logger.info(f"🔍 启动自动发现: 主节点 {host}:{port}{stale_note}，尝试连接...")
                result = self.connect_to_master(host, port)
                if result.get("status") == "connected":
                    logger.info(f"✅ 启动自动连接成功: {host}:{port}")
                else:
                    logger.info(f"启动自动连接失败: {result.get('reason', result.get('status'))}")
                    for alternate in self._discover_master_fallbacks(host, int(port)):
                        alt_host = alternate["master_host"]
                        alt_port = alternate["master_port"]
                        logger.info(
                            "启动自动连接回退: source=%s master=%s:%s",
                            alternate["source"], alt_host, alt_port,
                        )
                        if self.connect_to_master(alt_host, alt_port).get("status") == "connected":
                            break
            else:
                logger.info("启动自动发现: 未找到可用主节点，稍后可通过前端手动连接")
        except Exception as e:
            logger.warning(f"启动自动连接异常: {e}")


    def get_effective_node_id(self) -> str:
        """
        返回当前节点的有效 ID。

        主节点 → "master"
        从节点 → 使用配置的 NODE_ID，若为 "master"（默认值）则自动生成
        """
        effective_role = self._effective_role()
        node_id = self._runtime_setting('_configured_node_id')()
        if effective_role == "client" and (not node_id or node_id == "master"):
            return f"client_{__import__('socket').gethostname()}"
        return node_id


    def get_my_role(self) -> dict:
        """
        获取当前节点的角色信息。

        用于前端判断是否显示后台管理 Tab：
        - master 节点：完全开放
        - client 节点：需开启"分布式推理优化"后才显示自己的后台

        从节点使用本机 bootstrap 配置和 Tailnet 尝试发现主节点连接信息。

        Returns:
            { node_role, node_id, is_master, max_nodes, master_discovery, ... }
        """
        effective_role = self._effective_role()
        _effective_id = self.get_effective_node_id()
        # Phase 2.1+: 锁保护读取
        with self._nodes_lock:
            my_info = self.nodes.get(_effective_id)
        result = {
            "inference_mode": "distributed" if self.get_distributed_inference_enabled() else "local",
            "node_role": effective_role,
            "node_id": _effective_id,
            "is_master": effective_role == "master",
            "is_client": effective_role == "client",
            "max_nodes": self._max_nodes,
            "run_mode": self._runtime_setting('RUN_MODE'),
            "my_node": my_info.to_dict() if my_info else None,
            "tcp_server_running": self._tcp_server is not None and self._tcp_server._running,
        }
        result["is_provisional"] = False
        result["can_join_existing_master"] = False

        # 主节点：附加 MAC 身份验证状态
        if effective_role == "master":
            result["mac_addresses"] = getattr(self, '_mac_addresses', [])
            result["identity_verified"] = getattr(self, '_master_identity_verified', False)
            result["identity_reason"] = getattr(self, '_master_identity_reason', '')

        # 从节点：使用本机 bootstrap/Tailnet 自动发现主节点
        if effective_role == "client":
            try:
                discovery = self.discover_master()
                result["master_discovery"] = discovery
            except Exception:
                result["master_discovery"] = {"found": False}

        return result


    def update_max_nodes(self, new_max: int) -> dict:
        """
        动态调整最大节点数量（仅 master 可调用）。

        仅修改容量上限，不预创建空槽位。从节点通过 TCP 注册动态加入。

        Args:
            new_max: 新的最大节点数 (>= 1, 包含 master)

        Returns:
            { status, max_nodes, nodes_added, nodes_removed, ... }
        """
        import config as cfg

        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可修改最大节点数"}

        if new_max < 1:
            return {"status": "invalid", "reason": "max_nodes 至少为 1 (仅 master)"}

        old_max = self._max_nodes
        if new_max == old_max:
            return {"status": "unchanged", "max_nodes": old_max}

        cfg.MAX_NODES = new_max
        self._max_nodes = new_max

        # 清理残留幽灵节点（旧代码扩容时预创建的空槽位，从未连接过）
        # Phase 2.1+: 锁保护迭代+删除操作
        with self._nodes_lock:
            phantoms = []
            for nid, node in list(self.nodes.items()):
                if (nid != "master" and node.role != "master"
                        and not node.address and not node.hostname
                        and not node.connected_at and not node.last_heartbeat
                        and node.state == NodeState.OFFLINE):
                    phantoms.append(nid)
            for pid in phantoms:
                del self.nodes[pid]
                logger.info(f"  清理幽灵节点: {pid}")
        if phantoms:
            invalidate_transaction = getattr(
                self, "_invalidate_pipeline_load_transaction", None,
            )
            if callable(invalidate_transaction):
                invalidate_transaction(
                    reason_code="phantom_nodes_removed",
                    reason=(
                        "phantom nodes removed while updating cluster capacity: "
                        + ", ".join(sorted(phantoms))
                    ),
                )
            with self._layer_config_lock:
                # Phantom cleanup is a global control-plane reset. Keep the
                # assignment registry, retry state, transaction and active
                # route in one state: empty and therefore not routable.
                self._worker_assignments.clear()
                self._layer_config_retry_state.clear()
                self._pipeline_load_transaction = None
                self._active_pipeline_capacity_plan = None

        logger.info(f"最大节点数已更新: {old_max} → {new_max}"
                    + (f" (清理幽灵: {phantoms})" if phantoms else ""))

        return {
            "status": "ok",
            "max_nodes": new_max,
            "old_max": old_max,
            "nodes_added": [],
            "nodes_removed": [],
            "total_nodes": len(self.nodes),
        }


    @property
    def tcp_server(self):
        """获取 TCP 服务端实例"""
        return self._tcp_server


    def _verify_master_identity(self) -> None:
        """
        验证本机 MAC 地址是否与用户自持 SQLite 中的主节点 MAC 匹配。

        验证逻辑:
        - SQLite 中尚无 MAC 记录 → 首次启动，立即记录本机 MAC，"first_run"
        - 本机 MAC 与本地记录有交集 → 身份验证通过，"match"
        - 本机 MAC 与本地记录无交集 → 身份验证失败，"mac_mismatch"（可能是不同机器）

        验证结果存储在 self._master_identity_verified 和 self._master_identity_reason 中，
        前端可通过 get_my_role() 查询验证状态。
        """
        local_macs = sorted({
            str(mac).strip().lower()
            for mac in getattr(self, "_mac_addresses", [])
            if str(mac).strip()
        })
        if not local_macs:
            self._master_identity_verified = False
            self._master_identity_reason = "no_physical_mac"
            logger.error("主节点身份验证失败：未检测到可用物理网卡 MAC")
            return

        try:
            from local_store import (
                get_local_master_identity,
                set_local_master_identity,
            )

            stored = get_local_master_identity()
            stored_macs = stored.get("mac_addresses", [])
            if not stored_macs:
                set_local_master_identity(local_macs)
                self._master_identity_verified = True
                self._master_identity_reason = "first_run"
                logger.info("首次启动，已将本机 MAC 绑定到主节点 SQLite")
                return

            matched = sorted(set(local_macs).intersection(stored_macs))
            self._master_identity_verified = bool(matched)
            self._master_identity_reason = "match" if matched else "mac_mismatch"
        except Exception as e:
            logger.error("主节点 SQLite 身份验证异常: %s", e)
            self._master_identity_verified = False
            self._master_identity_reason = "local_store_unavailable"
            return

        if self._master_identity_reason == "match":
            logger.info("主节点身份验证通过：MAC 与主节点 SQLite 记录匹配 %s", matched)
        elif self._master_identity_reason == "mac_mismatch":
            logger.warning(
                f"⛔ 主节点身份验证失败！本机 MAC {local_macs} "
                f"与主节点 SQLite 记录 {stored_macs} 不匹配！"
                f"这可能意味着另一台机器正在尝试冒充主节点。"
                f"如需更换主节点机器，请在设置中使用「重置主节点身份」功能。"
            )


    def discover_master(self, *, skip_config: bool = False) -> dict:
        """
        发现主节点的连接信息（供从节点自动发现）。

        已连接节点优先使用本机保存的 bootstrap 配置；未保存配置时，
        通过同一 Tailnet 探测主节点。

        Returns:
            {
                "found": bool,
                "master_host": str,
                "master_port": int,
                "stale": bool,
                "source": "config" | "tailnet" | "none",
            }
        """
        # 已完成 bootstrap 的节点优先使用本地配置。
        import config as cfg
        settings = self._runtime_settings if self._runtime_settings is not None else vars(cfg)
        if (not skip_config
                and settings.get("CLIENT_MASTER_HOST")
                and settings["CLIENT_MASTER_HOST"] != "192.168.x.x"):
            return {
                "found": True,
                "master_host": settings["CLIENT_MASTER_HOST"],
                "master_port": settings["CLIENT_MASTER_PORT"],
                "stale": False,
                "source": "config",
            }

        try:
            from bootstrap import discover_master_via_tailnet
            return discover_master_via_tailnet(api_port=self._runtime_setting('_bootstrap_api_port')())
        except Exception as e:
            logger.debug("Tailnet 主节点发现失败: %s", e, exc_info=True)
            return {"found": False, "source": "none"}


    def _discover_master_fallbacks(
        self,
        attempted_host: str,
        attempted_port: int,
    ) -> list[dict]:
        """Return deterministic fallback endpoints after a primary attempt.

        The user-selected endpoint remains the primary config value.  The
        bootstrap endpoint is retained separately, then Tailnet discovery is
        last.  This prevents a background retry from overwriting an explicit
        IPv6 choice with a stale IPv4 bootstrap address.
        """
        from network_address import canonical_host

        attempted = (canonical_host(attempted_host), int(attempted_port or 0))
        candidates: list[dict] = []
        seen = {attempted}

        def _append(host: str, port: int, source: str, *, stale: bool = False) -> None:
            endpoint = (canonical_host(host), int(port or 0))
            if not endpoint[0] or not endpoint[1] or endpoint in seen:
                return
            seen.add(endpoint)
            candidates.append({
                "found": True,
                "master_host": endpoint[0],
                "master_port": endpoint[1],
                "stale": stale,
                "source": source,
            })

        try:
            from node_config import get_bootstrap_master_endpoint

            read_document = getattr(self, "_read_identity_document", None)
            bootstrap_endpoint = get_bootstrap_master_endpoint(read_document()) if read_document is not None else get_bootstrap_master_endpoint()
            if bootstrap_endpoint is not None:
                _append(
                    bootstrap_endpoint["host"],
                    bootstrap_endpoint["port"],
                    "bootstrap_config",
                )
        except Exception as exc:
            logger.debug("读取 bootstrap 回退地址失败: %s", exc, exc_info=True)

        tailnet = self.discover_master(skip_config=True)
        if tailnet.get("found"):
            _append(
                str(tailnet.get("master_host", "")),
                int(tailnet.get("master_port", 0) or 0),
                str(tailnet.get("source", "tailnet")),
                stale=bool(tailnet.get("stale", False)),
            )
        return candidates


    def get_distributed_inference_enabled(self) -> bool:
        """
        获取分布式推理开关状态。

        优先级: 运行时变量 > config.py 默认值。
        """
        if self._distributed_inference_enabled is not None:
            return self._distributed_inference_enabled
        return bool(self._runtime_setting("DISTRIBUTED_INFERENCE_ENABLED"))


    def set_distributed_inference_enabled(self, enabled: bool) -> dict:
        """Set participation and disconnect local workers when opting out."""
        with self._host.full_chat_execution_lock:
            return self._set_distributed_inference_enabled_locked(enabled)


    def _set_distributed_inference_enabled_locked(self, enabled: bool) -> dict:
        if enabled and not self.get_distributed_inference_enabled():
            # The model API unloads under this same lock before rejoining.
            # Other control surfaces must not advertise a resident local model.
            with self._host.full_chat_execution_lock:
                has_loaded = getattr(self._host, "has_loaded_model", None)
                loaded = bool(self._host.model_loaded) or bool(
                    callable(has_loaded) and has_loaded()
                )
                if loaded:
                    return {
                        "status": "error", "reason_code": "local_model_still_loaded",
                        "reason": "请先通过模型模式接口卸载本地模型，再加入集群。",
                    }
        self._distributed_inference_enabled = bool(enabled)
        self._client_reconnect_enabled = bool(enabled)
        if not enabled:
            supply_worker = getattr(self, "_model_supply_worker", None)
            if supply_worker is not None:
                supply_worker.cancel()
            supply = getattr(self, "_model_supply", None)
            if supply is not None:
                supply.cancel_active()
            self.release_pipeline_worker_for_local_model()
            client = getattr(self, "_tcp_client", None)
            if client is not None:
                client.disconnect()
        if enabled and self._effective_role() == "client":
            self._request_pipeline_worker_opt_in()
            if self._running and not getattr(self._tcp_client, "is_registered", False):
                threading.Thread(target=self._auto_connect_on_startup, name="explicit-cluster-rejoin", daemon=True).start()
        logger.info(f"分布式推理已{'启用' if enabled else '禁用'}")
        return {
            "status": "ok",
            "enabled": enabled,
            "message": f"分布式推理已{'启用' if enabled else '禁用'}",
        }


    def _request_pipeline_worker_opt_in(self) -> bool:
        """Ask the master to include this client in future layer assignments."""
        if not self.get_distributed_inference_enabled():
            return False
        with self._layer_config_lock:
            self._pipeline_worker_opted_out = False
        client = getattr(self, "_tcp_client", None)
        if not client or not getattr(client, "_running", False):
            return False
        try:
            from transport_port import MessageType

            client.send_data(
                {"node_id": self.get_effective_node_id()},
                MessageType.LAYER_WORKER_OPT_IN,
            )
            return True
        except Exception:
            logger.warning("通知主节点重新加入分层 worker 失败", exc_info=True)
            return False


    def reset_master_identity(self) -> dict:
        """
        重置主节点身份标识（仅主节点可调用）。

        用于以下场景：
        - 更换主节点机器（新机器的 MAC 与本机 SQLite 中记录不匹配）
        - 主节点更换了网卡
        - 需要清除旧的 MAC 记录重新绑定

        调用后立即把当前物理 MAC 绑定到主节点 SQLite，无需重启。
        """
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可重置身份标识"}

        try:
            # ★ R-R4（B5 / #8）：重置的语义是「**强制回到物理真相**」⇒ 必须以**实时探测的物理 MAC**
            #   为准，**不能**优先沿用 `self._mac_addresses` —— 那是启动时写入、也可能被别处改过的
            #   内存缓存；实测把缓存污染成错值后重置，会把**错值固化**进 SQLite，而本方法的 docstring
            #   明说「立即把当前**物理** MAC 绑定到主节点 SQLite」。
            #   ⚠️ 仍保留一层回落：**探测不到**物理网卡时才用缓存，避免探测失败把身份清空。
            from transport_port import get_mac_addresses

            macs = sorted({
                str(mac).strip().lower()
                for mac in get_mac_addresses()
                if str(mac).strip()
            })
            if not macs:
                macs = sorted({
                    str(mac).strip().lower()
                    for mac in getattr(self, "_mac_addresses", [])
                    if str(mac).strip()
                })
            if not macs:
                return {"status": "error", "reason": "未检测到可用物理网卡 MAC，无法重置身份"}

            from local_store import set_local_master_identity
            set_local_master_identity(macs)
            self._mac_addresses = macs
            self._master_identity_verified = True
            self._master_identity_reason = "reset"
            logger.warning("主节点身份已重置并重新绑定到本机 SQLite: %s", macs)
            return {
                "status": "ok",
                "message": "主节点身份已重置，并已立即绑定当前物理网卡 MAC。",
            }
        except Exception as e:
            logger.error("主节点 SQLite 身份重置失败: %s", e)
            return {"status": "error", "reason": f"主节点 SQLite 不可用: {e}"}


    def manual_register_node(self, node_id: str, hostname: str = "",
                             address: str = "", network_type: str = "unknown",
                             node_type: str = "pc") -> dict:
        """
        主节点手动注册一个从节点（无需 TCP 连接）。

        用于以下场景：
        - 管理员提前在后台录入从节点信息
        - 从节点尚未来得及通过 TCP 连接
        - 保留节点槽位供后续 TCP 激活

        手动注册的节点初始状态为 offline，待从节点 TCP 连接后自动变为 online。

        Args:
            node_id: 节点标识（如 "jetson-nano-01"）
            hostname: 主机名
            address: 预留地址（可选）
            network_type: 网络类型

        Returns:
            { status, node_id, message }
        """
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可手动注册从节点"}

        if node_id == "master":
            return {"status": "invalid", "reason": "不能注册名为 'master' 的节点"}

        with self._nodes_lock:
            if node_id in self.nodes:
                existing = self.nodes[node_id]
                if existing.role == "master":
                    return {"status": "invalid", "reason": f"'{node_id}' 是主节点，不可覆盖"}
                if (existing.is_available() or existing.connected_at or existing.last_heartbeat
                        or existing.device_info or existing.model_sha256):
                    return {
                        "status": "conflict",
                        "node_id": node_id,
                        "reason": "节点已通过自动注册建立真实连接记录，请先注销/删除后再手动重建",
                        "state": existing.state.value,
                    }
                existing.hostname = hostname or existing.hostname or node_id
                existing.address = address
                existing.network_type = network_type
                existing.node_type = node_type
                state_value = existing.state.value
                hostname_snapshot = existing.hostname
            else:
                existing = None

            if existing is not None:
                pass  # 更新路径：锁内修改完成
            else:
                # 检查容量：只统计在线/已注册节点（离线/幽灵不占位）
                online_non_master = [
                    n for n in self.nodes.values()
                    if n.role != "master" and (n.is_available() or n.address)
                ]
                if len(online_non_master) >= self._max_nodes - 1:
                    return {"status": "full", "reason": f"已达到最大在册从节点数量 ({self._max_nodes - 1})"}

                # 创建节点（初始 offline）
                node = NodeInfo(
                    node_id=node_id,
                    role=NodeRole.CLIENT,
                    node_type=node_type,
                    state=NodeState.OFFLINE,
                    hostname=hostname or node_id,
                    address=address,
                    network_type=network_type,
                )
                self.nodes[node_id] = node

        if existing is not None:
            logger.info(
                f"📝 手动注册节点已更新: {node_id} type={node_type} "
                f"(hostname={hostname_snapshot}, addr={address}, state={state_value})"
            )
            return {"status": "updated", "node_id": node_id,
                    "message": f"节点 '{node_id}' 已更新 (state={state_value})",
                    "state": state_value}

        logger.info(f"📝 主节点手动注册从节点: {node_id} type={node_type} (hostname={hostname}, addr={address})")
        return {
            "status": "registered",
            "node_id": node_id,
            "message": f"节点 '{node_id}' 已手动注册，等待 TCP 连接激活",
            "state": "offline",
        }


    def delete_node(self, node_id: str) -> dict:
        """
        删除离线节点记录（不同于 deregister：deregister 仅标记 offline）。

        仅允许删除非 master 且当前不在线的节点，常用于移除手动注册的
        Android/离线占位节点。
        """
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可删除节点"}
        if node_id == "master":
            return {"status": "invalid", "reason": "不能删除主节点"}
        # Phase 2.1+: 原子化 get+检查+pop，防止并发修改
        with self._nodes_lock:
            node = self.nodes.get(node_id)
            if node is None:
                return {"status": "not_found", "reason": f"节点 '{node_id}' 不存在"}
            if node.is_available():
                return {"status": "online", "reason": "节点在线，请先注销后删除"}

            old_node = self.nodes.pop(node_id)

        self._clear_layer_config_state(node_id)
        with self._layer_config_lock:
            self._pipeline_worker_opt_out.discard(node_id)

        if self._effective_role() == "master":
            self._push_node_update_to_all_clients(node_id, "remove", old_node)

        logger.info(
            f"🗑️ 节点已删除: {node_id} type={old_node.node_type} "
            f"hostname={old_node.hostname}"
        )
        return {"status": "deleted", "node_id": node_id,
                "message": f"节点 '{node_id}' 已删除"}


    def check_master_health(self) -> dict:
        """
        检查主节点是否在线。

        活跃 TCP 连接是唯一的在线事实源；断线时只返回本机保存的
        连接目标作为重连提示，不以远端数据库心跳推断在线状态。

        Returns:
            {
                "master_online": bool,
                "last_seen_seconds_ago": float | None,
                "stale": bool,
                "master_host": str,
                "master_port": int,
                "source": str,       # "tcp" | "tcp_disconnected" | "config" | "not_connected"
                "tcp_connected": bool | None,  # 本地 TCP 是否连通
            }
        """
        # ★ 第 1 级：本地 TCP 连接状态（秒级感知断连）
        tcp_client = getattr(self, '_tcp_client', None)
        tcp_connected = (
            tcp_client is not None
            and getattr(tcp_client, '_running', False)
            and getattr(tcp_client, 'is_registered', False)
            and getattr(tcp_client, 'sock', None) is not None
        )

        # 活跃且已注册的 TCP 连接是当前、直接的在线证据。
        if self._effective_role() == "client" and tcp_connected:
            return {
                "master_online": True,
                "last_seen_seconds_ago": 0.0,
                "stale": False,
                "master_host": getattr(tcp_client, "server_host", ""),
                "master_port": getattr(tcp_client, "server_port", 0),
                "source": "tcp",
                "tcp_connected": True,
            }

        configured_host = getattr(tcp_client, "server_host", "") or ""
        configured_port = getattr(tcp_client, "server_port", 0) or 0
        if not configured_host:
            try:
                import config as cfg
                settings = self._runtime_settings if self._runtime_settings is not None else vars(cfg)
                configured_host = str(settings.get("CLIENT_MASTER_HOST", "") or "")
                configured_port = int(settings.get("CLIENT_MASTER_PORT", 0) or 0)
            except Exception:
                configured_host = ""
                configured_port = 0
        if configured_host and configured_host != "192.168.x.x":
            return {
                "master_online": False,
                "last_seen_seconds_ago": None,
                "stale": True,
                "master_host": configured_host,
                "master_port": configured_port,
                "source": "tcp_disconnected" if tcp_client is not None else "config",
                "tcp_connected": tcp_connected,
            }
        return {
            "master_online": False,
            "last_seen_seconds_ago": None,
            "stale": True,
            "master_host": "",
            "master_port": 0,
            "source": "not_connected",
            "tcp_connected": tcp_connected,
        }


    def _start_client_health_monitor(self) -> None:
        """
        启动从节点后台线程：监控主节点是否在线。

        每 15 秒检查一次本机 TCP 连接状态。
        当检测到主节点从在线变为离线时，记录告警日志。
        当主节点恢复在线时，记录恢复日志并自动重连（如已配置）。
        """
        with self._client_health_start_lock:
            if (self._client_health_thread is not None
                    and self._client_health_thread.is_alive()):
                return

            self._start_client_health_monitor_locked()


    def _start_client_health_monitor_locked(self) -> None:
        """在 _client_health_start_lock 内初始化并启动唯一健康线程。"""

        # 从当前 TCP 连接状态初始化，避免启动时误触发“恢复重连”。
        try:
            initial_health = self.check_master_health()
            initial_online = initial_health.get("master_online", False)
        except Exception:
            initial_online = False
        self._client_master_was_online = initial_online
        self._client_master_online = initial_online
        self._client_reconnect_enabled = True

        self._client_master_down_since = 0.0         # 主节点首次检测到宕机的时间戳

        # 周期性重连：当主节点在线但本地 TCP 未连接时，每隔一定时间重试
        self._client_last_reconnect_attempt = 0.0    # 上次重连尝试的时间戳

        self._client_health_thread = threading.Thread(
            target=self._client_health_monitor_loop,
            name="client-master-health",
            daemon=True,
        )
        self._client_health_thread.start()
        logger.info("从节点主节点健康监控已启动（间隔 15s，重连间隔 60s）")


    def _client_health_monitor_loop(self) -> None:
        """从节点健康监控循环（后台 daemon 线程）"""
        while self._running:
            try:
                health = self.check_master_health()
                was_online = self._client_master_was_online
                is_online = health.get("master_online", False)

                self._client_master_online = is_online

                # ---- 检测主节点宕机 ----
                if was_online and not is_online:
                    self._client_master_down_since = time.time()
                    logger.warning(
                        f"⚠️ 检测到主节点宕机！上次心跳: "
                        f"{health.get('last_seen_seconds_ago', '?')}s 前"
                    )

                # ---- 检测主节点恢复 ----
                if not was_online and is_online:
                    # 是否真正观察到过宕机：监控线程启动早于 TCP 注册完成时，
                    # 首轮循环会出现 was_online=False → is_online=True 的
                    # 假"恢复"跳变（链路其实一直健康），此时不应打恢复日志
                    observed_down = self._client_master_down_since > 0
                    # 重置宕机追踪状态
                    self._client_master_down_since = 0.0

                    if observed_down:
                        logger.info(
                            f"✅ 主节点已恢复在线 "
                            f"({health.get('master_host')}:{health.get('master_port')})"
                        )
                    # 如果已有连接配置，尝试自动重连。
                    # ★ 仅在本地 TCP 确实未连接时才重连：监控线程启动早于
                    #   TCP 注册完成时，首轮循环会出现 was_online=False →
                    #   is_online=True 的假"恢复"跳变，此时链路本来就是
                    #   健康的，不应重连、更不应打"已自动重连"日志
                    if (self._client_reconnect_enabled
                            and not health.get("tcp_connected")):
                        host = health.get("master_host", "")
                        port = health.get("master_port", 0)
                        if host and port:
                            result = self.connect_to_master(host, port, _automatic=True)
                            if result.get("status") == "connected":
                                logger.info(f"🔄 已自动重连到主节点 {host}:{port}")

                # ---- 周期性重新发现 + 重连 ----
                # TCP 已断开时 health 会立即报告 offline，不能用 is_online
                # 作为重连前置条件，否则晚启动或换地址的主节点永远不会被发现。
                if (self._client_reconnect_enabled
                        and self._effective_role() == "client"):
                    tcp_client = getattr(self, '_tcp_client', None)
                    tcp_connected = (tcp_client is not None
                                     and getattr(tcp_client, '_running', False)
                                     and getattr(tcp_client, 'is_registered', False)
                                     and getattr(tcp_client, 'sock', None) is not None)
                    if not tcp_connected:
                        now = time.time()
                        last_attempt = getattr(self, '_client_last_reconnect_attempt', 0.0)
                        if now - last_attempt >= 60:  # 每 60 秒重试一次
                            self._client_last_reconnect_attempt = now
                            discovery = self.discover_master()
                            host = discovery.get("master_host", "")
                            port = int(discovery.get("master_port", 0) or 0)
                            if host and port:
                                logger.info(
                                    f"🔄 周期性重连尝试: {host}:{port}"
                                )
                                result = self.connect_to_master(host, port, _automatic=True)
                                if result.get("status") == "connected":
                                    logger.info(f"✅ 周期性重连成功: {host}:{port}")
                                    self._client_last_reconnect_attempt = 0.0  # 成功后重置
                                elif result.get("status") not in {"reconnecting", "retry_exhausted"}:
                                    for alternate in self._discover_master_fallbacks(host, port):
                                        alt_result = self.connect_to_master(
                                            alternate["master_host"],
                                            alternate["master_port"], _automatic=True,
                                        )
                                        if alt_result.get("status") == "connected":
                                            self._client_last_reconnect_attempt = 0.0
                                            break

                self._client_master_was_online = is_online
            except Exception as e:
                logger.debug(f"健康监控循环异常: {e}")

            time.sleep(15)


    def get_client_master_status(self) -> dict:
        """
        获取从节点视角下的主节点在线状态。

        Returns:
            { master_online, last_seen_ago, health }
        """
        health = self.check_master_health()
        return {
            "master_online": health.get("master_online", False),
            "last_seen_seconds_ago": health.get("last_seen_seconds_ago"),
            "master_host": health.get("master_host", ""),
            "master_port": health.get("master_port", 0),
            "stale": health.get("stale", True),
            "health": health,
        }
