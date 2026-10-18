"""Stable scheduler value objects shared by its implementation mixins."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class NodeState(str, Enum):
    """节点状态枚举。

    ★ 2026-10-05（DIST-2 收拢现状）：本枚举**实际只有两个值在用** ——
    `ONLINE` 与 `OFFLINE`；原先还有 `BUSY`（推理中）与 `ERROR`（异常），但
    **从未有任何代码把节点置成它们**（全仓仅 `scheduler.py` 有一处
    `state == NodeState.BUSY` 的比较，该分支恒假，连同"任务停止后恢复节点空闲"
    那段一起移除）。保留死枚举会让「可用性判据」看起来比实际复杂。

    **节点是否可用由两维系：**

    1. **本枚举**：`ONLINE` ⇔ `is_available()` 为真。置位点集中在
       `scheduler_cluster`（注册 / presence 心跳 / 远端节点表）与
       `scheduler._on_tcp_disconnect`（断连置 `OFFLINE`）。
    2. **心跳新鲜度**：`NodeInfo.is_heartbeat_fresh()`（阈值
       `WORKER_HEARTBEAT_MAX_AGE`）。TCP 半开时第 1 维要等巡检约 129s 才翻转，
       第 2 维才是在那段窗口里阻止"静默节点仍占容量"的判据。

    规划文档里提到的 `registered / admitted / ready / leased / draining` 那套状态
    **尚未进本枚举**，其语义目前分散在三个对象上：task-worker 的
    `accepted` / `selected_version`（`task_worker_adapter`）、`Reservation` /
    `lease_epoch`（`execution_provider`）。
    """
    ONLINE = "online"       # 在线（对端可达；是否真能派活另见心跳新鲜度）
    OFFLINE = "offline"     # 离线/断连


class NodeRole(str, Enum):
    """节点角色（字符串枚举，便于比较）"""
    MASTER = "master"
    CLIENT = "client"


#: 从节点心跳的**容忍上限**（秒）：超过它，节点被视为「不可用，不该再参与流水线
#: 就绪判定与容量规划」。取 45s 的 2 倍余量：App 侧
#: `AndroidPresenceStateMachine.heartbeatIntervalMs = 45_000`（可配 5–120s，见
#: `heartbeatIntervalSeconds.coerceIn(5, 120)`），而这里原先写死 10s ⇒ 45s 的间隔
#: 必然被判过期（实测 `11.4s > 10s`）。Route A 要求 Android 参与 readiness 后才暴露。
#:
#: ★ 2026-10-05（DIST-2「唯一超时来源」）：本常量由 `scheduler_pipeline` 上移到此处，
#: 因为**容量规划**（`scheduler._get_pipeline_capacity_nodes`）与 **readiness**
#: （`scheduler_pipeline._get_pipeline_readiness`）现在共用同一个阈值 —— 此前容量
#: 规划只看 `NodeInfo.is_available()`（`state == ONLINE`），完全不看心跳新鲜度，而
#: TCP 半开要等巡检约 129s 才置 OFFLINE ⇒ 静默节点在这段时间里一直占容量。
WORKER_HEARTBEAT_MAX_AGE = 120.0


#: task-worker **控制面** health 超时的**下界**（秒）。实际取值是
#: `max(这个下界, HEARTBEAT_INTERVAL * 4)`（见 `scheduler` 里构造
#: `TaskWorkerControlPlane` 处）—— 乘 4 是为了容忍若干次心跳抖动，下界则保证
#: 即使把 `HEARTBEAT_INTERVAL` 调到很小也不会把健康判定收得过紧。
#:
#: 为什么不直接用 `WORKER_HEARTBEAT_MAX_AGE`（120s）：**两者语义不同、层次不同**。
#: 控制面 health 决定「这个 provider 还要不要派 stage」（`healthy` /
#: `layer_stage_dispatch_enabled` / stage offer 准入），流水线层的
#: `WORKER_HEARTBEAT_MAX_AGE` 决定「这个节点还算不算数」（readiness 判定与容量
#: 规划）。**有意让前者更短**：先停止派活，再让节点退出规划，两层不同时翻转；
#: 若把两者并成同一个数，节点会在「仍被规划进层区间」的同时「已不再接受 stage」，
#: 边界反而更难推理。
TASK_WORKER_HEALTH_TIMEOUT_FLOOR_SECONDS = 30.0


#: Android **task-worker** 控制面心跳间隔（秒）。设备侧
#: `TaskWorkerClient.HEARTBEAT_INTERVAL_MS` 是唯一发送方，必须与它同值。
#: ★ 2026-10-07（DIST-NEXT-4）：此前这个数字只存在于设备侧硬编码里，主仓没有任何
#: 地方记录它 ⇒ 「设备心跳间隔 vs 主仓容忍上限」这类错配只能靠真机实测发现
#: （历史实测：presence 心跳 45s 而容忍上限写死 10s，`11.4s > 10s` 被判过期）。
ANDROID_TASK_WORKER_HEARTBEAT_INTERVAL_SECONDS = 15.0

#: Android presence（HTTP 心跳）间隔（秒）。`scheduler` 的
#: `ANDROID_HTTP_CLIENT_HEARTBEAT_INTERVAL_SECONDS` 由此派生，避免两处各写一份。
ANDROID_PRESENCE_HEARTBEAT_INTERVAL_SECONDS = 45.0


def worker_heartbeat_max_age_seconds() -> float:
    """worker 心跳容忍上限（秒）—— **容量规划与 readiness 的唯一来源**。

    返回 `WORKER_HEARTBEAT_MAX_AGE`（现值 120s）。该值必须同时满足：

    * ≥ 2 × Android task-worker 心跳（15s ⇒ 30s）；
    * ≥ 2 × Android presence 心跳（45s ⇒ 90s）；
    * < TCP 半开巡检上限（`tcp_comm.TCPServer.MAX_HEARTBEAT_MISSED` 派生）——
      让心跳判据**先**于 TCP 巡检翻转，静默节点不会在「仍占容量」的窗口里被派活。
    """
    return WORKER_HEARTBEAT_MAX_AGE


def task_worker_control_plane_health_timeout_seconds(
    heartbeat_interval_seconds: float,
) -> float:
    """task-worker 控制面 health 超时（秒）= `max(下界, 心跳间隔 × 4)`。

    故意**短于** [worker_heartbeat_max_age_seconds]：先停止派 stage，再让节点退出
    规划，两层不同时翻转（若并成同一个数，节点会在「仍被规划进层区间」的同时
    「已不再接受 stage」）。
    """
    return max(
        TASK_WORKER_HEALTH_TIMEOUT_FLOOR_SECONDS,
        float(heartbeat_interval_seconds) * 4.0,
    )


def worker_liveness_thresholds(
    heartbeat_interval_seconds: float,
) -> dict[str, float]:
    """一次给出全部相关阈值（诊断与回归测试用，便于打印/对比）。"""
    return {
        "pc_heartbeat_interval_seconds": float(heartbeat_interval_seconds),
        "android_task_worker_heartbeat_interval_seconds": (
            ANDROID_TASK_WORKER_HEARTBEAT_INTERVAL_SECONDS
        ),
        "android_presence_heartbeat_interval_seconds": (
            ANDROID_PRESENCE_HEARTBEAT_INTERVAL_SECONDS
        ),
        "control_plane_health_timeout_seconds": (
            task_worker_control_plane_health_timeout_seconds(
                heartbeat_interval_seconds,
            )
        ),
        "worker_heartbeat_max_age_seconds": worker_heartbeat_max_age_seconds(),
    }


def assert_worker_liveness_thresholds(heartbeat_interval_seconds: float) -> None:
    """启动期自检：各阈值必须互相自洽（错配直接 fail-loud）。

    把「心跳间隔 → 容忍上限」的关系固化成断言，下一次改阈值时立即暴露错配，
    而不是等真机实测出现「节点被判过期 / 静默占容量」。
    """
    thresholds = worker_liveness_thresholds(heartbeat_interval_seconds)
    max_age = thresholds["worker_heartbeat_max_age_seconds"]
    for key in (
        "android_task_worker_heartbeat_interval_seconds",
        "android_presence_heartbeat_interval_seconds",
    ):
        interval = thresholds[key]
        if max_age < interval * 2:
            raise ValueError(
                f"WORKER_HEARTBEAT_MAX_AGE={max_age}s must cover at least two "
                f"{key} ({interval}s)"
            )
    if thresholds["control_plane_health_timeout_seconds"] > max_age:
        raise ValueError(
            "task-worker control-plane health timeout must not exceed the worker "
            "heartbeat max age: stop dispatching before evicting the node"
        )


@dataclass
class NodeInfo:
    """节点信息（分布式模式下通过 TCP 注册填充）"""
    node_id: str
    role: str                      # 节点角色: "master" | "client"
    node_type: str = "pc"          # 设备平台: "pc" | "android"
    state: NodeState = NodeState.OFFLINE
    address: str = ""              # "ip:port" 字符串
    hostname: str = ""             # 客户端主机名
    device_info: dict = field(default_factory=dict)  # 客户端设备信息
    network_type: str = "unknown"  # 网络连接类型: wifi | ethernet | unknown
    connected_at: float = 0.0      # 连接/注册时间
    last_heartbeat: float = 0.0    # 上次心跳时间
    avg_rtt_ms: float = 0.0        # 滑动平均 RTT（指数加权，仅从节点有效）
    last_rtt_ms: float = 0.0       # 最近一次 RTT
    task_count: int = 0            # 已完成任务数
    error_count: int = 0           # 错误计数
    model_sha256: str = ""         # 模型 SHA256 校验值（阶段 7）
    presence_generation: int = 0
    presence_lease_id: str = ""
    presence_expires_at: float = 0.0

    def is_available(self) -> bool:
        return self.state == NodeState.ONLINE

    def heartbeat_age(self, now: Optional[float] = None) -> Optional[float]:
        """距上次心跳的秒数；从未有过任何时间基准时返回 `None`。

        基准优先取 `last_heartbeat`，未收到过心跳时回退到 `connected_at`。
        """
        base = self.last_heartbeat or self.connected_at
        if not base:
            return None
        return max(0.0, (time.time() if now is None else now) - base)

    def is_heartbeat_fresh(self, now: float, max_age: float) -> bool:
        """心跳是否仍在容忍窗口内。

        ★ 2026-10-05（DIST-2）：把「节点可用」从单一的 `NodeState` 扩到
        「状态 + 心跳新鲜度」。此前容量规划只看 `is_available()`
        （`state == ONLINE`），而 TCP 半开（对端进程已死、不发 FIN）要等巡检约
        129s 才置 `OFFLINE` ⇒ 静默节点在这段时间里**一直占容量**，会分配出错的
        层区间（实测：Y700 被带走后 master 仍把它算进规划，给出既非声明区间、
        层数也不对的结果）。

        `max_age` 由调用方注入（复用既有常量，不另造一套数字）。

        **时间基准完全未知时视为新鲜**：那是「未知」而不是「已过期」。真实路径
        不会出现这种情况（`register_node` 与 Android 注册都会立即写入
        `connected_at` 与 `last_heartbeat`），但测试与嵌入场景构造的节点可能两个
        字段都没填，不应因此被误判掉线。
        """
        age = self.heartbeat_age(now)
        return age is None or age <= max_age

    def to_dict(self) -> dict:
        """转为可序列化的字典"""
        return {
            "node_id": self.node_id,
            "role": self.role,
            "node_type": self.node_type,
            "state": self.state.value,
            "address": self.address,
            "hostname": self.hostname,
            "device_info": self.device_info,
            "network_type": self.network_type,
            "connected_at": self.connected_at,
            "last_heartbeat": self.last_heartbeat,
            "avg_rtt_ms": round(self.avg_rtt_ms, 1),
            "last_rtt_ms": round(self.last_rtt_ms, 1),
            "task_count": self.task_count,
            "error_count": self.error_count,
            "model_sha256": self.model_sha256,
            "presence_generation": self.presence_generation,
            "presence_expires_at": self.presence_expires_at,
            "is_available": self.is_available(),
        }


@dataclass
class InferenceTask:
    """单个推理任务"""
    task_id: str
    prompt: str
    state: str = "pending"         # pending | running | done | error
    start_time: float = 0.0
    end_time: float = 0.0
    result: Optional[str] = None
    error_msg: Optional[str] = None
    metrics: dict = field(default_factory=dict)  # 性能指标
