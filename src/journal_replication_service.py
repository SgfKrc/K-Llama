"""Minimal production wiring for durable journal replication (R-R11).

**为什么只需要"接线"**：复制契约（`src/journal_replication.py`，`JournalCheckpoint` +
`DurableJournalReplica`）、认证 TCP 帧（`src/journal_replication_transport.py`）与事件源
（`src/recovery_journal.py` 的 `workflow_events`）**三者都已存在**，而且 `TaskJournal` 的
`get_snapshot` / `list_events` **恰好就是** `ReplicationJournal` 契约要求的那两个方法 ——
`build_checkpoint()` 可以直接吃 `SQLiteTaskJournal`，不需要适配器。缺的只是**生产侧消费者**
（此前唯一消费者是 `tests/test_journal_replication.py`）。

**本模块做的**：
* 全局开关 `QLH_JOURNAL_REPLICATION_ENABLED`，**默认关**（关时任何调用都 fail-closed）；
* leader 侧 ``replicate_journal_once()``：`build_checkpoint()` + `send_checkpoint()`；
* follower 侧 ``JournalReplicaReceiver``：`DurableJournalReplica` + `JournalReplicationTcpServer`。

**本模块刻意不做的**（留给后续，别把它当已完成）：
* 不自动挂 `lifespan`、不随 `NODE_ROLE` 自动启停、不参与 leader 选举 —— 那些属 P4.5 的 HA
  生产启用（`NODE_ROLE=auto` 至今默认关，`production_availability_claim` 仍为 `false`）；
* 因此这里只提供**显式可调用**的入口，由 `src/api/routes_tasks.py` 的端点驱动。

**安全**：HMAC 只做认证，**不加密**；默认 bind loopback。生产部署需要受保护的传输与密钥托管
（见 `journal_replication_transport.py` 的模块 docstring）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from journal_replication import (
    DurableJournalReplica,
    JournalCheckpoint,
    JournalReplicationError,
    build_checkpoint,
)
from journal_replication_transport import (
    JournalReplicationTcpServer,
    JournalReplicationTransportError,
    request_checkpoint,
    send_checkpoint,
)

#: HMAC 密钥下限，与 `journal_replication_transport` 内部的判据一致。
MIN_SECRET_BYTES = 32


class JournalReplicationDisabled(RuntimeError):
    """`QLH_JOURNAL_REPLICATION_ENABLED` 未开启时使用复制路径，一律 fail-closed。"""


def _env_flag(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_text(env: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = env.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _env_port(env: Mapping[str, str], name: str, default: int = 0) -> int:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    # 0 合法 —— 表示"由系统分配端口"（测试与 loopback 场景用）。
    if not (0 <= value <= 65535):
        return default
    return value


def _env_timeout(env: Mapping[str, str], name: str, default: float = 5.0) -> float:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if not (0.1 <= value <= 600.0):
        return default
    return value


def _env_int(
    env: Mapping[str, str],
    name: str,
    default: int = 0,
    *,
    minimum: int = 0,
    maximum: int = 4096,
) -> int:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if not (minimum <= value <= maximum):
        return default
    return value


def _parse_quorum_peers(value: str) -> tuple[tuple[str, int], ...]:
    """解析 ``host:port,host:port``；**非法项直接丢弃**（绝不猜端口、绝不回退到默认端口）。"""

    peers: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for item in str(value or "").split(","):
        text = item.strip()
        if not text:
            continue
        host, _, port_text = text.rpartition(":")
        host = host.strip()
        try:
            port = int(port_text)
        except (TypeError, ValueError):
            continue
        if not host or not (0 < port <= 65535):
            continue
        entry = (host, port)
        if entry in seen:
            continue
        seen.add(entry)
        peers.append(entry)
    return tuple(peers)


@dataclass(frozen=True)
class JournalReplicationSettings:
    """复制链路的运行参数。`enabled` 默认 **False** —— 不显式开启就什么都不做。"""

    enabled: bool = False
    source_node_id: str = ""
    stream_id: str = "task-graph-journal"
    replica_path: str = ""
    bind_host: str = "127.0.0.1"
    bind_port: int = 0
    peer_host: str = ""
    peer_port: int = 0
    timeout_s: float = 5.0
    secret: bytes = b""
    #: #93：**回包前栅栏**的多数耐久目标。空 = 不做多数判定（退回单 peer 语义，行为不变）。
    quorum_peers: tuple[tuple[str, int], ...] = ()
    quorum_size: int = 0

    @property
    def secret_bytes(self) -> bytes:
        return bytes(self.secret)

    @property
    def required_quorum(self) -> int:
        """栅栏阈值：未显式指定时按**多数**取（``len(peers)//2 + 1``）；无对端 ⇒ 0。"""
        if not self.quorum_peers:
            return 0
        if self.quorum_size > 0:
            return min(self.quorum_size, len(self.quorum_peers))
        return len(self.quorum_peers) // 2 + 1

    def validate(self) -> None:
        """配置自检。**只在开启时才严格要求** —— 关着的时候允许缺项。"""

        if not self.enabled:
            return
        if not self.source_node_id:
            raise JournalReplicationError(
                "QLH_JOURNAL_REPLICATION_SOURCE_NODE_ID (or QLH_NODE_ID) is required"
            )
        if not self.stream_id:
            raise JournalReplicationError("QLH_JOURNAL_REPLICATION_STREAM_ID is required")
        if not self.replica_path:
            raise JournalReplicationError("QLH_JOURNAL_REPLICATION_REPLICA_PATH is required")
        if len(self.secret_bytes) < MIN_SECRET_BYTES:
            raise JournalReplicationError(
                f"QLH_JOURNAL_REPLICATION_SECRET must be at least {MIN_SECRET_BYTES} bytes"
            )
        if self.quorum_size > len(self.quorum_peers):
            raise JournalReplicationError(
                "QLH_JOURNAL_REPLICATION_QUORUM_SIZE cannot exceed the configured quorum peers"
            )

    def require_enabled(self) -> None:
        if not self.enabled:
            raise JournalReplicationDisabled(
                "journal replication is disabled; set QLH_JOURNAL_REPLICATION_ENABLED=1"
            )
        self.validate()

    def require_peer(self) -> None:
        """leader 侧推送需要显式对端。"""

        self.require_enabled()
        if not self.peer_host or self.peer_port <= 0:
            raise JournalReplicationError(
                "QLH_JOURNAL_REPLICATION_PEER_HOST / _PEER_PORT are required to push a checkpoint"
            )

    def require_quorum_peers(self) -> None:
        """#93：回包前栅栏需要**多个** replica 目标，否则「多数」没有意义。"""

        self.require_enabled()
        if not self.quorum_peers:
            raise JournalReplicationError(
                "QLH_JOURNAL_REPLICATION_QUORUM_PEERS is required for the quorum barrier"
            )

    def public(self) -> dict[str, Any]:
        """对外可见的状态 —— **绝不回显 secret**。"""

        return {
            "enabled": self.enabled,
            "source_node_id": self.source_node_id,
            "stream_id": self.stream_id,
            "replica_path": self.replica_path,
            "bind": {"host": self.bind_host, "port": self.bind_port},
            "peer": {"host": self.peer_host, "port": self.peer_port},
            "timeout_s": self.timeout_s,
            "secret_present": len(self.secret_bytes) >= MIN_SECRET_BYTES,
            "quorum": {
                "peers": [f"{host}:{port}" for host, port in self.quorum_peers],
                "size": self.quorum_size,
                "required": self.required_quorum,
            },
        }

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "JournalReplicationSettings":
        """从环境变量构造。

        刻意**不经过 `src/config.py`**：config 的常量在 import 时求值，测试里改 env 不会生效。
        本函数每次调用都读当前环境，既保持单一 env 命名空间，又可注入/可测。
        """

        env: Mapping[str, str] = os.environ if environ is None else environ
        secret_text = _env_text(env, "QLH_JOURNAL_REPLICATION_SECRET")
        return cls(
            enabled=_env_flag(env, "QLH_JOURNAL_REPLICATION_ENABLED", False),
            source_node_id=_env_text(
                env, "QLH_JOURNAL_REPLICATION_SOURCE_NODE_ID", "QLH_NODE_ID"
            ),
            stream_id=_env_text(
                env, "QLH_JOURNAL_REPLICATION_STREAM_ID", default="task-graph-journal"
            ),
            replica_path=_env_text(env, "QLH_JOURNAL_REPLICATION_REPLICA_PATH"),
            bind_host=_env_text(
                env, "QLH_JOURNAL_REPLICATION_BIND_HOST", default="127.0.0.1"
            ),
            bind_port=_env_port(env, "QLH_JOURNAL_REPLICATION_BIND_PORT", 0),
            peer_host=_env_text(env, "QLH_JOURNAL_REPLICATION_PEER_HOST"),
            peer_port=_env_port(env, "QLH_JOURNAL_REPLICATION_PEER_PORT", 0),
            timeout_s=_env_timeout(env, "QLH_JOURNAL_REPLICATION_TIMEOUT_S", 5.0),
            secret=secret_text.encode("utf-8"),
            quorum_peers=_parse_quorum_peers(
                _env_text(env, "QLH_JOURNAL_REPLICATION_QUORUM_PEERS")
            ),
            quorum_size=_env_int(env, "QLH_JOURNAL_REPLICATION_QUORUM_SIZE", 0),
        )


def replicate_journal_once(
    journal: Any,
    *,
    workflow_id: str,
    settings: JournalReplicationSettings | None = None,
) -> dict[str, Any]:
    """leader 侧：把一个 workflow 的 checkpoint 推到对端 replica 服务。

    ``journal`` 是鸭子类型 —— 只要有 ``get_snapshot()`` / ``list_events()`` 即可
    （``SQLiteTaskJournal`` 本来就满足 ``ReplicationJournal`` 契约，无需适配器）。
    失败一律抛出，由调用方决定 HTTP 语义；**不吞异常、不降级**。
    """

    resolved = settings or JournalReplicationSettings.from_env()
    resolved.require_peer()
    if not isinstance(workflow_id, str) or not workflow_id:
        raise JournalReplicationError("workflow_id is required")

    checkpoint: JournalCheckpoint = build_checkpoint(
        journal,
        source_node_id=resolved.source_node_id,
        stream_id=resolved.stream_id,
        workflow_id=workflow_id,
    )
    result = send_checkpoint(
        resolved.peer_host,
        resolved.peer_port,
        checkpoint,
        secret=resolved.secret_bytes,
        timeout=resolved.timeout_s,
    )
    return {
        "status": "replicated",
        "workflow_id": workflow_id,
        "source_node_id": resolved.source_node_id,
        "stream_id": resolved.stream_id,
        "durable_sequence": result.get("durable_sequence"),
        "checkpoint_digest": result.get("checkpoint_digest"),
        "event_count": len(checkpoint.records),
    }


def replicate_journal_to_quorum(
    journal: Any,
    *,
    workflow_id: str,
    settings: JournalReplicationSettings | None = None,
) -> dict[str, Any]:
    """#93「回包前栅栏」：把**同一个** checkpoint 推给多个 replica，达到多数耐久才回 OK。

    与 `replicate_journal_once` 的差别就是**不**在第一个 ACK 上返回：

    - 只做一次 `build_checkpoint` ⇒ 所有目标看到的是**同一个**快照与摘要（不会各拿一份）；
    - 单个目标失败**不抛**（要能继续收集多数），失败原因逐个记进 `failures`；
    - 返回 `accepted`（是否达到 `required_quorum`）与 `durable_acks`，HTTP 语义交给调用方；
      调用方在 `accepted=False` 时**必须**回错误，否则这道栅栏等于不存在。

    `send_checkpoint` 自身已校验对端回显的 `durable_sequence` / `checkpoint_digest`
    ⇒ 这里的一个 ACK 就等于「对端**耐久**持有到该点」。
    """

    resolved = settings or JournalReplicationSettings.from_env()
    resolved.require_quorum_peers()
    if not isinstance(workflow_id, str) or not workflow_id:
        raise JournalReplicationError("workflow_id is required")

    checkpoint: JournalCheckpoint = build_checkpoint(
        journal,
        source_node_id=resolved.source_node_id,
        stream_id=resolved.stream_id,
        workflow_id=workflow_id,
    )
    required = resolved.required_quorum
    durable: list[str] = []
    failures: dict[str, str] = {}
    for host, port in resolved.quorum_peers:
        label = f"{host}:{port}"
        try:
            send_checkpoint(
                host,
                port,
                checkpoint,
                secret=resolved.secret_bytes,
                timeout=resolved.timeout_s,
            )
        except (JournalReplicationTransportError, OSError) as exc:
            failures[label] = type(exc).__name__
            continue
        durable.append(label)
    accepted = len(durable) >= required
    return {
        "status": "replicated" if accepted else "quorum_not_reached",
        "accepted": accepted,
        "workflow_id": workflow_id,
        "source_node_id": resolved.source_node_id,
        "stream_id": resolved.stream_id,
        "durable_sequence": checkpoint.last_sequence,
        "checkpoint_digest": checkpoint.checkpoint_digest,
        "event_count": len(checkpoint.records),
        "durable_acks": len(durable),
        "required_quorum": required,
        "durable_peers": durable,
        "failures": failures,
    }


def recover_journal_prefix(
    replica: Any,
    *,
    workflow_id: str,
    settings: JournalReplicationSettings | None = None,
    mutation_guard=None,
    permission_check=None,
) -> dict[str, Any]:
    """#93：让**落后的副本主动追赶** —— 从对端拉回一个已自洽的 checkpoint 并应用。

    与栅栏（推）严格对偶：单个目标失败**不抛**（要能继续试别的目标），但**一个都没成功**
    时返回 `recovered=False` ⇒ 调用方必须继续拒绝接管，绝不允许「凑合上线」。
    """

    resolved = settings or JournalReplicationSettings.from_env()
    resolved.require_quorum_peers()
    if not isinstance(workflow_id, str) or not workflow_id:
        raise JournalReplicationError("workflow_id is required")

    applied: list[str] = []
    failures: dict[str, str] = {}
    durable_sequence: int | None = None
    for host, port in resolved.quorum_peers:
        label = f"{host}:{port}"
        try:
            checkpoint = request_checkpoint(
                host,
                port,
                source_node_id=resolved.source_node_id,
                stream_id=resolved.stream_id,
                workflow_id=workflow_id,
                secret=resolved.secret_bytes,
                timeout=resolved.timeout_s,
            )
            guarded = {"permission_check": permission_check} if permission_check is not None else {}
            apply = lambda: replica.apply_checkpoint(checkpoint, **guarded)
            result = mutation_guard(apply) if mutation_guard is not None else apply()
        except (JournalReplicationTransportError, JournalReplicationError, OSError) as exc:
            failures[label] = type(exc).__name__
            continue
        applied.append(label)
        durable_sequence = result.get("durable_sequence")
    recovered = bool(applied)
    return {
        "status": "recovered" if recovered else "unrecovered",
        "recovered": recovered,
        "workflow_id": workflow_id,
        "source_node_id": resolved.source_node_id,
        "stream_id": resolved.stream_id,
        "durable_sequence": durable_sequence,
        "applied_peers": applied,
        "failures": failures,
    }


class JournalReplicaReceiver:
    """follower 侧接收端：``DurableJournalReplica`` + 认证 TCP 服务。

    这是"能收"的那一半。**不自动常驻** —— 由调用方显式 ``start()`` / ``close()``。
    """

    def __init__(
        self,
        settings: JournalReplicationSettings | None = None,
        *,
        replica_path: str | None = None,
    ) -> None:
        resolved = settings or JournalReplicationSettings.from_env()
        resolved.require_enabled()
        path = replica_path or resolved.replica_path
        if not path:
            raise JournalReplicationError("a replica path is required")
        self.settings = resolved
        self.replica = DurableJournalReplica(
            path,
            source_node_id=resolved.source_node_id,
            stream_id=resolved.stream_id,
        )
        self.server = JournalReplicationTcpServer(
            self.replica,
            bind_host=resolved.bind_host,
            port=resolved.bind_port,
            secret=resolved.secret_bytes,
            timeout=resolved.timeout_s,
        )
        self.address: tuple[str, int] | None = None

    def start(self) -> tuple[str, int]:
        self.address = self.server.start()
        return self.address

    def close(self) -> None:
        try:
            self.server.close()
        finally:
            self.replica.close()

    def status(self, workflow_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "address": list(self.address) if self.address else None,
            "settings": self.settings.public(),
        }
        if workflow_id:
            payload["recovery_status"] = self.replica.recovery_status(workflow_id)
        return payload


__all__ = [
    "MIN_SECRET_BYTES",
    "JournalReplicaReceiver",
    "JournalReplicationDisabled",
    "JournalReplicationSettings",
    "JournalReplicationTransportError",
    "replicate_journal_once",
    "replicate_journal_to_quorum",
    "recover_journal_prefix",
]
