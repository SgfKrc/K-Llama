"""
网络通信模块 — TCP 主从通信、长连接、粘包处理、张量序列化
==========================================================
功能职责:
1. 主节点（服务端）监听、从节点（客户端）连接
2. 长连接维持、心跳检测、断线重连
3. 封包/解包（长度头 + 数据体）
4. 张量序列化 / 反序列化
5. 控制指令、特征张量收发

协议格式:
  数据包: [4字节长度头(大端序)] + [数据体(JSON字符串 或 二进制张量)]
  控制数据: JSON 字符串（指令、心跳、状态）
  推理数据: torch.save 序列化的二进制张量

依赖: socket, threading, json, torch
"""

from __future__ import annotations

import base64
from collections import OrderedDict
import json
import logging
import socket
import struct
import threading
import time
import hashlib
import hmac
import io
import ipaddress
import math
import os
import select
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable

from cluster_fence import ControlFence, ControlFenceError

try:  # torch is optional: the edge runtime omits it; tensor APIs then fail closed
    import torch
except ImportError:  # pragma: no cover - exercised by the edge venv
    torch = None  # type: ignore[assignment]


def _require_torch():
    """Return the torch module or fail closed (edge runtime has no torch)."""
    if torch is None:
        raise ConnectionError("tensor transport requires torch; the edge runtime omits it")
    return torch


from config import (
    SERVER_IP, SERVER_PORT, HEARTBEAT_INTERVAL,
    RECONNECT_MAX_RETRIES, RECONNECT_DELAY,
    AUTH_TIMESTAMP_WINDOW,
    PIPELINE_STEP_TIMEOUT,
)

try:
    from .network_path import (
        HeartbeatQualityWindow,
        DEFAULT_MAX_HEARTBEAT_RTT_MS,
    )
except ImportError:  # Script/PYTHONPATH=src compatibility.
    from network_path import (
        HeartbeatQualityWindow,
        DEFAULT_MAX_HEARTBEAT_RTT_MS,
    )

# 尝试导入 psutil 用于网络类型检测
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

logger = logging.getLogger(__name__)

_WINDOWS_NO_WINDOW = (
    getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
)


class _RegistrationRejected(Exception):
    """客户端注册被拒绝，调用方应关闭连接且不分发给上层。"""


# ================================================================
# 数据包格式常量
# ================================================================
HEADER_LEN = 4          # 长度头字节数（大端序 uint32）
MAX_PACKET_SIZE = 256 * 1024 * 1024  # 最大包大小 256MB（特征张量可能较大）
_CLIENT_SOCKET_TIMEOUT_SECONDS = HEARTBEAT_INTERVAL + 5
_TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")


def _protected_cluster_peer(host: str) -> bool:
    try:
        address = ipaddress.ip_address(str(host).split("%", 1)[0])
    except ValueError:
        return False
    return bool(
        address.is_loopback
        or address in _TAILNET_V4
        or address in _TAILNET_V6
    )


# ================================================================
# HMAC 集群认证（阶段 7 — 跨节点安全）
# ================================================================

def _get_cluster_secret() -> str:
    try:
        import config as cfg
        return getattr(cfg, "CLUSTER_SECRET", "") or os.environ.get("QLH_CLUSTER_SECRET", "")
    except Exception:
        return os.environ.get("QLH_CLUSTER_SECRET", "")


def _get_cluster_secret_epoch() -> int:
    try:
        import config as cfg
        value = getattr(
            cfg,
            "CLUSTER_SECRET_EPOCH",
            os.environ.get("QLH_CLUSTER_SECRET_EPOCH", "1"),
        )
    except Exception:
        value = os.environ.get("QLH_CLUSTER_SECRET_EPOCH", "1")
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


class RegistrationReplayGuard:
    """Bounded nonce fence shared by every registration on one server."""

    def __init__(self, *, max_entries: int = 4096) -> None:
        self.max_entries = max(128, int(max_entries))
        self._lock = threading.Lock()
        self._seen: OrderedDict[tuple[int, str], float] = OrderedDict()

    def consume(self, *, nonce: str, epoch: int, expires_at: float) -> bool:
        now = time.time()
        key = (int(epoch), str(nonce))
        with self._lock:
            while self._seen:
                oldest, expiry = next(iter(self._seen.items()))
                if expiry >= now and len(self._seen) < self.max_entries:
                    break
                self._seen.pop(oldest, None)
            if key in self._seen:
                return False
            self._seen[key] = float(expires_at)
            while len(self._seen) > self.max_entries:
                self._seen.popitem(last=False)
            return True


def build_auth_signature(
    node_id: str,
    timestamp: float = None,
    *,
    nonce: str | None = None,
    secret_epoch: int | None = None,
) -> dict:
    """
    为注册消息构建 HMAC 认证签名。

    签名算法: HMAC-SHA256(cluster_secret, "{node_id}:{timestamp}")
    防重放: timestamp 用于服务端时间窗口校验

    Returns:
        {"auth_timestamp": float, "auth_signature": str}
    """
    ts = time.time() if timestamp is None else float(timestamp)
    auth_nonce = nonce or base64.urlsafe_b64encode(os.urandom(18)).decode("ascii").rstrip("=")
    epoch = _get_cluster_secret_epoch() if secret_epoch is None else int(secret_epoch)
    message = f"v2\n{node_id}\n{ts:.6f}\n{auth_nonce}\n{epoch}".encode("utf-8")
    sig = hmac.new(
        _get_cluster_secret().encode("utf-8"), message, hashlib.sha256
    ).hexdigest()
    return {
        "auth_version": 2,
        "auth_timestamp": ts,
        "auth_nonce": auth_nonce,
        "auth_secret_epoch": epoch,
        "auth_signature": sig,
    }


def verify_auth_signature(
    node_id: str,
    auth_data: dict,
    *,
    replay_guard: RegistrationReplayGuard | None = None,
) -> tuple:
    """
    验证 HMAC 认证签名。

    检查:
    1. 时间戳在允许窗口内（防重放攻击）
    2. HMAC-SHA256 签名匹配

    Args:
        node_id: 注册节点 ID
        auth_data: {"auth_timestamp": float, "auth_signature": str}

    Returns:
        (ok: bool, reason: str)
    """
    if not auth_data:
        return False, "缺少认证签名（auth_timestamp/auth_signature）"

    if not isinstance(auth_data, dict):
        return False, "认证字段格式无效"
    ts = auth_data.get("auth_timestamp")
    nonce = auth_data.get("auth_nonce", "")
    epoch = auth_data.get("auth_secret_epoch")
    sig = auth_data.get("auth_signature", "")

    if ts is None:
        return False, "缺少时间戳（auth_timestamp）"
    if not sig:
        return False, "缺少签名（auth_signature）"
    if auth_data.get("auth_version") != 2:
        return False, "认证协议版本不受支持"
    if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(float(ts)):
        return False, "认证时间戳必须是有限数值"
    if not isinstance(nonce, str):
        return False, "认证 nonce 无效"
    try:
        nonce_raw = base64.urlsafe_b64decode(nonce + ("=" * (-len(nonce) % 4)))
    except (ValueError, TypeError):
        return False, "认证 nonce 无效"
    if len(nonce_raw) != 18:
        return False, "认证 nonce 无效"
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        return False, "认证密钥代次无效"
    current_epoch = _get_cluster_secret_epoch()
    if epoch != current_epoch:
        return False, "认证凭据已过期（cluster secret epoch mismatch）"

    secret = _get_cluster_secret()
    if not secret:
        return False, "主节点集群密钥未初始化，请先完成首次连接自动部署"

    # 1. 时间窗口校验（±5 分钟）
    now = time.time()
    drift = abs(now - ts)
    if drift > AUTH_TIMESTAMP_WINDOW:
        return False, (
            f"时间戳偏差过大（{drift:.0f}s > {AUTH_TIMESTAMP_WINDOW}s），"
            f"请检查系统时钟同步"
        )

    # 2. HMAC 签名校验
    message = f"v2\n{node_id}\n{float(ts):.6f}\n{nonce}\n{epoch}".encode("utf-8")
    expected = hmac.new(
        secret.encode("utf-8"), message, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, sig):
        return False, "HMAC 签名不匹配 — 集群密钥不一致"

    if replay_guard is not None and not replay_guard.consume(
        nonce=nonce,
        epoch=epoch,
        expires_at=float(ts) + AUTH_TIMESTAMP_WINDOW,
    ):
        return False, "认证 nonce 已被消费（replay rejected）"

    return True, "ok"


class MessageType(str, Enum):
    """消息类型枚举"""
    # 控制指令
    REGISTER = "register"           # 从节点注册
    HEARTBEAT = "heartbeat"         # 心跳
    HEARTBEAT_ACK = "heartbeat_ack" # 心跳应答
    TASK_START = "task_start"       # 推理任务开始
    TASK_STOP = "task_stop"         # 推理任务停止
    TASK_DONE = "task_done"         # 推理任务完成
    ERROR = "error"                 # 错误上报
    # 推理数据
    TENSOR = "tensor"               # 中间特征张量
    RESULT = "result"               # 最终推理结果（文本）
    # 状态
    STATUS_REQ = "status_req"       # 状态查询
    STATUS_RES = "status_res"       # 状态应答
    # 分布式推理调度
    INFER_FORWARD = "infer_forward"  # 从节点 → 主节点：转发推理请求
    INFER_RESULT = "infer_result"    # 主节点 → 从节点：推理结果回传
    INFER_CANCEL = "infer_cancel"    # 从节点 → 主节点：取消转发推理请求
    LAYER_CONFIG = "layer_config"    # 主节点 → 从节点：推送分层配置
    MODEL_PREPARE = "model_prepare"  # native segment prepare / commit barrier
    MODEL_SUPPLY = "model_supply"      # master-owned segment inventory / supply
    LAYER_CONFIG_ACK = "layer_config_ack"  # 从节点 → 主节点：模型层加载结果
    QWEN3_PIPELINE_DRY_RUN = "qwen3_pipeline_dry_run"
    QWEN3_PIPELINE_DRY_RUN_ACK = "qwen3_pipeline_dry_run_ack"
    LAYER_WORKER_OPT_OUT = "layer_worker_opt_out"  # 从节点 → 主节点：退出分层计算
    LAYER_WORKER_OPT_IN = "layer_worker_opt_in"  # 从节点 → 主节点：重新加入分层计算
    TASK_WORKER = "task_worker"      # PC Full Worker v2 控制面/后续 Stage envelope
    # 角色转让
    ROLE_TRANSFER = "role_transfer"          # 主节点 → 从节点：转让主节点身份
    ROLE_TRANSFER_ACK = "role_transfer_ack"  # 从节点 → 主节点：确认接收转让
    # 备用主节点
    SPARE_MASTER_DESIGNATE = "spare_master_designate"          # 主节点 → 从节点：指定为备用主节点
    SPARE_MASTER_DESIGNATE_ACK = "spare_master_designate_ack"  # 从节点 → 主节点：确认接收备用身份
    SPARE_MASTER_ACTIVATE = "spare_master_activate"            # 主节点 → 备用：激活暂代主节点职责
    SPARE_MASTER_ACTIVATE_ACK = "spare_master_activate_ack"    # 备用 → 主节点：确认激活
    SPARE_MASTER_DEACTIVATE = "spare_master_deactivate"        # 新主节点 → 备用：接管完成，退出暂代
    # 节点列表同步
    NODE_LIST_SYNC = "node_list_sync"            # 主节点 → 从节点：全量节点列表同步
    NODE_UPDATE = "node_update"                  # 主节点 → 从节点：单节点变更 (add/update/remove)
    # 分布式流水线推理
    LAYER_FORWARD = "layer_forward"              # 主→从：执行层前向传播
    LAYER_RESULT = "layer_result"                # 从→主：层前向传播结果
    CHAIN_FORWARD = "chain_forward"              # 主→从：下一逻辑段层前向转发
    CHAIN_FORWARD_ACK = "chain_forward_ack"      # 从→主：逻辑段接收/错误确认
    PIPELINE_DONE = "pipeline_done"              # 主→从：流水线任务完成（清理 KV 缓存）
    PIPELINE_ABORT = "pipeline_abort"            # 主→从：取消流水线任务
    PIPELINE_PAUSE = "pipeline_pause"            # 主→从：暂停流水线（二期协同抢占，协议预留）
    PIPELINE_RESUME = "pipeline_resume"          # 主→从：恢复流水线（二期协同抢占，协议预留）
    # L5: 多节点日志聚合
    LOG_REQUEST = "log_request"                  # 主→从：拉取从节点最近日志
    LOG_RESPONSE = "log_response"                # 从→主：返回最近日志条目


# ================================================================
# 网络类型检测
# ================================================================

# Control messages that can mutate cluster ownership or execution topology.
FENCED_CONTROL_MESSAGE_TYPES = frozenset({
    MessageType.TASK_START.value, MessageType.TASK_STOP.value,
    MessageType.LAYER_CONFIG.value, MessageType.LAYER_CONFIG_ACK.value,
    MessageType.MODEL_SUPPLY.value, MessageType.MODEL_PREPARE.value,
    MessageType.QWEN3_PIPELINE_DRY_RUN.value,
    MessageType.QWEN3_PIPELINE_DRY_RUN_ACK.value,
    MessageType.LAYER_WORKER_OPT_OUT.value, MessageType.LAYER_WORKER_OPT_IN.value,
    MessageType.TASK_WORKER.value, MessageType.ROLE_TRANSFER.value,
    MessageType.ROLE_TRANSFER_ACK.value, MessageType.SPARE_MASTER_DESIGNATE.value,
    MessageType.SPARE_MASTER_DESIGNATE_ACK.value,
    MessageType.SPARE_MASTER_ACTIVATE.value, MessageType.SPARE_MASTER_ACTIVATE_ACK.value,
    MessageType.SPARE_MASTER_DEACTIVATE.value, MessageType.NODE_LIST_SYNC.value,
    MessageType.NODE_UPDATE.value, MessageType.PIPELINE_PAUSE.value,
    MessageType.PIPELINE_RESUME.value,
})


def _is_fenced_control_message(msg_type: MessageType | str) -> bool:
    value = msg_type.value if isinstance(msg_type, MessageType) else str(msg_type)
    return value in FENCED_CONTROL_MESSAGE_TYPES


def _decorate_control_payload(data: Any, msg_type: MessageType, fence: ControlFence | None, *, peer=False) -> Any:
    if fence is None or not fence.enabled or not _is_fenced_control_message(msg_type):
        return data
    if not isinstance(data, dict):
        raise ControlFenceError("control_certificate_invalid", "fenced control messages require an object payload")
    payload = dict(data)
    if "control_certificate" not in payload:
        certificate = fence.current_certificate()
        if certificate is None:
            raise ControlFenceError("control_certificate_missing", "fenced control message has no current certificate")
        payload["control_certificate"] = certificate
    if peer:
        fence.admit_peer(payload.get("control_certificate"), action=f"tcp.send.{msg_type.value}")
    else:
        fence.admit(payload.get("control_certificate"), action=f"tcp.send.{msg_type.value}", source="tcp")
    return payload


def detect_network_type() -> str:
    """
    检测当前节点的主要网络连接类型。

    通过枚举活跃网络接口名称判断是 WiFi 还是以太网。
    同时也会检测实际的网络连接情况作为辅助判断。

    Returns:
        "wifi" | "ethernet" | "unknown"
    """
    detected = "unknown"

    if _HAS_PSUTIL:
        try:
            stats = psutil.net_if_stats()
            addrs = psutil.net_if_addrs()

            wifi_keywords = ['wi-fi', 'wlan', '无线', 'wifi', 'wireless']
            eth_keywords = ['eth', '以太', 'ethernet', 'en0', 'en']

            has_wifi = False
            has_eth = False

            for iface, stat in stats.items():
                iface_lower = iface.lower()
                if stat.isup:
                    # 检查接口名称
                    if any(kw in iface_lower for kw in wifi_keywords):
                        has_wifi = True
                    elif any(kw in iface_lower for kw in eth_keywords):
                        has_eth = True
                    # macOS: en0/en1 通常是以太网/WiFi
                    if iface_lower.startswith('en0'):
                        has_eth = True
                    elif iface_lower.startswith('en1'):
                        has_wifi = True

            # 尝试通过连接的默认路由接口判断
            if has_wifi and not has_eth:
                detected = "wifi"
            elif has_eth and not has_wifi:
                detected = "ethernet"
            elif has_wifi and has_eth:
                # 两者都有，尝试判断哪个是主连接
                try:
                    # 通过连接外部地址来判断使用的接口
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                    s.close()
                    # 查找该 IP 对应的接口
                    for iface, addr_list in addrs.items():
                        for addr in addr_list:
                            if addr.address == local_ip:
                                iface_lower = iface.lower()
                                if any(kw in iface_lower for kw in wifi_keywords):
                                    detected = "wifi"
                                    break
                                elif any(kw in iface_lower for kw in eth_keywords):
                                    detected = "ethernet"
                                    break
                        if detected != "unknown":
                            break
                except Exception:
                    logger.debug("默认路由接口检测失败，保持 unknown", exc_info=True)
            else:
                # 无法根据名称判断，通过实际接口类型判断
                for iface, stat in stats.items():
                    if stat.isup:
                        # en0 通常是以太网，en1 通常是WiFi (macOS)
                        # wlan 通常是WiFi (Linux), eth 通常是以太网
                        pass  # 已在上面处理

        except Exception:
            logger.debug("psutil 网络类型检测失败，回退到基本判断", exc_info=True)

    # 如果 psutil 不可用或检测失败，尝试通过系统命令检测
    if detected == "unknown":
        try:
            import platform
            if platform.system() == "Windows":
                result = subprocess.run(
                    ['netsh', 'interface', 'show', 'interface'],
                    capture_output=True, text=True, timeout=5,
                    encoding="utf-8", errors="replace",
                    creationflags=_WINDOWS_NO_WINDOW,
                )
                output = result.stdout.lower()
                if 'wi-fi' in output or 'wlan' in output:
                    detected = "wifi"
                if 'ethernet' in output or '以太网' in output:
                    detected = "ethernet"  # 以太网优先（更稳定）
            elif platform.system() == "Linux":
                # 通过 /sys/class/net 检测接口类型
                try:
                    net_dir = "/sys/class/net"
                    if os.path.isdir(net_dir):
                        for iface in os.listdir(net_dir):
                            iface_lower = iface.lower()
                            if iface_lower.startswith("wl"):
                                detected = "wifi"
                                break
                            elif (iface_lower.startswith("en")
                                  or iface_lower.startswith("eth")):
                                detected = "ethernet"
                                break
                except OSError:
                    pass
        except Exception:
            logger.debug("系统网络类型检测失败", exc_info=True)

    logger.debug(f"检测到网络类型: {detected}")
    return detected


def detect_lan_ip() -> str:
    """
    检测本机最优可达 IP 地址（供其他节点连接使用）。

    优先级（从高到低）:
    1. Tailscale / ZeroTier 虚拟组网 IP（100.x.y.z）— 最可靠，跨子网直连
    2. psutil 枚举物理网卡（以太网 > WiFi）— 同局域网直连
    3. UDP 默认路由 IP — 出网网卡 IP
    4. socket.gethostbyname(hostname) — 最后兜底

    Returns:
        IP 地址字符串，检测失败返回 "127.0.0.1"
    """

    def _usable_unicast_address(value: str) -> bool:
        """Reject loopback, APIPA/link-local and other non-advertisable addresses."""
        try:
            address = ipaddress.ip_address(value.split("%", 1)[0])
        except ValueError:
            return False
        return not any((
            address.is_loopback,
            address.is_link_local,
            address.is_unspecified,
            address.is_multicast,
            address.is_reserved,
        ))

    # ---- 策略 1: 优先检测 Tailscale / ZeroTier 虚拟组网 IP ----
    #   Tailscale 使用 100.64.0.0/10 (CGNAT) 地址段，接口名含 "tailscale"
    #   ZeroTier 接口名含 "zerotier"
    #   这些 IP 在组网内是全局可达的，不受物理子网隔离影响
    if _HAS_PSUTIL:
        try:
            import socket as _sock
            addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()

            # Tailscale 接口名称本身不足以证明地址可用于 Tailnet：未登录时
            # Windows 会留下 169.254/16 链路本地地址，不能被广告给其他节点。
            overlay_keywords = ('tailscale', 'zerotier', 'zt')
            # 100.64.0.0/10 范围: 100.64.0.0 ~ 100.127.255.255
            _TAILSCALE_LO = 0x64400000   # 100.64.0.0
            _TAILSCALE_HI = 0x647FFFFF   # 100.127.255.255

            overlay_candidates = []  # [(ip, iface, is_up, family)]

            for iface, addr_list in addrs.items():
                iface_lower = iface.lower()
                # Tailscale / ZeroTier 接口名匹配
                is_overlay = any(kw in iface_lower for kw in overlay_keywords)

                stat = stats.get(iface)
                is_up = stat.isup if stat else False

                for addr in addr_list:
                    if addr.family not in (_sock.AF_INET, _sock.AF_INET6):
                        continue
                    ip = addr.address
                    if not _usable_unicast_address(ip):
                        continue
                    # 100.64.0.0/10 范围的 IP 也视为 overlay（即使接口名不匹配）；
                    # Tailscale 还分配 fd7a:115c:a1e0::/48 的 IPv6 ULA，随接口名识别
                    in_tailscale_range = False
                    if addr.family == _sock.AF_INET:
                        try:
                            octets = [int(o) for o in ip.split(".")]
                            ip_int = (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]
                            in_tailscale_range = _TAILSCALE_LO <= ip_int <= _TAILSCALE_HI
                        except (ValueError, IndexError):
                            in_tailscale_range = False

                    in_tailscale_v6_range = False
                    if addr.family == _sock.AF_INET6:
                        try:
                            in_tailscale_v6_range = (
                                ipaddress.ip_address(ip.split("%", 1)[0])
                                in ipaddress.ip_network("fd7a:115c:a1e0::/48")
                            )
                        except ValueError:
                            pass

                    # Tailscale only accepts its assigned 100.64/10 or ULA address.
                    # ZeroTier has no fixed global prefix, so a usable address on its
                    # named adapter remains a valid overlay candidate.
                    is_zerotier = "zerotier" in iface_lower or iface_lower.startswith("zt")
                    if in_tailscale_range or in_tailscale_v6_range or (is_zerotier and is_overlay):
                        overlay_candidates.append((ip, iface, is_up, addr.family))

            if overlay_candidates:
                # 优先 IPv4，其次启用的接口
                overlay_candidates.sort(
                    key=lambda c: (0 if c[3] == _sock.AF_INET else 1, 0 if c[2] else 1)
                )
                ip = overlay_candidates[0][0]
                logger.info(f"检测到组网 IP (Tailscale/ZeroTier): {ip} (接口: {overlay_candidates[0][1]})")
                return ip
        except Exception:
            logger.debug("组网 IP (Tailscale/ZeroTier) 检测失败，尝试物理网卡", exc_info=True)

    # ---- 策略 2: psutil 枚举物理网络接口 ----
    if _HAS_PSUTIL:
        try:
            import socket as _sock
            addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()

            # 排除的接口前缀/标识（虚拟 / 容器 / 回环 / 组网）
            skip_prefixes = ('lo', 'veth', 'docker', 'br-', 'vmnet', 'virbr',
                             'tun', 'tap', 'wg', 'utun', 'anpi', 'bluetooth',
                             'tailscale', 'zerotier', 'zt')
            skip_markers = ('hyper-v', 'wsl', 'vbox', 'virtualbox')

            candidates = []  # [(ip, iface, is_up, is_wifi, family)]

            for iface, addr_list in addrs.items():
                iface_lower = iface.lower()
                if (any(iface_lower.startswith(p) for p in skip_prefixes)
                        or any(marker in iface_lower for marker in skip_markers)):
                    continue

                stat = stats.get(iface)
                is_up = stat.isup if stat else False

                is_wifi = any(kw in iface_lower for kw in
                              ('wi-fi', 'wlan', '无线', 'wifi', 'wireless'))

                for addr in addr_list:
                    if addr.family not in (_sock.AF_INET, _sock.AF_INET6):
                        continue
                    ip = addr.address
                    if not _usable_unicast_address(ip):
                        continue
                    candidates.append((ip, iface, is_up, is_wifi, addr.family))

            if candidates:
                # 优先级: IPv4 > IPv6；启用 + 以太网 > 启用 + WiFi > 启用 > 其他
                def _priority(c):
                    _, _, up, wifi, family = c
                    return (
                        0 if family == _sock.AF_INET else 1,
                        0 if up else 1,
                        0 if not wifi else 1,
                    )

                candidates.sort(key=_priority)
                ip = candidates[0][0]
                logger.info(f"检测到局域网 IP (psutil): {ip} (接口: {candidates[0][1]})")
                return ip
        except Exception:
            logger.debug("psutil 局域网 IP 检测失败，尝试 UDP 默认路由", exc_info=True)

    # ---- 策略 3: UDP 默认路由 IP ----
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        s.connect(("8.8.8.8", 53))
        ip = s.getsockname()[0]
        s.close()
        if ip and _usable_unicast_address(ip):
            logger.info(f"检测到局域网 IP (UDP): {ip}")
            return ip
    except Exception:
        logger.debug("UDP 默认路由 IP 检测失败，尝试 hostname", exc_info=True)

    # ---- 策略 3b: UDP IPv6 默认路由（纯 IPv6 网络兜底）----
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        s.connect(("2001:4860:4860::8888", 53))
        ip = s.getsockname()[0]
        s.close()
        if ip and _usable_unicast_address(ip):
            logger.info(f"检测到局域网 IP (UDP v6): {ip}")
            return ip
    except Exception:
        logger.debug("UDP IPv6 默认路由 IP 检测失败，尝试 hostname", exc_info=True)

    # ---- 策略 4: gethostbyname 兜底 ----
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and _usable_unicast_address(ip):
            logger.info(f"检测到局域网 IP (hostname): {ip} ({hostname})")
            return ip
    except Exception:
        logger.debug("hostname 局域网 IP 检测失败", exc_info=True)

    logger.warning("无法检测局域网 IP，回退到 127.0.0.1")
    return "127.0.0.1"


def get_mac_addresses() -> list[str]:
    """
    获取本机所有物理网卡的 MAC 地址列表。

    用于主节点身份验证：主节点首次启动时将 MAC 地址写入本机 SQLite，
    后续启动时验证本机 MAC 与本地记录是否匹配，防止 IP 变化
    导致身份混淆或其他机器冒充主节点。

    过滤策略:
    - 排除回环接口 (lo, Loopback)
    - 排除虚拟接口 (Docker, VMware, WSL, Hyper-V, VirtualBox)
    - 排除隧道/蓝牙/PPP 接口
    - 仅保留物理网卡 (WiFi + 以太网)

    Returns:
        MAC 地址字符串列表，如 ["aa:bb:cc:dd:ee:ff", "11:22:33:44:55:66"]
        检测失败返回空列表
    """
    macs: list[str] = []

    # 排除的接口关键字（虚拟 / 容器 / 隧道 / 蓝牙 / 回环）
    skip_keywords = (
        'loopback', 'pseudo', 'lookback',
        'docker', 'vmware', 'virtualbox', 'hyper-v', 'wsl',
        'veth', 'br-', 'virbr', 'vbox', 'vmnet',
        'tunnel', 'teredo', 'isatap', '6to4',
        'bluetooth', 'bluetooth',
        'usb', 'ndis', 'wan', 'ppp', 'pppoe',
    )

    # 策略 1: psutil 枚举（最可靠）
    if _HAS_PSUTIL:
        try:
            addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()

            for iface, addr_list in addrs.items():
                iface_lower = iface.lower()

                # 排除虚拟/回环接口
                if iface_lower.startswith('lo'):
                    continue
                if any(kw in iface_lower for kw in skip_keywords):
                    continue

                # 检查接口是否真实存在（不是纯软件接口）
                stat = stats.get(iface)
                if stat is None:
                    continue

                for addr in addr_list:
                    # AF_LINK 族（平台相关）— 在 Linux 上是 17，macOS 是 18
                    # psutil 中 MAC 地址的 family 可能是 AF_LINK 或 -1
                    if hasattr(addr, 'family') and str(addr.family) in ('17', '18', '-1', 'AF_LINK'):
                        mac = addr.address
                        if mac and mac != '00:00:00:00:00:00' and ':' in mac:
                            # psutil 有时将 MAC 放在 family=-1 的 address 中
                            pass
                    # 更稳健的方式：psutil 的 snicaddr 有 address 属性
                    # MAC 地址通常伴随 family == psutil.AF_LINK

            # 重新用更可靠的方式：遍历所有地址，取 family == AF_LINK 的
            import ctypes
            # AF_LINK 在不同平台的值不同，但 psutil 会将 MAC 地址的 family 设为 AF_LINK
            # 在 Windows 上 AF_LINK 通常不存在，MAC 地址的 family 可能是 -1

            for iface, addr_list in addrs.items():
                iface_lower = iface.lower()
                if iface_lower.startswith('lo'):
                    continue
                if any(kw in iface_lower for kw in skip_keywords):
                    continue

                stat = stats.get(iface)
                if stat is None:
                    continue

                for addr in addr_list:
                    family_val = getattr(addr, 'family', None)
                    # psutil.AF_LINK 通常用于 MAC 地址
                    # 在不同 OS 上值不同: Linux=17, macOS=18, Windows 可能没有
                    is_mac_addr = False
                    if hasattr(psutil, 'AF_LINK') and family_val == psutil.AF_LINK:
                        is_mac_addr = True
                    elif family_val == -1 or family_val == 17 or family_val == 18:
                        # Windows 或未知平台
                        is_mac_addr = True

                    if is_mac_addr:
                        mac = addr.address.strip().lower()
                        # 验证 MAC 格式 (6 组十六进制，用 : 或 - 分隔)
                        if mac and mac != '00:00:00:00:00:00':
                            import re
                            if re.match(r'^([0-9a-f]{2}[:-]){5}[0-9a-f]{2}$', mac):
                                macs.append(mac)

            if macs:
                logger.info(f"检测到物理网卡 MAC: {macs}")
                return macs
        except Exception as e:
            logger.debug(f"psutil MAC 检测失败: {e}", exc_info=True)

    # 策略 2: uuid.getnode() 兜底（返回单一 MAC）
    try:
        import uuid
        mac_int = uuid.getnode()
        if mac_int and mac_int != 0:
            mac = ':'.join(f'{(mac_int >> (i * 8)) & 0xff:02x}' for i in range(5, -1, -1))
            if mac != '00:00:00:00:00:00':
                logger.info(f"检测到 MAC (uuid): {mac}")
                return [mac]
    except Exception:
        logger.debug("uuid.getnode MAC 检测失败，尝试 Windows getmac", exc_info=True)

    # 策略 3: Windows subprocess 兜底
    try:
        import platform
        if platform.system() == "Windows":
            result = subprocess.run(
                ['getmac', '/fo', 'csv', '/v'],
                capture_output=True, text=True, timeout=5,
                encoding="utf-8", errors="replace",
                creationflags=_WINDOWS_NO_WINDOW,
            )
            import re
            for line in result.stdout.strip().split('\n')[1:]:  # skip header
                mac_match = re.findall(r'([0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2})', line)
                for mac in mac_match:
                    mac_lower = mac.lower()
                    if mac_lower != '00:00:00:00:00:00':
                        # 排除蓝牙、断开连接的设备（getmac 会显示 "Media disconnected"）
                        if 'media disconnected' not in line.lower():
                            macs.append(mac_lower)
            if macs:
                logger.info(f"检测到 MAC (getmac): {macs}")
                return macs
    except Exception:
        logger.debug("Windows getmac MAC 检测失败", exc_info=True)

    logger.warning("无法检测到任何有效 MAC 地址")
    return []


# ================================================================
# 封包 / 解包
# ================================================================

def pack_data(payload: bytes) -> bytes:
    """
    封包：4字节长度头（大端序）+ 数据体。

    Args:
        payload: 数据体字节流

    Returns:
        完整数据包字节流
    """
    header = struct.pack(">I", len(payload))
    return header + payload


def unpack_header(header: bytes) -> int:
    """
    解包长度头。

    Args:
        header: 4字节长度头

    Returns:
        数据体长度（字节数）
    """
    return struct.unpack(">I", header)[0]


def recv_exact(sock: socket.socket, n: int, mid_frame: bool = False) -> Optional[bytes]:
    """
    精确接收 n 字节数据（处理 TCP 粘包/拆包）。

    Args:
        sock: socket 对象
        n: 需要接收的字节数
        mid_frame: 本次读取是否处于一帧的中间（前面的长度头/数据体已被消费）。
                   为 True 时任何超时都视为字节流失步，抛出 ConnectionError；
                   为 False 时仅在已读到部分字节后超时才抛出 ConnectionError，
                   空闲超时（尚未读到任何字节）原样抛出 socket.timeout。

    Returns:
        接收到的字节流，连接断开时返回 None
    """
    data = b""
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
        except socket.timeout:
            if data or mid_frame:
                # 帧中途超时：部分字节已被消费，字节流已失去同步，
                # 继续接收只会把后续数据错当长度头。按连接故障处理，
                # 由上层走断开/重连路径恢复。
                raise ConnectionError(
                    f"接收超时导致帧不完整（已收 {len(data)}/{n} 字节），"
                    f"字节流失去同步"
                ) from None
            raise
        if not chunk:
            return None
        data += chunk
    return data


# ================================================================
# 张量序列化 / 反序列化
# ================================================================

def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """将张量序列化为字节流（torch.save 到内存 buffer）"""
    buffer = io.BytesIO()
    _require_torch().save(tensor, buffer)
    return buffer.getvalue()


def deserialize_tensor(data: bytes) -> torch.Tensor:
    """从字节流反序列化为张量（torch.load 从内存 buffer）"""
    buffer = io.BytesIO(data)
    return _require_torch().load(buffer)


# ---- 高速序列化路径（流水线隐藏状态传输优化） ----

def serialize_tensor_fast(tensor: torch.Tensor) -> bytes:
    """
    高速张量序列化，自动选择最优路径。

    小张量 (< 1MB): 使用 torch.save（兼容性好、含完整元数据）
    大张量 (≥ 1MB): 使用 numpy tobytes + 自定义头（零拷贝、速度快 ~3x）

    自定义头格式（大张量路径）:
        [4B  magic: b'TNR0' ]
        [1B  dtype_code      ]  # 0=float16, 1=float32, 2=int64, 3=int32
        [1B  ndim             ]
        [4B  dim0             ]
        [4B  dim1             ]
        [4B  dim2 (optional)  ]
        [4B  dim3 (optional)  ]
        [NB  raw_bytes        ]
    总开销: 10-22 字节

    小张量路径:
        [4B  magic: b'TNR1' ]
        [NB  torch.save 输出]
    """
    tensor_cpu = tensor.detach().cpu()
    nbytes = tensor_cpu.numel() * tensor_cpu.element_size()

    if nbytes < 1_000_000:
        # 小张量：torch.save（兼容性好）
        buf = io.BytesIO()
        _require_torch().save(tensor_cpu, buf)
        payload = buf.getvalue()
        return b'TNR1' + payload
    else:
        # 大张量：numpy tobytes（零拷贝）
        import numpy as np
        # Phase 5 review H3: 确保连续内存布局，避免非连续张量 .numpy() 崩溃
        arr = tensor_cpu.contiguous().numpy()
        dtype_map = {
            np.float16: 0, np.float32: 1,
            np.int64: 2, np.int32: 3,
        }
        dtype_code = dtype_map.get(arr.dtype.type)
        if dtype_code is None:
            # Phase 5.1: 拒绝不支持的 dtype，避免静默数据损坏
            raise ValueError(
                f"serialize_tensor_fast 不支持的 dtype: {arr.dtype} "
                f"(仅支持 float16/float32/int32/int64)"
            )
        ndim = arr.ndim
        shape = arr.shape

        header = struct.pack(
            f'>4s B B {"I" * ndim}',
            b'TNR0', dtype_code, ndim, *shape,
        )
        return header + arr.tobytes()


def deserialize_tensor_fast(data: bytes) -> torch.Tensor:
    """
    反序列化 serialize_tensor_fast 的输出。

    magic b'TNR0' → 大张量路径（numpy frombuffer）
    magic b'TNR1' → 小张量路径（torch.load）
    """
    if not data or len(data) < 4:
        raise ValueError("序列化数据为空或长度不足")

    magic = data[:4]
    if magic == b'TNR1':
        # 小张量路径
        buf = io.BytesIO(data[4:])
        # ★ torch ≥2.6 的 `torch.load` 默认 `weights_only=True` ⇒ 本路径（`TNR1` 小张量）
        #   会拒绝**我们自己**的序列化帧（实测 `UnpicklingError: Unsupported operand 26`，
        #   在跨机 relay 上表现为"末节点响应超时"，极难与"模型算错"区分）。
        #   这里承载的是**集群内部**的帧、走 HMAC 认证的 TCP 通道 ⇒ 恢复历史载入语义。
        return _require_torch().load(buf, weights_only=False)
    elif magic == b'TNR0':
        # 大张量路径
        import numpy as np
        dtype_map = {
            0: np.float16, 1: np.float32,
            2: np.int64, 3: np.int32,
        }
        dtype_code = data[4]
        ndim = data[5]
        dtype = dtype_map.get(dtype_code)
        if dtype is None:
            # Phase 5.1: 服务端拒绝未知 dtype code
            raise ValueError(
                f"deserialize_tensor_fast 收到未知 dtype_code: {dtype_code}"
            )
        # 解析 shape（每个维度 4 字节）
        shape = []
        offset = 6
        for _ in range(ndim):
            dim = struct.unpack_from('>I', data, offset)[0]
            shape.append(dim)
            offset += 4
        # 重建张量
        raw = data[offset:]
        arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
        return _require_torch().from_numpy(arr.copy())  # copy 确保内存连续
    else:
        # 未知格式，回退到 torch.load（weights_only 防 pickle 注入）
        buf = io.BytesIO(data)
        return _require_torch().load(buf, weights_only=True)


# ================================================================
# 消息构建与解析
# ================================================================

def build_message(msg_type: MessageType, data: Any = None,
                  binary_body: Optional[bytes] = None) -> bytes:
    """
    构建通信消息。

    Args:
        msg_type: 消息类型
        data: 消息载荷（dict/str 用于JSON，torch.Tensor 用于张量）
        binary_body: 可选二进制体（raw 字节，**不经 base64**）。与 `data` 同帧但
            独立成段：帧 = [长度][JSON meta] + [长度][binary_body]。层段 hidden 这类
            大块 f32 数据走此路径，省掉 base64 的 33% 膨胀与编解码开销；该帧的
            JSON 元数据（layer_range / hidden_spec / 摘要）仍由 `data` 承载。

    Returns:
        封包后的字节流
    """
    meta = {"type": msg_type.value}

    # ★ 票 21a（COMM-WIRE-01）：显式二进制体。与张量路径同一帧布局（元数据段 +
    #   数据段），但不要求 torch、也不做反序列化 ⇒ Android 等无 torch 对端可解析。
    #   未传 binary_body 时本分支不进入，逐字节与既有行为一致。
    if binary_body is not None:
        meta["format"] = "binary"
        meta["binary_bytes"] = len(binary_body)
        if data is not None:
            meta["data"] = data
        meta_bytes = json.dumps(meta).encode("utf-8")
        _validate_binary_length(meta, len(binary_body), len(meta_bytes))
        # 组合: [meta长度(4B)] + [meta JSON] + [体长度(4B)] + [raw 字节]
        return pack_data(meta_bytes) + pack_data(bytes(binary_body))

    if torch is not None and isinstance(data, torch.Tensor):
        meta["format"] = "tensor"
        tensor_bytes = serialize_tensor(data)
        meta_bytes = json.dumps(meta).encode("utf-8")
        # 组合: [meta长度(4B)] + [meta JSON] + [张量数据]
        meta_packed = pack_data(meta_bytes)
        tensor_packed = pack_data(tensor_bytes)
        return meta_packed + tensor_packed
    else:
        meta["format"] = "json"
        if data is not None:
            meta["data"] = data
        return pack_data(json.dumps(meta).encode("utf-8"))


def _validate_binary_length(meta: dict, body_length: int, meta_length: int) -> None:
    """Reject a bad declaration or oversized task-worker frame before body I/O."""
    declared = meta.get("binary_bytes")
    if (type(declared) is not int or declared != body_length
            or not 0 < body_length <= MAX_PACKET_SIZE):
        raise ConnectionError("invalid binary packet length")
    if meta.get("type") == MessageType.TASK_WORKER.value:
        from task_worker_protocol import MAX_BINARY_FRAME_BYTES

        if meta_length + HEADER_LEN + body_length > MAX_BINARY_FRAME_BYTES:
            raise ConnectionError("task worker binary message exceeds maximum size")


def parse_message(raw: bytes) -> dict:
    """
    解析通信消息。

    Args:
        raw: 原始数据包

    Returns:
        包含 type 和 data 的字典。张量消息额外包含 tensor 字段，
        二进制消息额外包含 bytes 字段（均在后续段中，由调用方继续接收）。
    """
    meta = json.loads(raw.decode("utf-8"))

    if meta.get("format") == "tensor":
        # 张量数据在后续包中，由调用方继续接收
        meta["_needs_tensor"] = True
    elif meta.get("format") == "binary":
        declared = meta.get("binary_bytes")
        if type(declared) is not int or not 0 < declared <= MAX_PACKET_SIZE:
            raise ConnectionError("invalid binary packet length")
        # ★ 票 21a：二进制体在后续段中，由调用方继续接收（与张量同一模式，
        #   但不做反序列化 —— 调用方拿到的是 raw 字节）。
        meta["_needs_binary"] = True
    return meta


# ================================================================
# IPv4/IPv6 双栈工具
# ================================================================

def parse_host_port(value: str, default_port: Optional[int] = None) -> tuple[str, Optional[int]]:
    """
    解析 "host:port" / "[fe80::1]:8888" / "fe80::1" / "host" 形式的地址。

    IPv6 字面量必须用方括号包裹才可携带端口（RFC 3986）；裸 IPv6
    字面量（含多个冒号）视为无端口地址。
    """
    value = (value or "").strip()
    if not value:
        return "", default_port
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return value, default_port
        host = value[1:end]
        rest = value[end + 1:]
        if rest.startswith(":"):
            port_str = rest[1:]
            if port_str.isdigit():
                return host, int(port_str)
        return host, default_port
    if value.count(":") == 1:
        host, _, port_str = value.rpartition(":")
        if host:
            if port_str.isdigit():
                return host, int(port_str)
            return host, default_port
    return value, default_port


def format_host_port(host: str, port: int) -> str:
    """格式化 "host:port"；IPv6 字面量自动加方括号。"""
    host = (host or "").strip()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{host}:{port}"


def _is_wildcard_host(host: str) -> bool:
    """是否为通配监听地址（0.0.0.0 / :: / 空）。"""
    return (host or "").strip() in ("", "0.0.0.0", "::")


def _ipv6_literal(host: str) -> bool:
    """是否为 IPv6 字面量（含冒号且未被方括号包裹）。"""
    host = (host or "").strip()
    return ":" in host and not host.startswith("[")


# ================================================================
# TCP 服务端（主节点）
# ================================================================

@dataclass
class ClientConn:
    """已连接客户端信息"""
    client_id: str               # 节点标识 "client1" / "client2"
    sock: socket.socket
    addr: tuple                  # TCP peer endpoint: (ip, ephemeral_port)
    advertised_host: str = ""    # 客户端自身可被其他节点连接的服务 host
    advertised_port: int = SERVER_PORT  # 客户端自身 TCP 服务监听端口
    advertised_address: str = "" # "host:port"，用于节点列表和链式转发
    role: str = ""               # 节点角色
    node_type: str = "pc"        # 设备平台: "pc" | "android"
    hostname: str = ""           # 客户端主机名
    device_info: dict = None     # 客户端设备信息
    network_type: str = "unknown"  # 网络连接类型: wifi | ethernet | unknown
    connected_at: float = 0.0    # 连接时间
    last_heartbeat: float = 0.0  # 上次心跳时间
    heartbeat_missed: int = 0    # 连续心跳丢失次数
    avg_rtt_ms: float = 0.0      # 客户端测得的到主节点 EWMA RTT
    last_rtt_ms: float = 0.0     # 客户端最近一次测得的 RTT
    registration_confirmed: bool = False  # scheduler 确认后才返回 registered ACK
    registration_epoch: int = 0            # monotonic identity epoch; changes on reconnect
    send_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def __post_init__(self):
        if self.device_info is None:
            self.device_info = {}
        if not self.advertised_host and self.addr:
            self.advertised_host = self.addr[0]
        if not self.advertised_port:
            self.advertised_port = SERVER_PORT
        if not self.advertised_address and self.advertised_host:
            self.advertised_address = format_host_port(self.advertised_host, self.advertised_port)
        if self.connected_at == 0.0:
            self.connected_at = time.time()
        if self.last_heartbeat == 0.0:
            self.last_heartbeat = time.time()


class TCPServer:
    """TCP 服务端：主节点使用，监听从节点连接"""

    # CPU-only pipeline workers may spend as long as the configured per-step
    # timeout in one layer forward.  Leave two additional heartbeat periods
    # for serialization and network transit before declaring the peer dead.
    MAX_HEARTBEAT_MISSED = max(
        10,
        math.ceil(PIPELINE_STEP_TIMEOUT / HEARTBEAT_INTERVAL) + 2,
    )

    def __init__(self, host: str = None, port: int = None):
        self.host = host or SERVER_IP
        self.port = SERVER_PORT if port is None else port
        self.sock: Optional[socket.socket] = None    # 主监听 socket（兼容旧引用）
        self._socks: list[socket.socket] = []        # 双栈监听 socket 列表
        self.clients: dict[str, ClientConn] = {}     # client_id -> ClientConn
        self._pending_registrations: dict[str, ClientConn] = {}
        self._clients_lock = threading.RLock()       # 保护 clients 的跨线程访问
        self._running = False
        self._accept_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._recv_threads: dict[str, threading.Thread] = {}
        self._registration_epochs: dict[str, int] = {}
        self._registration_context = threading.local()
        self._auth_replay_guard = RegistrationReplayGuard()
        self.on_message: Optional[Callable] = None    # 消息回调
        self.on_disconnect: Optional[Callable] = None # 断连回调
        # Called only after a successful REGISTER ACK has been written to the
        # socket.  Scheduler-side layer/config pushes must wait for this edge.
        self.on_registration_confirmed: Optional[Callable] = None
        self._control_fence: ControlFence | None = None

    def set_control_fence(self, fence: ControlFence | None) -> None:
        if fence is not None and not isinstance(fence, ControlFence):
            raise TypeError("control fence must be a ControlFence or None")
        self._control_fence = fence

    def _admit_inbound_control(self, msg: dict) -> None:
        fence = self._control_fence
        msg_type = msg.get("type", "")
        if fence is None or not fence.enabled or not _is_fenced_control_message(msg_type):
            return
        data = msg.get("data")
        certificate = data.get("control_certificate") if isinstance(data, dict) else None
        fence.admit(certificate, action=f"tcp.receive.{msg_type}", source="tcp")

    def start(self, on_message: Callable = None,
              on_disconnect: Callable = None,
              on_registration_confirmed: Callable = None) -> None:
        """
        启动 TCP 服务端，开始监听。

        Args:
            on_message: 收到消息时的回调函数
                        签名: (client_id: str, msg: dict, raw: bytes) -> None
            on_disconnect: 客户端断连时的回调函数
                          签名: (client_id: str) -> None
            on_registration_confirmed: REGISTER ACK 写入后调用的回调函数
                                       签名: (client_id: str) -> None

        回调均在 TCP 接收/心跳 I/O 线程同步执行。TCPServer 只持有显式
        注册的函数，不持有或反射控制面 owner；回调内部的控制面锁由
        调用方负责，TCPServer 不在持有 _clients_lock 时调用这些函数。
        """
        self.on_message = on_message
        self.on_disconnect = on_disconnect
        if on_registration_confirmed is not None:
            self.on_registration_confirmed = on_registration_confirmed
        self._socks = self._create_listen_sockets(self.host, self.port)
        self.sock = self._socks[0] if self._socks else None
        if self.port == 0 and self.sock is not None:
            self.port = int(self.sock.getsockname()[1])
        self._running = True
        bind_desc = ", ".join(
            f"{sock.getsockname()[0]}:{sock.getsockname()[1]}" for sock in self._socks
        )
        logger.info(f"TCP服务端启动: {self.host}:{self.port}（监听 {bind_desc}）")

        # 启动 accept 循环线程
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

        # 启动心跳检测线程
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    # ---- 双栈监听 ----

    @staticmethod
    def _resolve_bind_hosts(host: str) -> list[str]:
        """通配地址展开为 v4+v6 双栈；具体地址原样返回。"""
        if _is_wildcard_host(host):
            return ["0.0.0.0", "::"]
        return [(host or "").strip()]

    @classmethod
    def _create_listen_sockets(cls, host: str, port: int) -> list[socket.socket]:
        """创建监听 socket 列表（IPv6 socket 显式 V6ONLY=1，与 v4 socket 同端口共存）。"""
        socks: list[socket.socket] = []
        bind_port = port
        try:
            for bind_host in cls._resolve_bind_hosts(host):
                if bind_host == "::" or _ipv6_literal(bind_host):
                    family = socket.AF_INET6
                else:
                    family = socket.AF_INET
                sock = socket.socket(family, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if family == socket.AF_INET6:
                    try:
                        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                    except OSError:
                        pass
                sock.bind((bind_host, bind_port))
                if bind_port == 0:
                    bind_port = int(sock.getsockname()[1])
                sock.listen(2)  # 两台从节点
                sock.settimeout(1.0)  # accept 超时 1s，以便检查 _running
                socks.append(sock)
        except Exception:
            for sock in socks:
                sock.close()
            raise
        if not socks:
            raise OSError(f"无法创建监听 socket: {host}:{port}")
        return socks

    # ---- clients 线程安全访问 ----

    def _get_client(self, client_id: str) -> Optional[ClientConn]:
        """线程安全获取客户端连接。"""
        with self._clients_lock:
            return self.clients.get(client_id)

    def _set_client(self, client_id: str, conn: ClientConn) -> None:
        """线程安全设置客户端连接。"""
        with self._clients_lock:
            self.clients[client_id] = conn

    def _next_registration_epoch(self, client_id: str) -> int:
        """Allocate a new live epoch for every successful TCP registration."""
        with self._clients_lock:
            epoch = int(self._registration_epochs.get(client_id, 0)) + 1
            self._registration_epochs[client_id] = epoch
            return epoch

    def _pop_client(self, client_id: str) -> Optional[ClientConn]:
        """线程安全移除客户端连接；不存在时返回 None。"""
        with self._clients_lock:
            return self.clients.pop(client_id, None)

    def _pop_client_if_same(self, client_id: str,
                            expected: ClientConn) -> Optional[ClientConn]:
        """仅当当前连接对象仍是 expected 时才移除，避免误删新连接。"""
        with self._clients_lock:
            if self.clients.get(client_id) is expected:
                return self.clients.pop(client_id, None)
            return None

    def _client_ids_snapshot(self) -> list[str]:
        """线程安全获取客户端 ID 快照。"""
        with self._clients_lock:
            return list(self.clients.keys())

    def _client_items_snapshot(self) -> list[tuple[str, ClientConn]]:
        """线程安全获取客户端连接快照。"""
        with self._clients_lock:
            return list(self.clients.items())

    def _send_register_ack(self, conn: socket.socket, status: str,
                           client_id: str = "", reason: str = "") -> None:
        """发送 REGISTER ACK。"""
        data = {"status": status}
        if client_id:
            data["client_id"] = client_id
        if reason:
            data["reason"] = reason
        conn.sendall(build_message(MessageType.REGISTER, data))

    def confirm_registration(self, client_id: str) -> bool:
        """上层调度器确认注册成功后，向客户端返回 registered。"""
        # The scheduler callback runs synchronously in the registration
        # thread.  Prefer that exact socket so a concurrent reconnect cannot
        # redirect or invalidate the ACK merely by reusing the same client ID.
        client = getattr(self._registration_context, "client", None)
        if client is None or client.client_id != client_id:
            with self._clients_lock:
                client = self._pending_registrations.get(client_id)
                if client is None:
                    client = self.clients.get(client_id)
        if client is None:
            logger.warning("注册确认失败: client=%s 的连接已不存在", client_id)
            return False
        try:
            with client.send_lock:
                self._send_register_ack(client.sock, "registered", client_id=client_id)
            client.registration_confirmed = True
            with self._clients_lock:
                current = self.clients.get(client_id)
                if (current is None
                        or current.registration_epoch <= client.registration_epoch):
                    self.clients[client_id] = client
                if self._pending_registrations.get(client_id) is client:
                    self._pending_registrations.pop(client_id, None)
            return True
        except OSError as e:
            logger.warning(f"注册确认发送失败: {e}", exc_info=True)
            return False

    def reject_client(self, client_id: str, reason: str) -> None:
        """上层调度器拒绝注册时，返回 rejected 并关闭连接。"""
        client = self._pop_client(client_id)
        if client is None:
            return
        try:
            self._send_register_ack(client.sock, "rejected", client_id=client_id, reason=reason)
        except OSError as e:
            logger.debug(f"注册拒绝 ACK 发送失败: {e}", exc_info=True)
        try:
            client.sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        try:
            client.sock.close()
        except OSError:
            pass

    def disconnect_all_clients(
        self,
        reason: str,
        *,
        before_disconnect: Callable[[], Any] | None = None,
    ) -> tuple[Any, int]:
        """Atomically fence credentials and detach every authenticated peer."""
        with self._clients_lock:
            action_result = before_disconnect() if before_disconnect is not None else None
            client_items = list(self.clients.items())
            pending_items = list(self._pending_registrations.items())
            self.clients.clear()
            self._pending_registrations.clear()

        unique_connections: dict[int, ClientConn] = {}
        for _client_id, client in client_items + pending_items:
            unique_connections[id(client.sock)] = client
        for client in unique_connections.values():
            try:
                client.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.sock.close()
            except OSError:
                pass

        if self.on_disconnect:
            for client_id, _client in client_items:
                try:
                    self.on_disconnect(client_id)
                except Exception:
                    logger.error(
                        "disconnect callback failed during credential cutover: client=%s reason=%s",
                        client_id,
                        reason,
                        exc_info=True,
                    )
        logger.warning(
            "disconnected all cluster TCP clients: count=%s reason=%s",
            len(unique_connections),
            reason,
        )
        return action_result, len(unique_connections)

    # ---- Accept 循环 ----

    def _accept_loop(self) -> None:
        """在独立线程中循环 accept 客户端连接（双栈多 socket 轮询）"""
        logger.info("Accept 循环已启动，等待客户端连接...")
        while self._running:
            socks = list(self._socks)
            if not socks:
                break
            try:
                readable, _, _ = select.select(socks, [], [], 1.0)
            except OSError as e:
                if self._running:
                    logger.error(f"Accept select 异常: {e}", exc_info=True)
                break
            for sock in readable:
                try:
                    conn, addr = sock.accept()
                    logger.info(f"新连接: {addr}")
                    # 启动消息接收线程（暂时以 addr 为临时 ID，注册后更新）
                    temp_id = f"pending_{addr[1]}"
                    t = threading.Thread(
                        target=self._handle_client,
                        args=(conn, addr, temp_id),
                        daemon=True,
                    )
                    t.start()
                except socket.timeout:
                    continue
                except OSError as e:
                    if self._running:
                        logger.error(f"Accept 异常: {e}", exc_info=True)
                    break
        logger.info("Accept 循环已退出")

    # ---- 客户端消息处理 ----

    def _handle_client(self, conn: socket.socket, addr: tuple, temp_id: str) -> None:
        """
        接收并分发客户端消息。

        在独立线程中运行，持续接收消息直到连接断开。
        """
        # 注册完成前使用较短超时，防止握手阶段挂死；注册成功后再放宽
        conn.settimeout(HEARTBEAT_INTERVAL + 2)
        client_id = temp_id

        try:
            while self._running:
                # 接收长度头
                header = recv_exact(conn, HEADER_LEN)
                if header is None:
                    break  # 连接断开

                payload_len = unpack_header(header)
                if payload_len <= 0 or payload_len > MAX_PACKET_SIZE:
                    logger.warning(f"非法包长度: {payload_len}，断开 {addr}")
                    break

                payload = recv_exact(conn, payload_len, mid_frame=True)
                if payload is None:
                    break

                msg = parse_message(payload)

                # 处理张量附加数据
                if msg.get("_needs_tensor"):
                    tensor_header = recv_exact(conn, HEADER_LEN, mid_frame=True)
                    if tensor_header is None:
                        break
                    tensor_len = unpack_header(tensor_header)
                    if tensor_len <= 0 or tensor_len > MAX_PACKET_SIZE:
                        logger.warning(
                            f"非法张量包长度: {tensor_len}，断开 {addr}"
                        )
                        break
                    tensor_data = recv_exact(conn, tensor_len, mid_frame=True)
                    if tensor_data is None:
                        break
                    msg["tensor"] = deserialize_tensor(tensor_data)
                    del msg["_needs_tensor"]
                # ★ 票 21a：二进制体（raw 字节，不做反序列化）。
                elif msg.get("_needs_binary"):
                    binary_header = recv_exact(conn, HEADER_LEN, mid_frame=True)
                    if binary_header is None:
                        break
                    binary_len = unpack_header(binary_header)
                    _validate_binary_length(msg, binary_len, payload_len)
                    msg["binary"] = recv_exact(conn, binary_len, mid_frame=True)
                    if msg["binary"] is None:
                        break
                    del msg["_needs_binary"]

                msg_type = msg.get("type", "")
                registration_pending = False

                try:
                    self._admit_inbound_control(msg)
                except ControlFenceError as exc:
                    logger.warning("TCP control frame rejected code=%s type=%s", exc.code, msg_type)
                    try:
                        self.send_to_client(
                            client_id,
                            {"status": "rejected", "error_code": exc.code},
                            MessageType.ERROR,
                        )
                    except Exception:
                        pass
                    continue

                # ---- 消息分发 ----
                if msg_type == MessageType.REGISTER.value:
                    # 注册消息：提取客户端身份信息
                    try:
                        client_id = self._handle_registration(conn, addr, temp_id, msg)
                        registration_pending = True
                    except _RegistrationRejected as e:
                        logger.info(f"注册被拒，关闭连接: {addr[0]}:{addr[1]} — {e}")
                        try:
                            conn.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        break
                    # 更新 recv 线程映射
                    if temp_id in self._recv_threads:
                        del self._recv_threads[temp_id]
                    self._recv_threads[client_id] = threading.current_thread()
                    # 注册完成后放宽接收超时：离线判定交给心跳巡检
                    # （允许连续丢失 MAX_HEARTBEAT_MISSED 次心跳），
                    # 避免单次卡顿（模型加载/磁盘 IO/GC）就被误判离线踢出
                    conn.settimeout(
                        HEARTBEAT_INTERVAL * (self.MAX_HEARTBEAT_MISSED + 1) + 2
                    )

                elif msg_type == MessageType.HEARTBEAT.value:
                    # 心跳：回显客户端时间戳并回复 ACK
                    self._handle_heartbeat(client_id, msg)

                # 回调上层。TCPServer 只执行显式注册的事件，不反射回调
                # owner，也不改写控制面对象的任何属性。
                if self.on_message:
                    try:
                        self.on_message(client_id, msg)
                    except Exception as e:
                        logger.error(f"消息回调异常: {e}", exc_info=True)
                        if registration_pending:
                            self.reject_client(client_id, "调度器注册回调异常")
                            break
                elif registration_pending:
                    self.confirm_registration(client_id)

                registration_confirmed_now = False
                if registration_pending:
                    current = getattr(
                        self._registration_context, "client", None,
                    )
                    if current is None or current.sock is not conn:
                        break
                    if not current.registration_confirmed:
                        try:
                            self._send_register_ack(
                                current.sock,
                                "registered",
                                client_id=client_id,
                            )
                        except OSError:
                            self._pop_client_if_same(client_id, current)
                            break
                        else:
                            current.registration_confirmed = True
                            registration_confirmed_now = True
                            with self._clients_lock:
                                if self.clients.get(client_id) is None:
                                    self.clients[client_id] = current
                                if self._pending_registrations.get(client_id) is current:
                                    self._pending_registrations.pop(client_id, None)

                    # Scheduler callbacks may have already called
                    # confirm_registration().  The post-registration hook is
                    # an ACK edge, not an implementation-detail edge of the
                    # fallback path, so it must run for either confirmation
                    # route exactly once for this REGISTER frame.
                    registration_confirmed_now = current.registration_confirmed

                if (
                    registration_confirmed_now
                    and self.on_registration_confirmed is not None
                    and current is not None
                ):
                    try:
                        self.on_registration_confirmed(client_id)
                    except Exception as e:
                        # Registration is already valid; a post-ACK scheduling
                        # failure must not tear down the worker connection.
                        logger.warning(
                            "注册确认后的调度回调失败: client=%s error=%s",
                            client_id, e, exc_info=True,
                        )

        except socket.timeout:
            logger.info(f"客户端 {client_id} 接收超时")
        except (ConnectionResetError, ConnectionAbortedError) as e:
            logger.info(f"客户端 {client_id} 已关闭连接: {e}")
        except (ConnectionError, OSError) as e:
            logger.warning(f"客户端 {client_id} 连接异常: {e}", exc_info=True)
        finally:
            if hasattr(self._registration_context, "client"):
                del self._registration_context.client
            # 清理
            try:
                conn.close()
            except OSError:
                pass

            removed = None
            if client_id != temp_id:
                current = self._get_client(client_id)
                if current is not None and current.sock is conn:
                    removed = self._pop_client_if_same(client_id, current)
            with self._clients_lock:
                pending = self._pending_registrations.get(client_id)
                if pending is not None and pending.sock is conn:
                    self._pending_registrations.pop(client_id, None)

            if removed is not None:
                logger.info(f"客户端 {client_id} 已断开: {removed.addr}")
            else:
                logger.debug(f"未注册或已清理连接已断开: {client_id} addr={addr}")

            if client_id in self._recv_threads:
                del self._recv_threads[client_id]
            # 通知上层断连（仅对本线程实际移除的已注册连接通知一次）
            if self.on_disconnect and removed is not None:
                try:
                    self.on_disconnect(client_id)
                except Exception as e:
                    logger.error(f"断连回调异常: {e}", exc_info=True)

    def _handle_registration(self, conn: socket.socket, addr: tuple,
                             temp_id: str, msg: dict) -> str:
        """
        处理客户端注册消息。

        Returns:
            解析出的 client_id
        """
        data = msg.get("data", {})
        client_id = data.get("client_id", temp_id)
        role = data.get("role", "unknown")
        node_type = data.get("node_type", "pc")
        hostname = data.get("hostname", "unknown")
        device_info = data.get("device_info", {})
        network_type = data.get("network_type", "unknown")

        advertised_host = str(data.get("advertised_host") or "").strip()
        advertised_port_raw = data.get("advertised_port")
        advertised_address = str(data.get("advertised_address") or "").strip()
        if advertised_address:
            # 支持 "[fe80::1]:8888" / "fe80::1" / "1.2.3.4:8888" / "1.2.3.4"
            parsed_host, parsed_port = parse_host_port(advertised_address)
            if parsed_host:
                advertised_host = advertised_host or parsed_host
            if advertised_port_raw is None and parsed_port is not None:
                advertised_port_raw = parsed_port
        if advertised_host in ("", "0.0.0.0", "127.0.0.1", "::", "::1", "localhost"):
            advertised_host = addr[0]
        try:
            advertised_port = int(advertised_port_raw) if advertised_port_raw is not None else SERVER_PORT
        except (TypeError, ValueError):
            advertised_port = SERVER_PORT
        if not (1 <= advertised_port <= 65535):
            advertised_port = SERVER_PORT
        advertised_address = format_host_port(advertised_host, advertised_port)

        allow_unprotected = os.environ.get(
            "QLH_ALLOW_UNPROTECTED_CLUSTER_TCP", ""
        ).strip().lower() in {"1", "true", "yes"}
        if not allow_unprotected and not _protected_cluster_peer(addr[0]):
            reason = (
                "生产集群 TCP 只允许 loopback/Tailnet 受保护路径；"
                "直接 LAN 明文链路需显式不安全开关"
            )
            self._send_register_ack(conn, "rejected", reason=reason)
            raise _RegistrationRejected(reason)

        # ★ 阶段 7：HMAC 集群认证
        auth_data = data.get("auth", {})
        auth_ok, auth_reason = verify_auth_signature(
            client_id,
            auth_data,
            replay_guard=self._auth_replay_guard,
        )
        if not auth_ok:
            logger.error(
                f"⛔ 认证失败: {client_id}@{addr[0]}:{addr[1]} — {auth_reason}"
            )
            ack = build_message(MessageType.REGISTER, {
                "status": "rejected",
                "reason": f"认证失败: {auth_reason}",
            })
            try:
                conn.sendall(ack)
            except OSError as e:
                logger.debug(f"认证失败 ACK 发送失败: {e}", exc_info=True)
            raise _RegistrationRejected(f"认证失败: {auth_reason}")

        # ★ 安全：拒绝 client_id="master" 或 role="master" 的注册
        if client_id == "master":
            logger.error(
                f"⛔ 拒绝注册: client_id='master' 是保留字，"
                f"来源 {addr[0]}:{addr[1]} hostname={hostname}"
            )
            ack = build_message(MessageType.REGISTER, {
                "status": "rejected",
                "reason": "client_id 'master' 是保留字，请使用其他 ID",
            })
            try:
                conn.sendall(ack)
            except OSError as e:
                logger.debug(f"注册拒绝 ACK 发送失败: {e}", exc_info=True)
            raise _RegistrationRejected("client_id 'master' 是保留字")
        if role == "master":
            logger.error(
                f"⛔ 拒绝注册: 从节点不能声明 role='master'，"
                f"来源 {addr[0]}:{addr[1]} hostname={hostname}"
            )
            ack = build_message(MessageType.REGISTER, {
                "status": "rejected",
                "reason": "role 'master' 仅限于主节点自身",
            })
            try:
                conn.sendall(ack)
            except OSError as e:
                logger.debug(f"注册拒绝 ACK 发送失败: {e}", exc_info=True)
            raise _RegistrationRejected("从节点不能声明 role='master'")
        if node_type == "pipeline_peer":
            reason = "pipeline_peer 已禁用；worker 数据连接必须以 master 为中心"
            logger.error(
                "⛔ 拒绝注册: node_type='pipeline_peer' 已禁用，来源 %s:%s",
                addr[0], addr[1],
            )
            ack = build_message(MessageType.REGISTER, {
                "status": "rejected",
                "reason": reason,
            })
            try:
                conn.sendall(ack)
            except OSError as e:
                logger.debug("pipeline_peer 拒绝 ACK 发送失败: %s", e, exc_info=True)
            raise _RegistrationRejected(reason)
        # Android 节点只能作为 client
        if node_type == "android" and role == "master":
            logger.error(
                f"⛔ 拒绝注册: Android 节点不能担任 master 角色，"
                f"来源 {addr[0]}:{addr[1]}"
            )
            ack = build_message(MessageType.REGISTER, {
                "status": "rejected",
                "reason": "Android 节点不能担任 master 角色",
            })
            try:
                conn.sendall(ack)
            except OSError as e:
                logger.debug(f"注册拒绝 ACK 发送失败: {e}", exc_info=True)
            raise _RegistrationRejected("Android 节点不能担任 master 角色")

        client_conn = ClientConn(
            client_id=client_id,
            sock=conn,
            addr=addr,
            advertised_host=advertised_host,
            advertised_port=advertised_port,
            advertised_address=advertised_address,
            role=role,
            node_type=node_type,
            hostname=hostname,
            device_info=device_info,
            network_type=network_type,
            registration_epoch=self._next_registration_epoch(client_id),
        )
        # Bind before publishing the object.  An older reconnect callback may
        # concurrently remove the global mapping, but it cannot erase this
        # registration thread's exact socket identity.
        self._registration_context.client = client_conn
        # A reconnecting worker can briefly have more than one connect loop
        # racing (for example while the master restarts).  Never overwrite a
        # connection whose registration is still being confirmed: doing so
        # makes the scheduler ACK a different socket and all registrations
        # fail in a loop.  A fully registered connection may be replaced,
        # but the old socket is closed only after leaving the lock.
        previous = None
        with self._clients_lock:
            authenticated_epoch = int(auth_data.get("auth_secret_epoch", 0) or 0)
            if authenticated_epoch != _get_cluster_secret_epoch():
                reason = "cluster credential rotated during registration"
                try:
                    self._send_register_ack(
                        conn, "rejected", client_id=client_id, reason=reason,
                    )
                except OSError:
                    pass
                raise _RegistrationRejected(reason)
            previous = self.clients.get(client_id)
            if previous is not None and not previous.registration_confirmed:
                reason = "该节点已有注册正在确认"
                try:
                    self._send_register_ack(
                        conn, "rejected", client_id=client_id, reason=reason,
                    )
                except OSError as e:
                    logger.debug("重复注册拒绝 ACK 发送失败: %s", e, exc_info=True)
                raise _RegistrationRejected(reason)
            if previous is not None and previous.registration_confirmed:
                # A confirmed session may still be stale from the client's
                # perspective (for example after a dual-stack reconnect).
                # Adopt the new socket so its REGISTER ACK and subsequent
                # layer configuration are delivered to the live client. The
                # exact-socket cleanup below prevents the old recv thread from
                # removing this replacement or firing a false disconnect.
                logger.info(
                    "替换同一节点的已确认 TCP 会话: client=%s old_peer=%s new_peer=%s",
                    client_id,
                    getattr(previous, "addr", None),
                    addr,
                )
            self._pending_registrations[client_id] = client_conn
            self.clients[client_id] = client_conn
        if previous is not None and previous.sock is not conn:
            try:
                previous.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                previous.sock.close()
            except OSError:
                pass
        logger.info(
            f"✅ 节点注册成功: {client_id} role={role} "
            f"hostname={hostname} advertised={advertised_address} "
            f"peer={addr[0]}:{addr[1]}"
        )

        return client_id

    def _handle_heartbeat(self, client_id: str, msg: dict = None) -> None:
        """处理心跳消息：更新时间戳并回复 ACK（回显客户端时间戳用于 RTT 测量）"""
        conn = self._get_client(client_id)
        if conn:
            conn.last_heartbeat = time.time()
            conn.heartbeat_missed = 0
            # 提取客户端发送时间戳，原样回显
            echo_data = None
            if msg and isinstance(msg.get("data"), dict):
                data = msg["data"]
                reported_rtt = data.get("rtt_ms")
                try:
                    reported_rtt = float(reported_rtt)
                except (TypeError, ValueError):
                    reported_rtt = 0.0
                if (
                    math.isfinite(reported_rtt)
                    and 0.0 < reported_rtt <= DEFAULT_MAX_HEARTBEAT_RTT_MS
                ):
                    conn.last_rtt_ms = reported_rtt
                    conn.avg_rtt_ms = reported_rtt
                t_send = data.get("t_send", 0)
                if t_send:
                    echo_data = {"t_send": t_send}
            # 回复 ACK
            try:
                ack = build_message(MessageType.HEARTBEAT_ACK, echo_data)
                with conn.send_lock:
                    conn.sock.sendall(ack)
            except OSError as e:
                logger.debug(f"心跳 ACK 发送失败: client={client_id}, error={e}", exc_info=True)

    def send_to_client(self, client_id: str, data: Any,
                       msg_type: MessageType = MessageType.TENSOR,
                       *, registration_epoch: int | None = None,
                       binary_body: Optional[bytes] = None) -> None:
        """
        向指定从节点发送数据。

        Args:
            client_id: 从节点ID（"client1" / "client2"）
            data: 发送数据（dict / torch.Tensor）
            msg_type: 消息类型
            binary_body: 可选二进制体（raw 字节，见 `build_message`）。
        """
        conn = self._get_client(client_id)
        if conn is None:
            raise ConnectionError(f"从节点 {client_id} 未连接")
        data = _decorate_control_payload(data, msg_type, self._control_fence)
        packet = build_message(msg_type, data, binary_body=binary_body)
        try:
            # 心跳 ACK、控制消息和大张量会由不同线程发送，必须保持完整帧原子写入。
            with conn.send_lock:
                if registration_epoch is not None and (
                    not conn.registration_confirmed
                    or conn.registration_epoch != registration_epoch
                    or self._get_client(client_id) is not conn
                ):
                    raise ConnectionError(f"node {client_id} registration changed before send")
                conn.sock.sendall(packet)
        except OSError as e:
            raise ConnectionError(f"向 {client_id} 发送失败: {e}")

    def broadcast(self, data: Any,
                  msg_type: MessageType = MessageType.TASK_START) -> None:
        """向所有从节点广播消息"""
        for cid in self._client_ids_snapshot():
            try:
                self.send_to_client(cid, data, msg_type)
            except ConnectionError as e:
                logger.warning(f"广播跳过 {cid}: {e}", exc_info=True)

    def send_layer_config(
        self, client_id: str, assignment: dict, *, registration_epoch: int | None = None,
    ) -> None:
        """向指定从节点推送分层配置"""
        send_options = {"registration_epoch": registration_epoch} if registration_epoch is not None else {}
        self.send_to_client(client_id, assignment, MessageType.LAYER_CONFIG, **send_options)

    def is_authenticated_loopback_client(self, client_id: str) -> bool:
        """Return whether HMAC registration completed on a loopback socket."""
        client = self._get_client(client_id)
        if client is None or not client.registration_confirmed:
            return False
        try:
            return ipaddress.ip_address(client.addr[0]).is_loopback
        except ValueError:
            return False

    def get_registration_peer_epoch(self, client_id: str) -> int | None:
        """Return the exact live epoch, including the pre-REGISTER-ACK edge."""
        client = self._get_client(client_id)
        callback_client = getattr(self._registration_context, "client", None)
        if (callback_client is not None and callback_client.client_id == client_id
                and callback_client is not client):
            return None
        return int(client.registration_epoch) if client is not None else None

    def get_authenticated_peer_epoch(self, client_id: str) -> int | None:
        """Return the live registration epoch, or ``None`` when unauthenticated."""
        client = self._get_client(client_id)
        callback_client = getattr(self._registration_context, "client", None)
        if (callback_client is not None and callback_client.client_id == client_id
                and callback_client is not client):
            return None
        if client is None or not client.registration_confirmed:
            return None
        # Check confirmation and return the epoch from the same object. A
        # second lookup could substitute an unconfirmed reconnect after this
        # check, authorizing commands before that successor's HELLO_ACK.
        return int(client.registration_epoch)

    def is_authenticated_loopback_peer(self, client_id: str, epoch: int) -> bool:
        """Check loopback registration and the exact reconnect epoch."""
        try:
            expected = int(epoch)
        except (TypeError, ValueError):
            return False
        return self.is_authenticated_loopback_client(client_id) and (
            self.get_authenticated_peer_epoch(client_id) == expected
        )

    def send_qwen3_pipeline_dry_run(
        self, client_id: str, payload: dict,
    ) -> None:
        if not self.is_authenticated_loopback_client(client_id):
            raise ConnectionError(
                f"Qwen3 dry-run peer {client_id} is not authenticated loopback"
            )
        self.send_to_client(
            client_id, payload, MessageType.QWEN3_PIPELINE_DRY_RUN,
        )

    def broadcast_layer_config(self, assignments: dict) -> None:
        """
        向所有已连接从节点广播分层配置。

        Args:
            assignments: {client_id: {node_id, config_id, start_layer, end_layer,
                          has_embedding, has_lm_head, model_sha256}}

        每个客户端只接收自己的 assignment。接收端同时兼容旧版的
        ``{client_id: assignment}`` 外层映射。
        """
        for cid in self._client_ids_snapshot():
            if cid in assignments:
                try:
                    self.send_layer_config(cid, assignments[cid])
                except Exception as e:
                    logger.warning(f"分层配置推送跳过 {cid}: {e}", exc_info=True)

    # ---- 心跳检测 ----

    def _heartbeat_loop(self) -> None:
        """心跳检测循环：定期检查所有客户端心跳状态"""
        while self._running:
            time.sleep(HEARTBEAT_INTERVAL)
            now = time.time()
            for cid, conn in self._client_items_snapshot():
                elapsed = now - conn.last_heartbeat
                if elapsed > HEARTBEAT_INTERVAL * (self.MAX_HEARTBEAT_MISSED + 1):
                    conn.heartbeat_missed = self.MAX_HEARTBEAT_MISSED + 1
                    logger.warning(
                        f"⚠️ 节点 {cid} 心跳超时 ({elapsed:.0f}s)，"
                        f"连续丢失 {conn.heartbeat_missed} 次"
                    )
                    # 关闭连接，触发清理
                    try:
                        conn.sock.close()
                    except OSError:
                        pass
                    removed = self._pop_client_if_same(cid, conn)
                    # 通知上层（仅本线程实际移除连接时通知一次）
                    if removed is not None and self.on_disconnect:
                        try:
                            self.on_disconnect(cid)
                        except Exception as e:
                            logger.error(f"断连回调异常: {e}", exc_info=True)

    def get_client_ids(self) -> list:
        """获取所有已连接客户端 ID 列表"""
        return self._client_ids_snapshot()

    def get_client_info(self, client_id: str) -> Optional[dict]:
        """获取指定客户端的连接信息"""
        c = self._get_client(client_id)
        if c is None:
            return None
        peer_addr = f"{c.addr[0]}:{c.addr[1]}"
        advertised_addr = c.advertised_address or f"{c.advertised_host}:{c.advertised_port}"
        return {
            "client_id": c.client_id,
            # 兼容旧调用：addr 现在表示节点服务端点，不再是 TCP 临时源端口
            "addr": advertised_addr,
            "advertised_addr": advertised_addr,
            "advertised_host": c.advertised_host,
            "advertised_port": c.advertised_port,
            "peer_addr": peer_addr,
            "role": c.role,
            "node_type": c.node_type,
            "hostname": c.hostname,
            "device_info": c.device_info,
            "network_type": c.network_type,
            "connected_at": c.connected_at,
            "last_heartbeat": c.last_heartbeat,
            "heartbeat_missed": c.heartbeat_missed,
            "avg_rtt_ms": round(c.avg_rtt_ms, 3) if c.avg_rtt_ms > 0 else None,
            "last_rtt_ms": round(c.last_rtt_ms, 3) if c.last_rtt_ms > 0 else None,
            "registration_epoch": c.registration_epoch,
        }

    def stop(self) -> None:
        """停止服务端"""
        self._running = False
        with self._clients_lock:
            conns = list(self.clients.values())
            self.clients.clear()
        for conn in conns:
            try:
                conn.sock.close()
            except OSError:
                pass
        for sock in list(getattr(self, "_socks", []) or []):
            try:
                sock.close()
            except OSError:
                pass
        self._socks = []
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        logger.info("TCP服务端已停止")


# ================================================================
# TCP 客户端（从节点）
# ================================================================

_CLIENT_RECEIVE_CONTEXT = threading.local()


class TCPClient:
    """TCP 客户端：从节点使用，连接主节点"""

    MAX_RECONNECT_FAILURES = 8
    MAX_RECONNECT_BACKOFF_SECONDS = 30.0

    # A healthy peer must answer within at least one socket receive-timeout
    # window.  Counting heartbeat periods keeps the watchdog deterministic
    # when recv() repeatedly returns socket.timeout on a half-open connection.
    MAX_HEARTBEAT_ACK_MISSED = max(
        3,
        math.ceil(_CLIENT_SOCKET_TIMEOUT_SECONDS / HEARTBEAT_INTERVAL),
    )

    def __init__(self, server_host: str = None, server_port: int = None,
                 client_id: str = None, role: str = None,
                 node_type: str = "pc",
                 advertise_host: str = None,
                 advertise_port: int = None,
                 device_info: dict = None,
                 transport_runtime: Any = None):
        self.server_host = server_host or SERVER_IP
        self.server_port = server_port or SERVER_PORT
        self.client_id = client_id or f"client_{socket.gethostname()}"
        self.role = role or "client"
        self.node_type = node_type
        self.advertise_host = advertise_host
        self.advertise_port = advertise_port
        self.device_info = dict(device_info or {})
        # Optional Transport v2 runtime seam.  With no bridge injected the
        # established Legacy TCP path remains byte-for-byte unchanged.
        self.transport_runtime = transport_runtime
        self.sock: Optional[socket.socket] = None
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._recv_thread: Optional[threading.Thread] = None
        self.on_message: Optional[Callable] = None
        self.on_heartbeat: Optional[Callable] = None  # 心跳发送后回调（无参数）
        self.on_disconnect: Optional[Callable] = None
        self.on_reconnect: Optional[Callable] = None
        self._disconnect_callback_lock = threading.Lock()
        self._disconnect_notified = False
        self._connection_generation = 0
        self._registration_epoch = 0
        self._registered = False
        self.last_register_error: str = ""
        # 注册确认等待期间，本代连接可能被别的线程（上一代心跳/接收的收尾，或显式
        # disconnect()）关闭或置换。那种情况这一代已经作废，不是一次注册失败。
        self._last_registration_interrupted = False
        self.avg_rtt_ms: float = 0.0            # 滑动平均 RTT（指数加权）
        self._last_heartbeat_send: float = 0.0  # 最近一次心跳发送时间
        self._last_heartbeat_ack_send: float = 0.0
        self._heartbeat_ack_missed: int = 0
        self._heartbeat_quality = HeartbeatQualityWindow()
        self._connect_lock = threading.Lock()   # Phase 5.4: 防止并发 connect()
        self._send_lock = threading.Lock()      # 心跳和推理消息共享同一 TCP 字节流
        self._reconnect_state_lock = threading.Lock()
        self._reconnect_in_progress = False
        self._reconnect_retry_scheduled = False
        self._reconnect_retry_owner = None
        self._reconnect_intent_epoch = 0
        self._reconnect_enabled = True
        self._reconnect_context = threading.local()
        self.reconnect_failure_count = 0
        self.reconnect_state = "idle"
        self.last_reconnect_error = ""
        self.next_reconnect_at = 0.0
        self._control_fence: ControlFence | None = None

    @staticmethod
    def received_registration_owner() -> tuple | None:
        """Return the owner of this thread's current TCP callback, if any."""
        return getattr(_CLIENT_RECEIVE_CONTEXT, "owner", None)

    def capture_registration_owner(self) -> tuple | None:
        """Capture an authenticated (client, epoch, socket) under its publish lock."""
        with self._disconnect_callback_lock:
            if not self._registered or self.sock is None:
                return None
            return (self, self._registration_epoch, self.sock)

    def run_if_registration_owner(self, owner: tuple, action: Callable) -> Any:
        """Run a short state action atomically with registration/retirement.

        The action must not perform I/O, wait for a sender, or acquire the
        reconnect intent lock. Callers take their scheduler owner lock first.
        Socket sends happen afterwards, bound to owner[2], outside this lock.
        """
        with self._disconnect_callback_lock:
            if (owner[0] is not self or not self._registered
                    or owner[1] != self._registration_epoch
                    or owner[2] is not self.sock):
                return None
            return action()

    def set_control_fence(self, fence: ControlFence | None) -> None:
        if fence is not None and not isinstance(fence, ControlFence):
            raise TypeError("control fence must be a ControlFence or None")
        self._control_fence = fence

    def _admit_inbound_control(self, msg: dict) -> None:
        fence = self._control_fence
        msg_type = msg.get("type", "")
        if fence is None or not fence.enabled or not _is_fenced_control_message(msg_type):
            return
        data = msg.get("data")
        certificate = data.get("control_certificate") if isinstance(data, dict) else None
        fence.admit_peer(certificate, action=f"tcp.receive.{msg_type}", observe=True)

    def _transport_runtime_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        runtime = self.transport_runtime
        callback = getattr(runtime, method, None) if runtime is not None else None
        if not callable(callback):
            return None
        try:
            return callback(*args, **kwargs)
        except Exception:
            # Transport telemetry must never take down the established Legacy
            # TCP connection.  Explicit v2 send/receive calls are responsible
            # for surfacing their own contract errors to the caller.
            logger.debug("transport runtime hook failed: %s", method, exc_info=True)
            return None

    def get_transport_snapshot(self) -> dict:
        runtime = self.transport_runtime
        snapshot = getattr(runtime, "snapshot", None) if runtime is not None else None
        if not callable(snapshot):
            return {"mode": "legacy_tcp", "connected": bool(self._running and self.sock)}
        try:
            value = snapshot()
            return dict(value) if isinstance(value, dict) else {"mode": "legacy_tcp"}
        except Exception:
            return {"mode": "legacy_tcp", "snapshot_error": True}

    @staticmethod
    def _compute_local_model_sha256(
        model_path: str = None,
        model_id: str = None,
    ) -> str:
        """
        计算本地模型文件的 SHA256（用于注册时上报）。

        优先级:
        1. 读取已有的 .sha256 缓存文件
        2. 计算实际模型文件的 SHA256
        3. 无法获取时返回空字符串
        """
        import config as cfg
        from model_sync import compute_model_sha256, resolve_worker_model_path

        # 流水线只支持 PyTorch，必须优先上报 Safetensors/BIN 权重摘要。
        explicit_model = bool(model_path or model_id)
        if not model_path and model_id:
            try:
                model_path = resolve_worker_model_path(model_id)
            except ValueError:
                return ""
        model_path = model_path or getattr(cfg, 'MODEL_PATH', '')
        if model_path and os.path.isdir(model_path):
            try:
                result = compute_model_sha256(model_path)
                if result:
                    return result
            except Exception:
                pass

        # 非流水线场景保留 GGUF 摘要作为设备模型画像回退。
        if explicit_model:
            return ""
        gguf_path = getattr(cfg, 'GGUF_MODEL_PATH', '')
        if gguf_path and os.path.isfile(gguf_path):
            sha256_file = gguf_path + ".sha256"
            if os.path.isfile(sha256_file):
                try:
                    with open(sha256_file, "r") as f:
                        return f.read().strip().split()[0]
                except Exception:
                    pass
            try:
                h = hashlib.sha256()
                with open(gguf_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                result = h.hexdigest()
                with open(sha256_file, "w") as f:
                    f.write(f"{result}  {os.path.basename(gguf_path)}\n")
                return result
            except Exception:
                pass

        return ""

    @staticmethod
    def _connect_socket(server_host: str, server_port: int) -> socket.socket:
        """
        建立到主节点的 TCP 连接（自动选择 IPv4/IPv6）。

        使用 getaddrinfo 解析目标（支持域名、v4/v6 字面量、[v6] 方括号形式），
        优先尝试 IPv4 再尝试 IPv6（与旧行为一致，避免双栈下意外改走 v6）。
        全部失败时抛出最后一个异常。
        """
        host = (server_host or "").strip()
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        infos = socket.getaddrinfo(
            host, server_port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP,
        )
        infos.sort(key=lambda info: 0 if info[0] == socket.AF_INET else 1)
        last_error: Optional[OSError] = None
        for family, _socktype, _proto, _canonname, sockaddr in infos:
            sock: Optional[socket.socket] = None
            try:
                sock = socket.socket(family, socket.SOCK_STREAM)
                sock.settimeout(_CLIENT_SOCKET_TIMEOUT_SECONDS)
                sock.connect(sockaddr)
                return sock
            except OSError as e:
                last_error = e
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
        if last_error is not None:
            raise last_error
        raise OSError(f"无法解析主节点地址: {server_host}")

    def connect(self, on_message: Callable = None) -> bool:
        """
        连接主节点并完成注册。

        Args:
            on_message: 收到消息时的回调函数

        Returns:
            连接并注册是否成功
        """
        # Phase 5.4: 互斥锁防止并发连接导致多个 socket + 多组线程竞争
        if not self._connect_lock.acquire(blocking=False):
            logger.warning("connect() 已在执行中，忽略并发调用")
            return False
        try:
            with self._reconnect_state_lock:
                intent_epoch = getattr(self._reconnect_context, "epoch", None)
                if intent_epoch is None:
                    # Only an explicit connect starts a new intent after a manual stop.
                    self._reconnect_intent_epoch += 1
                    intent_epoch = self._reconnect_intent_epoch
                    self._reconnect_enabled = True
                    if self._reconnect_retry_owner is not None:
                        self._reconnect_retry_owner[1].set()
                    self._reconnect_retry_owner = None
                    self._reconnect_retry_scheduled = False
                    self.reconnect_failure_count = 0
                    self.last_reconnect_error = ""
                    self.next_reconnect_at = 0.0
                if not self._reconnect_enabled or intent_epoch != self._reconnect_intent_epoch:
                    return False
                self.reconnect_state = "connecting"
                self.on_message = on_message
                with self._disconnect_callback_lock:
                    self._registered = False
                self.last_register_error = ""
                self._last_registration_interrupted = False

            # 所有可能较慢的注册准备必须在建立 TCP 连接前完成。首次模型
            # 摘要和 Windows 网络检测都可能超过服务端的注册等待超时。
            import platform as _platform
            try:
                network_type = detect_network_type()
            except Exception:
                network_type = "unknown"
                logger.warning("网络类型检测失败，使用 unknown", exc_info=True)
            self._last_network_type = network_type

            registration_device_info = dict(self.device_info)
            if not registration_device_info:
                registration_device_info = {
                    "platform": _platform.system(),
                    "machine": _platform.machine(),
                    "python_version": _platform.python_version(),
                }
            try:
                model_sha256 = self._compute_local_model_sha256()
            except Exception:
                model_sha256 = ""
                logger.warning("本地模型摘要计算失败，注册时暂不上报", exc_info=True)
            hostname = _platform.node()

            for attempt in range(RECONNECT_MAX_RETRIES):
                with self._reconnect_state_lock:
                    if not self._reconnect_enabled or intent_epoch != self._reconnect_intent_epoch:
                        return False
                try:
                    opened_sock = self._connect_socket(self.server_host, self.server_port)
                    with self._reconnect_state_lock:
                        if not self._reconnect_enabled or intent_epoch != self._reconnect_intent_epoch:
                            opened_sock.close()
                            return False
                        with self._disconnect_callback_lock:
                            self.sock = opened_sock
                            self._running = True
                    logger.info(f"已连接主节点: {self.server_host}:{self.server_port}")

                    # 发送注册消息（携带身份、网络类型、服务端点和设备画像）
                    logger.info(f"检测到网络类型: {network_type}，准备向主节点注册")

                    advertised_host = (self.advertise_host or "").strip()
                    if not advertised_host:
                        try:
                            advertised_host = self.sock.getsockname()[0]
                        except Exception:
                            advertised_host = ""
                    if advertised_host in ("", "0.0.0.0", "127.0.0.1", "localhost"):
                        try:
                            advertised_host = detect_lan_ip()
                        except Exception:
                            advertised_host = advertised_host or "127.0.0.1"
                    advertised_port = self.advertise_port or SERVER_PORT
                    try:
                        advertised_port = int(advertised_port)
                    except (TypeError, ValueError):
                        advertised_port = SERVER_PORT
                    if not (1 <= advertised_port <= 65535):
                        advertised_port = SERVER_PORT
                    advertised_address = format_host_port(advertised_host, advertised_port)
                    logger.info(f"注册服务端点: {advertised_address}")

                    reg_data = {
                        "client_id": self.client_id,
                        "role": self.role,
                        "node_type": self.node_type,
                        "hostname": hostname,
                        "network_type": network_type,
                        "advertised_host": advertised_host,
                        "advertised_port": advertised_port,
                        "advertised_address": advertised_address,
                        "device_info": registration_device_info,
                        "model_sha256": model_sha256,
                        "auth": build_auth_signature(self.client_id),
                    }
                    # 本代连接自证：等待注册确认期间，self.sock 可能被别的线程置换
                    # 或关闭（disconnect()／上一代传输收尾）。必须能识别「这一代已作废」，
                    # 而不是当成一次注册失败 —— 否则一次正常断线会被放大成
                    # worker_reconnect_exhausted（2026-10-16 ROG 实测 WinError 10038）。
                    with self._disconnect_callback_lock:
                        register_sock = self.sock
                        register_generation = self._connection_generation
                    self.send_data(reg_data, MessageType.REGISTER)

                    # 等待注册确认
                    register_reason = "未收到注册确认"
                    registered = False
                    registration_interrupted = False
                    if register_sock is None or not self._running:
                        # 连接在发送 REGISTER 前后就被置换/关闭了：不在这条已作废的
                        # 连接上继续等待。
                        registration_interrupted = True
                        register_reason = "注册被中断（连接已由其他线程关闭）"
                    else:
                        try:
                            ack_msg = self.recv_data()
                            if ack_msg and ack_msg.get("type") == "register":
                                ack_data = ack_msg.get("data", {})
                                if ack_data.get("status") == "registered":
                                    registered = True
                                    logger.info(f"✅ 注册确认: {self.client_id} → {self.server_host}:{self.server_port}")
                                else:
                                    status = ack_data.get("status") or "unknown"
                                    reason = ack_data.get("reason") or ack_data.get("message") or status
                                    register_reason = f"注册被拒绝: {reason}"
                            elif ack_msg:
                                register_reason = f"注册确认类型异常: {ack_msg.get('type')}"
                        except (socket.timeout, OSError) as e:
                            with self._disconnect_callback_lock:
                                displaced = (
                                    self.sock is not register_sock
                                    or self._connection_generation != register_generation
                                )
                            registration_interrupted = displaced or not self._running
                            if registration_interrupted:
                                register_reason = "注册被中断（连接已由其他线程关闭）"
                                logger.info(
                                    "注册确认等待被中断: client=%s generation=%s→%s",
                                    self.client_id, register_generation,
                                    self._connection_generation,
                                )
                            else:
                                register_reason = f"注册确认等待失败: {e}"
                                logger.warning(f"注册确认等待超时: {e}", exc_info=True)

                    # A local shutdown can return EOF instead of raising on
                    # Windows. Even a valid ACK can finish after retirement.
                    # Check ownership for every receive outcome, not only the
                    # OSError path, before consuming the registration result.
                    with self._disconnect_callback_lock:
                        displaced = (
                            self.sock is not register_sock
                            or self._connection_generation != register_generation
                        )
                        if displaced or not self._running:
                            registered = False
                            registration_interrupted = True
                            register_reason = "注册被中断（连接已由其他线程关闭）"

                    if not registered:
                        self.last_register_error = register_reason
                        self._last_registration_interrupted = registration_interrupted
                        # Fence worker threads before closing a failed
                        # registration socket. Otherwise an old heartbeat can
                        # write to the closed descriptor and surface WinError
                        # 10038 during a reconnect race.
                        with self._disconnect_callback_lock:
                            if (self.sock is register_sock
                                    and self._connection_generation == register_generation):
                                self._connection_generation += 1
                                self._running = False
                                self._registered = False
                                self.sock = None
                            # Cleanup belongs to this attempt, even when a
                            # successor has already installed another socket.
                            failed_sock = register_sock
                        if failed_sock is not None:
                            try:
                                shutdown = getattr(failed_sock, "shutdown", None)
                                if callable(shutdown):
                                    shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass
                            try:
                                failed_sock.close()
                            except OSError:
                                pass
                        logger.warning(
                            "注册主节点失败: client=%s target=%s:%s reason=%s",
                            self.client_id, self.server_host, self.server_port,
                            register_reason,
                        )
                        if registration_interrupted:
                            # 本代连接已被别的线程置换或关闭：在同一批重试里再连一次
                            # 没有意义，交给上层按正常重连节奏安排下一次，且不计失败。
                            return False
                        return False

                    # Publish REGISTER and start its loops under the intent lock.
                    # A stop during socket connect or REGISTER can never publish a late success.
                    with self._reconnect_state_lock:
                        if not self._reconnect_enabled or intent_epoch != self._reconnect_intent_epoch:
                            opened_sock.close()
                            return False
                        with self._disconnect_callback_lock:
                            # Retirement can also happen between the receive
                            # check above and this publication lock.
                            if (self.sock is not register_sock
                                    or self._connection_generation != register_generation
                                    or not self._running):
                                self.last_register_error = "注册被中断（连接已由其他线程关闭）"
                                self._last_registration_interrupted = True
                                opened_sock.close()
                                return False
                            self._registered = True
                            self._last_registration_interrupted = False
                            self._registration_epoch += 1
                            self._connection_generation += 1
                            connection_generation = self._connection_generation
                            connection_sock = self.sock
                            self._disconnect_notified = False
                            self._last_heartbeat_send = 0.0
                            self._last_heartbeat_ack_send = 0.0
                            self._heartbeat_ack_missed = 0
                            self._heartbeat_quality.begin_generation(connection_generation)
                            self._transport_runtime_call("connected", connection_generation)
                        self.reconnect_failure_count = 0
                        self.reconnect_state = "ready"
                        self.last_reconnect_error = ""
                        self.next_reconnect_at = 0.0
                        self._heartbeat_thread = threading.Thread(
                            target=self._heartbeat_loop,
                            args=(connection_generation, connection_sock), daemon=True,
                        )
                        self._recv_thread = threading.Thread(
                            target=self._recv_loop,
                            args=(connection_generation, connection_sock), daemon=True,
                        )
                        self._heartbeat_thread.start()
                        self._recv_thread.start()
                    return True
                except ConnectionRefusedError:
                    logger.warning(
                        f"连接被拒绝 (尝试 {attempt+1}/{RECONNECT_MAX_RETRIES}) → "
                        f"{self.server_host}:{self.server_port}，{RECONNECT_DELAY}s 后重试..."
                    )
                    time.sleep(RECONNECT_DELAY)
                except socket.timeout:
                    logger.warning(
                        f"连接超时 (尝试 {attempt+1}/{RECONNECT_MAX_RETRIES}) → "
                        f"{self.server_host}:{self.server_port}，{RECONNECT_DELAY}s 后重试..."
                    )
                    time.sleep(RECONNECT_DELAY)
                except OSError as e:
                    logger.warning(
                        f"连接异常 (尝试 {attempt+1}/{RECONNECT_MAX_RETRIES}): {e} → "
                        f"{self.server_host}:{self.server_port}，{RECONNECT_DELAY}s 后重试...",
                        exc_info=True,
                    )
                    time.sleep(RECONNECT_DELAY)

            logger.error(f"无法连接主节点 {self.server_host}:{self.server_port}，已重试 {RECONNECT_MAX_RETRIES} 次")
            return False
        finally:
            self._connect_lock.release()

    def _recv_loop(
        self,
        connection_generation: int = None,
        connection_sock: socket.socket = None,
    ) -> None:
        """接收消息循环（在独立线程中运行）"""
        if connection_sock is None:
            connection_sock = self.sock
        try:
            while self._running and connection_sock:
                with self._disconnect_callback_lock:
                    if ((connection_generation is not None
                         and connection_generation != self._connection_generation)
                            or self.sock is not connection_sock):
                        break
                    received_owner = (self, self._registration_epoch, connection_sock)
                try:
                    msg = self.recv_data(connection_sock)
                    if msg is None:
                        break  # 连接断开
                    # The socket may have been replaced while recv_data blocked.
                    with self._disconnect_callback_lock:
                        if (self.sock is not connection_sock
                                or self._registration_epoch != received_owner[1]
                                or (connection_generation is not None
                                    and connection_generation != self._connection_generation)):
                            break
                    self._transport_runtime_call(
                        "observe_legacy_receive",
                        msg,
                        connection_generation=connection_generation,
                    )
                    # 内部处理：HEARTBEAT_ACK → 计算 RTT
                    if msg.get("type") == MessageType.HEARTBEAT_ACK.value:
                        self._handle_heartbeat_ack(msg, connection_generation)
                        continue  # 不向上层转发
                    try:
                        self._admit_inbound_control(msg)
                    except ControlFenceError as exc:
                        logger.warning("TCP control frame rejected code=%s type=%s", exc.code, msg.get("type"))
                        continue
                    if self.on_message:
                        previous_owner = getattr(_CLIENT_RECEIVE_CONTEXT, "owner", None)
                        _CLIENT_RECEIVE_CONTEXT.owner = received_owner
                        try:
                            self.on_message(msg)
                        except Exception as e:
                            logger.error(f"消息回调异常: {e}", exc_info=True)
                        finally:
                            _CLIENT_RECEIVE_CONTEXT.owner = previous_owner
                except socket.timeout:
                    continue
                except (ConnectionError, OSError) as e:
                    if not self._running:
                        break
                    self._transport_runtime_call("record_failure", e)
                    logger.warning(
                        f"客户端接收循环连接异常: {self.client_id} "
                        f"→ {self.server_host}:{self.server_port}, error={e}",
                        exc_info=True,
                    )
                    break
                except Exception as e:
                    # 帧解析/张量反序列化等未预期异常：不能让接收线程
                    # 静默死亡（否则连接成为僵尸），按连接故障断开处理
                    logger.error(
                        f"客户端接收循环未预期异常，按断连处理: {self.client_id} "
                        f"→ {self.server_host}:{self.server_port}, error={e}",
                        exc_info=True,
                    )
                    break
        finally:
            # 无论以何种方式退出循环，都必须完成清理并通知断连，
            # 否则上层健康监控无法感知断开、不会触发重连
            self._notify_disconnect(connection_generation, connection_sock=connection_sock)
            with self._disconnect_callback_lock:
                is_current = (
                    connection_generation is None
                    or connection_generation == self._connection_generation
                )
                retired_current = is_current and self.sock is connection_sock
                if retired_current:
                    self.sock = None
                    self._running = False
            if connection_sock is not None:
                try:
                    connection_sock.close()
                except OSError:
                    pass
            if retired_current:
                self._schedule_reconnect_retry()
            logger.info(f"接收循环已退出: {self.client_id}")

    def _notify_disconnect(
        self,
        connection_generation: int = None,
        connection_sock: socket.socket = None,
    ) -> None:
        with self._disconnect_callback_lock:
            if (connection_generation is not None
                    and connection_generation != self._connection_generation):
                return
            if connection_sock is not None and self.sock is not connection_sock:
                return
            if self._disconnect_notified:
                return
            self._disconnect_notified = True
            self._registered = False
            active_generation = self._connection_generation
        self._heartbeat_quality.record_disconnect(active_generation)
        if self.on_disconnect:
            try:
                self.on_disconnect()
            except Exception as exc:
                logger.debug("客户端断连回调异常: %s", exc, exc_info=True)

    def send_data(self, data: Any, msg_type: MessageType = MessageType.TENSOR,
                  connection_sock: socket.socket = None,
                  binary_body: Optional[bytes] = None) -> None:
        """向主节点发送数据"""
        data = _decorate_control_payload(data, msg_type, self._control_fence, peer=True)
        packet = build_message(msg_type, data, binary_body=binary_body)
        self._transport_runtime_call(
            "observe_legacy_send",
            packet,
            request_id=self.client_id,
            connection_generation=self._connection_generation,
        )
        with self._send_lock:
            active_sock = self.sock if connection_sock is None else connection_sock
            if active_sock is None:
                raise ConnectionError("未连接到主节点")
            # A worker that already captured a socket must never write after
            # reconnect moved ownership to another generation.  The caller can
            # then fence itself instead of surfacing an unclassified WinError.
            if connection_sock is not None and self.sock is not connection_sock:
                raise ConnectionError("连接已被新代际替换")
            try:
                active_sock.sendall(packet)
            except OSError as exc:
                self._transport_runtime_call("record_failure", exc)
                raise ConnectionError("主节点连接发送失败") from exc

    def recv_data(self, connection_sock: socket.socket = None) -> Optional[dict]:
        """
        接收主节点发来的数据（自动解包、还原张量/字符串）。

        Returns:
            解析后的消息字典，连接断开时返回 None
        """
        active_sock = connection_sock or self.sock
        if not active_sock:
            return None

        # 接收长度头
        header = recv_exact(active_sock, HEADER_LEN)
        if header is None:
            return None
        payload_len = unpack_header(header)
        if payload_len <= 0 or payload_len > MAX_PACKET_SIZE:
            raise ConnectionError(f"invalid packet length: {payload_len}")

        # 接收数据体
        payload = recv_exact(active_sock, payload_len, mid_frame=True)
        if payload is None:
            return None

        msg = parse_message(payload)

        # 如果消息携带张量，继续接收张量数据
        if msg.get("_needs_tensor"):
            tensor_header = recv_exact(active_sock, HEADER_LEN, mid_frame=True)
            if tensor_header is None:
                return None
            tensor_len = unpack_header(tensor_header)
            if tensor_len <= 0 or tensor_len > MAX_PACKET_SIZE:
                raise ConnectionError(
                    f"invalid tensor packet length: {tensor_len}"
                )
            tensor_data = recv_exact(active_sock, tensor_len, mid_frame=True)
            if tensor_data is None:
                return None
            msg["tensor"] = deserialize_tensor(tensor_data)
            del msg["_needs_tensor"]
        # ★ 票 21a：二进制体在同一帧的后续段里（raw 字节，不做反序列化）。
        elif msg.get("_needs_binary"):
            binary_header = recv_exact(active_sock, HEADER_LEN, mid_frame=True)
            if binary_header is None:
                return None
            binary_len = unpack_header(binary_header)
            _validate_binary_length(msg, binary_len, payload_len)
            binary_data = recv_exact(active_sock, binary_len, mid_frame=True)
            if binary_data is None:
                return None
            msg["binary"] = binary_data
            del msg["_needs_binary"]

        return msg

    def _handle_heartbeat_ack(
        self,
        msg: dict,
        connection_generation: int = None,
    ) -> None:
        """
        处理心跳应答：计算往返延迟 RTT。

        服务端在 HEARTBEAT_ACK 中回显客户端发送的时间戳，
        客户端收到后计算 RTT = now - t_send，并用指数加权滑动平均平滑。
        """
        data = msg.get("data", {})
        has_echo = isinstance(data, dict) and "t_send" in data
        t_echo = data.get("t_send") if has_echo else self._last_heartbeat_send
        t_now = time.time()
        generation = (
            self._connection_generation
            if connection_generation is None
            else connection_generation
        )
        with self._disconnect_callback_lock:
            if generation != self._connection_generation:
                return
            rtt_ms = self._heartbeat_quality.record_ack(
                generation, t_echo, t_now,
            )
            if rtt_ms is None:
                return
            self._last_heartbeat_ack_send = float(t_echo)
            self._heartbeat_ack_missed = 0

            # 指数加权滑动平均（α=0.1），平滑网络抖动
            if self.avg_rtt_ms > 0:
                self.avg_rtt_ms = 0.9 * self.avg_rtt_ms + 0.1 * rtt_ms
            else:
                self.avg_rtt_ms = rtt_ms

    def _heartbeat_loop(
        self,
        connection_generation: int = None,
        connection_sock: socket.socket = None,
    ) -> None:
        """心跳发送循环（含时间戳用于 RTT 测量）"""
        if connection_sock is None:
            connection_sock = self.sock
        while self._running and connection_sock:
            with self._disconnect_callback_lock:
                if (connection_generation is not None
                        and connection_generation != self._connection_generation):
                    break
                if self.sock is not connection_sock:
                    break
                generation = (
                    self._connection_generation
                    if connection_generation is None
                    else connection_generation
                )
                if (
                    self._last_heartbeat_send > 0
                    and self._last_heartbeat_ack_send
                    != self._last_heartbeat_send
                ):
                    self._heartbeat_ack_missed += 1
                else:
                    self._heartbeat_ack_missed = 0
                heartbeat_ack_timed_out = (
                    self._heartbeat_ack_missed
                    >= self.MAX_HEARTBEAT_ACK_MISSED
                )
                if not heartbeat_ack_timed_out:
                    self._last_heartbeat_send = time.time()
                    self._heartbeat_quality.record_send(
                        generation,
                        self._last_heartbeat_send,
                    )

            if heartbeat_ack_timed_out:
                logger.warning(
                    "heartbeat ACK timed out: client=%s generation=%s "
                    "missed=%s; reconnecting",
                    self.client_id, generation, self._heartbeat_ack_missed,
                )
                self._notify_disconnect(
                    generation,
                    connection_sock=connection_sock,
                )
                with self._disconnect_callback_lock:
                    should_reconnect = bool(
                        self._running
                        and generation == self._connection_generation
                        and self.sock is connection_sock
                    )
                if should_reconnect:
                    self._reconnect()
                break
            try:
                heartbeat_data = {"t_send": self._last_heartbeat_send}
                if self.avg_rtt_ms > 0:
                    heartbeat_data["rtt_ms"] = round(self.avg_rtt_ms, 3)
                self.send_data(
                    heartbeat_data,
                    MessageType.HEARTBEAT,
                    connection_sock=connection_sock,
                )
                if self.on_heartbeat:
                    try:
                        self.on_heartbeat()
                    except Exception as e:
                        logger.debug(f"on_heartbeat 回调异常: {e}", exc_info=True)
                time.sleep(HEARTBEAT_INTERVAL)
            except (ConnectionError, OSError) as e:
                with self._disconnect_callback_lock:
                    if (connection_generation is not None
                            and connection_generation != self._connection_generation):
                        break
                    if self.sock is not connection_sock:
                        break
                logger.warning(f"心跳发送失败，尝试重连: {e}", exc_info=True)
                self._reconnect()

    def _schedule_reconnect_retry(self) -> None:
        """One waiting owner; transfer to an active owner atomically under the same lock."""
        with self._reconnect_state_lock:
            caller_epoch = getattr(self._reconnect_context, "epoch", None)
            if (caller_epoch is not None and caller_epoch != self._reconnect_intent_epoch):
                return
            if (not self._reconnect_enabled or self._reconnect_in_progress
                    or self._reconnect_retry_owner is not None or self._registered
                    or self.reconnect_failure_count >= self.MAX_RECONNECT_FAILURES):
                return
            epoch = self._reconnect_intent_epoch
            wake = threading.Event()
            owner = (epoch, wake)
            self._reconnect_retry_owner = owner
            self._reconnect_retry_scheduled = True
            delay = min(
                RECONNECT_DELAY * 2 ** max(0, self.reconnect_failure_count - 1),
                self.MAX_RECONNECT_BACKOFF_SECONDS,
            )
            self.next_reconnect_at = time.time() + delay
            self.reconnect_state = "backing_off"

        def _retry() -> None:
            try:
                if not wake.wait(delay):
                    self._reconnect(_intent_epoch=epoch, _retry_owner=owner)
            finally:
                with self._reconnect_state_lock:
                    # Never clear the successor's ownership after handing this one off.
                    if self._reconnect_retry_owner is owner:
                        self._reconnect_retry_owner = None
                        self._reconnect_retry_scheduled = False

        threading.Thread(
            target=_retry, name=f"tcp-reconnect-{self.client_id}", daemon=True,
        ).start()

    def _reconnect(self, *, _intent_epoch=None, _retry_owner=None) -> None:
        """Retire the old generation, reconnect once, then hand failure to one delayed owner."""
        with self._reconnect_state_lock:
            epoch = self._reconnect_intent_epoch if _intent_epoch is None else _intent_epoch
            if (not self._reconnect_enabled or epoch != self._reconnect_intent_epoch
                    or self._reconnect_in_progress
                    or self.reconnect_failure_count >= self.MAX_RECONNECT_FAILURES):
                return
            if _retry_owner is not None:
                if self._reconnect_retry_owner is not _retry_owner:
                    return
                self._reconnect_retry_owner = None
                self._reconnect_retry_scheduled = False
            elif self._reconnect_retry_owner is not None:
                return
            # Clearing waiting ownership and claiming active ownership are one atomic operation.
            self._reconnect_in_progress = True
            registration_epoch_before = self._registration_epoch
            self.reconnect_state = "connecting"
            self.next_reconnect_at = 0.0
            with self._disconnect_callback_lock:
                self._connection_generation += 1
                self._running = False
                self._registered = False
                old_sock = self.sock
                self.sock = None
        retry = False
        try:
            logger.info("开始重连主节点: client=%s target=%s:%s intent_epoch=%s",
                        self.client_id, self.server_host, self.server_port, epoch)
            if old_sock is not None:
                try:
                    old_sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    old_sock.close()
                except OSError:
                    pass
            # Thread-local context preserves connect(on_message)'s public and injected contract.
            self._reconnect_context.epoch = epoch
            ok = self.connect(self.on_message)
            with self._reconnect_state_lock:
                if not self._reconnect_enabled or epoch != self._reconnect_intent_epoch:
                    # A newer explicit connect owns its own socket. Never retire that owner.
                    # Production connect fences its unpublished socket before returning.
                    if not self._reconnect_enabled:
                        with self._disconnect_callback_lock:
                            late_sock = self.sock
                            self.sock = None
                            self._running = self._registered = False
                        if late_sock is not None:
                            try:
                                late_sock.close()
                            except OSError:
                                pass
                    return
                if ok:
                    self.reconnect_failure_count = 0
                    self.reconnect_state = "ready"
                    self.last_reconnect_error = ""
                    logger.info("重连主节点成功: client=%s intent_epoch=%s", self.client_id, epoch)
                elif self._last_registration_interrupted:
                    # 连接在注册等待期间被主动置换或关闭：这不是一次注册失败，
                    # 不计入 reconnect_failure_count —— 计进去会把一次正常断线放大成
                    # worker_reconnect_exhausted（2026-10-16 ROG 实测：failures 累到 8
                    # 后进入每分钟一次的低频重试）。
                    # 但这一代确实作废了，必须自己安排下一次重连：中断路径不会发布
                    # 连接，lost_published_connection 恒为 False，不能指望那条兜底。
                    retry = True
                    self.reconnect_state = "backing_off"
                    self.last_reconnect_error = "registration_interrupted"
                    logger.info("重连未完成（注册被中断）: client=%s reason=%s",
                                self.client_id, self.last_register_error)
                else:
                    self.reconnect_failure_count += 1
                    retry = self.reconnect_failure_count < self.MAX_RECONNECT_FAILURES
                    self.reconnect_state = "backing_off" if retry else "retry_exhausted"
                    self.last_reconnect_error = "worker_reconnect_failed" if retry else "worker_reconnect_exhausted"
                    logger.error("%s: client=%s failures=%s registration_reason=%s",
                                 self.last_reconnect_error, self.client_id,
                                 self.reconnect_failure_count, self.last_register_error)
            # Transport registration alone does not restore the upper worker protocol.
            # Run outside the intent lock so an owner can schedule its fenced hello.
            if ok and self._registered and self.on_reconnect:
                try:
                    self.on_reconnect()
                except Exception:
                    logger.warning("event=tcp_reconnect_callback_failed client=%s",
                                   self.client_id, exc_info=True)
        finally:
            with self._reconnect_state_lock:
                self._reconnect_in_progress = False
                with self._disconnect_callback_lock:
                    lost_published_connection = (
                        self._registration_epoch > registration_epoch_before
                        and not self._registered
                    )
                # EOF can retire a newly accepted REGISTER while this owner is still active.
                # Its scheduling request was suppressed above; hand it off after releasing
                # active ownership, but only within this same, still-enabled connect intent.
                if (self._reconnect_enabled and epoch == self._reconnect_intent_epoch
                        and lost_published_connection):
                    retry = True
            # Keep this caller's intent until its successor is scheduled. A stopped/superseded
            # failure must not enqueue a retry into a newer explicit connect intent.
            try:
                if retry:
                    self._schedule_reconnect_retry()
            finally:
                self._reconnect_context.epoch = None

    @property
    def is_registered(self) -> bool:
        """是否已完成注册"""
        return self._registered

    @property
    def registration_epoch(self) -> int:
        """Monotonic local epoch for the current TCP registration identity."""
        return int(self._registration_epoch)

    def get_network_quality_snapshot(self) -> dict:
        """Return local heartbeat aggregates without endpoint data."""
        snapshot = self._heartbeat_quality.snapshot()
        snapshot["avg_rtt_ms"] = (
            round(self.avg_rtt_ms, 3) if self.avg_rtt_ms > 0 else None
        )
        return snapshot

    def disconnect(self) -> None:
        """断开连接"""
        # Advance the generation before closing so delayed receive/heartbeat
        # workers cannot operate on this socket after an explicit disconnect.
        with self._reconnect_state_lock:
            self._reconnect_intent_epoch += 1
            self._reconnect_enabled = False
            self.reconnect_state = "stopped"
            self.next_reconnect_at = 0.0
            if self._reconnect_retry_owner is not None:
                self._reconnect_retry_owner[1].set()
            self._reconnect_retry_owner = None
            self._reconnect_retry_scheduled = False
            with self._disconnect_callback_lock:
                self._connection_generation += 1
                self._running = False
                self._registered = False
                active_sock = self.sock
                self.sock = None
        if active_sock is not None:
            try:
                shutdown = getattr(active_sock, "shutdown", None)
                if callable(shutdown):
                    shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                active_sock.close()
            except OSError:
                pass
        self._notify_disconnect()
        logger.info("已断开主节点连接")
