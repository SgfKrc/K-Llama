"""Routes extracted from api_server; shared state remains facade-owned."""

from __future__ import annotations

from types import ModuleType

from fastapi import APIRouter
from api._routing import configure_route_module

router = APIRouter()
_api_module: ModuleType | None = None
_RESOLUTION_NAMES = (
    "AndroidPresenceRequest",
    "ClusterStatus",
    "ConnectToMasterRequest",
    "DistributedInferenceRequest",
    "FirstConnectBootstrapRequest",
    "LayerOverrideRequest",
    "Literal",
    "ManualRegisterRequest",
    "ModelRuntimeContractBindRequest",
    "ModelRuntimeSidecarActionRequest",
    "ModelRuntimeSidecarBeginRequest",
    "Qwen3LocalChainBeginRequest",
    "Qwen3LocalChainExecuteRequest",
    "Qwen3LocalChainParityRequest",
    "Request",
    "ResetIdentityRequest",
    "UpdateMaxNodesRequest",
)

def configure_api_module(module: ModuleType) -> None:
    configure_route_module(globals(), module, _RESOLUTION_NAMES)


def _require_static_master(action: str) -> None:
    if _api_module.scheduler._effective_role() != "master":
        raise _api_module.HTTPException(
            403,
            {"code": "not_master", "message": f"仅静态主节点可执行: {action}"},
        )

def exported_handlers() -> dict[str, object]:
    names = (
        "get_cluster_status", "get_cluster_nodes", "get_cluster_resources",
        "deregister_node", "delete_cluster_node", "get_cluster_config",
        "get_my_role", "update_max_nodes", "get_invite_info",
        "first_connect_bootstrap", "bootstrap_info", "connect_to_master",
        "manual_register_node", "register_android_presence",
        "heartbeat_android_presence", "check_master_health", "reset_master_identity", "discover_master",
        "get_control_plane_status",
        "get_queue_detail", "pause_queue", "resume_queue", "clear_queue",
        "cancel_queue_task", "get_distributed_inference_config",
        "set_distributed_inference_config", "get_layer_assignments",
        "get_pipeline_capacity_plan", "get_pipeline_reshard_status",
        "override_layer_assignments", "reset_layer_assignments",
        "get_model_runtime_sidecar_status", "get_model_runtime_contracts",
        "bind_model_runtime_contract", "begin_model_runtime_sidecar",
        "release_model_runtime_sidecar", "cancel_model_runtime_sidecar",
        "get_qwen3_local_chain_status", "begin_qwen3_local_chain",
        "run_qwen3_local_prefill", "run_qwen3_local_decode",
        "verify_qwen3_local_parity", "release_qwen3_local_chain",
        "cancel_qwen3_local_chain",
    )
    return {name: globals()[name] for name in names}

async def get_cluster_status():
    """
    获取集群整体状态。

    包含所有节点状态、TCP 连接信息、当前任务等。
    单机模式下返回 3 个默认节点（均为 online）。
    """
    return await _api_module.run_in_threadpool(_api_module.scheduler.get_status)

async def get_cluster_nodes():
    """
    获取所有节点详情列表。

    Returns:
        { nodes: [...], count: int, online_count: int }
    """
    nodes = await _api_module.run_in_threadpool(_api_module.scheduler.get_nodes)
    online_count = sum(1 for n in nodes if n["is_available"])
    return {
        "nodes": nodes,
        "count": len(nodes),
        "online_count": online_count,
        "offline_count": len(nodes) - online_count,
    }

async def get_cluster_resources():
    """Return the read-only aggregate CPU, RAM, and GPU resource view."""
    return await _api_module.run_in_threadpool(_api_module.scheduler.get_aggregate_resource_view)

async def deregister_node(node_id: str):
    """
    强制注销一个从节点。

    仅在分布式模式下有效；master 节点不可注销。
    """
    _require_static_master("deregister node")
    if node_id == "master":
        raise _api_module.HTTPException(400, "主节点不可注销")

    success = _api_module.scheduler.deregister_node(node_id)
    if not success:
        raise _api_module.HTTPException(404, f"节点 '{node_id}' 不存在")

    _api_module.logger.info(f"节点 {node_id} 已被强制注销")
    return {
        "status": "deregistered",
        "node_id": node_id,
    }

async def delete_cluster_node(node_id: str):
    """
    删除离线节点记录（区别于 deregister：deregister 仅标记离线）。

    用于移除手动注册的 Android / 离线占位节点。
    """
    _require_static_master("delete node")
    result = _api_module.scheduler.delete_node(node_id)
    status = result.get("status")
    if status == "denied":
        raise _api_module.HTTPException(403, result.get("reason", "权限不足"))
    if status == "invalid":
        raise _api_module.HTTPException(400, result.get("reason", "无效节点"))
    if status == "not_found":
        raise _api_module.HTTPException(404, result.get("reason", "节点不存在"))
    if status == "online":
        raise _api_module.HTTPException(409, result.get("reason", "节点在线，无法删除"))
    if status != "deleted":
        raise _api_module.HTTPException(500, result.get("reason", "删除节点失败"))
    return result

async def get_cluster_config():
    """
    获取分布式配置信息。

    包含网络配置、分层配置、模型配置、任务统计、当前节点角色。
    """
    return await _api_module.run_in_threadpool(_api_module.scheduler.get_config)

async def get_my_role():
    """
    获取当前节点的角色信息。

    用于前端判断：
    - master 节点：后台管理 Tab 完全开放
    - client 节点：需在设置中开启"分布式推理优化"后才可见
    """
    return await _api_module.run_in_threadpool(_api_module.scheduler.get_my_role)

async def update_max_nodes(req: UpdateMaxNodesRequest, request: Request):
    """
    动态调整最大节点数量（仅主节点可调用）。

    仅修改容量上限，不预创建空槽位。从节点通过 TCP 注册动态加入。
    """
    _require_static_master("update cluster capacity")
    result = _api_module.scheduler.update_max_nodes(req.max_nodes)
    if result.get("status") == "denied":
        raise _api_module.HTTPException(403, result.get("reason", "权限不足"))
    if result.get("status") == "invalid":
        raise _api_module.HTTPException(400, result.get("reason", "无效参数"))
    return result

async def get_invite_info():
    """
    获取主节点的邀请/连接信息（供从节点连接使用）。

    主节点调用此接口获取自身监听地址和端口，
    用户将此信息提供给从节点，从节点在后台管理中输入并连接。
    """
    return _api_module.scheduler.get_invite_info()

async def first_connect_bootstrap(req: FirstConnectBootstrapRequest, request: Request):
    """
    首次连接自动部署。

    安全边界：只接受 Tailscale / 受信 CIDR 来源。通过该接口下发集群密钥
    和主节点连接信息，客户端持久化后再走现有 TCP HMAC 注册。
    """
    if _api_module.os.environ.get("QLH_BOOTSTRAP_ENABLED", "true").strip().lower() in {"0", "false", "no"}:
        raise _api_module.HTTPException(403, "bootstrap disabled")

    peer_host = request.client.host if request.client else ""
    from bootstrap import (
        is_secure_bootstrap_source,
        is_trusted_bootstrap_source,
        normalize_node_id,
        normalize_node_type,
    )

    require_trusted = _api_module.os.environ.get("QLH_BOOTSTRAP_REQUIRE_TAILSCALE", "true").strip().lower()
    if require_trusted not in {"0", "false", "no"}:
        if not is_trusted_bootstrap_source(peer_host):
            raise _api_module.HTTPException(403, "source network is not trusted")
    if not is_secure_bootstrap_source(peer_host):
        raise _api_module.HTTPException(
            403,
            {
                "code": "protected_bootstrap_transport_required",
                "message": "credential bootstrap requires loopback or Tailnet overlay",
            },
        )

    if _api_module.scheduler._effective_role() != "master":
        raise _api_module.HTTPException(403, "only master can serve bootstrap")

    node_type = normalize_node_type(req.node_type)
    node_id = normalize_node_id(req.node_id, node_type)
    if node_id == "master":
        raise _api_module.HTTPException(400, "reserved node_id")

    from bootstrap_credentials import BootstrapCredentialError, seal_bootstrap_credential
    cluster_secret, cluster_secret_epoch = _api_module._bootstrap_cluster_credential()

    api_host = request.url.hostname or peer_host
    lan_ip = getattr(_api_module.scheduler, "_lan_ip", "") or ""
    from bootstrap import select_advertised_master_host
    from network_address import build_url, is_tailscale_ip

    master_tcp_host = select_advertised_master_host(api_host, lan_ip)
    master_api_host = api_host or master_tcp_host
    runtime_config = _api_module._get_app_runtime().resolve_config()
    master_api_port = request.url.port or runtime_config.api_port
    master_tcp_port = _api_module.scheduler.tcp_server.port if _api_module.scheduler.tcp_server else runtime_config.server_port

    cluster_id = runtime_config.environment_snapshot().get("QLH_CLUSTER_ID", "qlh-default")
    try:
        credential_envelope = seal_bootstrap_credential(
            public_key=req.credential_public_key,
            request_nonce=req.credential_request_nonce,
            requested_at=req.credential_requested_at,
            cluster_id=cluster_id,
            node_id=node_id,
            cluster_secret=cluster_secret,
            secret_epoch=cluster_secret_epoch,
        )
        import hashlib
        import json

        request_digest = hashlib.sha256(json.dumps(
            {
                "credential_public_key": req.credential_public_key,
                "credential_request_nonce": req.credential_request_nonce,
                "credential_requested_at": req.credential_requested_at,
                "node_id": node_id,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")).hexdigest()
        _api_module._get_bootstrap_nonce_ledger().consume(
            nonce=req.credential_request_nonce,
            request_digest=request_digest,
        )
    except BootstrapCredentialError as exc:
        status = 409 if exc.code in {"request_expired"} else 400
        raise _api_module.HTTPException(
            status, {"code": exc.code, "message": str(exc)}
        ) from exc
    except _api_module.BootstrapNonceError as exc:
        status = 409 if exc.code == "nonce_replayed" else 400
        raise _api_module.HTTPException(
            status, {"code": exc.code, "message": str(exc)}
        ) from exc

    hostname = req.hostname or node_id
    address = f"{peer_host}" if peer_host else ""
    register_result = _api_module.scheduler.manual_register_node(
        node_id=node_id,
        hostname=hostname,
        address=address,
        network_type="tailscale" if is_tailscale_ip(peer_host) else "trusted",
        node_type=node_type,
    )
    if register_result.get("status") in {"denied", "invalid", "full"}:
        status_code = 403 if register_result.get("status") == "denied" else 400
        raise _api_module.HTTPException(status_code, register_result.get("reason", "bootstrap registration failed"))

    # ★ 2026-09-19：不再按平台一刀切。Android 不能跑 PyTorch，**但能跑 llama.cpp/GGUF 引擎**，
    #   而 llama.cpp 现在也能做层前向（`forward_layers_from_hidden` / `forward_layers_to_hidden`）
    #   ⇒ 只要客户端**自报**了 `FORWARD_LAYERS` 能力，就承认它可以当流水线工作器；
    #   **未自报者退回旧行为（仅 pc）**，保证不会无意放行。
    pipeline_worker = _api_module._client_supports_forward_layers(node_type, req.capabilities)
    response = {
        "status": "ok",
        "cluster": {
            "cluster_id": cluster_id,
            "master_api_host": master_api_host,
            "master_api_port": master_api_port,
            "master_tcp_host": master_tcp_host,
            "master_tcp_port": master_tcp_port,
            "cluster_secret_epoch": cluster_secret_epoch,
        },
        "node": {
            "node_id": node_id,
            "role": "client",
            "node_type": node_type,
            "pipeline_worker": pipeline_worker,
        },
        "android": {
            "presence_interval_seconds": 45,
            # ★ 与此节点的实际能力一致（自报 FORWARD_LAYERS 的 Android 可为 True）。
            "pipeline_worker": pipeline_worker,
            "model_manifest_url": build_url(
                "http", master_api_host, master_api_port, "/api/models/downloadable"
            ),
        },
        "credential_envelope": credential_envelope,
    }
    _api_module.logger.info(
        "首次连接部署: node_id=%s type=%s peer=%s host=%s api=%s:%s tcp=%s:%s",
        node_id, node_type, peer_host, hostname,
        master_api_host, master_api_port, master_tcp_host, master_tcp_port,
    )
    return response

async def bootstrap_info(request: Request):
    """Minimal discovery endpoint for peers already admitted to the Tailnet."""
    peer_host = request.client.host if request.client else ""
    from bootstrap import is_trusted_bootstrap_source

    if not is_trusted_bootstrap_source(peer_host):
        raise _api_module.HTTPException(403, "source network is not trusted")
    role = _api_module.scheduler.get_my_role()
    return {
        "status": "ok",
        "is_master": bool(role.get("is_master")),
        "node_id": role.get("node_id", ""),
        "master_api_port": _api_module.API_PORT,
        "master_tcp_port": _api_module.scheduler.tcp_server.port if _api_module.scheduler.tcp_server else _api_module.SERVER_PORT,
    }

async def connect_to_master(req: ConnectToMasterRequest):
    """
    从节点主动连接主节点（从节点的「连接主节点」按钮触发）。

    调用后本节点将通过 TCP 向指定主节点发起注册，
    注册成功后主节点的节点列表中将出现本节点。
    """
    if _api_module.scheduler._effective_role() == "master":
        raise _api_module.HTTPException(403, "静态主节点不能切换为从节点")

    # connect_to_master 内部含多次 TCP 重试（最长可达数十秒），
    # 必须放入线程池执行，否则会阻塞事件循环冻结所有 HTTP 接口
    result = await _api_module.run_in_threadpool(
        _api_module.scheduler.connect_to_master,
        req.master_host,
        req.master_port,
        persist_preference=True,
    )
    if result.get("status") == "denied":
        raise _api_module.HTTPException(403, result.get("reason", "仅从节点可连接主节点"))
    if result.get("status") == "bootstrap_failed":
        raise _api_module.HTTPException(400, result.get("reason", "首次连接自动部署失败"))
    if result.get("status") == "failed":
        raise _api_module.HTTPException(400, result.get("reason", "连接失败"))
    if result.get("status") == "error":
        raise _api_module.HTTPException(500, result.get("reason", "连接异常"))
    return result

async def manual_register_node(req: ManualRegisterRequest):
    """
    主节点手动注册一个从节点（无需 TCP 连接）。

    管理员可在后台管理页面提前录入从节点信息。
    手动注册的节点初始状态为 offline，待从节点通过 TCP 连接后自动变为 online。

    如果从节点主动通过「连接主节点」发起 TCP 注册，也会自动加入节点列表，
    无需手动注册。此接口用于管理员提前规划节点或预留槽位。
    """
    result = _api_module.scheduler.manual_register_node(
        node_id=req.node_id,
        hostname=req.hostname,
        address=req.address,
        network_type=req.network_type,
        node_type=req.node_type,
    )
    if result.get("status") == "denied":
        raise _api_module.HTTPException(403, result.get("reason", "仅主节点可手动注册"))
    if result.get("status") == "invalid":
        raise _api_module.HTTPException(400, result.get("reason", "无效参数"))
    if result.get("status") == "full":
        raise _api_module.HTTPException(400, result.get("reason", "节点容量已满"))
    if result.get("status") == "exists":
        return result  # 已存在不报错，返回当前状态
    return result

async def register_android_presence(req: AndroidPresenceRequest, request: Request):
    """Android Full 薄客户端在线登记/心跳（不是 TCP worker 注册）。"""
    http_peer = request.client.host if request.client else ""
    result = _api_module.scheduler.register_android_client(
        node_id=req.node_id,
        hostname=req.hostname,
        address=req.address,
        network_type=req.network_type,
        device_info=req.device_info,
        client_mode=req.client_mode,
        app_variant=req.app_variant,
        app_version=req.app_version,
        http_peer=http_peer,
    )
    if result.get("status") == "denied":
        raise _api_module.HTTPException(403, result.get("reason", "仅主节点可登记 Android 客户端"))
    if result.get("status") == "invalid":
        raise _api_module.HTTPException(400, result.get("reason", "无效 Android 节点"))
    return result

async def heartbeat_android_presence(req: AndroidPresenceRequest, request: Request):
    """Refresh an Android presence lease; this endpoint never re-registers a node."""
    http_peer = request.client.host if request.client else ""
    result = _api_module.scheduler.heartbeat_android_client(
        node_id=req.node_id,
        presence_generation=req.presence_generation,
        presence_lease_id=req.presence_lease_id,
        http_peer=http_peer,
    )
    if result.get("status") == "denied":
        raise _api_module.coded_http_error(403, result.get("error_code", "not_master"), result.get("reason", "仅主节点可接收 Android 心跳"))
    if result.get("status") == "invalid":
        raise _api_module.coded_http_error(400, result.get("error_code", "invalid_node_id"), result.get("reason", "无效 Android 节点"))
    if result.get("status") == "rejected":
        raise _api_module.coded_http_error(409, result.get("error_code", "presence_rejected"), result.get("reason", "Android presence 被拒绝"))
    return result

async def check_master_health():
    """
    检查主节点是否在线（通过数据库心跳时间戳）。

    从节点前端周期性调用此接口（配合 5 秒轮询），
    当检测到主节点宕机时显示告警横幅。
    主节点自身调用时返回本地运行状态。

    Returns:
        { master_online, last_seen_seconds_ago, stale, master_host, master_port }
    """
    if _api_module.scheduler._effective_role() == "master":
        # 主节点自身：直接返回在线
        return {
            "master_online": True,
            "last_seen_seconds_ago": 0,
            "stale": False,
            "master_host": getattr(_api_module.scheduler, '_lan_ip', '') or _api_module.SERVER_IP,
            "master_port": _api_module.SERVER_PORT,
            "source": "self",
        }
    return _api_module.scheduler.get_client_master_status()

async def discover_master():
    """
    发现主节点的连接信息（从节点自动发现）。

    已连接节点优先使用本机 bootstrap 配置；未保存配置时，通过同一
    Tailnet 探测主节点。旧远端数据库仅为兼容回退，前端可自动填充
    连接表单。

     Returns:
         {
             "found": bool,           # 是否找到主节点
             "master_host": str,      # 主节点 IP
             "master_port": int,      # 主节点端口
             "master_mac_addresses": [str],  # 主节点 MAC 地址（身份标识）
             "stale": bool,           # 心跳是否过期 (>120s)
             "source": str,           # "config" | "tailnet" | "none"
         }
    """
    return _api_module.scheduler.discover_master()

async def reset_master_identity(
    req: ResetIdentityRequest,
):
    """
    重置主节点身份标识（仅主节点可调用）。

    用于更换主节点机器或网卡后，替换主节点 SQLite 中旧的 MAC 地址记录。
    需要输入确认字符串 'reset' 以防止误操作。

    调用成功后立即绑定当前物理 MAC，无需重启后端服务。
    """
    _require_static_master("reset master identity")
    if req.confirm.strip().lower() != "reset":
        raise _api_module.HTTPException(400, "请输入 'reset' 确认重置操作")
    result = _api_module.scheduler.reset_master_identity()
    if result.get("status") == "denied":
        raise _api_module.HTTPException(403, result.get("reason", "权限不足"))
    if result.get("status") == "error":
        raise _api_module.HTTPException(500, result.get("reason", "操作失败"))
    return result

async def get_control_plane_status():
    """Report the statically configured role authority."""
    role = _api_module.scheduler._effective_role()
    return {
        "enabled": False,
        "available": role in {"master", "client"},
        "authority": "static_role",
        "role": role,
        "leader_id": _api_module.scheduler.get_effective_node_id() if role == "master" else None,
    }

async def get_queue_detail():
    """
    获取推理调度队列完整详情。

    返回三级队列（Q0/Q1/Q2）中每个任务的序列化信息，
    含优先级、等待时间、预估耗时、老化状态、抢占统计。
    仅主节点可用。
    """
    if not _api_module.scheduler._effective_role() == "master":
        raise _api_module.HTTPException(403, "仅主节点可查看请求队列")
    return _api_module.scheduler._execution_app.requests.queue_snapshot()


async def pause_queue(request: Request):
    """暂停接受新请求。仅主节点。"""
    _require_static_master("pause queue")
    _api_module.scheduler._execution_app.requests.pause()
    return {"success": True, "paused": True}

async def resume_queue(request: Request):
    """恢复接受新请求。仅主节点。"""
    _require_static_master("resume queue")
    _api_module.scheduler._execution_app.requests.resume()
    return {"success": True, "paused": False}

async def clear_queue(request: Request):
    """清空所有排队任务（不影响执行中的任务）。仅主节点。"""
    _require_static_master("clear queue")
    count = _api_module.scheduler._execution_app.requests.clear()
    return {"success": True, "cleared": count}

async def cancel_queue_task(task_id: str, request: Request):
    """
    取消指定排队任务。

    执行中的流水线任务会在当前 token step 完成后通过 PIPELINE_ABORT 中止。
    仅主节点。
    """
    _require_static_master("cancel queue task")
    ok = _api_module.scheduler._execution_app.requests.cancel(task_id)
    if ok:
        return _api_module.CancelTaskResponse(success=True, task_id=task_id, message="任务已取消")
    else:
        return _api_module.CancelTaskResponse(
            success=False, task_id=task_id,
            message="任务不存在或已经完成，无法取消"
        )



async def get_distributed_inference_config():
    """
    获取分布式推理开关状态。
    """
    from config import DISTRIBUTED_INFERENCE_ENABLED
    return {
        "enabled": _api_module.scheduler.get_distributed_inference_enabled(),
        "default": DISTRIBUTED_INFERENCE_ENABLED,
    }

async def set_distributed_inference_config(req: DistributedInferenceRequest):
    """
    设置分布式推理开关。

    - 主节点：控制是否接收从节点连接和协调分布式推理
    - 从节点：控制是否将推理请求转发给主节点
    """
    result = await _api_module.run_in_threadpool(
        _api_module._set_node_inference_mode,
        "distributed" if req.enabled else "local",
    )
    if result.get("status") == "error":
        raise _api_module.HTTPException(500, result.get("reason", "设置失败"))
    return result

async def get_layer_assignments():
    """
    获取当前模型分层配置。

    Returns:
        {
            "total": 24,
            "strategy": "dynamic" | "manual",
            "assignments": [{node_id, role, start_layer, end_layer,
                             has_embedding, has_lm_head, score}],
            "computed_at": timestamp | null,
        }
    """
    return await _api_module.run_in_threadpool(_api_module.scheduler.get_layer_assignments)

async def get_pipeline_capacity_plan():
    """Return the metadata-only, all-or-nothing pipeline capacity plan.

    The response is a read-only admission/transaction projection. It never
    downloads or materializes model weights.
    """
    return await _api_module.run_in_threadpool(_api_module.scheduler.get_pipeline_capacity_plan)

async def get_pipeline_reshard_status():
    """Return the address-free, epoch-fenced automatic recovery state."""
    return await _api_module.run_in_threadpool(_api_module.scheduler.get_pipeline_reshard_status)

async def override_layer_assignments(req: LayerOverrideRequest):
    """
    手动覆盖模型分层配置（仅主节点可调用）。

    验证规则:
      - 所有区间必须从 0 开始连续覆盖到 24
      - node_id 必须是已注册节点
      - 区间不能重叠
    """
    result = _api_module.scheduler.override_layer_assignments([
        {"node_id": a.node_id, "start_layer": a.start_layer, "end_layer": a.end_layer}
        for a in req.assignments
    ])
    if result.get("status") == "denied":
        raise _api_module.HTTPException(403, result.get("reason", "仅主节点可修改"))
    if result.get("status") == "invalid":
        raise _api_module.HTTPException(400, result.get("reason", "分层配置无效"))
    if result.get("status") == "error":
        raise _api_module.HTTPException(500, result.get("reason", "操作失败"))
    return result

async def reset_layer_assignments():
    """
    重置分层配置，清除手动覆盖，恢复自动（dynamic）策略。

    仅主节点可调用。
    """
    if _api_module.scheduler._effective_role() != "master":
        raise _api_module.HTTPException(403, "仅主节点可重置分层配置")
    return _api_module.scheduler.reset_layer_assignments()

async def get_model_runtime_sidecar_status():
    return await _api_module.run_in_threadpool(_api_module.scheduler.get_model_runtime_sidecar_status)

async def get_model_runtime_contracts():
    return await _api_module.run_in_threadpool(_api_module.scheduler.get_model_runtime_contracts)

async def bind_model_runtime_contract(req: ModelRuntimeContractBindRequest):
    try:
        return await _api_module.run_in_threadpool(
            _api_module.scheduler.bind_model_runtime_contract, req.profile, req.model_id,
        )
    except _api_module.HTTPException:
        raise
    except Exception as exc:
        _api_module._raise_qwen3_local_http(exc)

async def begin_model_runtime_sidecar(req: ModelRuntimeSidecarBeginRequest):
    try:
        if req.contract_id:
            return await _api_module.run_in_threadpool(
                _api_module.scheduler.begin_model_runtime_sidecar,
                req.profile,
                req.contract,
                contract_id=req.contract_id,
            )
        return await _api_module.run_in_threadpool(
            _api_module.scheduler.begin_model_runtime_sidecar, req.profile, req.contract,
        )
    except Exception as exc:
        _api_module._raise_qwen3_local_http(exc)

async def release_model_runtime_sidecar(req: ModelRuntimeSidecarActionRequest):
    try:
        return await _api_module.run_in_threadpool(_api_module.scheduler.release_model_runtime_sidecar, req.profile)
    except Exception as exc:
        _api_module._raise_qwen3_local_http(exc)

async def cancel_model_runtime_sidecar(profile: Literal["qwen3_sidecar", "gemma4_pipeline"]):
    try:
        return await _api_module.run_in_threadpool(_api_module.scheduler.cancel_model_runtime_sidecar, profile)
    except Exception as exc:
        _api_module._raise_qwen3_local_http(exc)

async def get_qwen3_local_chain_status():
    return await _api_module.run_in_threadpool(_api_module._require_qwen3_local_master().get_qwen3_local_chain_status)

async def begin_qwen3_local_chain(req: Qwen3LocalChainBeginRequest):
    try:
        return await _api_module.run_in_threadpool(
            _api_module._require_qwen3_local_master().begin_qwen3_local_sidecar_chain,
            req.contract,
        )
    except _api_module.HTTPException:
        raise
    except Exception as exc:
        _api_module._raise_qwen3_local_http(exc)

async def run_qwen3_local_prefill(req: Qwen3LocalChainExecuteRequest):
    try:
        return await _api_module.run_in_threadpool(
            _api_module._require_qwen3_local_master().run_qwen3_local_prefill,
            input_ref=req.input_ref, batch_size=req.batch_size,
            sequence_length=req.sequence_length,
        )
    except _api_module.HTTPException:
        raise
    except Exception as exc:
        _api_module._raise_qwen3_local_http(exc)

async def run_qwen3_local_decode(req: Qwen3LocalChainExecuteRequest):
    try:
        return await _api_module.run_in_threadpool(
            _api_module._require_qwen3_local_master().run_qwen3_local_decode,
            input_ref=req.input_ref, batch_size=req.batch_size,
            sequence_length=req.sequence_length,
        )
    except _api_module.HTTPException:
        raise
    except Exception as exc:
        _api_module._raise_qwen3_local_http(exc)

async def verify_qwen3_local_parity(req: Qwen3LocalChainParityRequest):
    try:
        return await _api_module.run_in_threadpool(
            _api_module._require_qwen3_local_master().verify_qwen3_local_cpu_parity,
            reference_prefill=req.reference_prefill,
            reference_decode=req.reference_decode,
            rtol=req.rtol, atol=req.atol,
        )
    except _api_module.HTTPException:
        raise
    except Exception as exc:
        _api_module._raise_qwen3_local_http(exc)

async def release_qwen3_local_chain():
    try:
        return await _api_module.run_in_threadpool(
            _api_module._require_qwen3_local_master().release_qwen3_local_sidecar_chain,
        )
    except _api_module.HTTPException:
        raise
    except Exception as exc:
        _api_module._raise_qwen3_local_http(exc)

async def cancel_qwen3_local_chain():
    try:
        return await _api_module.run_in_threadpool(
            _api_module._require_qwen3_local_master().cancel_qwen3_local_sidecar_chain,
        )
    except _api_module.HTTPException:
        raise
    except Exception as exc:
        _api_module._raise_qwen3_local_http(exc)

def register_routes() -> None:
    router.add_api_route('/api/cluster/status', get_cluster_status, methods=['GET'], response_model=ClusterStatus, response_model_exclude_unset=True)
    router.add_api_route('/api/cluster/nodes', get_cluster_nodes, methods=['GET'])
    router.add_api_route('/api/cluster/resources', get_cluster_resources, methods=['GET'])
    router.add_api_route('/api/cluster/nodes/{node_id}/deregister', deregister_node, methods=['POST'])
    router.add_api_route('/api/cluster/nodes/{node_id}', delete_cluster_node, methods=['DELETE'])
    router.add_api_route('/api/cluster/config', get_cluster_config, methods=['GET'])
    router.add_api_route('/api/cluster/my-role', get_my_role, methods=['GET'])
    router.add_api_route('/api/cluster/config/max-nodes', update_max_nodes, methods=['PUT'])
    router.add_api_route('/api/cluster/invite', get_invite_info, methods=['GET'])
    router.add_api_route('/api/bootstrap/first-connect', first_connect_bootstrap, methods=['POST'])
    router.add_api_route('/api/bootstrap/info', bootstrap_info, methods=['GET'])
    router.add_api_route('/api/cluster/connect', connect_to_master, methods=['POST'])
    router.add_api_route('/api/cluster/nodes/register', manual_register_node, methods=['POST'])
    router.add_api_route('/api/cluster/android/register', register_android_presence, methods=['POST'])
    router.add_api_route('/api/cluster/android/heartbeat', heartbeat_android_presence, methods=['POST'])
    router.add_api_route('/api/cluster/master-health', check_master_health, methods=['GET'])
    router.add_api_route('/api/cluster/discover', discover_master, methods=['GET'])
    router.add_api_route('/api/cluster/reset-identity', reset_master_identity, methods=['POST'])
    router.add_api_route('/api/cluster/control-plane', get_control_plane_status, methods=['GET'])
    router.add_api_route('/api/cluster/queue', get_queue_detail, methods=['GET'])
    router.add_api_route('/api/cluster/queue/pause', pause_queue, methods=['POST'])
    router.add_api_route('/api/cluster/queue/resume', resume_queue, methods=['POST'])
    router.add_api_route('/api/cluster/queue/clear', clear_queue, methods=['POST'])
    router.add_api_route('/api/cluster/queue/task/{task_id}', cancel_queue_task, methods=['DELETE'])
    router.add_api_route('/api/cluster/config/distributed-inference', get_distributed_inference_config, methods=['GET'])
    router.add_api_route('/api/cluster/config/distributed-inference', set_distributed_inference_config, methods=['PUT'])
    router.add_api_route('/api/cluster/layers', get_layer_assignments, methods=['GET'])
    router.add_api_route('/api/cluster/pipeline-capacity', get_pipeline_capacity_plan, methods=['GET'])
    router.add_api_route('/api/cluster/pipeline-reshard', get_pipeline_reshard_status, methods=['GET'])
    router.add_api_route('/api/cluster/layers', override_layer_assignments, methods=['PUT'])
    router.add_api_route('/api/cluster/layers', reset_layer_assignments, methods=['DELETE'])
    router.add_api_route('/api/cluster/model-runtime/sidecars', get_model_runtime_sidecar_status, methods=['GET'])
    router.add_api_route('/api/cluster/model-runtime/contracts', get_model_runtime_contracts, methods=['GET'])
    router.add_api_route('/api/cluster/model-runtime/contracts/bind', bind_model_runtime_contract, methods=['POST'])
    router.add_api_route('/api/cluster/model-runtime/sidecars/begin', begin_model_runtime_sidecar, methods=['POST'])
    router.add_api_route('/api/cluster/model-runtime/sidecars/release', release_model_runtime_sidecar, methods=['POST'])
    router.add_api_route('/api/cluster/model-runtime/sidecars/{profile}', cancel_model_runtime_sidecar, methods=['DELETE'])
    router.add_api_route('/api/cluster/qwen3/local-chain', get_qwen3_local_chain_status, methods=['GET'])
    router.add_api_route('/api/cluster/qwen3/local-chain/begin', begin_qwen3_local_chain, methods=['POST'])
    router.add_api_route('/api/cluster/qwen3/local-chain/prefill', run_qwen3_local_prefill, methods=['POST'])
    router.add_api_route('/api/cluster/qwen3/local-chain/decode', run_qwen3_local_decode, methods=['POST'])
    router.add_api_route('/api/cluster/qwen3/local-chain/parity', verify_qwen3_local_parity, methods=['POST'])
    router.add_api_route('/api/cluster/qwen3/local-chain/release', release_qwen3_local_chain, methods=['POST'])
    router.add_api_route('/api/cluster/qwen3/local-chain', cancel_qwen3_local_chain, methods=['DELETE'])
