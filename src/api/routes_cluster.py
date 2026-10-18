"""Routes extracted from api_server; shared state remains facade-owned."""

from __future__ import annotations

from types import ModuleType

from fastapi import APIRouter, Depends
import auth_service
from api._routing import configure_route_module

router = APIRouter()
_api_module: ModuleType | None = None
_RESOLUTION_NAMES = (
    "AndroidPresenceRequest",
    "CastVoteRequest",
    "ClusterJoinConsume",
    "ClusterJoinGrantIssue",
    "ClusterJoinRequestCreate",
    "ClusterSecretRotateRequest",
    "ClusterStatus",
    "ConnectToMasterRequest",
    "ControlCertificateRequest",
    "CreateReviewRequest",
    "DistributedInferenceRequest",
    "FirstConnectBootstrapRequest",
    "LayerOverrideRequest",
    "Literal",
    "ManualRegisterRequest",
    "ModelRuntimeContractBindRequest",
    "ModelRuntimeSidecarActionRequest",
    "ModelRuntimeSidecarBeginRequest",
    "Optional",
    "Qwen3LocalChainBeginRequest",
    "Qwen3LocalChainExecuteRequest",
    "Qwen3LocalChainParityRequest",
    "Request",
    "ResetIdentityRequest",
    "SetQueueStrategyRequest",
    "SpareMasterRequest",
    "TaskGraphConfigRequest",
    "TransferMasterRequest",
    "UpdateMaxNodesRequest",
)

def configure_api_module(module: ModuleType) -> None:
    configure_route_module(globals(), module, _RESOLUTION_NAMES)

def exported_handlers() -> dict[str, object]:
    return {name: globals()[name] for name in ['get_cluster_status', 'get_cluster_nodes', 'get_cluster_resources', 'deregister_node', 'delete_cluster_node', 'get_cluster_config', 'get_my_role', 'update_max_nodes', 'get_invite_info', 'create_cluster_join_request', 'issue_cluster_join_grant', 'rotate_cluster_secret', 'consume_cluster_join_grant', 'first_connect_bootstrap', 'bootstrap_info', 'connect_to_master', 'manual_register_node', 'register_android_presence', 'heartbeat_android_presence', 'check_master_health', 'discover_master', 'reset_master_identity', 'get_control_plane_status', 'get_cluster_management_score', 'install_control_plane_certificate', 'quorum_voter_rpc', 'get_queue_detail', 'set_queue_strategy', 'pause_queue', 'resume_queue', 'clear_queue', 'cancel_queue_task', 'get_task_graph_config', 'set_task_graph_config', 'get_distributed_inference_config', 'set_distributed_inference_config', 'get_layer_assignments', 'get_pipeline_capacity_plan', 'get_pipeline_reshard_status', 'override_layer_assignments', 'reset_layer_assignments', 'get_model_runtime_sidecar_status', 'get_model_runtime_contracts', 'bind_model_runtime_contract', 'begin_model_runtime_sidecar', 'release_model_runtime_sidecar', 'cancel_model_runtime_sidecar', 'get_qwen3_local_chain_status', 'begin_qwen3_local_chain', 'run_qwen3_local_prefill', 'run_qwen3_local_decode', 'verify_qwen3_local_parity', 'release_qwen3_local_chain', 'cancel_qwen3_local_chain', 'transfer_master_role', 'get_transfer_logs', 'get_spare_master', 'designate_spare_master', 'clear_spare_master', 'get_spare_master_logs', 'create_review_ticket', 'cast_review_vote', 'list_review_tickets', 'get_review_ticket', 'check_can_vote', 'trigger_expire_check', 'delete_review_ticket', 'delete_resolved_review_tickets']}

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

async def deregister_node(
    node_id: str,
    _principal=Depends(auth_service.require_authenticated_role("admin")),
):
    """
    强制注销一个从节点。

    仅在分布式模式下有效；master 节点不可注销。
    """
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

async def delete_cluster_node(
    node_id: str,
    _principal=Depends(auth_service.require_authenticated_role("admin")),
):
    """
    删除离线节点记录（区别于 deregister：deregister 仅标记离线）。

    用于移除手动注册的 Android / 离线占位节点。
    """
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
    if _api_module.control_fence.enabled:
        _api_module.control_fence.require_current_permit(action="cluster.config.max_nodes")
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

async def create_cluster_join_request(req: ClusterJoinRequestCreate):
    """Create a client-only request code on the node that wants to join."""
    try:
        current_node_id = _api_module.scheduler.get_effective_node_id()
        target_node_id = req.target_node_id or current_node_id
        # A provisional master is still reported as ``master`` until the
        # explicit role transition. Predict the same stable ID that
        # Scheduler.get_effective_node_id() will use after client activation.
        if target_node_id == "master":
            target_node_id = "client_%s" % __import__("socket").gethostname()
        elif req.target_node_id and target_node_id != current_node_id:
            raise _api_module.JoinContractError(
                "target_node_id must match this node identity", code="request_mismatch"
            )
        cluster_id = req.cluster_id.strip() or _api_module.os.environ.get("QLH_CLUSTER_ID", "qlh-default")
        keypair = _api_module.generate_join_keypair()
        request = _api_module.build_join_request(
            master_endpoint=req.master_endpoint,
            cluster_id=cluster_id,
            target_node_id=target_node_id,
            target_public_key=keypair.public_key,
            request_ttl_seconds=req.request_ttl_seconds,
            capabilities=req.capabilities,
        )
        ledger = _api_module._get_join_ledger()
        ledger.save_pending_request(request, keypair)
        code = _api_module.encode_join_request(request)
        return {
            "status": "created",
            "request": request,
            "request_code": code,
            "qr_payload": code,
            "target_node_id": target_node_id,
            "expires_at": request["request_expires_at"],
            "storage": "sqlite",
        }
    except _api_module.JoinContractError as exc:
        raise _api_module.HTTPException(400, {"code": exc.code, "message": str(exc)}) from exc
    except Exception as exc:
        _api_module.logger.error("cluster join request creation failed: %s", exc, exc_info=True)
        raise _api_module.HTTPException(503, "本地入群请求存储不可用") from exc

async def issue_cluster_join_grant(
    req: ClusterJoinGrantIssue,
    request: Request,
    principal=Depends(auth_service.require_role("admin")),
):
    """Issue a short-lived client-only grant after local TOTP confirmation."""
    if _api_module.scheduler._effective_role() != "master":
        raise _api_module.HTTPException(403, "仅主节点可签发入群授权")
    _api_module.auth_service.verify_totp_confirmation(
        principal, req.otp_code, source=_api_module.auth_service.request_source(request)
    )
    try:
        if bool(req.request_code) == bool(req.request):
            raise _api_module.JoinContractError("request_code 或 request 必须且只能提供一个", code="invalid_request")
        join_request = (
            _api_module.decode_join_request(req.request_code)
            if req.request_code else dict(req.request or {})
        )
        if req.request is not None:
            _api_module.encode_join_request(join_request)
        ledger = _api_module._get_join_ledger()
        key_id, keypair = ledger.get_or_create_issuer_keypair()
        grant = _api_module.issue_join_grant(
            join_request,
            issuer_key_id=key_id,
            issuer_private_key=_api_module.load_join_private_key(keypair.private_key),
            issuer_public_key=keypair.public_key,
            ttl_seconds=req.ttl_seconds,
        )
        code = _api_module.encode_join_grant(grant)
        return {
            "status": "issued",
            "grant_code": code,
            "qr_payload": code,
            "issuer_key_id": key_id,
            "issuer_public_key": keypair.public_key,
            "target_node_id": join_request["target_node_id"],
            "expires_at": grant["payload"]["expires_at"],
            "auth_method": "totp",
        }
    except _api_module.JoinContractError as exc:
        status = 409 if exc.code in {"request_expired", "nonce_replayed"} else 400
        raise _api_module.HTTPException(status, {"code": exc.code, "message": str(exc)}) from exc
    except Exception as exc:
        _api_module.logger.error("cluster join grant issuance failed: %s", exc, exc_info=True)
        raise _api_module.HTTPException(503, "本地主节点入群密钥不可用") from exc

async def rotate_cluster_secret(
    req: ClusterSecretRotateRequest,
    request: Request,
    principal=Depends(auth_service.require_role("admin")),
):
    """Rotate the cluster root after Tailnet revocation and drop live peers."""
    if _api_module.scheduler._effective_role() != "master":
        raise _api_module.HTTPException(403, "only master can rotate cluster credentials")
    if not req.tailnet_revocation_completed:
        raise _api_module.HTTPException(
            409,
            {
                "code": "tailnet_revocation_required",
                "message": "remove the revoked node from the Tailnet before rotation",
            },
        )
    _api_module.auth_service.verify_totp_confirmation(
        principal,
        req.otp_code,
        source=_api_module.auth_service.request_source(request),
    )

    try:
        tcp_server = getattr(_api_module.scheduler, "_tcp_server", None)
        if tcp_server is not None:
            rotation, disconnected = await _api_module.run_in_threadpool(
                tcp_server.disconnect_all_clients,
                "cluster credential rotation",
                before_disconnect=_api_module._rotate_cluster_credential,
            )
        else:
            rotation = await _api_module.run_in_threadpool(
                _api_module._rotate_cluster_credential,
            )
            disconnected = 0
        _secret, epoch = rotation
        return {
            "status": "rotated",
            "cluster_secret_epoch": epoch,
            "disconnected_clients": disconnected,
            "rebootstrap_required": True,
        }
    except Exception as exc:
        _api_module.logger.error(
            "cluster credential rotation failed: %s", exc, exc_info=True,
        )
        raise _api_module.HTTPException(
            503,
            {
                "code": "cluster_credential_rotation_failed",
                "message": "cluster credential rotation failed",
            },
        ) from exc


async def consume_cluster_join_grant(req: ClusterJoinConsume):
    """Verify a one-time grant, switch this node to client, and connect."""
    try:
        grant = _api_module.decode_join_grant(req.grant_code)
        payload = grant["payload"]
        pending = _api_module._get_join_ledger().load_pending_request(str(payload.get("request_digest", "")))
        if pending is None:
            raise _api_module.JoinContractError("本节点没有对应的待处理入群请求", code="request_not_found")
        expected_request, _target_keypair = pending
        current_node_id = _api_module.scheduler.get_effective_node_id()
        if current_node_id != expected_request.get("target_node_id"):
            raise _api_module.JoinContractError("授权目标节点与本节点不匹配", code="request_mismatch")
        issuer_public_key = str(payload.get("issuer_public_key") or "")
        ledger = _api_module._get_join_ledger()
        # The one-time authorization must be consumed before role, config or
        # network side effects. A failed connection requires a fresh grant;
        # replay can never enter the switch/connect path.
        verified = _api_module.verify_and_consume_join_grant(
            req.grant_code,
            issuer_public_key=issuer_public_key,
            expected_request=expected_request,
            ledger=ledger,
        )
        ledger.delete_pending_request(str(expected_request["request_digest"]))
        master_host, master_port = _api_module._join_endpoint_parts(str(payload["master_endpoint"]))
        if _api_module.scheduler._effective_role() == "master":
            if not _api_module.scheduler.can_join_existing_master():
                raise _api_module.JoinContractError("当前主节点不允许降级加入其他集群", code="role_switch_denied")
            switch_result = await _api_module.run_in_threadpool(
                _api_module.scheduler.activate_client_mode, master_host, master_port
            )
            if switch_result.get("status") == "denied":
                raise _api_module.JoinContractError(
                    switch_result.get("reason", "无法切换为从节点"), code="role_switch_denied"
                )
            connection_result = switch_result.get("connect_result") or switch_result
        else:
            connection_result = await _api_module.run_in_threadpool(
                _api_module.scheduler.connect_to_master, master_host, master_port
            )
        if connection_result.get("status") not in {"connected", "unchanged", "switched"}:
            raise _api_module.JoinContractError(
                connection_result.get("reason", "连接主节点失败"), code="connect_failed"
            )
        return {
            "status": "connected",
            "role": "client",
            "node_id": verified["target_node_id"],
            "master_endpoint": verified["master_endpoint"],
            "message": "已验证一次性授权并降级为从节点",
        }
    except _api_module.JoinContractError as exc:
        status = 409 if exc.code in {"nonce_replayed", "request_not_found", "role_switch_denied", "connect_failed"} else 400
        raise _api_module.HTTPException(status, {"code": exc.code, "message": str(exc)}) from exc
    except Exception as exc:
        _api_module.logger.error("cluster join grant consumption failed: %s", exc, exc_info=True)
        raise _api_module.HTTPException(500, "入群授权消费失败") from exc

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
        _api_module._get_join_ledger().consume_bootstrap_request(
            nonce=req.credential_request_nonce,
            request_digest=request_digest,
        )
    except BootstrapCredentialError as exc:
        status = 409 if exc.code in {"request_expired"} else 400
        raise _api_module.HTTPException(
            status, {"code": exc.code, "message": str(exc)}
        ) from exc
    except _api_module.JoinContractError as exc:
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
    force_bootstrap = False
    if _api_module.scheduler._effective_role() == "master":
        if not req.switch_to_client or not _api_module.scheduler.can_join_existing_master():
            raise _api_module.HTTPException(403, "当前主节点已确认或已有从节点，不能切换为从节点")
        # 角色切换可能阻塞在角色迁移锁上，放入线程池避免卡死事件循环
        switch_result = await _api_module.run_in_threadpool(_api_module.scheduler.activate_client_mode)
        if switch_result.get("status") == "denied":
            raise _api_module.HTTPException(409, switch_result.get("reason", "无法切换为从节点"))
        force_bootstrap = True

    # connect_to_master 内部含多次 TCP 重试（最长可达数十秒），
    # 必须放入线程池执行，否则会阻塞事件循环冻结所有 HTTP 接口
    result = await _api_module.run_in_threadpool(
        _api_module.scheduler.connect_to_master,
        req.master_host,
        req.master_port,
        force_bootstrap=force_bootstrap,
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
    _principal=Depends(auth_service.require_authenticated_role("admin")),
):
    """
    重置主节点身份标识（仅主节点可调用）。

    用于更换主节点机器或网卡后，替换主节点 SQLite 中旧的 MAC 地址记录。
    需要输入确认字符串 'reset' 以防止误操作。

    调用成功后立即绑定当前物理 MAC，无需重启后端服务。
    """
    if req.confirm.strip().lower() != "reset":
        raise _api_module.HTTPException(400, "请输入 'reset' 确认重置操作")
    result = _api_module.scheduler.reset_master_identity()
    if result.get("status") == "denied":
        raise _api_module.HTTPException(403, result.get("reason", "权限不足"))
    if result.get("status") == "error":
        raise _api_module.HTTPException(500, result.get("reason", "操作失败"))
    return result

async def get_control_plane_status():
    """Read-only fencing status; it remains available while writes are fenced."""
    result = _api_module.control_fence.snapshot()
    result["auto_role"] = _api_module.scheduler.get_auto_role_snapshot()
    return result

async def get_cluster_management_score():
    """Return the versioned, read-only management-capability score snapshot."""

    def _build_snapshot() -> dict[str, Any]:
        try:
            import config as _config

            secret = str(getattr(_config, "CLUSTER_SECRET", "") or "")
        except Exception:
            secret = _api_module.os.environ.get("QLH_CLUSTER_SECRET", "")
        return _api_module.build_management_score_snapshot(
            _api_module.scheduler.get_nodes(),
            signing_secret=secret,
        )

    return await _api_module.run_in_threadpool(_build_snapshot)

async def install_control_plane_certificate(req: ControlCertificateRequest):
    """Install a certificate already issued by the quorum protocol."""
    if _api_module.scheduler._effective_role() != "master":
        raise _api_module.HTTPException(403, {"code": "not_master", "message": "only master may install a control certificate"})
    try:
        result = _api_module.control_fence.install_certificate(req.certificate)
    except _api_module.ControlFenceError as exc:
        status = 503 if exc.code == "control_fence_unavailable" else 409
        raise _api_module.HTTPException(status, {"code": exc.code, "message": str(exc)}) from exc
    return {"status": "installed", **result}


async def quorum_voter_rpc(request: Request):
    """Serve the authenticated local voter used by an auto-role peer."""
    runtime = (
        getattr(_api_module, "auto_role_runtime", None)
        or getattr(_api_module.scheduler, "_auto_role_runtime", None)
    )
    if runtime is None:
        raise _api_module.HTTPException(503, {"code": "quorum_unavailable", "message": "automatic role is not configured"})
    if not runtime.authenticate(request.headers.get("X-QLH-Quorum-Secret")):
        raise _api_module.HTTPException(403, {"code": "quorum_auth_failed", "message": "voter authentication failed"})
    try:
        payload = await request.json()
    except Exception as exc:
        raise _api_module.HTTPException(400, {"code": "quorum_unavailable", "message": "voter request is invalid"}) from exc
    return runtime.handle_rpc(payload)

async def get_queue_detail():
    """
    获取推理调度队列完整详情。

    返回三级队列（Q0/Q1/Q2）中每个任务的序列化信息，
    含优先级、等待时间、预估耗时、老化状态、抢占统计。
    仅主节点可用。
    """
    if not _api_module.scheduler._effective_role() == "master":
        raise _api_module.HTTPException(403, "仅主节点可查看请求队列")
    return _api_module.scheduler.pipeline_queue.get_queue_detail()

async def set_queue_strategy(req: SetQueueStrategyRequest, request: Request):
    """切换调度策略: fifo | mlfq。仅主节点。"""
    if _api_module.control_fence.enabled:
        _api_module.control_fence.require_current_permit(action="cluster.queue.strategy")
    if not _api_module.scheduler._effective_role() == "master":
        raise _api_module.HTTPException(403, "仅主节点可切换调度策略")
    try:
        _api_module.scheduler.pipeline_queue.set_strategy(req.strategy)
        return {"success": True, "strategy": req.strategy}
    except ValueError as e:
        raise _api_module.HTTPException(400, str(e))

async def pause_queue(request: Request):
    """暂停接受新请求。仅主节点。"""
    if _api_module.control_fence.enabled:
        _api_module.control_fence.require_current_permit(action="cluster.queue.pause")
    if not _api_module.scheduler._effective_role() == "master":
        raise _api_module.HTTPException(403, "仅主节点可暂停请求队列")
    _api_module.scheduler.pipeline_queue.pause()
    return {"success": True, "paused": True}

async def resume_queue(request: Request):
    """恢复接受新请求。仅主节点。"""
    if _api_module.control_fence.enabled:
        _api_module.control_fence.require_current_permit(action="cluster.queue.resume")
    if not _api_module.scheduler._effective_role() == "master":
        raise _api_module.HTTPException(403, "仅主节点可恢复请求队列")
    _api_module.scheduler.pipeline_queue.resume()
    return {"success": True, "paused": False}

async def clear_queue(request: Request):
    """清空所有排队任务（不影响执行中的任务）。仅主节点。"""
    if _api_module.control_fence.enabled:
        _api_module.control_fence.require_current_permit(action="cluster.queue.clear")
    if not _api_module.scheduler._effective_role() == "master":
        raise _api_module.HTTPException(403, "仅主节点可清空请求队列")
    count = _api_module.scheduler.pipeline_queue.clear()
    return {"success": True, "cleared": count}

async def cancel_queue_task(task_id: str, request: Request):
    """
    取消指定排队任务。

    执行中的流水线任务会在当前 token step 完成后通过 PIPELINE_ABORT 中止。
    仅主节点。
    """
    if _api_module.control_fence.enabled:
        _api_module.control_fence.require_current_permit(action="cluster.queue.cancel")
    if not _api_module.scheduler._effective_role() == "master":
        raise _api_module.HTTPException(403, "仅主节点可取消队列任务")
    ok = _api_module.scheduler.pipeline_queue.cancel_task(task_id)
    if ok:
        return _api_module.CancelTaskResponse(success=True, task_id=task_id, message="任务已取消")
    else:
        return _api_module.CancelTaskResponse(
            success=False, task_id=task_id,
            message="任务不存在或已经完成，无法取消"
        )

async def get_task_graph_config():
    """Read task-graph experiment switches without claiming physical readiness."""
    return _api_module._task_graph_feature_settings()

async def set_task_graph_config(req: TaskGraphConfigRequest):
    """Toggle local task-graph experiments; production Worker dispatch stays gated."""
    if _api_module.scheduler._effective_role() != "master":
        raise _api_module.HTTPException(403, "仅主节点可切换任务链实验开关")
    if req.enabled is None and req.worker_experimental_enabled is None:
        raise _api_module.HTTPException(400, "至少提供一个任务链开关")
    try:
        return _api_module._set_task_graph_runtime_settings(
            task_graph_enabled=req.enabled,
            task_worker_experimental_enabled=req.worker_experimental_enabled,
        )
    except _api_module.HTTPException:
        raise
    except Exception as exc:
        _api_module.logger.error("任务链运行时开关更新失败", exc_info=True)
        raise _api_module.HTTPException(500, f"任务链实验开关更新失败: {exc}") from exc

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

async def transfer_master_role(req: TransferMasterRequest):
    """
    将主节点身份转让给指定从节点（仅主节点可调用）。

    流程:
      1. 主节点通过 TCP 向目标从节点发送 ROLE_TRANSFER 消息
      2. 从节点保存升级日志、返回 ACK
      3. 主节点保存降级日志、更新数据库中的主节点信息
      4. 建议双方重启以应用新角色

    注意: 转让后需要重启服务才能生效：
      - 原主节点重启后以从节点模式运行
      - 新主节点重启后以主节点模式运行
    """
    # 内部同步等待从节点 ACK（最长 15s），放入线程池避免阻塞事件循环
    result = await _api_module.run_in_threadpool(
        _api_module.scheduler.transfer_master_role, req.target_node_id
    )
    if result.get("status") == "denied":
        raise _api_module.HTTPException(403, result.get("reason", "权限不足"))
    if result.get("status") == "invalid":
        raise _api_module.HTTPException(400, result.get("reason", "参数无效"))
    if result.get("status") == "timeout":
        raise _api_module.HTTPException(408, result.get("reason", "超时"))
    if result.get("status") == "error":
        raise _api_module.HTTPException(500, result.get("reason", "操作失败"))
    return result

async def get_transfer_logs():
    """
    获取角色转让日志（降级 + 升级）。

    Returns:
        { logs: [{direction, from_role, to_role, related_node, timestamp, ...}] }
    """
    logs = _api_module.scheduler.get_transfer_logs()
    return {"logs": logs, "count": len(logs)}

async def get_spare_master():
    """
    获取当前备用主节点信息。

    Returns:
        { spare_master: {node_id, hostname, address, designated_at, is_online, state} | null }
    """
    spare = _api_module.scheduler.get_spare_master()
    return {"spare_master": spare}

async def designate_spare_master(req: SpareMasterRequest):
    """
    指定一个在线从节点为备用主节点（仅主节点可调用）。

    规则:
      - 集群节点数 ≥ 2
      - 目标节点必须在线且为 client

    Returns:
        { status, message, spare_master, ... }
    """
    # 内部同步等待从节点 ACK（最长 15s），放入线程池避免阻塞事件循环
    result = await _api_module.run_in_threadpool(
        _api_module.scheduler.designate_spare_master, req.target_node_id
    )
    if result.get("status") == "denied":
        raise _api_module.HTTPException(403, result.get("reason", "权限不足"))
    if result.get("status") == "invalid":
        raise _api_module.HTTPException(400, result.get("reason", "参数无效"))
    if result.get("status") == "timeout":
        raise _api_module.HTTPException(408, result.get("reason", "超时"))
    if result.get("status") == "duplicate":
        return result  # 不抛异常，返回已有信息
    if result.get("status") == "error":
        raise _api_module.HTTPException(500, result.get("reason", "操作失败"))
    return result

async def clear_spare_master():
    """
    清除备用主节点指定（仅主节点可调用）。

    Returns:
        { status, message }
    """
    result = _api_module.scheduler.clear_spare_master()
    if result.get("status") == "denied":
        raise _api_module.HTTPException(403, result.get("reason", "权限不足"))
    return result

async def get_spare_master_logs():
    """
    获取备用主节点操作日志。

    Returns:
        { logs: [{direction, timestamp, details, ...}] }
    """
    logs = _api_module.scheduler.get_spare_master_logs()
    return {"logs": logs, "count": len(logs)}

async def create_review_ticket(req: CreateReviewRequest):
    """
    创建主节点转让审查工单（仅 master 可用）。

    需要先指定备用主节点。
    创建成功后通过 TUI/API 查询和处理工单。
    """
    if _api_module.scheduler._effective_role() != "master":
        raise _api_module.HTTPException(status_code=403, detail="仅主节点可创建审查工单")

    # 检查 spare master
    spare = _api_module.scheduler.get_spare_master()
    if not spare or not spare.get("node_id"):
        raise _api_module.HTTPException(
            status_code=400,
            detail="未指定备用主节点。请先在「备用主节点」中指定后再创建审查工单。",
        )

    try:
        from review import ReviewManager
        review_mgr = ReviewManager()
        ticket = await _api_module.run_in_threadpool(
            review_mgr.create_ticket,
            created_by=_api_module.scheduler.get_effective_node_id(),
            target_node_id=req.target_node_id,
            reason=req.reason,
            timeout_hours=req.timeout_hours,
        )
        if ticket is None:
            raise _api_module.HTTPException(status_code=503, detail="主节点 SQLite 不可用，无法创建审查工单")
        return ticket.to_dict()
    except _api_module.HTTPException:
        raise
    except Exception as e:
        _api_module.logger.error(f"创建审查工单失败: {e}", exc_info=True)
        raise _api_module.HTTPException(status_code=500, detail=f"创建审查工单失败: {e}")

async def cast_review_vote(req: CastVoteRequest):
    """
    对审查工单投票（仅 PC 独显节点可投票）。

    投票值: -1（阻止）、0（弃权）、+1（赞同）。
    同一节点重复投票会更新之前的投票。

    阈值: >= +2 通过，<= -2 阻止。
    """
    node_id = _api_module.scheduler.get_effective_node_id()

    # 验证投票资格
    can_vote, reason = _api_module.scheduler.can_node_vote(node_id)
    if not can_vote:
        raise _api_module.HTTPException(status_code=403, detail=reason)

    if req.vote not in (-1, 0, 1):
        raise _api_module.HTTPException(status_code=400, detail="投票值必须为 -1、0 或 +1")

    try:
        from review import ReviewManager
        review_mgr = ReviewManager()
        ticket = await _api_module.run_in_threadpool(
            review_mgr.cast_vote,
            ticket_id=req.ticket_id,
            voter_node_id=node_id,
            vote_value=req.vote,
            comment=req.comment,
        )
        if ticket is None:
            raise _api_module.HTTPException(status_code=404, detail=f"工单 '{req.ticket_id}' 不存在或已关闭")
        return ticket.to_dict()
    except _api_module.HTTPException:
        raise
    except Exception as e:
        _api_module.logger.error(f"投票失败: {e}", exc_info=True)
        raise _api_module.HTTPException(status_code=500, detail=f"投票失败: {e}")

async def list_review_tickets(
    status: Optional[str] = None,
    limit: int = 20,
    summary: bool = False,
):
    """列出审查工单。可选过滤: ?status=pending"""
    try:
        from review import ReviewManager
        review_mgr = ReviewManager()
        safe_limit = max(1, min(int(limit), 8 if summary else 100))
        tickets = review_mgr.list_tickets(status)[:safe_limit]
        if summary:
            tickets = [
                {
                    "ticket_id": ticket.ticket_id,
                    "status": ticket.status.value,
                    "created_at": _api_module._workflow_safe_timestamp(ticket.created_at),
                    "target_node_id": str(ticket.target_node_id or ""),
                    "score": max(-100, min(int(ticket.score or 0), 100)),
                    "expires_at": _api_module._workflow_safe_timestamp(ticket.expires_at),
                    "resolved_at": _api_module._workflow_safe_timestamp(ticket.resolved_at),
                    "vote_count": min(len(ticket.votes), 100),
                }
                for ticket in tickets
            ]
        return {
            "tickets": [t if isinstance(t, dict) else t.to_dict() for t in tickets],
            "count": len(tickets),
        }
    except Exception as e:
        raise _api_module.HTTPException(status_code=500, detail=f"获取工单列表失败: {e}")

async def get_review_ticket(ticket_id: str):
    """获取单个审查工单详情。"""
    try:
        from review import ReviewManager
        review_mgr = ReviewManager()
        ticket = review_mgr.get_ticket(ticket_id)
        if ticket is None:
            raise _api_module.HTTPException(status_code=404, detail=f"工单 '{ticket_id}' 不存在")
        return ticket.to_dict()
    except _api_module.HTTPException:
        raise
    except Exception as e:
        raise _api_module.HTTPException(status_code=500, detail=f"获取工单失败: {e}")

async def check_can_vote():
    """检查当前节点是否有审查投票资格。"""
    node_id = _api_module.scheduler.get_effective_node_id()
    can_vote, reason = _api_module.scheduler.can_node_vote(node_id)
    return {
        "node_id": node_id,
        "can_vote": can_vote,
        "reason": reason,
    }

async def trigger_expire_check():
    """手动触发审查工单过期检查。"""
    if _api_module.scheduler._effective_role() != "master":
        raise _api_module.HTTPException(status_code=403, detail="仅主节点可执行此操作")
    try:
        from review import ReviewManager
        review_mgr = ReviewManager()
        expired = await _api_module.run_in_threadpool(review_mgr.resolve_expired)
        return {"expired": expired, "count": len(expired)}
    except Exception as e:
        raise _api_module.HTTPException(status_code=500, detail=f"过期检查失败: {e}")

async def delete_review_ticket(ticket_id: str):
    """删除单个审查工单（所有状态均可）。"""
    if _api_module.scheduler._effective_role() != "master":
        raise _api_module.HTTPException(status_code=403, detail="仅主节点可执行此操作")
    try:
        from review import ReviewManager
        ok = ReviewManager().delete_ticket(ticket_id)
        if not ok:
            raise _api_module.HTTPException(status_code=404, detail=f"工单 {ticket_id} 不存在或删除失败")
        return {"status": "deleted", "ticket_id": ticket_id}
    except _api_module.HTTPException:
        raise
    except Exception as e:
        raise _api_module.HTTPException(status_code=500, detail=f"删除工单失败: {e}")

async def delete_resolved_review_tickets():
    """批量删除所有已解决（approved/rejected/expired）的审查工单。"""
    if _api_module.scheduler._effective_role() != "master":
        raise _api_module.HTTPException(status_code=403, detail="仅主节点可执行此操作")
    try:
        from review import ReviewManager
        count = ReviewManager().delete_resolved()
        return {"status": "deleted", "count": count}
    except Exception as e:
        raise _api_module.HTTPException(status_code=500, detail=f"批量删除工单失败: {e}")


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
    router.add_api_route('/api/cluster/join/request', create_cluster_join_request, methods=['POST'])
    router.add_api_route('/api/cluster/join/grant', issue_cluster_join_grant, methods=['POST'])
    router.add_api_route('/api/cluster/credentials/rotate', rotate_cluster_secret, methods=['POST'])
    router.add_api_route('/api/cluster/join/consume', consume_cluster_join_grant, methods=['POST'])
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
    router.add_api_route('/api/cluster/management-score', get_cluster_management_score, methods=['GET'])
    router.add_api_route('/api/cluster/control-plane/certificate', install_control_plane_certificate, methods=['POST'])
    router.add_api_route('/api/cluster/quorum/voter', quorum_voter_rpc, methods=['POST'])
    router.add_api_route('/api/cluster/queue', get_queue_detail, methods=['GET'])
    router.add_api_route('/api/cluster/queue/strategy', set_queue_strategy, methods=['POST'])
    router.add_api_route('/api/cluster/queue/pause', pause_queue, methods=['POST'])
    router.add_api_route('/api/cluster/queue/resume', resume_queue, methods=['POST'])
    router.add_api_route('/api/cluster/queue/clear', clear_queue, methods=['POST'])
    router.add_api_route('/api/cluster/queue/task/{task_id}', cancel_queue_task, methods=['DELETE'])
    router.add_api_route('/api/cluster/config/task-graph', get_task_graph_config, methods=['GET'])
    router.add_api_route('/api/cluster/config/task-graph', set_task_graph_config, methods=['PUT'])
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
    router.add_api_route('/api/cluster/transfer-master', transfer_master_role, methods=['POST'])
    router.add_api_route('/api/cluster/transfer-logs', get_transfer_logs, methods=['GET'])
    router.add_api_route('/api/cluster/spare-master', get_spare_master, methods=['GET'])
    router.add_api_route('/api/cluster/spare-master', designate_spare_master, methods=['POST'])
    router.add_api_route('/api/cluster/spare-master', clear_spare_master, methods=['DELETE'])
    router.add_api_route('/api/cluster/spare-master/logs', get_spare_master_logs, methods=['GET'])
    router.add_api_route('/api/cluster/review/create', create_review_ticket, methods=['POST'])
    router.add_api_route('/api/cluster/review/vote', cast_review_vote, methods=['POST'])
    router.add_api_route('/api/cluster/review/tickets', list_review_tickets, methods=['GET'])
    router.add_api_route('/api/cluster/review/tickets/{ticket_id}', get_review_ticket, methods=['GET'])
    router.add_api_route('/api/cluster/review/can-vote', check_can_vote, methods=['GET'])
    router.add_api_route('/api/cluster/review/expire-check', trigger_expire_check, methods=['POST'])
    router.add_api_route('/api/cluster/review/tickets/{ticket_id}', delete_review_ticket, methods=['DELETE'])
    router.add_api_route('/api/cluster/review/tickets', delete_resolved_review_tickets, methods=['DELETE'])
