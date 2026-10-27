# -*- coding: utf-8 -*-
"""scheduler-svc HTTP 壳（微服务架构改造计划 §4.2，透传路径契约）。

设计（§4.2 T3 落地决策）：
  - 网关对 /api/cluster/*、/api/device/* 做 1:1 透传（去 /api 前缀）
    → scheduler-svc 提供 /cluster/*、/device/* 端点，字段对齐
    gateway/test/fake-scheduler.ts + docs/TUI适配实施计划.md §2.2
  - 端点实现 = api_server 对应薄壳端点的等价复制（api_server.py:5169-5928
    cluster 区 + 1820-1960 device 区），scheduler → 注入实例；源文件不动
  - 节点角色为静态配置，不提供选主、角色转让、备用主节点或转让审查端点
  - bootstrap 域属 control-svc（阶段 3），不在此实现
  - device 画像：scheduler 启动时不自动检测，壳内惰性检测并缓存
    （device_profiler.get_profile()），同步 scheduler._local_device_profile

并行共存（§1.4）：本文件独立实现，现有 Python 后端零改动。
"""
import asyncio
import logging
import threading
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from api_errors import coded_http_error, install_http_error_handler
from release_contract import PRODUCT_VERSION

logger = logging.getLogger("scheduler_svc_http")

router = APIRouter()

# ---- 请求模型（复制自 api_server.py:801-813，字段保真） ----
class UpdateMaxNodesRequest(BaseModel):
    max_nodes: int = Field(..., ge=1, le=64, description="新的最大节点数（包含 master）")


class ConnectToMasterRequest(BaseModel):
    master_host: str = Field(..., description="主节点 IP 地址", min_length=1)
    master_port: int = Field(8888, ge=1, le=65535, description="主节点端口")


class Qwen3LocalChainBeginRequest(BaseModel):
    contract: dict


class Qwen3LocalChainExecuteRequest(BaseModel):
    input_ref: str = Field(..., min_length=1, max_length=2048)
    batch_size: int = Field(..., ge=1, le=64)
    sequence_length: int = Field(..., ge=1, le=1048576)


class Qwen3LocalChainParityRequest(BaseModel):
    reference_prefill: str = Field(..., min_length=1, max_length=2048)
    reference_decode: str = Field(..., min_length=1, max_length=2048)
    rtol: float = Field(default=1e-4, ge=0.0, le=1.0)
    atol: float = Field(default=1e-5, ge=0.0, le=1.0)


class ModelRuntimeSidecarBeginRequest(BaseModel):
    profile: Literal["qwen3_sidecar", "gemma4_pipeline"]
    contract: Optional[dict] = None
    contract_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


class ModelRuntimeSidecarActionRequest(BaseModel):
    profile: Literal["qwen3_sidecar", "gemma4_pipeline"]


class ModelRuntimeContractBindRequest(BaseModel):
    profile: Literal["qwen3_sidecar", "gemma4_pipeline"]
    model_id: str = Field(..., min_length=1, max_length=128)


# ---- scheduler 实例注入（build_scheduler_app 时设置） ----
_scheduler_holder: Optional[object] = None
_scheduler_lock = threading.RLock()


def _scheduler():
    sched = _scheduler_holder
    if sched is None:
        raise HTTPException(503, "scheduler 未就绪")
    return sched


def set_scheduler(scheduler) -> None:
    global _scheduler_holder
    with _scheduler_lock:
        _scheduler_holder = scheduler


def reset_scheduler() -> None:
    """清空注入实例（测试隔离；多实例/并行测试互不串扰）。"""
    global _scheduler_holder
    with _scheduler_lock:
        _scheduler_holder = None


# ---- device 画像缓存（壳内惰性检测，与 api_server 的 device_profile 等价） ----
_device_profile_cache: Optional[dict] = None


def _detect_device_profile() -> dict:
    from device_profiler import get_profile

    profiler = get_profile()
    return profiler.to_dict()


def _with_compat_device_fields(profile: dict) -> dict:
    """补 TUI/前端兼容别名字段（不改 device_profiler 本体，api_server 路径不受影响）。

    消费方按以下形状读取（tui_admin.py:1141-1151、tui-contract 用例 33）：
      - 顶层 hostname            ← platform.hostname
      - os.system / os.release   ← platform.os / platform.os_version
      - cpu.model / cpu.brand    ← cpu.model_name
      - memory                   ← ram（同值别名）
    真实 to_dict() 中这些值在 platform/cpu 内层，字段名也不同；此处做
    只增不改的兼容包装，保持 device_profiler 共享库冻结不动。
    """
    result = dict(profile)
    platform = profile.get("platform") or {}
    if "hostname" not in result:
        result["hostname"] = platform.get("hostname", "")
    if "os" not in result:
        result["os"] = {
            "system": platform.get("os", ""),
            "release": platform.get("os_version", ""),
        }
    cpu = profile.get("cpu")
    if isinstance(cpu, dict):
        cpu_compat = dict(cpu)
        if "model" not in cpu_compat and "model_name" in cpu_compat:
            cpu_compat["model"] = cpu_compat["model_name"]
        if "brand" not in cpu_compat and "model_name" in cpu_compat:
            cpu_compat["brand"] = cpu_compat["model_name"]
        result["cpu"] = cpu_compat
    ram = profile.get("ram")
    if "memory" not in result and isinstance(ram, dict):
        result["memory"] = ram
    return result


# =====================================================================
# cluster 域（复制自 api_server.py:5169-5928 薄壳端点，scheduler → 注入）
# =====================================================================
@router.get("/cluster/status")
async def get_cluster_status():
    """
    获取集群整体状态。

    包含所有节点状态、TCP 连接信息、当前任务等。
    单机模式下返回 3 个默认节点（均为 online）。
    壳层增强：补 run_mode/node_role/node_id/max_nodes 四字段
    （网关 /api/status 聚合依赖，对齐 TUI §2.2 契约；scheduler 源不动）。
    """
    sched = _scheduler()
    result = sched.get_status()
    if not isinstance(result, dict):
        result = {}
    role = {}
    try:
        role = sched.get_my_role() or {}
    except Exception:
        pass
    result.setdefault("run_mode", role.get("run_mode") or "standalone")
    result.setdefault("node_role", role.get("node_role") or "")
    result.setdefault("node_id", role.get("node_id") or "")
    result.setdefault("max_nodes", role.get("max_nodes") or 1)
    return result


@router.get("/cluster/nodes")
async def get_cluster_nodes():
    """
    获取所有节点详情列表。

    Returns:
        { nodes: [...], count: int, online_count: int }
    """
    nodes = _scheduler().get_nodes()
    online_count = sum(1 for n in nodes if n["is_available"])
    return {
        "nodes": nodes,
        "count": len(nodes),
        "online_count": online_count,
        "offline_count": len(nodes) - online_count,
    }


@router.post("/cluster/nodes/{node_id}/deregister")
async def deregister_node(node_id: str):
    """
    强制注销一个从节点。

    仅在分布式模式下有效；master 节点不可注销。
    """
    if node_id == "master":
        raise HTTPException(400, "主节点不可注销")

    success = _scheduler().deregister_node(node_id)
    if not success:
        raise HTTPException(404, f"节点 '{node_id}' 不存在")

    logger.info(f"节点 {node_id} 已被强制注销")
    return {
        "status": "deregistered",
        "node_id": node_id,
    }


@router.delete("/cluster/nodes/{node_id}")
async def delete_cluster_node(node_id: str):
    """
    删除离线节点记录（区别于 deregister：deregister 仅标记离线）。

    用于移除手动注册的 Android / 离线占位节点。
    """
    result = _scheduler().delete_node(node_id)
    status = result.get("status")
    if status == "denied":
        raise HTTPException(403, result.get("reason", "权限不足"))
    if status == "invalid":
        raise HTTPException(400, result.get("reason", "无效节点"))
    if status == "not_found":
        raise HTTPException(404, result.get("reason", "节点不存在"))
    if status == "online":
        raise HTTPException(409, result.get("reason", "节点在线，无法删除"))
    if status != "deleted":
        raise HTTPException(500, result.get("reason", "删除节点失败"))
    return result


@router.get("/cluster/config")
async def get_cluster_config():
    """
    获取分布式配置信息。

    包含网络配置、分层配置、模型配置、任务统计、当前节点角色。
    """
    return _scheduler().get_config()


@router.get("/cluster/my-role")
async def get_my_role():
    """
    获取当前节点的角色信息。

    用于前端判断：
    - master 节点：后台管理 Tab 完全开放
    - client 节点：需在设置中开启"分布式推理优化"后才可见
    """
    return _scheduler().get_my_role()


@router.put("/cluster/config/max-nodes")
async def update_max_nodes(req: UpdateMaxNodesRequest):
    """
    动态调整最大节点数量（仅主节点可调用）。

    仅修改容量上限，不预创建空槽位。从节点通过 TCP 注册动态加入。
    """
    result = _scheduler().update_max_nodes(req.max_nodes)
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "权限不足"))
    if result.get("status") == "invalid":
        raise HTTPException(400, result.get("reason", "无效参数"))
    return result


@router.get("/cluster/invite")
async def get_invite_info():
    """
    获取主节点的邀请/连接信息（供从节点连接使用）。

    主节点调用此接口获取自身监听地址和端口，
    用户将此信息提供给从节点，从节点在后台管理中输入并连接。
    """
    return _scheduler().get_invite_info()


@router.post("/cluster/connect")
async def connect_to_master(req: ConnectToMasterRequest):
    """
    从节点主动连接主节点（从节点的「连接主节点」按钮触发）。

    调用后本节点将通过 TCP 向指定主节点发起注册，
    注册成功后主节点的节点列表中将出现本节点。
    """
    if _scheduler()._effective_role() == "master":
        raise HTTPException(403, "节点角色为静态配置；请将本节点配置为 client 并重启")

    # connect_to_master 内部含多次 TCP 重试（最长可达数十秒），
    # 必须放入线程池执行，否则会阻塞事件循环冻结所有 HTTP 接口
    result = await run_in_threadpool(
        _scheduler().connect_to_master,
        req.master_host,
        req.master_port,
        persist_preference=True,
    )
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "仅从节点可连接主节点"))
    if result.get("status") == "bootstrap_failed":
        raise HTTPException(400, result.get("reason", "首次连接自动部署失败"))
    if result.get("status") == "failed":
        raise HTTPException(400, result.get("reason", "连接失败"))
    if result.get("status") == "error":
        raise HTTPException(500, result.get("reason", "连接异常"))
    return result


class ManualRegisterRequest(BaseModel):
    node_id: str = Field(..., min_length=1, max_length=64, description="节点标识")
    hostname: str = Field(default="", description="主机名")
    address: str = Field(default="", description="预留 IP:Port")
    network_type: str = Field(default="unknown", description="网络类型: wifi | ethernet | unknown")
    node_type: str = Field(default="pc", description="设备平台: pc | android")


class AndroidPresenceRequest(BaseModel):
    node_id: str = Field(..., min_length=1, max_length=64, description="Android 稳定节点标识")
    hostname: str = Field(default="", description="Android 设备名")
    address: str = Field(default="", description="HTTP 客户端地址（可选，仅展示）")
    network_type: str = Field(default="unknown", description="网络类型: wifi | mobile | ethernet | vpn | other | unknown")
    device_info: dict = Field(default_factory=dict, description="Android 设备画像/运行状态")
    client_mode: str = Field(default="thin", description="客户端模式: thin | full")
    app_variant: str = Field(default="full", description="Android flavor: full | lite")
    app_version: str = Field(default="", description="App 版本")


@router.post("/cluster/nodes/register")
async def manual_register_node(req: ManualRegisterRequest):
    """
    主节点手动注册一个从节点（无需 TCP 连接）。

    管理员可在后台管理页面提前录入从节点信息。
    手动注册的节点初始状态为 offline，待从节点通过 TCP 连接后自动变为 online。

    如果从节点主动通过「连接主节点」发起 TCP 注册，也会自动加入节点列表，
    无需手动注册。此接口用于管理员提前规划节点或预留槽位。
    """
    result = _scheduler().manual_register_node(
        node_id=req.node_id,
        hostname=req.hostname,
        address=req.address,
        network_type=req.network_type,
        node_type=req.node_type,
    )
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "仅主节点可手动注册"))
    if result.get("status") == "invalid":
        raise HTTPException(400, result.get("reason", "无效参数"))
    if result.get("status") == "full":
        raise HTTPException(400, result.get("reason", "节点容量已满"))
    if result.get("status") == "exists":
        return result  # 已存在不报错，返回当前状态
    return result


@router.post("/cluster/android/register")
async def register_android_presence(req: AndroidPresenceRequest, request: Request):
    """Android Full 薄客户端在线登记/心跳（不是 TCP worker 注册）。"""
    http_peer = request.client.host if request.client else ""
    result = _scheduler().register_android_client(
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
        raise HTTPException(403, result.get("reason", "仅主节点可登记 Android 客户端"))
    if result.get("status") == "invalid":
        raise HTTPException(400, result.get("reason", "无效 Android 节点"))
    return result


@router.post("/cluster/android/heartbeat")
async def heartbeat_android_presence(req: AndroidPresenceRequest, request: Request):
    """Android Full 薄客户端心跳；实现与 register 相同，重复调用会刷新 last_heartbeat。"""
    return await register_android_presence(req, request)


@router.get("/cluster/master-health")
async def check_master_health():
    """
    检查主节点是否在线（通过数据库心跳时间戳）。

    从节点前端周期性调用此接口（配合 5 秒轮询），
    当检测到主节点宕机时显示告警横幅。
    主节点自身调用时返回本地运行状态。

    Returns:
        { master_online, last_seen_seconds_ago, stale, master_host, master_port }
    """
    if _scheduler()._effective_role() == "master":
        # 主节点自身：直接返回在线
        import config as _cfg

        return {
            "master_online": True,
            "last_seen_seconds_ago": 0,
            "stale": False,
            "master_host": getattr(_scheduler(), '_lan_ip', '') or getattr(
                _cfg, "SERVER_IP", "",
            ),
            "master_port": getattr(_cfg, "SERVER_PORT", 8888),
            "source": "self",
        }
    return _scheduler().get_client_master_status()


@router.get("/cluster/discover")
async def discover_master():
    """
    从数据库查询主节点的连接信息（从节点自动发现）。

    从节点启动后调用此接口，尝试在数据库中查找已注册的主节点。
    如果找到且在 120 秒内有心跳，则返回主节点地址，
    前端可自动填充连接表单。

     Returns:
         {
             "found": bool,           # 是否在数据库中找到主节点
             "master_host": str,      # 主节点 IP
             "master_port": int,      # 主节点端口
             "master_mac_addresses": [str],  # 主节点 MAC 地址（身份标识）
             "stale": bool,           # 心跳是否过期 (>120s)
             "source": str,           # "database" | "config" | "none"
         }
    """
    return _scheduler().discover_master()


class ResetIdentityRequest(BaseModel):
    confirm: str = Field(default="", description="输入 'reset' 确认重置")


@router.post("/cluster/reset-identity")
async def reset_master_identity(req: ResetIdentityRequest):
    """
    重置主节点身份标识（仅主节点可调用）。

    用于更换主节点机器或网卡后，清除数据库中旧的 MAC 地址记录。
    需要输入确认字符串 'reset' 以防止误操作。

    调用后需重启主节点后端服务，新的 MAC 地址将在下次启动时自动记录。
    """
    if req.confirm.strip().lower() != "reset":
        raise HTTPException(400, "请输入 'reset' 确认重置操作")
    result = _scheduler().reset_master_identity()
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "权限不足"))
    if result.get("status") == "error":
        raise HTTPException(500, result.get("reason", "操作失败"))
    return result


# ============================================================
# 有界 FIFO 请求队列 API
# ============================================================

class CancelTaskResponse(BaseModel):
    success: bool
    task_id: str
    message: str = ""


@router.get("/cluster/queue")
async def get_queue_detail():
    """
    获取推理调度队列完整详情。

    返回有界 FIFO 队列、终局决定和物理执行状态。
    仅主节点可用。
    """
    if not _scheduler()._effective_role() == "master":
        raise HTTPException(403, "仅主节点可查看请求队列")
    return _scheduler()._execution_app.requests.queue_snapshot()




@router.post("/cluster/queue/pause")
async def pause_queue():
    """暂停接受新请求。仅主节点。"""
    if not _scheduler()._effective_role() == "master":
        raise HTTPException(403, "仅主节点可暂停请求队列")
    _scheduler()._execution_app.requests.pause()
    return {"success": True, "paused": True}


@router.post("/cluster/queue/resume")
async def resume_queue():
    """恢复接受新请求。仅主节点。"""
    if not _scheduler()._effective_role() == "master":
        raise HTTPException(403, "仅主节点可恢复请求队列")
    _scheduler()._execution_app.requests.resume()
    return {"success": True, "paused": False}


@router.post("/cluster/queue/clear")
async def clear_queue():
    """清空所有排队任务（不影响执行中的任务）。仅主节点。"""
    if not _scheduler()._effective_role() == "master":
        raise HTTPException(403, "仅主节点可清空请求队列")
    count = _scheduler()._execution_app.requests.clear()
    return {"success": True, "cleared": count}


@router.delete("/cluster/queue/task/{task_id}")
async def cancel_queue_task(task_id: str):
    """
    取消指定排队任务。

    执行中的流水线任务会在当前 token step 完成后通过 PIPELINE_ABORT 中止。
    仅主节点。
    """
    if not _scheduler()._effective_role() == "master":
        raise HTTPException(403, "仅主节点可取消队列任务")
    ok = _scheduler()._execution_app.requests.cancel(task_id)
    if ok:
        return CancelTaskResponse(success=True, task_id=task_id, message="任务已取消")
    else:
        return CancelTaskResponse(
            success=False, task_id=task_id,
            message="任务不存在或已经完成，无法取消"
        )


# ============================================================
# 分布式推理开关 API
# ============================================================

@router.get("/cluster/config/distributed-inference")
async def get_distributed_inference_config():
    """
    获取分布式推理开关状态。
    """
    from config import DISTRIBUTED_INFERENCE_ENABLED
    return {
        "enabled": _scheduler().get_distributed_inference_enabled(),
        "default": DISTRIBUTED_INFERENCE_ENABLED,
    }


class DistributedInferenceRequest(BaseModel):
    enabled: bool = Field(..., description="是否启用分布式推理")


@router.put("/cluster/config/distributed-inference")
async def set_distributed_inference_config(req: DistributedInferenceRequest):
    """
    设置分布式推理开关。

    - 主节点：控制是否接收从节点连接和协调分布式推理
    - 从节点：控制是否将推理请求转发给主节点
    """
    result = await run_in_threadpool(
        _scheduler().set_distributed_inference_enabled, req.enabled,
    )
    if result.get("status") == "error":
        if result.get("reason_code") == "local_model_still_loaded":
            raise coded_http_error(409, "LOCAL_MODEL_STILL_LOADED", result["reason"])
        raise HTTPException(500, result.get("reason", "设置失败"))
    return result


# ============================================================
# 动态模型分层 API
# ============================================================

@router.get("/cluster/layers")
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
    return _scheduler().get_layer_assignments()


class LayerOverrideItem(BaseModel):
    node_id: str = Field(..., description="节点标识")
    start_layer: int = Field(..., ge=0, description="起始层（含）")
    end_layer: int = Field(..., ge=1, description="结束层（不含）")


class LayerOverrideRequest(BaseModel):
    assignments: list[LayerOverrideItem] = Field(..., min_length=1, description="分层覆盖列表")


@router.put("/cluster/layers")
async def override_layer_assignments(req: LayerOverrideRequest):
    """
    手动覆盖模型分层配置（仅主节点可调用）。

    验证规则:
      - 所有区间必须从 0 开始连续覆盖到 24
      - node_id 必须是已注册节点
      - 区间不能重叠
    """
    result = _scheduler().override_layer_assignments([
        {"node_id": a.node_id, "start_layer": a.start_layer, "end_layer": a.end_layer}
        for a in req.assignments
    ])
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "仅主节点可修改"))
    if result.get("status") == "invalid":
        raise HTTPException(400, result.get("reason", "分层配置无效"))
    if result.get("status") == "error":
        raise HTTPException(500, result.get("reason", "操作失败"))
    return result


@router.delete("/cluster/layers")
async def reset_layer_assignments():
    """
    重置分层配置，清除手动覆盖，恢复自动（dynamic）策略。

    仅主节点可调用。
    """
    if _scheduler()._effective_role() != "master":
        raise HTTPException(403, "仅主节点可重置分层配置")
    return _scheduler().reset_layer_assignments()


def _require_qwen3_local_master():
    sched = _scheduler()
    if sched._effective_role() != "master":
        raise HTTPException(403, "Qwen3 local chain is available only on the master node")
    return sched


def _raise_qwen3_local_http(exc: Exception) -> None:
    code = str(getattr(exc, "reason_code", "qwen3_local_chain_rejected"))
    message = str(getattr(exc, "reason", str(exc)))[:2048]
    status = 403 if "master" in code.lower() or "master" in message.lower() else 409 if any(token in code or token in message.lower() for token in (
        "phase", "stale", "active", "fenced", "duplicate",
    )) else 400
    raise HTTPException(status, {"code": code, "message": message}) from exc


@router.get("/cluster/model-runtime/sidecars")
async def get_model_runtime_sidecar_status():
    return await run_in_threadpool(_scheduler().get_model_runtime_sidecar_status)


@router.get("/cluster/model-runtime/contracts")
async def get_model_runtime_contracts():
    return await run_in_threadpool(_scheduler().get_model_runtime_contracts)


@router.post("/cluster/model-runtime/contracts/bind")
async def bind_model_runtime_contract(req: ModelRuntimeContractBindRequest):
    try:
        return await run_in_threadpool(
            _scheduler().bind_model_runtime_contract, req.profile, req.model_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_qwen3_local_http(exc)


@router.post("/cluster/model-runtime/sidecars/begin")
async def begin_model_runtime_sidecar(req: ModelRuntimeSidecarBeginRequest):
    try:
        if req.contract_id:
            return await run_in_threadpool(
                _scheduler().begin_model_runtime_sidecar,
                req.profile,
                req.contract,
                contract_id=req.contract_id,
            )
        return await run_in_threadpool(
            _scheduler().begin_model_runtime_sidecar, req.profile, req.contract,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_qwen3_local_http(exc)


@router.post("/cluster/model-runtime/sidecars/release")
async def release_model_runtime_sidecar(req: ModelRuntimeSidecarActionRequest):
    try:
        return await run_in_threadpool(_scheduler().release_model_runtime_sidecar, req.profile)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_qwen3_local_http(exc)


@router.delete("/cluster/model-runtime/sidecars/{profile}")
async def cancel_model_runtime_sidecar(profile: Literal["qwen3_sidecar", "gemma4_pipeline"]):
    try:
        return await run_in_threadpool(_scheduler().cancel_model_runtime_sidecar, profile)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_qwen3_local_http(exc)


@router.get("/cluster/qwen3/local-chain")
async def get_qwen3_local_chain_status():
    return await run_in_threadpool(_require_qwen3_local_master().get_qwen3_local_chain_status)


@router.post("/cluster/qwen3/local-chain/begin")
async def begin_qwen3_local_chain(req: Qwen3LocalChainBeginRequest):
    try:
        return await run_in_threadpool(
            _require_qwen3_local_master().begin_qwen3_local_sidecar_chain,
            req.contract,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_qwen3_local_http(exc)


@router.post("/cluster/qwen3/local-chain/prefill")
async def run_qwen3_local_prefill(req: Qwen3LocalChainExecuteRequest):
    try:
        return await run_in_threadpool(
            _require_qwen3_local_master().run_qwen3_local_prefill,
            input_ref=req.input_ref, batch_size=req.batch_size,
            sequence_length=req.sequence_length,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_qwen3_local_http(exc)


@router.post("/cluster/qwen3/local-chain/decode")
async def run_qwen3_local_decode(req: Qwen3LocalChainExecuteRequest):
    try:
        return await run_in_threadpool(
            _require_qwen3_local_master().run_qwen3_local_decode,
            input_ref=req.input_ref, batch_size=req.batch_size,
            sequence_length=req.sequence_length,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_qwen3_local_http(exc)


@router.post("/cluster/qwen3/local-chain/parity")
async def verify_qwen3_local_parity(req: Qwen3LocalChainParityRequest):
    try:
        return await run_in_threadpool(
            _require_qwen3_local_master().verify_qwen3_local_cpu_parity,
            reference_prefill=req.reference_prefill,
            reference_decode=req.reference_decode,
            rtol=req.rtol, atol=req.atol,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_qwen3_local_http(exc)


@router.post("/cluster/qwen3/local-chain/release")
async def release_qwen3_local_chain():
    try:
        return await run_in_threadpool(
            _require_qwen3_local_master().release_qwen3_local_sidecar_chain,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_qwen3_local_http(exc)


@router.delete("/cluster/qwen3/local-chain")
async def cancel_qwen3_local_chain():
    try:
        return await run_in_threadpool(
            _require_qwen3_local_master().cancel_qwen3_local_sidecar_chain,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_qwen3_local_http(exc)


# =====================================================================
# device 域（复制自 api_server.py:1820-1960，宿主适配：无 kv_cache/
# model_host —— 推理在 inference-svc，壳只负责画像存储与配置应用）
# =====================================================================
@router.get("/device/profile")
async def get_device_profile():
    """获取完整设备画像（与 api_server /api/device/profile 字段一致）。"""
    global _device_profile_cache
    if _device_profile_cache is None:
        try:
            _device_profile_cache = await asyncio.to_thread(
                _detect_device_profile,
            )
            sched = _scheduler()
            try:
                sched.update_local_device_profile(_device_profile_cache)
            except Exception:
                pass
        except Exception as exc:
            raise coded_http_error(
                500, "DEVICE_PROFILE_DETECTION_FAILED", f"设备检测失败: {exc}",
            )
    return _with_compat_device_fields(_device_profile_cache)


@router.post("/device/auto-configure")
async def auto_configure():
    """根据设备画像自动应用推荐配置（配置应用保留在 scheduler-svc；
    KV 缓存重建属 inference-svc 范畴，壳内跳过——api_server 的
    kv_cache/model_host 部分由 inference-svc /v1/models 承接）。"""
    global _device_profile_cache
    if _device_profile_cache is None:
        try:
            _device_profile_cache = await asyncio.to_thread(
                _detect_device_profile,
            )
        except Exception as exc:
            raise coded_http_error(
                500, "DEVICE_PROFILE_DETECTION_FAILED", f"设备检测失败: {exc}",
            )
    sched = _scheduler()
    try:
        sched.update_local_device_profile(_device_profile_cache)
    except Exception:
        pass

    rec = _device_profile_cache.get("recommendations", [])
    warnings = _device_profile_cache.get("warnings", [])
    tier = _device_profile_cache.get("tier", "laptop")
    score = _device_profile_cache.get("score_total", 50)

    from device_profiler import get_profile

    profiler = get_profile()
    config = profiler.recommend_config()

    # 应用推荐配置到 config（KV 缓存/模型侧由 inference-svc 在 load 时消费）
    import config as cfg

    cfg.PAGE_SIZE = config["page_size"]
    cfg.MAX_PAGE_NUM = config["max_pages"]
    cfg.MAX_SEQ_LEN = config["max_seq_len"]

    logger.info(f"自适应配置已应用: {config['description']}")
    return {
        "status": "configured",
        "tier": tier,
        "score": score,
        "applied_config": config,
        "recommendations": rec,
        "warnings": warnings,
    }


class SelectGpuRequest(BaseModel):
    gpu_index: int = Field(..., ge=0, description="GPU 列表中要切换到的序号")


@router.post("/device/select-gpu")
async def select_gpu(req: SelectGpuRequest):
    """切换推理 GPU（更新 profiler 选中项与缓存画像；模型重载属
    inference-svc /v1/models/load 范畴）。"""
    global _device_profile_cache
    if _device_profile_cache is None:
        raise coded_http_error(
            400,
            "DEVICE_PROFILE_NOT_READY",
            "设备画像未就绪，请先调用 GET /device/profile",
        )

    gpus = _device_profile_cache.get("gpus", [])
    if req.gpu_index < 0 or req.gpu_index >= len(gpus):
        raise coded_http_error(
            400,
            "DEVICE_GPU_INDEX_INVALID",
            f"无效的 GPU 序号: {req.gpu_index}。"
            f"可用范围: 0-{len(gpus) - 1}（共 {len(gpus)} 个 GPU）",
        )

    from device_profiler import get_profile

    profiler = get_profile()
    if not profiler.select_gpu(req.gpu_index):
        raise HTTPException(500, "GPU 切换失败")
    _device_profile_cache = profiler.to_dict()
    sched = _scheduler()
    try:
        sched.update_local_device_profile(_device_profile_cache)
    except Exception:
        pass

    selected = gpus[req.gpu_index]
    logger.info(
        f"GPU 已切换: [{req.gpu_index}] {selected['name']} "
        f"({selected['gpu_type']}, CUDA: {selected['cuda_available']})"
    )
    return {
        "status": "switched",
        "selected_gpu_index": req.gpu_index,
        "selected_gpu": {
            "name": selected["name"],
            "gpu_type": selected["gpu_type"],
            "cuda_available": selected["cuda_available"],
            "vram_total_gb": selected["vram_total_gb"],
        },
        "device": profiler.recommend_config()["device"],
        "warning": "切换 GPU 后需要重新加载模型才能生效。",
    }


def build_scheduler_app(scheduler) -> "FastAPI":
    """构造 scheduler-svc HTTP 应用（:8020，QLH_SCHEDULER_HTTP_PORT）。"""
    from fastapi import FastAPI

    set_scheduler(scheduler)
    app = FastAPI(title="scheduler-svc", version=PRODUCT_VERSION)
    install_http_error_handler(app)
    app.include_router(router)
    transfer_runtime = getattr(scheduler, "_qwen3_artifact_transfer_runtime", None)
    peer_verifier = getattr(scheduler, "_qwen3_peer_request_verifier", None)
    network_coordinator = getattr(scheduler, "_qwen3_network_transfer_coordinator", None)
    if transfer_runtime is not None and peer_verifier is not None:
        from qwen3_pipeline_data_plane import router as qwen3_transfer_router
        from qwen3_pipeline_peer_auth import Qwen3PeerAuthMiddleware

        app.state.qwen3_artifact_transfer = transfer_runtime
        app.add_middleware(Qwen3PeerAuthMiddleware, verifier=peer_verifier)
        app.include_router(qwen3_transfer_router)
        if network_coordinator is not None:
            from qwen3_pipeline_control import router as qwen3_network_control_router
            app.state.qwen3_network_transfer_coordinator = network_coordinator
            app.include_router(qwen3_network_control_router)
    else:
        app.state.qwen3_artifact_transfer_reason = "scheduler_transfer_not_configured"
    return app
