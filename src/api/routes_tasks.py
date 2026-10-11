"""Routes extracted from api_server; shared state remains facade-owned."""

from __future__ import annotations

from types import ModuleType

from fastapi import APIRouter
from api._routing import configure_route_module
from journal_replication import JournalReplicationError
from journal_replication_service import (
    JournalReplicationDisabled,
    JournalReplicationSettings,
    JournalReplicationTransportError,
    replicate_journal_once,
)

router = APIRouter()
_api_module: ModuleType | None = None
_RESOLUTION_NAMES = ("Optional",)

def configure_api_module(module: ModuleType) -> None:
    configure_route_module(globals(), module, _RESOLUTION_NAMES)

def exported_handlers() -> dict[str, object]:
    return {name: globals()[name] for name in ['list_workflows', 'cleanup_task_journal', 'replicate_journal', 'journal_replication_status', 'get_workflow', 'cancel_workflow']}

async def list_workflows(limit: int = 20, session_id: str = "", summary: bool = False):
    if summary:
        # Mobile audit is deliberately bounded server-side, independent of a
        # caller-provided page size.
        limit = max(1, min(int(limit), 8))
    role = _api_module.scheduler._effective_role()
    provider_error = ""
    if _api_module.TASK_GRAPH_ENABLED and role == "master":
        try:
            _api_module._ensure_local_task_provider()
            _api_module._sync_remote_task_worker_providers()
        except _api_module.ProviderError as exc:
            provider_error = f"{exc.code}: {exc}"
    journal = _api_module.task_graph_coordinator.journal_status()
    try:
        workflows = _api_module.task_graph_coordinator.list(
            limit=limit, session_id=session_id,
        )
    except _api_module.TaskGraphUnavailable:
        journal = _api_module.task_graph_coordinator.journal_status()
        workflows = _api_module.task_graph_coordinator.list(
            limit=limit, session_id=session_id,
        )
    public_journal = _api_module._public_task_journal(journal)
    workflows = [
        (_api_module._public_workflow_summary(workflow, journal) if summary else _api_module._public_workflow(workflow, journal))
        for workflow in workflows
    ]
    provider_status = _api_module.task_graph_coordinator.provider_status()
    local_provider_ready = any(
        item.get("provider_id") == "local_full_model"
        and bool(item.get("healthy"))
        and bool(item.get("available"))
        for item in provider_status
    )
    local_available = bool(
        _api_module.TASK_GRAPH_ENABLED
        and role == "master"
        and _api_module.NODE_ROLE != "auto"
        and bool(journal.get("available", False))
        and not provider_error
    )
    if summary:
        return {
            "enabled": bool(_api_module.TASK_GRAPH_ENABLED),
            "available": bool(journal.get("available", False)),
            "role": role,
            "workflows": workflows,
            "journal": {
                key: public_journal.get(key)
                for key in ("available", "record_count", "retention_days")
                if key in public_journal
            },
        }
    return {
        "enabled": _api_module.TASK_GRAPH_ENABLED,
        # ``available`` retains provider-readiness semantics for API clients;
        # the UI uses ``local_available`` so physical Worker admission cannot
        # lock the local experiment selector.
        "available": bool(local_available and local_provider_ready),
        "local_available": local_available,
        "local_provider_ready": local_provider_ready,
        "role": role,
        "templates": ["dual_candidate"],
        "providers": _api_module.task_graph_coordinator.provider_ids(),
        "provider_status": provider_status,
        "provider_error": provider_error,
        "worker_protocol": _api_module.scheduler.get_task_worker_protocol_status(),
        "controls": _api_module._task_graph_feature_settings(),
        "journal": public_journal,
        "workflows": workflows,
}

async def cleanup_task_journal(
    max_age_days: Optional[int] = None,
    max_records: Optional[int] = None,
):
    if _api_module.scheduler._effective_role() != "master":
        raise _api_module.HTTPException(403, "只有主节点可以清理任务图 journal。")
    if _api_module.NODE_ROLE == "auto":
        raise _api_module.HTTPException(503, "自动选主尚无 journal 多数提交水位，禁止清理。")
    if not _api_module.TASK_GRAPH_ENABLED:
        raise _api_module.HTTPException(409, "任务链实验未启用。")
    age_days = (
        _api_module.TASK_GRAPH_RETENTION_DAYS
        if max_age_days is None else max(0, min(int(max_age_days), 3650))
    )
    records = (
        _api_module.TASK_GRAPH_RETENTION_MAX_RECORDS
        if max_records is None else max(0, min(int(max_records), 100000))
    )
    try:
        result = _api_module.task_graph_coordinator.cleanup_journal(
            max_age_days=age_days,
            max_records=records,
        )
    except _api_module.TaskGraphUnavailable as exc:
        raise _api_module.HTTPException(503, str(exc)) from exc
    return {
        "status": "completed",
        "policy": {
            "max_age_days": age_days,
            "max_records": records,
        },
        "result": result,
    }

async def replicate_journal(workflow_id: str):
    """把一个 workflow 的 journal checkpoint 推到对端副本服务（R-R11 的最小可用接线）。

    三重 fail-closed，顺序刻意如此：非主节点 **403**、未开启 **409**、journal 不可用 **503**，
    然后才是真正推送（对端拒绝 ⇒ **502**，参数/契约问题 ⇒ **400**）。
    """

    if _api_module.scheduler._effective_role() != "master":
        raise _api_module.HTTPException(403, "只有主节点可以推送 journal 副本。")
    settings = JournalReplicationSettings.from_env()
    if not settings.enabled:
        raise _api_module.HTTPException(
            409, "journal 复制未启用（QLH_JOURNAL_REPLICATION_ENABLED）。"
        )
    journal = getattr(_api_module.task_graph_coordinator, "journal", None)
    if journal is None:
        raise _api_module.HTTPException(503, "任务图 journal 不可用，无法取快照。")
    try:
        return replicate_journal_once(journal, workflow_id=workflow_id, settings=settings)
    except JournalReplicationDisabled as exc:
        raise _api_module.HTTPException(409, str(exc)) from exc
    except JournalReplicationTransportError as exc:
        raise _api_module.HTTPException(502, str(exc)) from exc
    except JournalReplicationError as exc:
        raise _api_module.HTTPException(400, str(exc)) from exc


async def journal_replication_status():
    """复制链路的只读状态（**不回显 secret**）：运维据此确认「到底开没开、对端是谁」。"""

    settings = JournalReplicationSettings.from_env()
    journal = getattr(_api_module.task_graph_coordinator, "journal", None)
    return {
        "enabled": settings.enabled,
        "settings": settings.public(),
        "journal_available": journal is not None,
        "role": _api_module.scheduler._effective_role(),
        "note": (
            "最小可用接线：不自动常驻、不随 NODE_ROLE 自动启停；"
            "复制由 POST /api/workflows/journal/replicate 显式驱动。"
        ),
    }


async def get_workflow(workflow_id: str):
    try:
        workflow = _api_module.task_graph_coordinator.get(workflow_id)
        journal = _api_module.task_graph_coordinator.journal_status()
        return _api_module._public_workflow(workflow, journal)
    except _api_module.WorkflowNotFound as exc:
        raise _api_module.HTTPException(404, f"工作流不存在: {workflow_id}") from exc
    except _api_module.TaskGraphUnavailable as exc:
        raise _api_module.HTTPException(503, str(exc)) from exc

async def cancel_workflow(workflow_id: str):
    if _api_module.scheduler._effective_role() != "master":
        raise _api_module.HTTPException(403, "只有主节点可以取消工作流。")
    if _api_module.NODE_ROLE == "auto":
        raise _api_module.HTTPException(503, "自动选主尚无 journal 多数提交水位，禁止修改工作流。")
    try:
        workflow = _api_module.task_graph_coordinator.request_cancel(workflow_id)
    except _api_module.TaskGraphUnavailable as exc:
        raise _api_module.HTTPException(503, str(exc)) from exc
    except _api_module.TaskGraphError as exc:
        raise _api_module.HTTPException(400, str(exc)) from exc
    if workflow is None:
        return {
            "status": "cancel_pending",
            "workflow": {
                "workflow_id": workflow_id,
                "state": "pending_registration",
                "cancel_requested": True,
            },
        }
    return {
        "status": (
            "cancel_requested"
            if workflow["state"] not in {"completed", "failed", "cancelled"}
            else workflow["state"]
        ),
        "workflow": workflow,
    }


def register_routes() -> None:
    router.add_api_route('/api/workflows', list_workflows, methods=['GET'])
    router.add_api_route('/api/workflows/journal/cleanup', cleanup_task_journal, methods=['POST'])
    # 复制端点刻意排在 `/api/workflows/{workflow_id}` **之前**（固定段先注册更稳妥）。
    router.add_api_route('/api/workflows/journal/replicate', replicate_journal, methods=['POST'])
    router.add_api_route('/api/workflows/journal/replication', journal_replication_status, methods=['GET'])
    router.add_api_route('/api/workflows/{workflow_id}', get_workflow, methods=['GET'])
    router.add_api_route('/api/workflows/{workflow_id}/cancel', cancel_workflow, methods=['POST'])
