"""A single selection entry point, served locally or relayed to the bound master."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response
from api.routes_model_supply import ModelSupplyIntent, _error, prepare_supply
from model_supply import ModelSupplyError
from model_intent import ModelIntentCoordinator
from model_intent_gateway import get_binding, forward_intent
from model_intent_access import require_model_intent_source
import model_config as mc
from model_pipeline_availability import (
    ModelPipelineAvailabilityError, model_pipeline_catalog_payload,
    require_model_pipeline_available,
)

router = APIRouter()
_api_module = None
_coordinator = None


def configure_api_module(module):
    global _api_module
    _api_module = module


def exported_handlers():
    return {}


def _retirement_identity(scheduler):
    with scheduler._layer_config_lock:
        return tuple(None if not state else (
            state.get("config_id"), state.get("generation"), state.get("candidate_id"),
            state.get("phase", state.get("transaction_phase")),
        ) for state in (
            scheduler._pipeline_load_transaction,
            scheduler._active_pipeline_capacity_plan,
            (getattr(scheduler, "_pipeline_recovery_state", None)
             if getattr(scheduler, "_pipeline_recovery_pending", False) else None),
        ))


async def _retire_native_plan():
    api = _api_module
    scheduler = api.scheduler
    expected = _retirement_identity(scheduler)
    if not any(expected):
        # Only cleanup debt remains. Never issue another unload after retirement.
        return

    def retire():
        # This owner must join native cleanup outside full-chat, as on UI unload.
        api._wait_for_worker_prepare_retirement("local_model_load")
        with api.model_host.full_chat_execution_lock:
            # The original transition fence also excludes ordinary prepare, whose
            # transaction admission does not take full-chat. Use its nesting rules.
            scheduler._begin_layer_config_model_change()
            try:
                scheduler._model_supply._check_master()
                if _retirement_identity(scheduler) != expected:
                    raise ModelSupplyError("INTENT_RETIRE_SUPERSEDED", "The native model changed while retirement waited; retry the selection")
                return api._unload_model_under_model_lock(retire_worker_runtime=False)
            finally:
                scheduler._end_layer_config_model_change()

    result = await run_in_threadpool(retire)
    if not isinstance(result, dict) or result.get("success") is not True or result.get("loaded") is not False:
        raise ModelSupplyError("INTENT_RETIRE_FAILED", "Native model retirement failed; retry the selection")
    # The coordinator waits for the scheduler's identity-bound remote cleanup ACKs.


def coordinator():
    global _coordinator
    if _coordinator is None or _coordinator.scheduler is not _api_module.scheduler:
        _coordinator = ModelIntentCoordinator(_api_module.scheduler, lambda operation_id: prepare_supply(operation_id, None),
                                              retire=_retire_native_plan)
    return _coordinator


def _master():
    return _api_module.scheduler._effective_role() == "master"


async def catalog(request: Request):
    require_model_intent_source(request, _api_module.scheduler)
    binding = get_binding(_api_module.scheduler, _api_module.API_PORT)
    if not _master():
        result = await forward_intent(_api_module.scheduler, request, "catalog")
        if isinstance(result, dict):
            result["binding"] = binding
        return result
    # Remote selectors never need the master's filesystem paths.
    allowed = {"model_id", "name", "description", "model_type", "is_builtin", "is_experimental",
               "is_available", "unavailable_reason", "available_formats", "has_safetensors", "has_gguf",
               "preferred_engine", "supported_engines", "quant_types", "default_quant_type", "max_context",
               "recommended_vram_gb", "pipeline_engine_availability",
               "pipeline_auto_engine", "pipeline_auto_unavailable_reason"}
    models = await run_in_threadpool(lambda: [
        {key: value for key, value in model_pipeline_catalog_payload(
            entry, _api_module._model_api_payload(entry),
        ).items() if key in allowed}
        for entry in _api_module._get_all_model_configs()
    ])
    current = coordinator().current()
    return {"models": models, "active_model_id": current["intent"]["model_id"] if current else None, "binding": binding}


async def connect(request: Request):
    """Use existing authenticated discovery/bootstrap, with no caller-supplied host."""
    require_model_intent_source(request, _api_module.scheduler)
    scheduler = _api_module.scheduler
    if not scheduler.get_distributed_inference_enabled():
        raise HTTPException(409, {"code": "INTENT_LOCAL_MODE", "message": "Switch to distributed inference to join the cluster"})
    binding = get_binding(scheduler, _api_module.API_PORT)
    if binding["status"] == "connected" and not (_master() and scheduler.can_join_existing_master()):
        return binding
    found = await run_in_threadpool(scheduler.discover_master)
    if not found.get("found"):
        raise HTTPException(409, {"code": "INTENT_MASTER_NOT_FOUND", "message": "Start the master on an approved network and retry discovery"})
    from api.routes_cluster import connect_to_master
    req = _api_module.ConnectToMasterRequest(master_host=found["master_host"], master_port=found["master_port"], switch_to_client=True)
    await connect_to_master(req)
    return get_binding(scheduler, _api_module.API_PORT)


def _preflight_selection(intent):
    model = mc.get_model_config(intent.model_id, _api_module._get_registered_experimental_models())
    require_model_pipeline_available(model, intent.engine)


async def create(intent: ModelSupplyIntent, request: Request):
    require_model_intent_source(request, _api_module.scheduler)
    if not _master():
        return await forward_intent(_api_module.scheduler, request, "", intent.model_dump())
    try:
        # Reject only definite local incompatibility, before create can retire
        # the active native plan. Acquisition and content proof stay with supply.
        await run_in_threadpool(_preflight_selection, intent)
        key = request.headers.get("Idempotency-Key", "")
        if key:
            import re
            if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", key) is None:
                raise ModelSupplyError("INTENT_IDEMPOTENCY_KEY_INVALID", "Invalid idempotency key")
        return await coordinator().create(intent.model_dump(), **({"idempotency_key": key} if key else {}))
    except (ModelSupplyError, ModelPipelineAvailabilityError) as exc:
        raise _error(exc) from exc


async def current(request: Request):
    require_model_intent_source(request, _api_module.scheduler)
    if not _master():
        return await forward_intent(_api_module.scheduler, request, "current")
    try:
        return coordinator().current()
    except ModelSupplyError as exc:
        raise _error(exc) from exc


async def get(operation_id: str, request: Request):
    require_model_intent_source(request, _api_module.scheduler)
    if not _master():
        return await forward_intent(_api_module.scheduler, request, operation_id)
    try:
        return coordinator().get(operation_id)
    except ModelSupplyError as exc:
        raise _error(exc) from exc


async def cancel(operation_id: str, request: Request):
    require_model_intent_source(request, _api_module.scheduler)
    if not _master():
        return await forward_intent(_api_module.scheduler, request, operation_id)
    try:
        return await run_in_threadpool(coordinator().cancel, operation_id)
    except ModelSupplyError as exc:
        raise _error(exc) from exc


def register_routes():
    router.add_api_route("/api/models/intent/catalog", catalog, methods=["GET"])
    router.add_api_route("/api/models/intent/connect", connect, methods=["POST"])
    router.add_api_route("/api/models/intent/current", current, methods=["GET"])
    router.add_api_route("/api/models/intent", create, methods=["POST"])
    router.add_api_route("/api/models/intent/{operation_id}", get, methods=["GET"])
    router.add_api_route("/api/models/intent/{operation_id}", cancel, methods=["DELETE"])

async def shutdown():
    if _coordinator is not None:
        await _coordinator.close()
