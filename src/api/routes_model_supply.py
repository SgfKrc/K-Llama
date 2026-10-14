"""Master model selection -> verified segment supply, without executor readiness claims."""
from __future__ import annotations
from types import ModuleType
from typing import Literal

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse, Response

from model_supply import ModelSupplyError

router = APIRouter()
_api_module: ModuleType | None = None


class ModelSupplyIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    engine: Literal["auto", "llama_cpp", "pytorch"] = "auto"
    quant_type: str = Field(default="", max_length=32)
    strategy: Literal["auto", "distribute", "local_cut"] = "auto"
    node_ids: list[str] = Field(default_factory=list, max_length=32)
    timeout_seconds: int = Field(default=3600, ge=30, le=86400)


def configure_api_module(module):
    global _api_module
    _api_module = module


def exported_handlers():
    return {}


def _error(exc):
    status = 404 if exc.code.endswith("NOT_FOUND") else 403 if exc.code.endswith("FORBIDDEN") else 409
    return HTTPException(status, {"code": exc.code, "message": exc.message})


def _public(operation):
    operation = dict(operation)
    operation.pop("master_model_path", None)
    return operation


async def create_supply(intent: ModelSupplyIntent, request: Request):
    _api_module.require_model_api_source(request)
    try:
        return _public(await run_in_threadpool(_api_module.scheduler._model_supply.create, intent.model_dump()))
    except ModelSupplyError as exc:
        raise _error(exc) from exc


async def get_supply(operation_id: str, request: Request):
    _api_module.require_model_api_source(request)
    try:
        return _public(_api_module.scheduler._model_supply.get(operation_id))
    except ModelSupplyError as exc:
        raise _error(exc) from exc


async def cancel_supply(operation_id: str, request: Request):
    _api_module.require_model_api_source(request)
    try:
        return _public(_api_module.scheduler._model_supply.cancel(operation_id))
    except ModelSupplyError as exc:
        raise _error(exc) from exc


async def prepare_supply(operation_id: str, request: Request):
    """Reuse the existing model transaction; supply success itself never loads."""
    _api_module.require_model_api_source(request)
    try:
        with _api_module.scheduler._model_supply.prepare_guard(operation_id) as source:
            from model_load_resolver import ModelLoadResolution
            resolution = ModelLoadResolution(
                model_id=source["model_id"], requested_engine=source["engine"], engine=source["engine"],
                artifact_kind="gguf" if source["engine"] == "llama_cpp" else "safetensors",
                model_path=source["model_path"], requested_quant=source["quant_type"],
                quant_type=source["quant_type"], runtime="llama_cpp_native" if source["engine"] == "llama_cpp" else "pytorch_cuda",
                runtime_quant=source["quant_type"],
                layer_range_mode="precut_artifact" if source["engine"] == "llama_cpp" else "dynamic",
                reason_code="SUPPLY_ASSETS_AVAILABLE", reason="Use verified supplied segment")
            from api.routes_models import _prepare_pipeline_model_resolution
            result = await _prepare_pipeline_model_resolution(resolution, operation_id=operation_id)
            result["supply_operation_id"] = operation_id
            return result
    except ModelSupplyError as exc:
        raise _error(exc) from exc

async def supply_file(operation_id: str, node_id: str, filename: str, request: Request):
    try:
        path = _api_module.scheduler._model_supply.authorize_file(
            operation_id, node_id, filename, request.headers.get("X-QLH-Supply-Token", ""))
    except ModelSupplyError as exc:
        raise _error(exc) from exc
    size = path.stat().st_size
    start, end, status = 0, size - 1, 200
    raw = request.headers.get("range", "")
    if raw:
        import re
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", raw)
        if not match:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        start = int(match[1])
        end = min(int(match[2]), size - 1) if match[2] else size - 1
        if start < 0 or start >= size or end < start:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        status = 206
    def chunks():
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                # Cancellation/rejoin revokes even an already open response.
                try:
                    _api_module.scheduler._model_supply.authorize_file(
                        operation_id, node_id, filename, request.headers.get("X-QLH-Supply-Token", ""))
                except ModelSupplyError:
                    return
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    return
                remaining -= len(chunk)
                yield chunk
    headers = {"Accept-Ranges": "bytes", "Content-Length": str(max(0, end - start + 1))}
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(chunks(), status_code=status, headers=headers, media_type="application/octet-stream")


def register_routes():
    router.add_api_route("/api/models/supply", create_supply, methods=["POST"])
    router.add_api_route("/api/models/supply/{operation_id}", get_supply, methods=["GET"])
    router.add_api_route("/api/models/supply/{operation_id}", cancel_supply, methods=["DELETE"])
    router.add_api_route("/api/models/supply/{operation_id}/prepare", prepare_supply, methods=["POST"])
    router.add_api_route("/api/models/supply/{operation_id}/files/{node_id}/{filename:path}", supply_file, methods=["GET"])
