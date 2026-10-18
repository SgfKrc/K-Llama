"""Thin HTTP and SSE adapters over versioned owner observations."""
import asyncio
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from qlh_core.bootstrap.client_snapshots import ClientSnapshots
router = APIRouter()
_api_module = None
def configure_api_module(module):
    global _api_module
    _api_module = module
def exported_handlers():
    return {}
async def client_state():
    return await asyncio.to_thread(ClientSnapshots(_api_module.scheduler).current)
async def client_events(request: Request):
    async def events():
        previous = None
        while not await request.is_disconnected():
            value = await client_state()
            signature = (value.get("revision"), value.get("queue", {}).get("revision"), value.get("participation"), value.get("runnable"), value.get("validity"), value.get("reason_detail"))
            if signature != previous:
                yield "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"
                previous = signature
            else:
                yield ": keepalive\n\n"
            await asyncio.sleep(0.5)
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
async def recent_requests(limit: int = 20):
    return _api_module.scheduler._execution_app.requests.recent(limit)
def register_routes():
    router.add_api_route("/api/requests", recent_requests, methods=["GET"])
    router.add_api_route("/api/client/state", client_state, methods=["GET"])
    router.add_api_route("/api/client/events", client_events, methods=["GET"])
