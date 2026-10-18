"""Version error for retired workflow product endpoints."""
from fastapi import APIRouter
from api_errors import coded_http_error
router = APIRouter()
def configure_api_module(module):
    pass
def exported_handlers():
    return {}
async def retired_workflow(path: str = ""):
    raise coded_http_error(410, "CLIENT_FEATURE_REMOVED", {"code": "CLIENT_FEATURE_REMOVED", "feature": "workflow", "replacement": "linear_execution", "contract_version": 2})
def register_routes():
    router.add_api_route("/api/workflows", retired_workflow, methods=["GET", "POST"], include_in_schema=False)
    router.add_api_route("/api/workflows/{path:path}", retired_workflow, methods=["GET", "POST", "DELETE"], include_in_schema=False)
    for path in ("/api/experimental/speculative", "/api/experimental/speculative/capability", "/api/cluster/config/task-graph", "/api/cluster/queue/strategy"):
        router.add_api_route(path, retired_workflow, methods=["GET", "POST"], include_in_schema=False)
