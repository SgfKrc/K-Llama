"""Routes extracted from api_server; shared state remains facade-owned.

★ 2026-10-18：`/api/diagnostics/product-source` 与 `/api/diagnostics/product-runtime`
原先直接以 `@app.get(...)` 写在 `api_server.py` 里，违反
`REFACTOR-LARGEFILE-04/05` 的结构约束（端点不得由门面持有，
`tests/test_refactor_largefile_baseline.py::test_api_facade_no_longer_owns_endpoint_definitions`
就是守这条的）。这里按既有模式搬进独立 router，行为不变：
`api_server.product_source_observation` 等名字仍由 `exported_handlers()` 回填。
"""

from __future__ import annotations

from types import ModuleType

from fastapi import APIRouter
from api._routing import configure_route_module

router = APIRouter()
_api_module: ModuleType | None = None
_RESOLUTION_NAMES = ("Request",)


def configure_api_module(module: ModuleType) -> None:
    configure_route_module(globals(), module, _RESOLUTION_NAMES)


def exported_handlers() -> dict[str, object]:
    return {
        name: globals()[name]
        for name in ("product_source_observation", "product_runtime_observation")
    }


def product_source_observation(request: Request, nonce: str):
    """Observe installed/source bytes before any model has been prepared."""
    _api_module._require_product_observation_request(request, nonce)
    import psutil

    process = psutil.Process(_api_module.os.getpid())
    role = "master" if _api_module.scheduler._effective_role() == "master" else "pc"
    try:
        source = _api_module._product_runtime_source.observe_source(role)
    except ValueError as exc:
        raise _api_module.HTTPException(
            409, {"code": "PRODUCT_SOURCE_UNAVAILABLE", "message": str(exc)},
        ) from exc
    return {
        "nonce": nonce,
        "pid": process.pid,
        "created_at": process.create_time(),
        "captured_at": _api_module.time.time(),
        "source": source,
        "errors": [],
    }


def product_runtime_observation(request: Request, nonce: str):
    """Explicit lab observation over the existing loopback API and auth boundary."""
    _api_module._require_product_observation_request(request, nonce)
    from product_runtime_receipt import capture_runtime
    import psutil

    process = psutil.Process(_api_module.os.getpid())
    role = "master" if _api_module.scheduler._effective_role() == "master" else "pc"
    # The helper reads the existing runtime owners with their established locks;
    # source rehash happens separately and never constructs a model manager.
    # `capture_runtime` 需要一个模块对象来取运行态 owner —— 这里传门面模块本身，
    # 与搬迁前的 `sys.modules[__name__]` 完全等价。
    try:
        runtime = capture_runtime(_api_module)
        source = _api_module._product_runtime_source.observe_source(role)
    except ValueError as exc:
        raise _api_module.HTTPException(
            409, {"code": "PRODUCT_OBSERVATION_UNAVAILABLE", "message": str(exc)},
        ) from exc
    return {
        "nonce": nonce,
        "pid": process.pid,
        "created_at": process.create_time(),
        "captured_at": _api_module.time.time(),
        "runtime": runtime,
        "source": source,
        "errors": [],
    }


def register_routes() -> None:
    router.add_api_route(
        '/api/diagnostics/product-source', product_source_observation, methods=['GET'])
    router.add_api_route(
        '/api/diagnostics/product-runtime', product_runtime_observation, methods=['GET'])
