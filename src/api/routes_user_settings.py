"""Local user-preference routes, independent of account authentication."""

from __future__ import annotations

from types import ModuleType

from fastapi import APIRouter
from pydantic import BaseModel, Field
from api._routing import configure_route_module

router = APIRouter()
_api_module: ModuleType | None = None


class UserSettingsRequest(BaseModel):
    settings: dict = Field(default_factory=dict, description="完整的用户设置 JSON")


def configure_api_module(module: ModuleType) -> None:
    configure_route_module(globals(), module, ())


def exported_handlers() -> dict[str, object]:
    return {
        "get_user_settings": get_user_settings,
        "update_user_settings": update_user_settings,
    }


async def get_user_settings():
    try:
        settings = _api_module._local_store.get_local_user_settings()
        return {"settings": settings, "source": "sqlite"}
    except Exception as exc:
        _api_module.logger.error("读取 SQLite 用户设置失败: %s", exc)
        raise _api_module.HTTPException(503, f"本地设置存储不可用: {exc}") from exc


async def update_user_settings(req: UserSettingsRequest):
    try:
        _api_module._local_store.set_local_user_settings(req.settings)
    except Exception as exc:
        _api_module.logger.error("存储 SQLite 用户设置失败: %s", exc)
        raise _api_module.HTTPException(503, f"本地设置存储失败: {exc}") from exc

    return {
        "status": "ok",
        "source": "sqlite",
        "legacy_exported": False,
        "synced_fields": list(req.settings.keys()),
    }


def register_routes() -> None:
    router.add_api_route("/api/user/settings", get_user_settings, methods=["GET"])
    router.add_api_route("/api/user/settings", update_user_settings, methods=["PUT"])
