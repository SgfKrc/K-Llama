"""Routes extracted from api_server; shared state remains facade-owned."""

from __future__ import annotations

from types import ModuleType

from fastapi import APIRouter, Depends, Header
import auth_service
from api._routing import configure_route_module

router = APIRouter()
_api_module: ModuleType | None = None
_RESOLUTION_NAMES = (
    "CreateUserRequest",
    "LoginRequest",
    "Optional",
    "PatchUserRequest",
    "Request",
    "TotpVerifyRequest",
    "UserSettingsRequest",
)

def configure_api_module(module: ModuleType) -> None:
    configure_route_module(globals(), module, _RESOLUTION_NAMES)

def exported_handlers() -> dict[str, object]:
    return {name: globals()[name] for name in ['auth_capability', 'auth_login', 'auth_logout', 'auth_me', 'auth_totp_provision', 'auth_totp_verify', 'list_users', 'create_user', 'patch_user', 'delete_user', 'get_user_settings', 'update_user_settings']}

async def auth_capability():
    """认证能力探测（**本地**，不再是 control-svc 反代）。

    返回：
        available       认证实现是否可用（本进程内）
        required        是否强制登录（`QLH_AUTH_REQUIRED`）
        bootstrap_open  是否处于首次引导（库中尚无任何账户）
        user_count      账户数
    """
    return _api_module.auth_service.auth_capability_payload()

async def auth_login(req: LoginRequest, request: Request):
    """登录并签发 Bearer token（**明文只在本次响应返回**）。

    ⚠️ 首次引导（库中无账户）时，请改用 `POST /api/users` 创建第一个 admin —— 该端点
    在引导期**无需登录**。
    """
    token, rec = _api_module.auth_service.login(
        req.username, req.password, totp_code=req.totp_code,
        source=_api_module.auth_service.request_source(request),
    )
    return {
        "status": "ok",
        "token": token,
        "username": rec.username,
        "role": rec.role,
        "expires_at": rec.expires_at,
    }

async def auth_logout(authorization: Optional[str] = Header(default=None)):
    """吊销当前 Bearer 登录态。"""
    token = _api_module.auth_service._extract_bearer(authorization)
    revoked = _api_module.auth_service.logout(token) if token else False
    return {"status": "ok", "revoked": bool(revoked)}

async def auth_me(principal=Depends(auth_service.require_session)):
    """返回当前主体（未强制认证时为 anonymous）。"""
    return _api_module._principal_payload(principal)

async def auth_totp_provision(
    principal=Depends(auth_service.require_authenticated_role("admin", "operator")),
):
    """为**当前主体**生成并绑定 TOTP 密钥，返回 `otpauth://` URI 供 Auth App 扫码。

    ⚠️ 重新调用会**覆盖**既有密钥（旧 Auth App 条目随即失效）。
    """
    secret = _api_module.auth_app.generate_secret()
    _api_module.auth_service.get_auth_store().bind_totp(principal.username, secret)
    _api_module.auth_service.get_totp_verifier().reset()
    return {
        "status": "ok",
        "username": principal.username,
        "secret": secret,
        "otpauth_uri": _api_module.auth_app.provisioning_uri(secret, account=principal.username),
        "algorithm": _api_module.auth_app.TOTP_ALGORITHM,
        "digits": _api_module.auth_app.TOTP_DIGITS,
        "period": _api_module.auth_app.TOTP_INTERVAL_SECONDS,
        "reauth_required": True,
    }

async def auth_totp_verify(
    req: TotpVerifyRequest,
    request: Request,
    principal=Depends(auth_service.require_authenticated_role("admin", "operator")),
):
    """校验一次当前主体的 TOTP（用于确认 Auth App 绑定成功）。"""
    secret = _api_module.auth_service.get_auth_store().get_totp_secret(principal.username)
    if not secret:
        raise _api_module.HTTPException(404, {"code": "totp_not_bound", "message": "尚未绑定 Auth App"})
    try:
        ok = _api_module.auth_service.get_totp_verifier().verify(
            secret, req.code, account=principal.username,
            source=_api_module.auth_service.request_source(request),
        )
    except _api_module.auth_app.TotpRateLimitedError as exc:
        raise _api_module.HTTPException(429, {"code": exc.code, "message": str(exc)}) from exc
    except _api_module.auth_app.TotpReplayError as exc:
        raise _api_module.HTTPException(401, {"code": exc.code, "message": str(exc)}) from exc
    except _api_module.auth_app.TotpError as exc:
        raise _api_module.HTTPException(400, {"code": exc.code, "message": str(exc)}) from exc
    return {"status": "ok", "verified": bool(ok)}

async def list_users(request: Request = None,
                     principal=Depends(auth_service.require_authenticated_role("admin"))):
    """列出账户（不含任何口令/密钥材料）。"""
    _api_module.require_model_api_source(request)
    users = _api_module.auth_service.get_auth_store().list_users()
    return {"users": [
        {"username": u.username, "role": u.role, "disabled": u.disabled,
         "totp_bound": u.totp_bound, "created_at": u.created_at}
        for u in users
    ]}

async def create_user(req: CreateUserRequest, request: Request = None,
                      principal=Depends(auth_service.require_session)):
    """创建账户。

    ⚠️ **首次引导**（库中尚无账户）时，本端点**无需登录**，但**只允许创建 admin**
    （否则全新部署无法进入）；此后必须由 admin 调用。
    """
    store = _api_module.auth_service.get_auth_store()
    bootstrap = store.count_users() == 0
    if bootstrap:
        if req.role != "admin":
            raise _api_module.HTTPException(400, {
                "code": "bootstrap_requires_admin",
                "message": "首次引导只允许创建 admin 账户",
            })
    else:
        _api_module.require_model_api_source(request)
        if getattr(principal, "username", "anonymous") == "anonymous":
            raise _api_module.HTTPException(401, {
                "code": "auth_required",
                "message": "账户创建需要实名 admin 登录",
            })
        if not getattr(principal, "is_admin", False):
            raise _api_module.HTTPException(403, {"code": "insufficient_role", "message": "需要 admin 角色"})
    try:
        rec = store.create_user(req.username, req.password, role=req.role)
    except AuthStoreError as exc:
        code = 409 if exc.code == "user_exists" else 400
        raise _api_module.HTTPException(code, {"code": exc.code, "message": str(exc)}) from exc
    return {"status": "created", "username": rec.username, "role": rec.role,
            "bootstrap": bootstrap}

async def patch_user(username: str, req: PatchUserRequest, request: Request = None,
                     principal=Depends(auth_service.require_authenticated_role("admin"))):
    """修改账户：角色 / 禁用 / 重置口令。禁用会**立即吊销**该用户全部登录态。"""
    _api_module.require_model_api_source(request)
    store = _api_module.auth_service.get_auth_store()
    if store.get_user(username) is None:
        raise _api_module.HTTPException(404, {"code": "user_not_found", "message": f"用户不存在: {username}"})
    # ⚠️ 不允许把最后一个 admin 降级或禁用，避免锁死管理面
    if (req.role not in (None, "admin") or req.disabled is True):
        admins = [u for u in store.list_users() if u.role == "admin" and not u.disabled]
        if len(admins) <= 1 and any(u.username == username for u in admins):
            raise _api_module.HTTPException(400, {
                "code": "last_admin_protected",
                "message": "不能降级或禁用最后一个可用 admin",
            })
    changed = {}
    try:
        if req.role is not None:
            changed["role"] = store.set_role(username, req.role)
        if req.disabled is not None:
            changed["disabled"] = store.set_disabled(username, bool(req.disabled))
        if req.password is not None:
            changed["password"] = store.set_password(username, req.password)
    except AuthStoreError as exc:
        raise _api_module.HTTPException(400, {"code": exc.code, "message": str(exc)}) from exc
    if not any(changed.values()):
        raise _api_module.HTTPException(400, {"code": "nothing_to_change", "message": "未提供任何可变更字段"})
    return {"status": "updated", "username": username, "changed": changed}

async def delete_user(username: str, request: Request = None,
                      principal=Depends(auth_service.require_authenticated_role("admin"))):
    """删除账户（及其 TOTP 绑定与登录态，由外键级联）。"""
    _api_module.require_model_api_source(request)
    store = _api_module.auth_service.get_auth_store()
    admins = [u for u in store.list_users() if u.role == "admin" and not u.disabled]
    if len(admins) <= 1 and any(u.username == username for u in admins):
        raise _api_module.HTTPException(400, {
            "code": "last_admin_protected",
            "message": "不能删除最后一个可用 admin",
        })
    if not store.delete_user(username):
        raise _api_module.HTTPException(404, {"code": "user_not_found", "message": f"用户不存在: {username}"})
    return {"status": "deleted", "username": username}

async def get_user_settings():
    """从主节点 SQLite 读取完整的用户偏好设置。"""
    try:
        settings = _api_module._local_store.get_local_user_settings()
        return {"settings": settings, "source": "sqlite"}
    except Exception as e:
        _api_module.logger.error(f"读取 SQLite 用户设置失败: {e}")
        raise _api_module.HTTPException(503, f"本地设置存储不可用: {e}")

async def update_user_settings(req: UserSettingsRequest):
    """写入用户本人主节点上的 SQLite。"""
    try:
        settings = req.settings
        _api_module._local_store.set_local_user_settings(settings)
    except Exception as e:
        _api_module.logger.error(f"存储 SQLite 用户设置失败: {e}")
        raise _api_module.HTTPException(503, f"本地设置存储失败: {e}")

    return {
        "status": "ok",
        "source": "sqlite",
        "legacy_exported": False,
        "synced_fields": list(settings.keys()),
    }


def register_routes() -> None:
    router.add_api_route('/api/auth/capability', auth_capability, methods=['GET'])
    router.add_api_route('/api/auth/login', auth_login, methods=['POST'])
    router.add_api_route('/api/auth/logout', auth_logout, methods=['POST'])
    router.add_api_route('/api/auth/me', auth_me, methods=['GET'])
    router.add_api_route('/api/auth/totp/provision', auth_totp_provision, methods=['POST'])
    router.add_api_route('/api/auth/totp/verify', auth_totp_verify, methods=['POST'])
    router.add_api_route('/api/users', list_users, methods=['GET'])
    router.add_api_route('/api/users', create_user, methods=['POST'])
    router.add_api_route('/api/users/{username}', patch_user, methods=['PATCH'])
    router.add_api_route('/api/users/{username}', delete_user, methods=['DELETE'])
    router.add_api_route('/api/user/settings', get_user_settings, methods=['GET'])
    router.add_api_route('/api/user/settings', update_user_settings, methods=['PUT'])
