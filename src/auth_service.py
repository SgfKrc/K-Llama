"""★ Auth 服务层（2026-09-19，G：monolith 内实现；抛弃控制面）。

## 定位
`auth_store.py` 只管表读写；本模块把它组装成**服务语义**并与 FastAPI 对接：
* **单例存储**（复用 `local_store.initialize_local_store()` 的同一个 SQLite）
* **Bearer 解析**（`Authorization: Bearer <token>` ⇒ `SessionRecord`）
* **FastAPI 依赖**：`require_session`（任意登录）、`require_role(...)`（角色门）
* **首次引导**：库里**一个用户都没有**时，允许**无需登录**创建第一个 `admin`
  （否则全新部署无法进入 —— 这是原控制面「bootstrap」能力的本地替代）

## 与原控制面的差异（有意为之）
* **无独立进程/端口** ⇒ 不会出现 `503 Auth control service unavailable`；
* **无网络跳** ⇒ 认证与业务在同一事务边界内；
* **数据同库** ⇒ 备份/迁移只需一个 SQLite 文件。

## ⚠️ 边界
* 本模块**不管来源 IP** —— 那是 `model_api_access.require_model_api_source` 的职责，
  两者**叠加**使用（本机 loopback 默认可信；远程需显式 CIDR **且** 登录）。
"""

from __future__ import annotations

import os
import secrets
import threading
from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, Request

import auth_app
from auth_store import (
    ROLE_ADMIN,
    ROLE_OPERATOR,
    ROLE_VIEWER,
    SESSION_TTL_SECONDS,
    VALID_ROLES,
    AuthStore,
    AuthStoreError,
    SessionRecord,
)

__all__ = [
    "get_auth_store",
    "get_totp_verifier",
    "is_bootstrap_open",
    "auth_required",
    "auth_capability_payload",
    "login",
    "verify_totp_confirmation",
    "logout",
    "resolve_bearer",
    "require_session",
    "require_role",
    "require_authenticated_role",
    "AuthPrincipal",
]

_store_lock = threading.RLock()
_store_instance: Optional[AuthStore] = None
_verifier: Optional[auth_app.TotpVerifier] = None


def _auth_db_path() -> str:
    """与 `local_store` 用**同一个** SQLite 文件（避免多一处需备份的状态）。"""
    import local_store

    return local_store.initialize_local_store()


def get_auth_store() -> AuthStore:
    global _store_instance
    with _store_lock:
        if _store_instance is None:
            _store_instance = AuthStore(_auth_db_path())
        return _store_instance


def get_totp_verifier() -> auth_app.TotpVerifier:
    global _verifier
    with _store_lock:
        if _verifier is None:
            _verifier = auth_app.TotpVerifier()
        return _verifier


def _reset_for_tests() -> None:
    """仅供测试：丢弃单例（重新指向新库/新库文件）。"""
    global _store_instance, _verifier
    with _store_lock:
        _store_instance = None
        _verifier = None


def is_bootstrap_open() -> bool:
    """库中没有任何用户 ⇒ 允许无登录创建第一个 admin（首次引导）。"""
    try:
        return get_auth_store().count_users() == 0
    except Exception:  # noqa: BLE001 —— 存储不可用时按“不开放”处理（fail-closed）
        return False


def _auth_required() -> bool:
    return os.environ.get("QLH_AUTH_REQUIRED", "").strip().lower() in {"1", "true", "on", "yes"}


def auth_required() -> bool:
    """Return whether HTTP business APIs must carry a valid Bearer session."""
    return _auth_required()


def auth_capability_payload() -> dict:
    """`/api/auth/capability` 的**本地**实现（不再是反代探测）。

    语义：
      * `available`  —— 认证**实现已存在于本进程**（恒为 True，除非存储不可用）
      * `required`   —— 是否**强制**要求登录（由 `QLH_AUTH_REQUIRED` 决定；默认不强制）
      * `bootstrap_open` —— 是否处于首次引导（尚无任何账户）
    """
    try:
        store = get_auth_store()
        users = store.count_users()
        available = True
    except Exception as exc:  # noqa: BLE001
        return {
            "required": True, "enforced": True, "available": False,
            "mode": "local_totp", "policy_version": "n1a-v1", "service": "api_server",
            "bootstrap_open": False, "user_count": 0,
            "reason_code": "auth_store_unavailable", "reason": str(exc)[:160],
        }
    required = _auth_required()
    return {
        "required": required,
        "enforced": required,
        "available": available,
        "mode": "local_totp",
        "policy_version": "n1a-v1",
        "service": "api_server",
        "bootstrap_open": users == 0,
        "user_count": users,
    }


@dataclass(frozen=True)
class AuthPrincipal:
    username: str
    role: str
    token: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN


def request_source(request: Optional[Request] = None) -> str:
    """Return the direct peer address; forwarded headers are not trusted."""
    client = getattr(request, "client", None) if request is not None else None
    return str(getattr(client, "host", "") or "unknown")[:256]


def login(username: str, password: str, *, totp_code: Optional[str] = None,
          source: Optional[str] = None,
          ttl_seconds: int = SESSION_TTL_SECONDS) -> tuple[str, SessionRecord]:
    """校验口令（**若已绑定 TOTP 则必须同时提供有效一次性码**）并签发登录态。

    Raises:
        HTTPException(401): 口令错误 / TOTP 缺失或不正确。
        HTTPException(429): TOTP 失败过多（冷却期）。
    """
    store = get_auth_store()
    if not store.verify_password(username, password):
        raise HTTPException(401, {"code": "invalid_credentials", "message": "用户名或口令不正确"})

    secret = store.get_totp_secret(username)
    if secret:
        if not totp_code:
            raise HTTPException(401, {
                "code": "totp_required",
                "message": "该账户已绑定 Auth App，请同时提供 6 位验证码",
            })
        try:
            ok = get_totp_verifier().verify(
                secret, totp_code, account=username, source=source)
        except auth_app.TotpRateLimitedError as exc:
            raise HTTPException(429, {"code": exc.code, "message": str(exc)}) from exc
        except auth_app.TotpReplayError as exc:
            raise HTTPException(401, {"code": exc.code, "message": str(exc)}) from exc
        except auth_app.TotpError as exc:
            raise HTTPException(400, {"code": exc.code, "message": str(exc)}) from exc
        if not ok:
            raise HTTPException(401, {"code": "totp_invalid", "message": "验证码不正确"})

    try:
        return store.issue_session(username, ttl_seconds=ttl_seconds)
    except AuthStoreError as exc:
        raise HTTPException(403, {"code": exc.code, "message": str(exc)}) from exc


def verify_totp_confirmation(
    principal: AuthPrincipal, code: Optional[str], *, source: Optional[str] = None
) -> None:
    """Require a real local identity and consume one Auth App/TOTP code.

    The default compatibility mode may expose an anonymous principal for
    read-only local use, but a high-risk approval must never treat it as a signer.
    """
    if principal is None or principal.username == "anonymous":
        raise HTTPException(
            403,
            {"code": "auth_required", "message": "需要已登录账户进行 Auth App 确认"},
        )
    secret = get_auth_store().get_totp_secret(principal.username)
    if not secret:
        raise HTTPException(
            501,
            {
                "code": "auth_control_plane_unavailable",
                "message": "当前账户尚未配置 Auth App/TOTP，拒绝签发入群授权",
            },
        )
    if not code or not code.strip():
        raise HTTPException(
            403,
            {"code": "totp_required", "message": "需要输入当前 Auth App 一次性验证码"},
        )
    try:
        verified = get_totp_verifier().verify(
            secret, code, account=principal.username, source=source)
    except auth_app.TotpRateLimitedError as exc:
        raise HTTPException(429, {"code": "rate_limited", "message": str(exc)}) from exc
    except auth_app.TotpReplayError as exc:
        raise HTTPException(403, {"code": "totp_replayed", "message": str(exc)}) from exc
    except auth_app.TotpError as exc:
        raise HTTPException(403, {"code": "totp_invalid", "message": str(exc)}) from exc
    if not verified:
        raise HTTPException(403, {"code": "totp_invalid", "message": "验证码不正确"})


def logout(token: str) -> bool:
    return get_auth_store().revoke_session(token)


def _extract_bearer(authorization: Optional[str]) -> str:
    if not authorization:
        return ""
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return ""


def resolve_bearer(authorization: Optional[str]) -> Optional[AuthPrincipal]:
    """把 `Authorization: Bearer <token>` 解析成主体；无效返回 None。"""
    token = _extract_bearer(authorization)
    if not token:
        return None
    rec = get_auth_store().resolve_session(token)
    if rec is None:
        return None
    return AuthPrincipal(rec.username, rec.role, token)


def require_session(authorization: Optional[str] = Header(default=None)) -> AuthPrincipal:
    """FastAPI 依赖：要求**已登录**。

    ⚠️ `QLH_AUTH_REQUIRED` 未开启时（默认），**首次引导阶段**与**本机 loopback**均放行 ——
    后者由各端点叠加的 `require_model_api_source` 负责；这样默认部署行为与改造前一致。
    """
    principal = resolve_bearer(authorization)
    if principal is not None:
        return principal
    if not _auth_required() or is_bootstrap_open():
        # 未强制认证：给一个「匿名」主体，保留既有可用性
        return AuthPrincipal("anonymous", ROLE_ADMIN)
    raise HTTPException(401, {"code": "auth_required", "message": "需要登录（Bearer token）"})


def require_role(*roles: str):
    """FastAPI 依赖工厂：要求登录且角色属于 `roles`。"""
    allowed = frozenset(roles)

    def _dep(authorization: Optional[str] = Header(default=None)) -> AuthPrincipal:
        principal = resolve_bearer(authorization)
        if principal is None:
            if not _auth_required():
                return AuthPrincipal("anonymous", ROLE_ADMIN)
            raise HTTPException(401, {"code": "auth_required", "message": "需要登录（Bearer token）"})
        if principal.role not in allowed:
            raise HTTPException(403, {
                "code": "insufficient_role",
                "message": f"需要角色之一: {sorted(allowed)}",
            })
        return principal

    return _dep


def require_authenticated_role(*roles: str):
    """Require a named session even when global compatibility auth is off."""
    allowed = frozenset(roles)

    def _dep(authorization: Optional[str] = Header(default=None)) -> AuthPrincipal:
        principal = resolve_bearer(authorization)
        if principal is None:
            raise HTTPException(401, {
                "code": "auth_required",
                "message": "该高风险操作需要实名登录（Bearer token）",
            })
        if principal.role not in allowed:
            raise HTTPException(403, {
                "code": "insufficient_role",
                "message": f"需要角色之一: {sorted(allowed)}",
            })
        return principal

    return _dep
