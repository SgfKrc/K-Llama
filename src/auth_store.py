"""★ Auth 存储层（2026-09-19，monolith 内实现 G；抛弃控制面）。

## 为什么单独成模块
原设计把账户 / 会话 / TOTP 放在**独立的 control-svc（`127.0.0.1:8030`）**里，主仓只做反代。
微服务改造叫停后该服务已不存在 ⇒ 现在**在 monolith 进程内实现**，但**复用同一个用户级
SQLite**（`local_store.initialize_local_store()` 返回的路径）—— 不新建数据库文件，
避免多一处需要备份/迁移的状态。

## 表
* `auth_users`    —— 账户（用户名 / 口令哈希 / 角色 / 禁用位 / 时间戳）
* `auth_totp`     —— 每账户的 TOTP 加密记录引用（Auth App 绑定）
* `auth_sessions` —— 登录态（**只存 token 的 SHA256**，不存明文）

## 口令
`hashlib.pbkdf2_hmac("sha256", ...)` + 每用户随机盐（**纯标准库**，不引入 bcrypt/argon2）。
校验用 `hmac.compare_digest`（常数时间）。

## ⚠️ 安全边界
* 本模块**不做**来源校验（那是 `model_api_access.py` / `api_server` 的职责）。
* **明文 token 只在签发时返回一次**；库里只有哈希 —— 泄库不等于可冒充。
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from local_secret_store import LocalSecretStore

__all__ = [
    "ROLE_ADMIN",
    "ROLE_OPERATOR",
    "ROLE_VIEWER",
    "VALID_ROLES",
    "SESSION_TTL_SECONDS",
    "AuthStoreError",
    "UserRecord",
    "SessionRecord",
    "AuthStore",
]

ROLE_ADMIN = "admin"
ROLE_OPERATOR = "operator"
ROLE_VIEWER = "viewer"
VALID_ROLES = (ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER)

SESSION_TTL_SECONDS = 12 * 3600      # 登录态有效期（12h）
_PBKDF2_ROUNDS = 200_000
_SALT_BYTES = 16
_TOTP_SECRET_REF = "local-secret-store:v1"


class AuthStoreError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class UserRecord:
    username: str
    role: str
    disabled: bool
    created_at: float
    password_changed_at: float
    totp_bound: bool = False


@dataclass(frozen=True)
class SessionRecord:
    username: str
    role: str
    created_at: float
    expires_at: float
    last_used_at: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS auth_users (
    username            TEXT PRIMARY KEY,
    password_hash       TEXT NOT NULL,
    salt                TEXT NOT NULL,
    role                TEXT NOT NULL,
    disabled            INTEGER NOT NULL DEFAULT 0,
    created_at          REAL NOT NULL,
    password_changed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_totp (
    username     TEXT PRIMARY KEY,
    secret       TEXT NOT NULL,
    confirmed_at REAL,
    FOREIGN KEY (username) REFERENCES auth_users(username) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash   TEXT PRIMARY KEY,
    username     TEXT NOT NULL,
    created_at   REAL NOT NULL,
    expires_at   REAL NOT NULL,
    last_used_at REAL NOT NULL,
    FOREIGN KEY (username) REFERENCES auth_users(username) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(username);
"""


def _hash_password(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), _PBKDF2_ROUNDS
    ).hex()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _totp_secret_record(username: str) -> str:
    digest = hashlib.sha256(username.encode("utf-8")).hexdigest()
    return f"auth_totp:{digest}"


def _restrict_database_files(path: str) -> None:
    for candidate in (Path(path), Path(path + "-wal"), Path(path + "-shm")):
        if not candidate.exists():
            continue
        try:
            candidate.chmod(0o600)
        except OSError:
            pass


class AuthStore:
    """Auth 表的读写（**线程安全**：一把 `RLock` + 每次操作新建连接）。"""

    def __init__(self, db_path: str, *, secret_store_path: str | None = None):
        self._db_path = db_path
        default_secret_path = Path(db_path).with_name("auth_secrets.json")
        self._secret_store = LocalSecretStore(secret_store_path or default_secret_path)
        self._lock = threading.RLock()
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA)
                conn.commit()
            finally:
                conn.close()
                _restrict_database_files(self._db_path)

    # ---------------------------------------------------------------- 内部
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA secure_delete = ON")
        _restrict_database_files(self._db_path)
        return conn

    @staticmethod
    def _revoke_all_sessions_in_transaction(conn: sqlite3.Connection, username: str) -> int:
        cur = conn.execute("DELETE FROM auth_sessions WHERE username = ?", (username,))
        return int(cur.rowcount)

    # ---------------------------------------------------------------- 账户
    def count_users(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                return int(conn.execute("SELECT COUNT(*) FROM auth_users").fetchone()[0])
            finally:
                conn.close()

    def create_user(self, username: str, password: str, *,
                    role: str = ROLE_VIEWER) -> UserRecord:
        name = (username or "").strip()
        if not name:
            raise AuthStoreError("invalid_username", "用户名不能为空")
        if role not in VALID_ROLES:
            raise AuthStoreError("invalid_role", f"角色必须是 {VALID_ROLES}")
        if len(password or "") < 8:
            raise AuthStoreError("weak_password", "口令至少 8 位")
        now = time.time()
        salt = secrets.token_hex(_SALT_BYTES)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO auth_users (username, password_hash, salt, role, disabled,"
                    " created_at, password_changed_at) VALUES (?,?,?,?,0,?,?)",
                    (name, _hash_password(password, salt), salt, role, now, now),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                raise AuthStoreError("user_exists", f"用户已存在: {name}") from exc
            finally:
                conn.close()
        return UserRecord(name, role, False, now, now, False)

    def list_users(self) -> list[UserRecord]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT u.username, u.role, u.disabled, u.created_at, u.password_changed_at,"
                    "       (t.username IS NOT NULL) AS totp_bound"
                    "  FROM auth_users u LEFT JOIN auth_totp t ON t.username = u.username"
                    " ORDER BY u.username"
                ).fetchall()
                return [UserRecord(r["username"], r["role"], bool(r["disabled"]),
                                   r["created_at"], r["password_changed_at"],
                                   bool(r["totp_bound"])) for r in rows]
            finally:
                conn.close()

    def get_user(self, username: str) -> Optional[UserRecord]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT u.username, u.role, u.disabled, u.created_at, u.password_changed_at,"
                    "       (t.username IS NOT NULL) AS totp_bound"
                    "  FROM auth_users u LEFT JOIN auth_totp t ON t.username = u.username"
                    " WHERE u.username = ?", (username,)
                ).fetchone()
                if row is None:
                    return None
                return UserRecord(row["username"], row["role"], bool(row["disabled"]),
                                  row["created_at"], row["password_changed_at"],
                                  bool(row["totp_bound"]))
            finally:
                conn.close()

    def verify_password(self, username: str, password: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT password_hash, salt, disabled FROM auth_users WHERE username = ?",
                    (username,),
                ).fetchone()
            finally:
                conn.close()
        if row is None or row["disabled"]:
            # ⚠️ 即使用户不存在也做一次等价的哈希运算，避免用时间侧信道枚举用户名
            _hash_password(password or "", secrets.token_hex(_SALT_BYTES))
            return False
        return hmac.compare_digest(_hash_password(password or "", row["salt"]),
                                   row["password_hash"])

    def set_disabled(self, username: str, disabled: bool) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("UPDATE auth_users SET disabled = ? WHERE username = ?",
                                   (1 if disabled else 0, username))
                if cur.rowcount:
                    # Any account-state transition invalidates every old bearer.
                    self._revoke_all_sessions_in_transaction(conn, username)
                conn.commit()
                return bool(cur.rowcount)
            finally:
                conn.close()

    def set_role(self, username: str, role: str) -> bool:
        if role not in VALID_ROLES:
            raise AuthStoreError("invalid_role", f"角色必须是 {VALID_ROLES}")
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("UPDATE auth_users SET role = ? WHERE username = ?",
                                   (role, username))
                if cur.rowcount:
                    self._revoke_all_sessions_in_transaction(conn, username)
                conn.commit()
                return bool(cur.rowcount)
            finally:
                conn.close()

    def set_password(self, username: str, password: str) -> bool:
        if len(password or "") < 8:
            raise AuthStoreError("weak_password", "口令至少 8 位")
        salt = secrets.token_hex(_SALT_BYTES)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "UPDATE auth_users SET password_hash = ?, salt = ?, password_changed_at = ?"
                    " WHERE username = ?",
                    (_hash_password(password, salt), salt, time.time(), username))
                if cur.rowcount:
                    self._revoke_all_sessions_in_transaction(conn, username)
                conn.commit()
                return bool(cur.rowcount)
            finally:
                conn.close()

    def delete_user(self, username: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM auth_users WHERE username = ?", (username,))
                conn.commit()
                deleted = bool(cur.rowcount)
            finally:
                conn.close()
            if deleted:
                try:
                    self._secret_store.delete(_totp_secret_record(username))
                except Exception:
                    # The account/session deletion already committed. An
                    # encrypted orphan must not turn success into a false
                    # failure signal or resurrect the deleted identity.
                    pass
            return deleted

    # ---------------------------------------------------------------- TOTP
    def bind_totp(self, username: str, secret: str) -> None:
        value = str(secret or "").strip()
        if not value:
            raise AuthStoreError("invalid_totp_secret", "TOTP 密钥不能为空")
        with self._lock:
            record_name = _totp_secret_record(username)
            previous = self._secret_store.get(record_name)
            self._secret_store.set(record_name, value)
            conn = None
            try:
                conn = self._connect()
                conn.execute(
                    "INSERT INTO auth_totp (username, secret, confirmed_at) VALUES (?,?,?)"
                    " ON CONFLICT(username) DO UPDATE SET secret = excluded.secret,"
                    " confirmed_at = excluded.confirmed_at",
                    (username, _TOTP_SECRET_REF, time.time()),
                )
                self._revoke_all_sessions_in_transaction(conn, username)
                conn.commit()
            except Exception:
                if previous:
                    self._secret_store.set(record_name, previous)
                else:
                    self._secret_store.delete(record_name)
                raise
            finally:
                if conn is not None:
                    conn.close()

    def get_totp_secret(self, username: str) -> Optional[str]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT secret FROM auth_totp WHERE username = ?",
                                   (username,)).fetchone()
                if row is None:
                    return None
                stored_value = str(row["secret"] or "")
                record_name = _totp_secret_record(username)
                if stored_value == _TOTP_SECRET_REF:
                    secret = self._secret_store.get(record_name)
                    if not secret:
                        raise AuthStoreError(
                            "totp_secret_unavailable",
                            "TOTP 密钥引用存在但加密记录不可用",
                        )
                    return secret

                # Migrate legacy plaintext rows on first use. Future database
                # snapshots expose only a reference, never the TOTP seed.
                self._secret_store.set(record_name, stored_value)
                conn.execute(
                    "UPDATE auth_totp SET secret = ? WHERE username = ?",
                    (_TOTP_SECRET_REF, username),
                )
                conn.commit()
                return stored_value
            finally:
                conn.close()

    def clear_totp(self, username: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM auth_totp WHERE username = ?", (username,))
                conn.commit()
                cleared = bool(cur.rowcount)
            finally:
                conn.close()
            try:
                self._secret_store.delete(_totp_secret_record(username))
            except Exception:
                # The binding row is already gone. Keep the externally visible
                # result truthful; the unreachable encrypted record is inert.
                pass
            return cleared

    # ---------------------------------------------------------------- 登录态
    def issue_session(self, username: str, *,
                      ttl_seconds: int = SESSION_TTL_SECONDS) -> tuple[str, SessionRecord]:
        """签发登录态。**返回的 token 是明文，只在这一刻可见**；库里只存 SHA256。"""
        token = secrets.token_urlsafe(32)
        now = time.time()
        expires = now + max(60, int(ttl_seconds))
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT role FROM auth_users WHERE username = ? AND disabled = 0",
                                   (username,)).fetchone()
                if row is None:
                    raise AuthStoreError("user_unavailable", f"用户不可用: {username}")
                conn.execute(
                    "INSERT INTO auth_sessions (token_hash, username, created_at, expires_at,"
                    " last_used_at) VALUES (?,?,?,?,?)",
                    (_hash_token(token), username, now, expires, now),
                )
                conn.commit()
                role = row["role"]
            finally:
                conn.close()
        return token, SessionRecord(username, role, now, expires, now)

    def resolve_session(self, token: str) -> Optional[SessionRecord]:
        """校验登录态；过期即删除并返回 None。命中时刷新 `last_used_at`。"""
        if not token:
            return None
        digest = _hash_token(token)
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT s.username, s.created_at, s.expires_at, s.last_used_at,"
                    "       u.role, u.disabled"
                    "  FROM auth_sessions s JOIN auth_users u ON u.username = s.username"
                    " WHERE s.token_hash = ?", (digest,)
                ).fetchone()
                if row is None:
                    return None
                if row["disabled"] or now >= row["expires_at"]:
                    conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (digest,))
                    conn.commit()
                    return None
                conn.execute("UPDATE auth_sessions SET last_used_at = ? WHERE token_hash = ?",
                             (now, digest))
                conn.commit()
                return SessionRecord(row["username"], row["role"], row["created_at"],
                                     row["expires_at"], now)
            finally:
                conn.close()

    def revoke_session(self, token: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?",
                                   (_hash_token(token),))
                conn.commit()
                return bool(cur.rowcount)
            finally:
                conn.close()

    def revoke_all_sessions(self, username: str) -> int:
        with self._lock:
            conn = self._connect()
            try:
                revoked = self._revoke_all_sessions_in_transaction(conn, username)
                conn.commit()
                return revoked
            finally:
                conn.close()

    def purge_expired_sessions(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM auth_sessions WHERE expires_at <= ?",
                                   (time.time(),))
                conn.commit()
                return int(cur.rowcount)
            finally:
                conn.close()
