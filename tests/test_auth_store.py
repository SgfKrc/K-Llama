"""★ `auth_store.py` 测试（2026-09-19，G：monolith 内实现）。

覆盖：账户 CRUD、口令校验（含常数时间比较的存在性）、角色、禁用、TOTP 绑定、
登录态签发/解析/吊销/过期清理，以及**「库里不存明文 token」**这一安全性质。
"""

from __future__ import annotations

import sqlite3

import pytest

from auth_store import (
    ROLE_ADMIN,
    ROLE_VIEWER,
    AuthStore,
    AuthStoreError,
)


@pytest.fixture()
def store(tmp_path):
    return AuthStore(str(tmp_path / "auth.sqlite"))


class TestUsers:
    def test_create_and_get(self, store):
        rec = store.create_user("koakuma", "hunter2hunter2", role=ROLE_ADMIN)
        assert rec.username == "koakuma" and rec.role == ROLE_ADMIN
        got = store.get_user("koakuma")
        assert got is not None and got.role == ROLE_ADMIN and got.disabled is False
        assert store.count_users() == 1

    def test_duplicate_rejected(self, store):
        store.create_user("a", "password123")
        with pytest.raises(AuthStoreError) as ei:
            store.create_user("a", "password123")
        assert ei.value.code == "user_exists"

    def test_weak_password_rejected(self, store):
        with pytest.raises(AuthStoreError) as ei:
            store.create_user("b", "short")
        assert ei.value.code == "weak_password"

    def test_blank_username_rejected(self, store):
        with pytest.raises(AuthStoreError) as ei:
            store.create_user("   ", "password123")
        assert ei.value.code == "invalid_username"

    def test_invalid_role_rejected(self, store):
        with pytest.raises(AuthStoreError) as ei:
            store.create_user("c", "password123", role="root")
        assert ei.value.code == "invalid_role"

    def test_list_users_reports_totp_binding(self, store):
        store.create_user("u1", "password123")
        store.create_user("u2", "password123")
        store.bind_totp("u2", "GEZDGNBVGY3TQOJQ")
        by_name = {u.username: u for u in store.list_users()}
        assert by_name["u1"].totp_bound is False
        assert by_name["u2"].totp_bound is True

    def test_set_role_and_delete(self, store):
        store.create_user("d", "password123")
        assert store.set_role("d", ROLE_ADMIN) is True
        assert store.get_user("d").role == ROLE_ADMIN
        assert store.delete_user("d") is True
        assert store.get_user("d") is None

    def test_delete_cascades_totp_and_sessions(self, store):
        store.create_user("e", "password123")
        store.bind_totp("e", "GEZDGNBVGY3TQOJQ")
        token, _ = store.issue_session("e")
        assert store.delete_user("e") is True
        assert store.get_totp_secret("e") is None
        assert store.resolve_session(token) is None


class TestPasswords:
    def test_verify_ok(self, store):
        store.create_user("f", "correct-horse")
        assert store.verify_password("f", "correct-horse") is True

    def test_verify_wrong(self, store):
        store.create_user("f", "correct-horse")
        assert store.verify_password("f", "wrong") is False

    def test_verify_unknown_user_is_false_not_error(self, store):
        assert store.verify_password("nope", "whatever") is False

    def test_set_password_invalidates_old(self, store):
        store.create_user("g", "oldpassword")
        old_token, _ = store.issue_session("g")
        assert store.set_password("g", "newpassword") is True
        assert store.verify_password("g", "oldpassword") is False
        assert store.verify_password("g", "newpassword") is True
        assert store.resolve_session(old_token) is None


class TestDisabled:
    def test_disabled_blocks_password(self, store):
        store.create_user("h", "password123")
        assert store.set_disabled("h", True) is True
        assert store.verify_password("h", "password123") is False

    def test_disabled_revokes_sessions(self, store):
        store.create_user("h", "password123")
        tokens = [store.issue_session("h")[0] for _ in range(2)]
        assert all(store.resolve_session(token) is not None for token in tokens)
        store.set_disabled("h", True)
        assert all(store.resolve_session(token) is None for token in tokens)

    def test_re_enable_restores_password(self, store):
        store.create_user("h", "password123")
        store.set_disabled("h", True)
        store.set_disabled("h", False)
        assert store.verify_password("h", "password123") is True


class TestRoleChanges:
    def test_role_change_revokes_all_existing_sessions(self, store):
        store.create_user("role-user", "password123")
        tokens = [store.issue_session("role-user")[0] for _ in range(2)]
        assert store.set_role("role-user", ROLE_ADMIN) is True
        assert all(store.resolve_session(token) is None for token in tokens)


class TestSessions:
    def test_issue_and_resolve(self, store):
        store.create_user("i", "password123", role=ROLE_ADMIN)
        token, rec = store.issue_session("i")
        assert token and rec.username == "i" and rec.role == ROLE_ADMIN
        resolved = store.resolve_session(token)
        assert resolved is not None and resolved.username == "i"

    def test_plaintext_token_not_stored(self, store, tmp_path):
        """★ 安全性质：库里只应有 SHA256，不得出现明文 token。"""
        store.create_user("j", "password123")
        token, _ = store.issue_session("j")
        blob = (tmp_path / "auth.sqlite").read_bytes()
        assert token.encode() not in blob, "明文登录态 token 不得落库"

    def test_unknown_token_rejected(self, store):
        assert store.resolve_session("not-a-real-token") is None

    def test_empty_token_rejected(self, store):
        assert store.resolve_session("") is None

    def test_expired_session_purged(self, store):
        store.create_user("k", "password123")
        token, _ = store.issue_session("k", ttl_seconds=60)
        # 手工把过期时间推到过去
        conn = sqlite3.connect(str(store._db_path))
        conn.execute("UPDATE auth_sessions SET expires_at = ?", (0.0,))
        conn.commit()
        conn.close()
        assert store.resolve_session(token) is None

    def test_revoke_session(self, store):
        store.create_user("l", "password123")
        token, _ = store.issue_session("l")
        assert store.revoke_session(token) is True
        assert store.resolve_session(token) is None

    def test_revoke_all(self, store):
        store.create_user("m", "password123")
        t1, _ = store.issue_session("m")
        t2, _ = store.issue_session("m")
        assert store.revoke_all_sessions("m") == 2
        assert store.resolve_session(t1) is None and store.resolve_session(t2) is None

    def test_purge_expired(self, store):
        store.create_user("n", "password123")
        store.issue_session("n", ttl_seconds=60)
        conn = sqlite3.connect(str(store._db_path))
        conn.execute("UPDATE auth_sessions SET expires_at = ?", (0.0,))
        conn.commit()
        conn.close()
        assert store.purge_expired_sessions() == 1


class TestTotp:
    def test_bind_get_clear(self, store):
        store.create_user("o", "password123")
        assert store.get_totp_secret("o") is None
        store.bind_totp("o", "GEZDGNBVGY3TQOJQ")
        assert store.get_totp_secret("o") == "GEZDGNBVGY3TQOJQ"
        # 重复绑定应覆盖而非报错
        store.bind_totp("o", "MFRGGZDFMZTWQ2LK")
        assert store.get_totp_secret("o") == "MFRGGZDFMZTWQ2LK"
        assert store.clear_totp("o") is True
        assert store.get_totp_secret("o") is None

    def test_binding_totp_revokes_all_existing_sessions(self, store):
        store.create_user("totp-session", "password123")
        tokens = [store.issue_session("totp-session")[0] for _ in range(2)]

        store.bind_totp("totp-session", "JBSWY3DPEHPK3PXP")

        assert all(store.resolve_session(token) is None for token in tokens)

    def test_totp_seed_is_not_stored_in_plaintext_sqlite(self, store, tmp_path):
        secret = "JBSWY3DPEHPK3PXP"
        store.create_user("secure-totp", "password123")
        store.bind_totp("secure-totp", secret)

        conn = sqlite3.connect(str(tmp_path / "auth.sqlite"))
        try:
            stored = conn.execute(
                "SELECT secret FROM auth_totp WHERE username = ?", ("secure-totp",),
            ).fetchone()[0]
        finally:
            conn.close()

        assert stored == "local-secret-store:v1"
        assert secret.encode() not in (tmp_path / "auth.sqlite").read_bytes()
        assert secret not in (tmp_path / "auth_secrets.json").read_text(encoding="utf-8")
        assert store.get_totp_secret("secure-totp") == secret

    def test_legacy_plaintext_totp_seed_migrates_on_read(self, store, tmp_path):
        secret = "MFRGGZDFMZTWQ2LK"
        store.create_user("legacy-totp", "password123")
        conn = sqlite3.connect(str(tmp_path / "auth.sqlite"))
        try:
            conn.execute(
                "INSERT INTO auth_totp (username, secret, confirmed_at) VALUES (?,?,?)",
                ("legacy-totp", secret, 1.0),
            )
            conn.commit()
        finally:
            conn.close()

        assert store.get_totp_secret("legacy-totp") == secret
        conn = sqlite3.connect(str(tmp_path / "auth.sqlite"))
        try:
            stored = conn.execute(
                "SELECT secret FROM auth_totp WHERE username = ?", ("legacy-totp",),
            ).fetchone()[0]
        finally:
            conn.close()
        assert stored == "local-secret-store:v1"


class TestSchemaIdempotent:
    def test_reopen_keeps_schema_and_data(self, tmp_path):
        path = str(tmp_path / "auth.sqlite")
        s1 = AuthStore(path)
        s1.create_user("p", "password123")
        s2 = AuthStore(path)          # 重开不应报错
        assert s2.count_users() == 1
        assert s2.verify_password("p", "password123") is True
