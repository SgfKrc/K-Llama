"""★ G 集成测试：monolith 内认证（2026-09-19）。

验证 `api_server` 上的本地认证端点（**不再是 control-svc 反代**）：
引导 → 登录 → 鉴权 → TOTP → 账户管理 → 反代层确已移除。

⚠️ 每个用例用**独立临时库**（`auth_service._reset_for_tests()` + monkeypatch 路径）。
"""

from __future__ import annotations

import base64
import importlib

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import auth_service

    db = str(tmp_path / "auth.sqlite")
    monkeypatch.setattr(auth_service, "_auth_db_path", lambda: db)
    auth_service._reset_for_tests()
    import api_server

    importlib.reload(auth_service)  # 确保单例清空后重建
    auth_service._reset_for_tests()
    monkeypatch.setattr(auth_service, "_auth_db_path", lambda: db)
    with TestClient(api_server.app) as c:
        yield c
    auth_service._reset_for_tests()


class TestNoProxyLayer:
    """原反代层必须已消失（这是本次改造的核心）。"""

    def test_proxy_helpers_gone(self):
        import api_server

        src = open(api_server.__file__, encoding="utf-8").read()
        for token in ("_proxy_control_request", "QLH_CONTROL_URL",
                      "proxy_auth_request", "proxy_users_root", "control_url"):
            assert token not in src, f"反代层残留: {token}"

    def test_env_no_longer_referenced(self):
        import os

        import api_server  # noqa: F401

        assert os.environ.get("QLH_CONTROL_URL") in (None, ""), (
            "改造后不应再要求配置 QLH_CONTROL_URL"
        )


class TestCapability:
    def test_bootstrap_open_when_empty(self, client):
        r = client.get("/api/auth/capability")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert body["service"] == "api_server"
        assert body["bootstrap_open"] is True
        assert body["user_count"] == 0
        assert body["mode"] == "local_totp"

    def test_bootstrap_closed_after_first_user(self, client):
        client.post("/api/users", json={"username": "root", "password": "password123",
                                        "role": "admin"})
        body = client.get("/api/auth/capability").json()
        assert body["bootstrap_open"] is False
        assert body["user_count"] == 1


class TestBootstrapAndLogin:
    def test_first_user_must_be_admin(self, client):
        r = client.post("/api/users", json={"username": "x", "password": "password123",
                                            "role": "viewer"})
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "bootstrap_requires_admin"

    def test_bootstrap_then_login(self, client):
        r = client.post("/api/users", json={"username": "root", "password": "password123",
                                            "role": "admin"})
        assert r.status_code == 200 and r.json()["bootstrap"] is True

        r = client.post("/api/auth/login", json={"username": "root", "password": "password123"})
        assert r.status_code == 200
        body = r.json()
        assert body["role"] == "admin" and body["token"]

        # 带 token 访问 /api/auth/me
        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {body['token']}"})
        assert me.status_code == 200 and me.json()["username"] == "root"

    def test_bad_password_rejected(self, client):
        client.post("/api/users", json={"username": "root", "password": "password123",
                                        "role": "admin"})
        r = client.post("/api/auth/login", json={"username": "root", "password": "wrong"})
        assert r.status_code == 401

    def test_logout_revokes(self, client):
        client.post("/api/users", json={"username": "root", "password": "password123",
                                        "role": "admin"})
        token = client.post("/api/auth/login",
                            json={"username": "root", "password": "password123"}).json()["token"]
        hdr = {"Authorization": f"Bearer {token}"}
        assert client.get("/api/auth/me", headers=hdr).json()["username"] == "root"
        assert client.post("/api/auth/logout", headers=hdr).json()["revoked"] is True
        # 吊销后不再是实名主体
        assert client.get("/api/auth/me", headers=hdr).json()["username"] == "anonymous"


class TestTotp:
    def _login(self, client, username="root", password="password123"):
        client.post("/api/users", json={"username": username, "password": password,
                                        "role": "admin"})
        token = client.post("/api/auth/login",
                            json={"username": username, "password": password}).json()["token"]
        return {"Authorization": f"Bearer {token}"}

    def test_provision_then_login_requires_code(self, client):
        hdr = self._login(client)
        r = client.post("/api/auth/totp/provision", headers=hdr)
        assert r.status_code == 200
        secret = r.json()["secret"]
        assert r.json()["otpauth_uri"].startswith("otpauth://totp/")
        assert r.json()["reauth_required"] is True
        assert client.get("/api/auth/me", headers=hdr).json()["username"] == "anonymous"

        # 绑定后：只给口令 ⇒ 401 且提示需要验证码
        r2 = client.post("/api/auth/login", json={"username": "root", "password": "password123"})
        assert r2.status_code == 401
        assert r2.json()["detail"]["code"] == "totp_required"

        # 给出正确验证码 ⇒ 通过
        import auth_app

        code = auth_app.totp(secret)
        r3 = client.post("/api/auth/login", json={"username": "root", "password": "password123",
                                                  "totp_code": code})
        assert r3.status_code == 200, r3.text

    def test_verify_endpoint(self, client):
        hdr = self._login(client)
        secret = client.post("/api/auth/totp/provision", headers=hdr).json()["secret"]
        import auth_app
        import auth_service

        code = auth_app.totp(secret)
        token = client.post(
            "/api/auth/login",
            json={"username": "root", "password": "password123", "totp_code": code},
        ).json()["token"]
        auth_service.get_totp_verifier().reset()
        ok = client.post(
            "/api/auth/totp/verify",
            json={"code": code},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert ok.status_code == 200 and ok.json()["verified"] is True

    def test_wrong_code_rejected(self, client):
        hdr = self._login(client)
        secret = client.post("/api/auth/totp/provision", headers=hdr).json()["secret"]
        import auth_app

        token = client.post(
            "/api/auth/login",
            json={
                "username": "root",
                "password": "password123",
                "totp_code": auth_app.totp(secret),
            },
        ).json()["token"]
        r = client.post(
            "/api/auth/totp/verify",
            json={"code": "000000"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200 and r.json()["verified"] is False


class TestUserAdmin:
    def _admin_headers(self, client):
        client.post("/api/users", json={"username": "root", "password": "password123",
                                        "role": "admin"})
        token = client.post("/api/auth/login",
                            json={"username": "root", "password": "password123"}).json()["token"]
        return {"Authorization": f"Bearer {token}"}

    def test_list_users_hides_secrets(self, client):
        hdr = self._admin_headers(client)
        client.post("/api/users", json={"username": "u2", "password": "password123"},
                    headers=hdr)
        body = client.get("/api/users", headers=hdr).json()
        names = {u["username"] for u in body["users"]}
        assert {"root", "u2"} <= names
        blob = str(body)
        assert "password" not in blob and "secret" not in blob and "salt" not in blob

    def test_set_role_and_disable(self, client):
        hdr = self._admin_headers(client)
        client.post("/api/users", json={"username": "u3", "password": "password123"}, headers=hdr)
        user_token = client.post(
            "/api/auth/login",
            json={"username": "u3", "password": "password123"},
        ).json()["token"]
        user_hdr = {"Authorization": f"Bearer {user_token}"}
        r = client.patch("/api/users/u3", json={"role": "operator"}, headers=hdr)
        assert r.status_code == 200 and r.json()["changed"]["role"] is True
        assert client.get("/api/auth/me", headers=user_hdr).json()["username"] == "anonymous"

        user_token = client.post(
            "/api/auth/login",
            json={"username": "u3", "password": "password123"},
        ).json()["token"]
        user_hdr = {"Authorization": f"Bearer {user_token}"}
        r_password = client.patch(
            "/api/users/u3", json={"password": "newpassword123"}, headers=hdr,
        )
        assert r_password.status_code == 200
        assert client.get("/api/auth/me", headers=user_hdr).json()["username"] == "anonymous"

        r2 = client.patch("/api/users/u3", json={"disabled": True}, headers=hdr)
        assert r2.status_code == 200
        # 被禁用后无法登录
        assert client.post("/api/auth/login",
                           json={"username": "u3", "password": "password123"}).status_code == 401

    def test_last_admin_protected(self, client):
        hdr = self._admin_headers(client)
        r = client.patch("/api/users/root", json={"disabled": True}, headers=hdr)
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "last_admin_protected"
        r2 = client.delete("/api/users/root", headers=hdr)
        assert r2.status_code == 400
        assert r2.json()["detail"]["code"] == "last_admin_protected"

    def test_delete_user(self, client):
        hdr = self._admin_headers(client)
        client.post("/api/users", json={"username": "u4", "password": "password123"}, headers=hdr)
        r = client.delete("/api/users/u4", headers=hdr)
        assert r.status_code == 200 and r.json()["status"] == "deleted"
        assert client.delete("/api/users/u4", headers=hdr).status_code == 404

    def test_non_admin_cannot_manage(self, client):
        hdr = self._admin_headers(client)
        client.post("/api/users", json={"username": "v1", "password": "password123",
                                        "role": "viewer"}, headers=hdr)
        vt = client.post("/api/auth/login",
                         json={"username": "v1", "password": "password123"}).json()["token"]
        vh = {"Authorization": f"Bearer {vt}"}
        assert client.get("/api/users", headers=vh).status_code == 403

    def test_optional_auth_mode_cannot_anonymously_take_over_accounts(self, client):
        client.post(
            "/api/users",
            json={"username": "root", "password": "password123", "role": "admin"},
        )

        assert client.get("/api/users").status_code == 401
        assert client.post(
            "/api/users",
            json={"username": "attacker", "password": "password123", "role": "admin"},
        ).status_code == 401
        assert client.patch(
            "/api/users/root", json={"password": "attacker-password"},
        ).status_code == 401
        assert client.delete("/api/users/root").status_code == 401
        assert client.post("/api/auth/totp/provision").status_code == 401
        assert client.post(
            "/api/auth/login",
            json={"username": "root", "password": "attacker-password"},
        ).status_code == 401
