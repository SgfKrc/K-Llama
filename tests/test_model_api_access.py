"""Model API network-boundary tests."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
from model_api_access import is_model_api_source_trusted, require_model_api_source


def _request(host: str):
    return SimpleNamespace(client=SimpleNamespace(host=host))


def test_model_api_defaults_to_loopback_only(monkeypatch):
    # ★ 2026-10-09（脆性修复）：本机会为 **tailscale 协作**配 `QLH_MODEL_API_TRUSTED_CIDRS`
    #   （实测值 `100.64.0.0/10,fd7a:115c:a1e0::/48`），此时 `100.64.x` 属于"**显式受信**"
    #   ⇒ 该断言就会失败。本用例验的是**默认（未配置任何 CIDR）**时的行为，因此必须**隔离环境**
    #   —— 否则它会依赖"跑测试的机器有没有配过这个变量"，属于脆性测试（换网/加 tailscale 就红）。
    monkeypatch.delenv("QLH_MODEL_API_TRUSTED_CIDRS", raising=False)
    assert is_model_api_source_trusted(_request("127.0.0.1")) is True
    assert is_model_api_source_trusted(_request("100.64.10.20")) is False
    assert is_model_api_source_trusted(SimpleNamespace(client=None)) is False


def test_model_api_accepts_explicit_remote_cidr():
    request = _request("100.64.10.20")
    assert is_model_api_source_trusted(request, trusted_cidrs="100.64.0.0/10") is True


def test_api_server_uses_resolved_runtime_cidrs(monkeypatch):
    from types import SimpleNamespace

    runtime = SimpleNamespace(resolve_config=lambda: SimpleNamespace(
        environment_snapshot=lambda: {"QLH_MODEL_API_TRUSTED_CIDRS": "100.64.0.0/10"}
    ))
    monkeypatch.setattr(api_server, "_get_app_runtime", lambda: runtime)

    assert api_server.require_model_api_source(_request("100.64.10.20")) == "100.64.10.20"
    with pytest.raises(HTTPException) as exc:
        api_server.require_model_api_source(_request("192.0.2.10"))
    assert exc.value.status_code == 403


def test_model_api_rejects_untrusted_source_with_stable_code():
    with pytest.raises(HTTPException) as exc:
        require_model_api_source(_request("192.0.2.10"))
    assert exc.value.status_code == 403
    assert exc.value.detail["code"] == "MODEL_API_SOURCE_UNTRUSTED"


@pytest.mark.parametrize("path", ["/api/models/registry", "/api/models/downloads"])
def test_model_api_http_routes_reject_remote_source(path):
    with TestClient(api_server.app, client=("192.0.2.10", 50000)) as client:
        response = client.get(path)
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "MODEL_API_SOURCE_UNTRUSTED"
