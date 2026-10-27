"""Static cluster-role controls exposed by the TUI."""

from __future__ import annotations

from typing import Any

import tui_api
from tui_shared import API_PATHS, COMMAND_SPECS


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def get(self, path: str, **kw: Any) -> dict:
        self.calls.append(("get", path, kw))
        return {"ok": True}

    def post(self, path: str, body: Any = None, **kw: Any) -> dict:
        self.calls.append(("post", path, body, kw))
        return {"ok": True}


def test_static_master_health_path():
    assert API_PATHS["cluster_master_health"] == "/cluster/master-health"
    api = FakeApi()
    tui_api.master_health(api)
    assert api.calls[0][0:2] == ("get", "/cluster/master-health")


def test_reset_master_identity_requires_confirmation_token():
    assert API_PATHS["cluster_reset_identity"] == "/cluster/reset-identity"
    api = FakeApi()
    tui_api.reset_master_identity(api)
    assert api.calls[0][0:3] == (
        "post", "/cluster/reset-identity", {"confirm": "reset"},
    )


def test_ha_command_only_exposes_static_master_maintenance():
    ha = next(spec for spec in COMMAND_SPECS if spec["name"] == "/ha")
    assert ha["args"] == "health | reset-identity"
    assert "quorum" not in ha["desc"].lower()
    assert "transfer" not in ha["desc"].lower()
