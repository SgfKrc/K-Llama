"""Immutable startup inputs and node participation ports."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RuntimeConfig:
    state_dir: str
    node_config_path: str
    secret_store_path: str
    node_id: str
    node_role: str = "client"
    run_mode: str = "distributed"
    api_port: int = 8000
    server_port: int = 8888
    server_ip: str = "0.0.0.0"
    quant_type: str = "int4"
    log_dir: str = ""
    environment: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    revision: int = 1
    settings_json: str = field(default="{}", repr=False)
    legacy_document: str = field(default="", repr=False)

    def legacy_settings(self) -> dict[str, Any]:
        import json
        return json.loads(self.settings_json)

    def environment_snapshot(self) -> dict[str, str]:
        return dict(self.environment)


@dataclass(frozen=True)
class NodeIncarnation:
    node_id: str
    boot_id: str
    credential_epoch: int


@dataclass(frozen=True)
class ParticipationSnapshot:
    inference_mode: str = "distributed"
    revision: int = 0
    eligible: bool = True


class ParticipationPort(Protocol):
    def snapshot(self) -> ParticipationSnapshot: ...
    def set_mode(self, mode: str) -> Any: ...
