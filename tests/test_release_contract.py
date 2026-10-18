from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

import distributed_completion
import node_config
import release_contract


ROOT = Path(__file__).resolve().parents[1]


def _clear_role_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "QLH_NODE_ROLE",
        "QLH_RELEASE_PROFILE_ENFORCE",
        "QLH_RELEASE_CONTRACT_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_canonical_versions_are_exported() -> None:
    contract = json.loads((ROOT / "release-contract.json").read_text(encoding="utf-8"))
    assert release_contract.PRODUCT_VERSION == contract["version"]["product"]
    assert release_contract.ANDROID_VERSION_CODE == contract["version"]["android_code"]


def test_release_profile_overrides_stale_product_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = {
        "QLH_RELEASE_PROFILE_ENFORCE": "1",
        "QLH_ROUTE_A_STAGE_OFFER": "0",
        "QLH_TASK_GRAPH_ENABLED": "1",
    }
    applied = release_contract.apply_release_profile_to_env(environment)
    assert applied == release_contract.FIXED_RELEASE_ENV
    assert environment["QLH_ROUTE_A_STAGE_OFFER"] == "1"
    assert "QLH_TASK_GRAPH_ENABLED" not in release_contract.FIXED_RELEASE_ENV
    assert environment["QLH_TASK_WORKER_EXPERIMENTAL_ENABLED"] == "1"


def test_release_runtime_feature_conflict_is_named_and_side_effect_free(monkeypatch):
    import api_server
    monkeypatch.setenv("QLH_RELEASE_PROFILE_ENFORCE", "1")
    before = api_server.TASK_WORKER_EXPERIMENTAL_ENABLED
    with pytest.raises(HTTPException) as error:
        api_server._apply_runtime_feature_settings({"QLH_TASK_WORKER_EXPERIMENTAL_ENABLED": False})
    assert error.value.status_code == 409
    assert error.value.detail["code"] == "RELEASE_PROFILE_FEATURE_LOCKED"
    assert api_server.TASK_WORKER_EXPERIMENTAL_ENABLED == before


def test_runtime_feature_lock_only_rejects_release_conflicts(monkeypatch):
    import api_server
    monkeypatch.setenv("QLH_RELEASE_PROFILE_ENFORCE", "1")
    assert api_server._apply_runtime_feature_settings({"QLH_TASK_WORKER_EXPERIMENTAL_ENABLED": True}) == {"QLH_TASK_WORKER_EXPERIMENTAL_ENABLED": True}
    with pytest.raises(ValueError):
        api_server._apply_runtime_feature_settings({"QLH_TASK_GRAPH_ENABLED": True})


def test_packaged_role_is_deterministic_without_user_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_role_env(monkeypatch)
    monkeypatch.setenv("QLH_RELEASE_PROFILE_ENFORCE", "1")
    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(tmp_path / "missing.json"))
    assert node_config.resolve_initial_node_role() == "master"


def test_unconfirmed_persisted_role_does_not_override_release_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "node_config.json"
    config_path.write_text(
        json.dumps({"node": {"role": "client", "role_confirmed": False}}),
        encoding="utf-8",
    )
    _clear_role_env(monkeypatch)
    monkeypatch.setenv("QLH_RELEASE_PROFILE_ENFORCE", "1")
    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(config_path))
    assert node_config.resolve_initial_node_role() == "master"


def test_invalid_explicit_role_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QLH_NODE_ROLE", "typo")
    with pytest.raises(ValueError, match="invalid node role"):
        node_config.resolve_initial_node_role()


def test_distributed_required_rejects_local_or_external_completion() -> None:
    with pytest.raises(distributed_completion.DistributedCompletionError):
        distributed_completion.validate_distributed_completion(
            "distributed_required",
            {"distributed_used": False, "execution_mode": "external_api"},
        )


def test_release_distributed_required_accepts_route_a_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QLH_RELEASE_PROFILE_ENFORCE", "1")
    distributed_completion.validate_distributed_completion(
        "distributed_required",
        {
            "distributed_used": True,
            "execution_mode": "route_a_stage_offer_v3",
            "workers_used": ["worker-1"],
            "layer_segments": [[16, 24]],
        },
    )


def test_release_distributed_required_rejects_non_product_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QLH_RELEASE_PROFILE_ENFORCE", "1")
    with pytest.raises(distributed_completion.DistributedCompletionError, match="不允许"):
        distributed_completion.validate_distributed_completion(
            "distributed_required",
            {"distributed_used": True, "execution_mode": "task_graph"},
        )
