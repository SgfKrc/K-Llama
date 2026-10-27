import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("credential_source", ["plaintext", "encrypted"])
@pytest.mark.parametrize("config_entry", ["legacy", "runtime"])
def test_legacy_auto_migration_preserves_cluster_credential_and_node_identity(
    monkeypatch, tmp_path, credential_source, config_entry,
):
    import node_config
    from local_secret_store import LocalSecretStore
    from qlh_core.adapters.settings import IdentityStore, SettingsStore

    config_path = tmp_path / "node.json"
    secret_path = tmp_path / "secrets.json"
    secret = "static-role-migration-credential-32chars"
    document = {
        "bootstrapped": True,
        "node": {"node_id": "former-candidate", "role": "auto", "role_confirmed": True},
        "cluster": {"cluster_id": "fixture", "cluster_secret_epoch": 7},
    }
    if credential_source == "plaintext":
        document["cluster"]["cluster_secret"] = secret
    else:
        LocalSecretStore(secret_path).set("cluster_secret", json.dumps({
            "schema": "qlh.cluster-credential.v1", "secret": secret, "epoch": 7,
        }))
        document["cluster"]["cluster_secret_ref"] = "local-secret-store:v1"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    environment = {
        "QLH_NODE_CONFIG_PATH": str(config_path),
        "QLH_NODE_SECRET_STORE_PATH": str(secret_path),
        "QLH_STATE_DIR": str(tmp_path / "state"),
        "QLH_RELEASE_PROFILE_ENFORCE": "0",
    }
    monkeypatch.delenv("QLH_NODE_ROLE", raising=False)
    monkeypatch.delenv("QLH_CLUSTER_SECRET", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    if config_entry == "legacy":
        migrated = node_config.load_node_config()
        assert migrated["node"]["role"] == "client"
        assert node_config.resolve_initial_node_role() == "client"
        assert node_config.get_local_cluster_secret() == secret
    else:
        store = SettingsStore(environment, dotenv_path=tmp_path / "absent.env")
        config = store.load()
        assert config.node_role == "client"
        assert json.loads(config_path.read_text("utf-8"))["node"]["role"] == "auto"
        resolved, incarnation = IdentityStore(str(config_path), str(secret_path)).open(config)
        assert incarnation.node_id == "former-candidate"
        assert incarnation.credential_epoch == 7
        assert resolved.environment_snapshot()["QLH_CLUSTER_SECRET"] == secret
        assert resolved.environment_snapshot()["QLH_CLUSTER_SECRET_EPOCH"] == "7"

    persisted = json.loads(config_path.read_text("utf-8"))
    assert persisted["node"] == {
        "node_id": "former-candidate", "role": "client", "role_confirmed": True,
    }
    assert persisted["cluster"]["cluster_secret_epoch"] == 7
    assert "cluster_secret" not in persisted["cluster"]
    assert secret not in Path(secret_path).read_text("utf-8")
