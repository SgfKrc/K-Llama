"""Local node bootstrap configuration.

The regular .env file is intentionally not bundled with installers because it
contains secrets.  This module provides a small non-source-controlled runtime
configuration file used after a trusted first-connect bootstrap.
"""

from __future__ import annotations

import copy
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from network_address import canonical_host
from local_secret_store import LocalSecretStore
from release_contract import (
    DEFAULT_PACKAGED_NODE_ROLE,
    is_release_environment_locked,
    release_profile_enforced,
)


_ROLE_ALIASES = {
    "master": "master",
    "auto": "auto",
    "slave": "client",
    "worker": "client",
    "client": "client",
}

_CLUSTER_SECRET_RECORD = "cluster_secret"
_CLUSTER_SECRET_REF = "local-secret-store:v1"
_CLUSTER_CREDENTIAL_SCHEMA = "qlh.cluster-credential.v1"


def _parse_node_role(value: Any, *, source: str) -> str:
    raw = str(value or "").strip().lower()
    try:
        return _ROLE_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(f"invalid node role from {source}: {value!r}") from exc


def get_app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_node_config_path() -> Path:
    override = os.environ.get("QLH_NODE_CONFIG_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    # Keep identity outside the checkout in both frozen and source mode.  A
    # source checkout is disposable (and patch delivery may hard-reset/clean
    # it), while the node identity and cluster secret belong to the user.
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        return base / "QLH-Edge-Inference" / "node_config.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "qlh" / "node_config.json"
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "qlh" / "node_config.json"


def get_node_secret_store_path() -> Path:
    override = os.environ.get("QLH_NODE_SECRET_STORE_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return get_node_config_path().with_name("node_secrets.json")


def _secret_store() -> LocalSecretStore:
    return LocalSecretStore(get_node_secret_store_path())


def _coerce_cluster_secret_epoch(value: Any) -> int:
    try:
        epoch = int(value or 1)
    except (TypeError, ValueError):
        epoch = 1
    return max(1, epoch)


def _store_cluster_credential(value: str, epoch: int) -> None:
    secret = str(value or "").strip()
    if not secret:
        raise ValueError("cluster secret must not be blank")
    payload = json.dumps(
        {
            "schema": _CLUSTER_CREDENTIAL_SCHEMA,
            "secret": secret,
            "epoch": _coerce_cluster_secret_epoch(epoch),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    _secret_store().set(_CLUSTER_SECRET_RECORD, payload)


def _read_stored_cluster_credential(*, fallback_epoch: int = 1) -> tuple[str, int]:
    raw = _secret_store().get(_CLUSTER_SECRET_RECORD).strip()
    if not raw:
        return "", _coerce_cluster_secret_epoch(fallback_epoch)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if payload is not None:
        if not isinstance(payload, dict) or payload.get("schema") != _CLUSTER_CREDENTIAL_SCHEMA:
            raise ValueError("stored cluster credential schema is invalid")
        secret = str(payload.get("secret", "") or "").strip()
        if not secret:
            raise ValueError("stored cluster credential is missing its secret")
        return secret, _coerce_cluster_secret_epoch(payload.get("epoch", 1))

    # v1 stored only the secret string and projected the epoch into public
    # node_config.json. Upgrade it in place so all future reads are atomic.
    epoch = _coerce_cluster_secret_epoch(fallback_epoch)
    _store_cluster_credential(raw, epoch)
    return raw, epoch


def _environment_cluster_secret_epoch() -> int:
    return _coerce_cluster_secret_epoch(os.environ.get("QLH_CLUSTER_SECRET_EPOCH", 1))


def get_local_cluster_secret() -> str:
    """Read the encrypted credential first; environment is import-only fallback."""
    stored, _epoch = _read_stored_cluster_credential(
        fallback_epoch=_environment_cluster_secret_epoch(),
    )
    if stored:
        return stored
    return os.environ.get("QLH_CLUSTER_SECRET", "").strip()


def _sanitized_config_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Move legacy/plain credentials out of JSON before any write."""
    payload = copy.deepcopy(data)
    cluster = payload.get("cluster") if isinstance(payload.get("cluster"), dict) else {}
    cluster = dict(cluster)
    plaintext = str(cluster.pop("cluster_secret", "") or "").strip()
    epoch = _coerce_cluster_secret_epoch(cluster.get("cluster_secret_epoch", 1))
    stored_secret, stored_epoch = _read_stored_cluster_credential(fallback_epoch=epoch)
    if plaintext and (not stored_secret or epoch > stored_epoch):
        _store_cluster_credential(plaintext, epoch)
        stored_secret, stored_epoch = plaintext, epoch
    if not stored_secret:
        imported_secret = os.environ.get("QLH_CLUSTER_SECRET", "").strip()
        if imported_secret:
            imported_epoch = _environment_cluster_secret_epoch()
            _store_cluster_credential(imported_secret, imported_epoch)
            stored_secret, stored_epoch = imported_secret, imported_epoch
    if stored_secret:
        cluster["cluster_secret_ref"] = _CLUSTER_SECRET_REF
        cluster["cluster_secret_epoch"] = stored_epoch
    payload["cluster"] = cluster
    return payload


def resolve_initial_node_role() -> str:
    """Resolve an explicit/confirmed role, then the release/development default."""
    explicit = os.environ.get("QLH_NODE_ROLE", "").strip().lower()
    if explicit:
        return _parse_node_role(explicit, source="QLH_NODE_ROLE")
    data = load_node_config()
    node = data.get("node") if isinstance(data.get("node"), dict) else {}
    configured = str(node.get("role", "")).strip().lower()
    confirmed = bool(node.get("role_confirmed", False) or data.get("bootstrapped", False))
    if configured and confirmed:
        return _parse_node_role(configured, source="node_config.json")
    if release_profile_enforced():
        return DEFAULT_PACKAGED_NODE_ROLE
    return "client"


def load_node_config() -> dict[str, Any]:
    path = get_node_config_path()
    legacy_path = get_app_root() / "node_config.json"
    if path.is_file():
        candidate = path
    elif legacy_path != path and legacy_path.is_file():
        candidate = legacy_path
    else:
        return {}

    try:
        with candidate.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}

    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    legacy_plaintext = bool(str(cluster.get("cluster_secret", "") or "").strip())
    if candidate != path or legacy_plaintext:
        # Migration failures are security failures. Never continue to another
        # stale config source or silently keep plaintext as the active truth.
        write_node_config(data)
        if candidate != path:
            _retire_legacy_node_config(candidate, data)
        data = _sanitized_config_payload(data)
    return data


def _write_node_config_path(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _sanitized_config_payload(data)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(tmp_path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        tmp_path.chmod(0o600)
    except OSError:
        pass
    os.replace(tmp_path, path)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def _retire_legacy_node_config(path: Path, data: dict[str, Any]) -> None:
    """Remove an installation-directory config or scrub it atomically."""
    try:
        path.unlink()
        return
    except FileNotFoundError:
        return
    except OSError:
        # Program Files policies, antivirus scanners or backup agents can
        # deny deletion transiently. Replacing the legacy file with the same
        # sanitized payload still removes the plaintext credential.
        _write_node_config_path(path, data)


def write_node_config(data: dict[str, Any]) -> Path:
    return _write_node_config_path(get_node_config_path(), data)


def _set_env_value(name: str, value: Any, *, overwrite: bool = False) -> None:
    if value is None:
        return
    value_str = str(value).strip()
    if not value_str:
        return
    if overwrite:
        os.environ[name] = value_str
    else:
        os.environ.setdefault(name, value_str)


def _normalize_master_endpoint(host: str | None, port: int | str | None) -> dict[str, Any] | None:
    """Return a canonical, safe-to-persist master TCP endpoint."""
    normalized_host = canonical_host(host)
    if not normalized_host:
        return None
    try:
        normalized_port = int(port or 0)
    except (TypeError, ValueError):
        return None
    if not 1 <= normalized_port <= 65535:
        return None

    address_family = "hostname"
    try:
        import ipaddress

        address = ipaddress.ip_address(normalized_host.split("%", 1)[0])
        address_family = "ipv6" if address.version == 6 else "ipv4"
    except ValueError:
        pass
    return {
        "host": normalized_host,
        "port": normalized_port,
        "address_family": address_family,
    }


def get_preferred_master_endpoint(
    config_data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Read the user-selected endpoint without trusting malformed old state."""
    data = config_data if config_data is not None else load_node_config()
    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    preferred = (
        cluster.get("preferred_master_endpoint")
        if isinstance(cluster.get("preferred_master_endpoint"), dict)
        else {}
    )
    return _normalize_master_endpoint(preferred.get("host"), preferred.get("port"))


def get_bootstrap_master_endpoint(
    config_data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Read the bootstrap-provided endpoint kept as a fallback to a preference."""
    data = config_data if config_data is not None else load_node_config()
    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    return _normalize_master_endpoint(
        cluster.get("master_tcp_host") or cluster.get("master_host"),
        cluster.get("master_tcp_port") or cluster.get("master_port"),
    )


def persist_preferred_master_endpoint(host: str, port: int) -> dict[str, Any]:
    """Persist an explicit successful connection without replacing bootstrap data.

    The selected endpoint belongs to the user-owned node configuration.  The
    original bootstrap endpoint remains available for deterministic recovery
    when the preferred address cannot be reached.
    """
    endpoint = _normalize_master_endpoint(host, port)
    if endpoint is None:
        raise ValueError("invalid master endpoint")

    data = load_node_config()
    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    data["cluster"] = {
        **cluster,
        "preferred_master_endpoint": endpoint,
    }
    write_node_config(data)
    apply_node_config_to_env(data, overwrite=True)
    _sync_loaded_module_attr("config", "CLIENT_MASTER_HOST", endpoint["host"])
    _sync_loaded_module_attr("config", "CLIENT_MASTER_PORT", endpoint["port"])
    return endpoint


def apply_node_config_to_env(
    config_data: dict[str, Any] | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    data = config_data if config_data is not None else load_node_config()
    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    projected_epoch = _coerce_cluster_secret_epoch(cluster.get("cluster_secret_epoch", 1))
    stored_secret, stored_epoch = _read_stored_cluster_credential(
        fallback_epoch=projected_epoch,
    )
    if stored_secret:
        # The encrypted pair is the single credential authority. A stale .env
        # left behind by an older deployment must never roll back a rotation.
        os.environ["QLH_CLUSTER_SECRET"] = stored_secret
        os.environ["QLH_CLUSTER_SECRET_EPOCH"] = str(stored_epoch)
    if not data:
        return {}

    node = data.get("node") if isinstance(data.get("node"), dict) else {}
    preferred_endpoint = get_preferred_master_endpoint(data)
    configured_host = (
        preferred_endpoint["host"]
        if preferred_endpoint is not None
        else cluster.get("master_tcp_host") or cluster.get("master_host")
    )
    configured_port = (
        preferred_endpoint["port"]
        if preferred_endpoint is not None
        else cluster.get("master_tcp_port") or cluster.get("master_port")
    )
    # An explicit successful connection is user-owned runtime state.  It must
    # win over a stale installer or source-checkout .env bootstrap address.
    endpoint_overwrite = overwrite or preferred_endpoint is not None

    _set_env_value("QLH_NODE_ROLE", node.get("role"), overwrite=overwrite)
    _set_env_value("QLH_NODE_ID", node.get("node_id"), overwrite=overwrite)
    _set_env_value("QLH_NODE_TYPE", node.get("node_type"), overwrite=overwrite)
    if not stored_secret:
        _set_env_value("QLH_CLUSTER_SECRET", get_local_cluster_secret(), overwrite=overwrite)
        _set_env_value(
            "QLH_CLUSTER_SECRET_EPOCH",
            projected_epoch,
            overwrite=overwrite,
        )
    _set_env_value(
        "QLH_MASTER_HOST",
        configured_host,
        overwrite=endpoint_overwrite,
    )
    _set_env_value(
        "QLH_MASTER_PORT",
        configured_port,
        overwrite=endpoint_overwrite,
    )
    _set_env_value(
        "QLH_CLIENT_MASTER_HOST",
        configured_host,
        overwrite=endpoint_overwrite,
    )
    _set_env_value(
        "QLH_CLIENT_MASTER_PORT",
        configured_port,
        overwrite=endpoint_overwrite,
    )
    _set_env_value("QLH_MASTER_API_HOST", cluster.get("master_api_host"), overwrite=overwrite)
    _set_env_value("QLH_MASTER_API_PORT", cluster.get("master_api_port"), overwrite=overwrite)
    _set_env_value(
        "QLH_API_PORT",
        cluster.get("master_api_port") if node.get("role") == "master" else None,
        overwrite=overwrite,
    )
    # Development feature gates remain user preferences. A release profile is
    # an artifact-owned upper bound and cannot be changed by stale user state.
    features = data.get("features") if isinstance(data.get("features"), dict) else {}
    if not is_release_environment_locked("QLH_TASK_GRAPH_ENABLED"):
        _set_env_value(
            "QLH_TASK_GRAPH_ENABLED", features.get("task_graph_enabled"), overwrite=True,
        )
    if not is_release_environment_locked("QLH_TASK_WORKER_EXPERIMENTAL_ENABLED"):
        _set_env_value(
            "QLH_TASK_WORKER_EXPERIMENTAL_ENABLED",
            features.get("task_worker_experimental_enabled"),
            overwrite=True,
        )
    return data


def build_bootstrap_config(response: dict[str, Any]) -> dict[str, Any]:
    cluster = response.get("cluster") if isinstance(response.get("cluster"), dict) else {}
    node = response.get("node") if isinstance(response.get("node"), dict) else {}
    existing = load_node_config()
    existing_cluster = (
        existing.get("cluster") if isinstance(existing.get("cluster"), dict) else {}
    )
    features = existing.get("features") if isinstance(existing.get("features"), dict) else {}
    result = {
        "bootstrapped": True,
        "cluster": {
            "cluster_id": cluster.get("cluster_id", "qlh-default"),
            "master_api_host": cluster.get("master_api_host", ""),
            "master_api_port": int(cluster.get("master_api_port", 8000) or 8000),
            "master_tcp_host": cluster.get("master_tcp_host", ""),
            "master_tcp_port": int(cluster.get("master_tcp_port", 8888) or 8888),
            "cluster_secret_ref": _CLUSTER_SECRET_REF,
            "cluster_secret_epoch": int(cluster.get("cluster_secret_epoch", 1) or 1),
        },
        "node": {
            "node_id": node.get("node_id", ""),
            "role": node.get("role", "client"),
            "node_type": node.get("node_type", "pc"),
            "pipeline_worker": bool(node.get("pipeline_worker", True)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if features:
        result["features"] = dict(features)
    # A preference is valid only inside the same cluster.  A successful join
    # to another cluster must not silently retain its previous master.
    existing_cluster_id = str(existing_cluster.get("cluster_id", "") or "")
    response_cluster_id = str(cluster.get("cluster_id", "qlh-default") or "")
    if not existing_cluster_id or existing_cluster_id == response_cluster_id:
        preferred_endpoint = get_preferred_master_endpoint(existing)
        if preferred_endpoint is not None:
            result["cluster"]["preferred_master_endpoint"] = preferred_endpoint
    return result


def persist_bootstrap_response(response: dict[str, Any]) -> Path:
    cluster = response.get("cluster") if isinstance(response.get("cluster"), dict) else {}
    secret = str(cluster.get("cluster_secret", "") or "").strip()
    if not secret:
        raise ValueError("bootstrap response did not contain a decrypted cluster credential")
    epoch = _coerce_cluster_secret_epoch(cluster.get("cluster_secret_epoch", 1))
    _store_cluster_credential(secret, epoch)
    os.environ["QLH_CLUSTER_SECRET"] = secret
    os.environ["QLH_CLUSTER_SECRET_EPOCH"] = str(epoch)
    config_data = build_bootstrap_config(response)
    path = write_node_config(config_data)
    apply_node_config_to_env(config_data, overwrite=True)
    return path


def ensure_local_cluster_secret() -> str:
    """Return an existing cluster secret or create one in the encrypted store."""
    data = load_node_config()
    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    projected_epoch = _coerce_cluster_secret_epoch(cluster.get("cluster_secret_epoch", 1))
    secret, epoch = _read_stored_cluster_credential(fallback_epoch=projected_epoch)
    if secret:
        os.environ["QLH_CLUSTER_SECRET"] = secret
        os.environ["QLH_CLUSTER_SECRET_EPOCH"] = str(epoch)
        return secret

    secret = os.environ.get("QLH_CLUSTER_SECRET", "").strip() or secrets.token_urlsafe(32)
    epoch = (
        _environment_cluster_secret_epoch()
        if os.environ.get("QLH_CLUSTER_SECRET", "").strip()
        else projected_epoch
    )
    _store_cluster_credential(secret, epoch)
    node = data.get("node") if isinstance(data.get("node"), dict) else {}
    explicit_role = os.environ.get("QLH_NODE_ROLE", "").strip()
    role_confirmed = bool(node.get("role_confirmed", False) or data.get("bootstrapped", False))
    if not data and explicit_role:
        role_confirmed = True
    persisted_node = {
        **node,
        "role_confirmed": role_confirmed,
        "node_id": node.get("node_id", os.environ.get("QLH_NODE_ID", "master")),
        "node_type": node.get("node_type", os.environ.get("QLH_NODE_TYPE", "pc")),
        "pipeline_worker": bool(node.get("pipeline_worker", True)),
    }
    if explicit_role:
        persisted_node["role"] = _parse_node_role(explicit_role, source="QLH_NODE_ROLE")
    data.update({
        "bootstrapped": bool(data.get("bootstrapped", False)),
        "cluster": {
            **cluster,
            "cluster_id": cluster.get("cluster_id", "qlh-default"),
            "cluster_secret_ref": _CLUSTER_SECRET_REF,
            "cluster_secret_epoch": epoch,
        },
        "node": persisted_node,
    })
    write_node_config(data)
    os.environ["QLH_CLUSTER_SECRET"] = secret
    os.environ["QLH_CLUSTER_SECRET_EPOCH"] = str(epoch)
    return secret


def get_local_cluster_secret_epoch(config_data: dict[str, Any] | None = None) -> int:
    data = config_data if config_data is not None else load_node_config()
    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    projected_epoch = _coerce_cluster_secret_epoch(cluster.get("cluster_secret_epoch", 1))
    stored_secret, stored_epoch = _read_stored_cluster_credential(
        fallback_epoch=projected_epoch,
    )
    if stored_secret:
        return stored_epoch
    if os.environ.get("QLH_CLUSTER_SECRET", "").strip():
        return _environment_cluster_secret_epoch()
    return projected_epoch


def rotate_local_cluster_secret() -> tuple[str, int]:
    """Rotate the local cluster root and advance its credential generation.

    Operators must first remove the revoked device from the Tailnet trust root,
    then rotate on the master and re-bootstrap every remaining node.  Old
    nodes fail registration because their epoch and HMAC secret no longer
    match the master.
    """
    old_secret = ensure_local_cluster_secret()
    data = load_node_config()
    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    old_epoch = get_local_cluster_secret_epoch(data)
    epoch = old_epoch + 1
    secret = secrets.token_urlsafe(32)
    _store_cluster_credential(secret, epoch)
    data["cluster"] = {
        **cluster,
        "cluster_secret_ref": _CLUSTER_SECRET_REF,
        "cluster_secret_epoch": epoch,
    }
    try:
        write_node_config(data)
    except Exception:
        # Keep the encrypted record and the public epoch coherent when the
        # config replacement fails (disk full, ACL denial, antivirus race).
        _store_cluster_credential(old_secret, old_epoch)
        raise
    os.environ["QLH_CLUSTER_SECRET"] = secret
    os.environ["QLH_CLUSTER_SECRET_EPOCH"] = str(epoch)
    _sync_loaded_module_attr("config", "CLUSTER_SECRET", secret)
    _sync_loaded_module_attr("config", "CLUSTER_SECRET_EPOCH", epoch)
    return secret, epoch


def _sync_loaded_module_attr(module_name: str, attr: str, value: Any) -> None:
    module = sys.modules.get(module_name)
    if module is not None:
        try:
            setattr(module, attr, value)
        except Exception:
            pass


def apply_runtime_config(response: dict[str, Any]) -> None:
    """Update already-imported runtime modules after bootstrap."""
    cluster = response.get("cluster") if isinstance(response.get("cluster"), dict) else {}
    node = response.get("node") if isinstance(response.get("node"), dict) else {}
    runtime_secret = str(cluster.get("cluster_secret", "") or "").strip()
    runtime_epoch = _coerce_cluster_secret_epoch(cluster.get("cluster_secret_epoch", 1))
    if runtime_secret:
        _store_cluster_credential(runtime_secret, runtime_epoch)
        os.environ["QLH_CLUSTER_SECRET"] = runtime_secret
    if cluster.get("cluster_secret_epoch") is not None:
        os.environ["QLH_CLUSTER_SECRET_EPOCH"] = str(runtime_epoch)
    apply_node_config_to_env(build_bootstrap_config(response), overwrite=True)
    try:
        import config as cfg

        if runtime_secret:
            cfg.CLUSTER_SECRET = runtime_secret
        cfg.CLUSTER_SECRET_EPOCH = runtime_epoch
        if cluster.get("master_tcp_host"):
            cfg.CLIENT_MASTER_HOST = str(cluster["master_tcp_host"])
        if cluster.get("master_tcp_port"):
            cfg.CLIENT_MASTER_PORT = int(cluster["master_tcp_port"])
        if node.get("node_id"):
            _node_id = str(node["node_id"])
            try:
                from node_runtime import node_runtime
                node_runtime.set_node_id(_node_id)
            except Exception:
                pass
            _sync_loaded_module_attr("scheduler", "NODE_ID", _node_id)
        if node.get("role"):
            _role = str(node["role"])
            try:
                from node_runtime import node_runtime
                node_runtime.set_node_role(_role)
            except Exception:
                pass
            _sync_loaded_module_attr("scheduler", "NODE_ROLE", _role)
    except Exception:
        pass
