"""Explicit settings and identity IO, isolated by caller-supplied paths."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import secrets
import socket
import sys
import threading
import uuid

from local_secret_store import LocalSecretStore
from qlh_core.contracts.runtime import NodeIncarnation, RuntimeConfig


_IDENTITY_LOCKS_GUARD = threading.Lock()
_IDENTITY_LOCKS: dict[str, threading.RLock] = {}


def _identity_lock(path: Path):
    key = os.path.normcase(str(path.expanduser().resolve()))
    with _IDENTITY_LOCKS_GUARD:
        return _IDENTITY_LOCKS.setdefault(key, threading.RLock())


class IdentityStore:
    """Use the existing public document and encrypted credential pair."""
    def __init__(self, document: str, secret_store: str, *, legacy_document="") -> None:
        self.path = Path(document)
        self.legacy_path = Path(legacy_document) if legacy_document else None
        self.secrets = LocalSecretStore(secret_store)
        self._lock = _identity_lock(self.path)

    def read_document(self) -> dict:
        with self._lock:
            candidate = self.path
            if not candidate.exists():
                candidate = self.legacy_path
                if candidate is None or not candidate.exists():
                    return {}
            value = json.loads(candidate.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError("node configuration must be an object")
            return value

    def _write(self, document: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(document, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)

    def persist_participation(self, mode):
        if mode not in {"distributed", "local"}:
            raise ValueError("participation_mode_invalid")
        with self._lock:
            document = self.read_document()
            node = dict(document.get("node") or {})
            node["desired_inference_mode"] = mode
            document["node"] = node
            self._write(document)

    def credential(self, document: dict, environment: dict) -> tuple[str, int]:
        cluster = document.get("cluster") or {}
        raw = self.secrets.get("cluster_secret")
        epoch = max(1, int(cluster.get("cluster_secret_epoch") or environment.get("QLH_CLUSTER_SECRET_EPOCH") or 1))
        if raw:
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                # Legacy string upgrade happens only during explicit startup.
                value = {"schema": "qlh.cluster-credential.v1", "secret": raw, "epoch": epoch}
            if not isinstance(value, dict) or value.get("schema") != "qlh.cluster-credential.v1":
                raise ValueError("stored cluster credential schema is invalid")
            secret = str(value.get("secret") or "").strip()
            if not secret:
                raise ValueError("stored cluster credential is missing its secret")
            return secret, max(1, int(value["epoch"]))
        secret = str(cluster.get("cluster_secret") or environment.get("QLH_CLUSTER_SECRET") or "").strip()
        return secret, epoch

    def open(self, config: RuntimeConfig) -> tuple[RuntimeConfig, NodeIncarnation]:
        with self._lock:
            document = self.read_document()
            environment = config.environment_snapshot()
            secret, epoch = self.credential(document, environment)
            if not secret and config.node_role == "master" and environment.get("QLH_RELEASE_PROFILE_ENFORCE") == "1":
                secret = secrets.token_urlsafe(32)
            if secret:
                self.secrets.set("cluster_secret", json.dumps({
                    "schema": "qlh.cluster-credential.v1", "secret": secret, "epoch": epoch,
                }, sort_keys=True))
                cluster = dict(document.get("cluster") or {})
                cluster.pop("cluster_secret", None)
                cluster.update(cluster_secret_ref="local-secret-store:v1", cluster_secret_epoch=epoch)
                document["cluster"] = cluster
                environment.update(QLH_CLUSTER_SECRET=secret, QLH_CLUSTER_SECRET_EPOCH=str(epoch))
            node = dict(document.get("node") or {})
            node["node_id"] = config.node_id
            if str(node.get("role") or "").strip().lower() == "auto":
                node.update(role="client", role_confirmed=True)
            document["node"] = node
            pending = document.get("identity_migration") or {}
            if pending.get("legacy_document"):
                self.legacy_path = Path(pending["legacy_document"])
            migrating = self.legacy_path is not None and self.legacy_path != self.path and self.legacy_path.exists()
            if migrating:
                document["identity_migration"] = {"legacy_document": str(self.legacy_path)}
            self._write(document)
            if migrating:
                # Keep an installation copy as a sanitized recovery document;
                # the user-directory document becomes the single active source.
                legacy = IdentityStore(str(self.legacy_path), str(config.secret_store_path))
                clean_copy = {key: value for key, value in document.items() if key != "identity_migration"}
                legacy._write(clean_copy)
            if "identity_migration" in document:
                document.pop("identity_migration")
                self._write(document)
            from config import build_settings
            values = serializable_settings(build_settings(environment))
            values["LOG_DIR"] = config.log_dir
            final = replace(config, environment=tuple(sorted(environment.items())),
                            settings_json=json.dumps(values, sort_keys=True), legacy_document="")
            return final, NodeIncarnation(config.node_id, uuid.uuid4().hex, epoch)


    def _persist_config(self, config, document, secret, epoch):
        """Publish credentials before their public projection; failures stay visible."""
        previous_credential = self.secrets.get("cluster_secret")
        credential = json.dumps({
            "schema": "qlh.cluster-credential.v1", "secret": secret, "epoch": epoch,
        }, sort_keys=True) if secret else previous_credential
        credential_changed = credential != previous_credential
        if credential_changed:
            self.secrets.set("cluster_secret", credential)
        cluster = dict(document.get("cluster") or {})
        cluster.pop("cluster_secret", None)
        if secret:
            cluster.update(cluster_secret_ref="local-secret-store:v1", cluster_secret_epoch=epoch)
        document["cluster"] = cluster
        try:
            self._write(document)
        except Exception:
            if credential_changed:
                if previous_credential:
                    self.secrets.set("cluster_secret", previous_credential)
                else:
                    self.secrets.delete("cluster_secret")
            raise
        environment = config.environment_snapshot()
        node = document.get("node") or {}
        for key, value in {
            "QLH_NODE_ID": node.get("node_id"), "QLH_NODE_ROLE": node.get("role"),
            "QLH_NODE_TYPE": node.get("node_type"),
            "QLH_MASTER_API_HOST": cluster.get("master_api_host"),
            "QLH_MASTER_API_PORT": cluster.get("master_api_port"),
        }.items():
            if value is not None and str(value):
                environment[key] = str(value)
        from node_config import get_preferred_master_endpoint, get_bootstrap_master_endpoint
        endpoint = get_preferred_master_endpoint(document) or get_bootstrap_master_endpoint(document)
        if endpoint is not None:
            environment.update(QLH_CLIENT_MASTER_HOST=endpoint["host"], QLH_CLIENT_MASTER_PORT=str(endpoint["port"]),
                               QLH_MASTER_HOST=endpoint["host"], QLH_MASTER_PORT=str(endpoint["port"]))
        if secret:
            environment.update(QLH_CLUSTER_SECRET=secret, QLH_CLUSTER_SECRET_EPOCH=str(epoch))
        from config import build_settings
        values = serializable_settings(build_settings(environment))
        values["LOG_DIR"] = config.log_dir
        return replace(config, node_id=environment["QLH_NODE_ID"], node_role=environment["QLH_NODE_ROLE"],
                       environment=tuple(sorted(environment.items())), settings_json=json.dumps(values, sort_keys=True),
                       revision=config.revision + 1)

    def accept_bootstrap(self, config, response):
        """Accept an already decrypted join response into this node's identity pair."""
        from node_config import build_bootstrap_config
        with self._lock:
            existing = self.read_document()
            cluster = response.get("cluster") or {}
            secret = str(cluster.get("cluster_secret") or "").strip()
            epoch = cluster.get("cluster_secret_epoch", 1)
            if not secret or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
                raise ValueError("bootstrap response contains an invalid cluster credential")
            previous = existing.get("cluster") or {}
            old_secret, old_epoch = self.credential(existing, config.environment_snapshot())
            if previous.get("cluster_id") and previous.get("cluster_id") == cluster.get("cluster_id"):
                if epoch < old_epoch or (epoch == old_epoch and old_secret and secret != old_secret):
                    raise ValueError("bootstrap credential would regress or fork the stored epoch")
            document = {**existing, **build_bootstrap_config(response, existing=existing)}
            node = document["node"]
            if not node.get("node_id") or node.get("role") not in {"client", "master"}:
                raise ValueError("bootstrap response contains an invalid node identity")
            return self._persist_config(config, document, secret, epoch)

    def persist_preferred_master_endpoint(self, config, host, port):
        from node_config import _normalize_master_endpoint
        endpoint = _normalize_master_endpoint(host, port)
        if endpoint is None:
            raise ValueError("invalid master endpoint")
        with self._lock:
            document = self.read_document()
            document["cluster"] = {**(document.get("cluster") or {}), "preferred_master_endpoint": endpoint}
            secret, epoch = self.credential(document, config.environment_snapshot())
            return self._persist_config(config, document, secret, epoch), endpoint

    def rotate_cluster_credential(self, config):
        with self._lock:
            document = self.read_document()
            _secret, epoch = self.credential(document, config.environment_snapshot())
            secret, epoch = secrets.token_urlsafe(32), epoch + 1
            return self._persist_config(config, document, secret, epoch), (secret, epoch)

    def ensure_cluster_credential(self, config):
        with self._lock:
            document = self.read_document()
            secret, epoch = self.credential(document, config.environment_snapshot())
            if not secret:
                secret = secrets.token_urlsafe(32)
            cluster = dict(document.get("cluster") or {})
            cluster.setdefault("cluster_id", config.environment_snapshot().get("QLH_CLUSTER_ID", "qlh-default"))
            document["cluster"] = cluster
            return self._persist_config(config, document, secret, epoch), (secret, epoch)


class SettingsStore:
    """Precedence: defaults < .env < public config < explicit environment.

    Stored encrypted credentials and artifact-owned release gates always win.
    All reads are explicit; the returned snapshot does not observe later env edits.
    """
    def __init__(self, environment=None, *, dotenv_path=None, default_node_role=None) -> None:
        self._environment = dict(os.environ if environment is None else environment)
        self.dotenv_path = Path(dotenv_path) if dotenv_path is not None else None
        self._default_node_role = default_node_role

    def _read_inputs(self):
        """Resolve public settings without opening the encrypted credential store."""
        environment = {}
        if self.dotenv_path is not None:
            candidates = [self.dotenv_path]
        else:
            from node_config import get_app_root
            import config as legacy_config
            candidates = [get_app_root() / ".env", Path(legacy_config.__file__).parent / ".env", Path.cwd() / ".env"]
        from dotenv import dotenv_values
        from dotenv.variables import parse_variables
        for candidate in dict.fromkeys(candidates):
            if candidate.is_file():
                for key, value in dotenv_values(candidate, interpolate=False).items():
                    if value is not None:
                        variables = {**environment, **self._environment}
                        environment[key] = "".join(atom.resolve(variables) for atom in parse_variables(value))
                break
        environment.update(self._environment)
        # Path resolution uses the explicit environment over .env values.
        if sys.platform == "win32":
            base = Path(environment.get("LOCALAPPDATA") or Path.home() / "AppData/Local") / "QLH-Edge-Inference"
            default_config = base / "node_config.json"
            default_state = base / "state"
        elif sys.platform == "darwin":
            base = Path.home() / "Library/Application Support/qlh"
            default_config, default_state = base / "node_config.json", base / "state"
        else:
            default_config = Path(environment.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "qlh/node_config.json"
            default_state = Path(environment.get("XDG_STATE_HOME") or Path.home() / ".local/state") / "qlh-edge-inference"
        state = Path(environment.get("QLH_STATE_DIR") or default_state).expanduser().absolute()
        document = Path(environment.get("QLH_NODE_CONFIG_PATH") or default_config).expanduser().absolute()
        secret_path = Path(environment.get("QLH_NODE_SECRET_STORE_PATH") or document.with_name("node_secrets.json")).expanduser().absolute()
        legacy_document = ""
        if not environment.get("QLH_NODE_CONFIG_PATH"):
            from node_config import get_app_root
            candidate = get_app_root() / "node_config.json"
            if candidate != document and candidate.is_file() and not document.exists():
                legacy_document = str(candidate)
        identity = IdentityStore(str(document), str(secret_path), legacy_document=legacy_document)
        public = identity.read_document()
        cleanup = public.get("identity_migration") or {}
        if cleanup.get("legacy_document"):
            legacy_document = str(cleanup["legacy_document"])
        node = public.get("node") or {}
        cluster = public.get("cluster") or {}
        confirmed = bool(public.get("bootstrapped") or node.get("role_confirmed"))
        defaults = {}
        desired_mode = node.get("desired_inference_mode")
        if desired_mode is not None:
            if desired_mode not in {"distributed", "local"}:
                raise ValueError("participation_mode_invalid")
            defaults["QLH_DISTRIBUTED_INFERENCE_ENABLED"] = "1" if desired_mode == "distributed" else "0"
        if confirmed and node.get("role"):
            defaults["QLH_NODE_ROLE"] = node["role"]
        if node.get("node_id"):
            defaults["QLH_NODE_ID"] = node["node_id"]
        for name, value in (("QLH_MASTER_API_HOST", cluster.get("master_api_host")),
                            ("QLH_MASTER_API_PORT", cluster.get("master_api_port")),
                            ("QLH_API_PORT", cluster.get("master_api_port") if node.get("role") == "master" else None)):
            if value:
                defaults[name] = str(value)
        preferred = cluster.get("preferred_master_endpoint") or {}
        defaults.update({k: str(v) for k, v in {
            "QLH_CLIENT_MASTER_HOST": preferred.get("host") or cluster.get("master_tcp_host") or cluster.get("master_host"),
            "QLH_CLIENT_MASTER_PORT": preferred.get("port") or cluster.get("master_tcp_port"),
        }.items() if v})
        environment = {**environment, **defaults, **self._environment}
        if preferred.get("host") and preferred.get("port"):
            environment["QLH_CLIENT_MASTER_HOST"] = str(preferred["host"])
            environment["QLH_CLIENT_MASTER_PORT"] = str(preferred["port"])
        features = public.get("features") or {}
        for key, field in (("QLH_TASK_WORKER_EXPERIMENTAL_ENABLED", "task_worker_experimental_enabled"),):
            if field in features:
                environment[key] = "1" if features[field] else "0"
        from release_contract import apply_release_profile_to_env, DEFAULT_PACKAGED_NODE_ROLE, release_profile_enforced
        apply_release_profile_to_env(environment)
        if release_profile_enforced(environment):
            environment["QLH_RELEASE_PROFILE_ENFORCE"] = "1"
        role = str(environment.get("QLH_NODE_ROLE") or (DEFAULT_PACKAGED_NODE_ROLE if release_profile_enforced(environment) else self._default_node_role or "client")).lower()
        if role == "auto":
            role = "client"
        role = {"worker": "client", "slave": "client"}.get(role, role)
        if role not in {"master", "client"}:
            raise ValueError("invalid node role")
        run_mode = environment.get("QLH_RUN_MODE", "distributed")
        if run_mode not in {"single", "distributed"}:
            raise ValueError("invalid run mode")
        node_id = str(environment.get("QLH_NODE_ID") or ("master" if role == "master" else "client_" + socket.gethostname()))
        environment.update(QLH_STATE_DIR=str(state), QLH_NODE_CONFIG_PATH=str(document),
                           QLH_NODE_SECRET_STORE_PATH=str(secret_path), QLH_NODE_ID=node_id, QLH_NODE_ROLE=role,
                           QLH_RUN_MODE=run_mode)
        return environment, identity, public, legacy_document

    def load_listener(self) -> dict[str, str]:
        """Read listener discovery inputs before deciding whether startup is needed."""
        return self._read_inputs()[0]

    def load(self) -> RuntimeConfig:
        environment, identity, public, legacy_document = self._read_inputs()
        secret, epoch = identity.credential(public, environment)
        if secret:
            environment.update(QLH_CLUSTER_SECRET=secret, QLH_CLUSTER_SECRET_EPOCH=str(epoch))
        state, document, secret_path = (Path(environment[key]) for key in
            ("QLH_STATE_DIR", "QLH_NODE_CONFIG_PATH", "QLH_NODE_SECRET_STORE_PATH"))
        role, node_id, run_mode = (environment[key] for key in
            ("QLH_NODE_ROLE", "QLH_NODE_ID", "QLH_RUN_MODE"))
        def port(name, default):
            result = int(environment.get(name) or default)
            if not 1 <= result <= 65535:
                raise ValueError("invalid port: " + name)
            return result
        from config import build_settings
        resolved = build_settings(environment)
        values = serializable_settings(resolved)
        values["LOG_DIR"] = str(state / "logs")
        return RuntimeConfig(str(state), str(document), str(secret_path), node_id, role,
                             run_mode, port("QLH_API_PORT", 8000), port("QLH_SERVER_PORT", 8888),
                             environment.get("QLH_SERVER_IP", "0.0.0.0"), str(resolved["QUANT_TYPE"]),
                             str(state / "logs"), tuple(sorted(environment.items())), settings_json=json.dumps(values, sort_keys=True), legacy_document=legacy_document)


def serializable_settings(resolved):
    return {name: value for name, value in resolved.items()
            if name.isupper() and isinstance(value, (str, int, float, bool, dict, list, tuple, type(None)))}
