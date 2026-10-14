"""One-time, client-only cluster join grants.

This module deliberately stops at the cryptographic join contract. Auth App
verification happens at the API service boundary; this layer never receives a
TOTP seed, code, or boolean approval flag. The resulting short-lived Ed25519
grant can be rendered as both a text code and a QR payload. A SQLite ledger
makes nonce consumption survive restart.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from local_secret_store import LocalSecretStore, LocalSecretStoreError

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
except Exception as exc:  # pragma: no cover - depends on runtime packaging
    InvalidSignature = Exception  # type: ignore[assignment,misc]
    Ed25519PrivateKey = None  # type: ignore[assignment,misc]
    Ed25519PublicKey = None  # type: ignore[assignment,misc]
    serialization = None  # type: ignore[assignment]
    _CRYPTO_IMPORT_ERROR = exc
else:
    _CRYPTO_IMPORT_ERROR = None


SCHEMA_VERSION = "qlh.cluster.join-grant.v1"
GRANT_PREFIX = "qlhjoin1"
REQUEST_PREFIX = "qlhjoinreq1"
MAX_GRANT_LENGTH = 16 * 1024
MAX_REQUEST_LENGTH = 16 * 1024
DEFAULT_CAPABILITIES = ("presence", "task")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SAFE_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


class JoinContractError(ValueError):
    """Fail-closed local join contract error with a stable reason code."""

    def __init__(self, message: str, *, code: str = "join_invalid") -> None:
        self.code = code
        super().__init__(message)


def _require_crypto() -> None:
    if Ed25519PrivateKey is None:
        raise JoinContractError(
            f"cryptography is unavailable: {_CRYPTO_IMPORT_ERROR}",
            code="crypto_unavailable",
        )


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: Any, *, field: str, expected_length: int | None = None) -> bytes:
    if not isinstance(value, str) or not value:
        raise JoinContractError(f"{field} must be base64url text", code="invalid_encoding")
    try:
        raw = base64.urlsafe_b64decode(value + ("=" * (-len(value) % 4)))
    except (UnicodeEncodeError, ValueError, TypeError) as exc:
        raise JoinContractError(f"{field} is not valid base64url", code="invalid_encoding") from exc
    if expected_length is not None and len(raw) != expected_length:
        raise JoinContractError(f"{field} has invalid length", code="invalid_encoding")
    return raw


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _safe_id(value: Any, *, field: str, pattern: re.Pattern[str] = _SAFE_ID) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise JoinContractError(f"{field} is invalid", code="invalid_field")
    return value


def _normalize_endpoint(value: Any) -> str:
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
        raise JoinContractError("master_endpoint is invalid", code="invalid_endpoint")
    parsed = urlsplit(f"//{value}")
    if parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise JoinContractError("master_endpoint must be host:port", code="invalid_endpoint")
    try:
        port = parsed.port
    except ValueError as exc:
        raise JoinContractError("master_endpoint port is invalid", code="invalid_endpoint") from exc
    if not parsed.hostname or port is None or not 1 <= port <= 65535:
        raise JoinContractError("master_endpoint must include a valid port", code="invalid_endpoint")
    host = parsed.hostname.lower()
    if ":" in host:
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def _now_seconds(value: int | float | None) -> int:
    if value is None:
        return int(time.time())
    if isinstance(value, bool):
        raise JoinContractError("time value is invalid", code="invalid_time")
    return int(value)


def _utc_text(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class JoinKeyPair:
    """Raw Ed25519 key material encoded for local config/QR-safe transport."""

    private_key: str
    public_key: str


def generate_join_keypair() -> JoinKeyPair:
    """Generate the target node keypair; the private key never enters a grant."""
    _require_crypto()
    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    return JoinKeyPair(
        private_key=_b64(
            private.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
        ),
        public_key=_b64(
            public.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ),
    )


def load_join_private_key(encoded: str) -> Ed25519PrivateKey:
    """Load a raw Ed25519 private key kept in the user-owned local store."""
    _require_crypto()
    try:
        return Ed25519PrivateKey.from_private_bytes(
            _unb64(encoded, field="private_key", expected_length=32)
        )
    except (ValueError, TypeError) as exc:
        raise JoinContractError("private key is invalid", code="invalid_key") from exc


def public_key_for_private(private_key: Ed25519PrivateKey) -> str:
    """Return the URL-safe raw public key for a private key object."""
    _require_crypto()
    if not isinstance(private_key, Ed25519PrivateKey):
        raise JoinContractError("private key is invalid", code="invalid_key")
    return _b64(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )


def _normalize_capabilities(value: Any) -> list[str]:
    if value is None:
        value = DEFAULT_CAPABILITIES
    if not isinstance(value, (list, tuple)) or not value or len(value) > 16:
        raise JoinContractError("capabilities are invalid", code="invalid_capabilities")
    result = []
    for item in value:
        item = _safe_id(item, field="capability", pattern=re.compile(r"^[a-z][a-z0-9._:-]{0,31}$"))
        if item == "master" or item.startswith("admin"):
            raise JoinContractError("client grant requests forbidden capability", code="role_escalation")
        result.append(item)
    if len(set(result)) != len(result):
        raise JoinContractError("capabilities must be unique", code="invalid_capabilities")
    return sorted(result)


def build_join_request(
    *,
    master_endpoint: str,
    cluster_id: str,
    target_node_id: str,
    target_public_key: str,
    requested_at: int | float | None = None,
    request_ttl_seconds: int = 600,
    capabilities: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build the target-side request that an administrator approves."""
    requested = _now_seconds(requested_at)
    if isinstance(request_ttl_seconds, bool) or not 60 <= int(request_ttl_seconds) <= 3600:
        raise JoinContractError("request TTL is outside the allowed window", code="invalid_ttl")
    _unb64(target_public_key, field="target_public_key", expected_length=32)
    target_id = _safe_id(target_node_id, field="target_node_id")
    if target_id == "master":
        raise JoinContractError("client join target cannot be master", code="role_escalation")
    request = {
        "schema_version": SCHEMA_VERSION,
        "master_endpoint": _normalize_endpoint(master_endpoint),
        "cluster_id": _safe_id(cluster_id, field="cluster_id"),
        "target_node_id": target_id,
        "target_public_key": target_public_key,
        "capabilities": _normalize_capabilities(capabilities),
        "requested_at": requested,
        "request_expires_at": requested + int(request_ttl_seconds),
        "request_nonce": _b64(secrets.token_bytes(18)),
    }
    request["request_digest"] = hashlib.sha256(_canonical(request)).hexdigest()
    return request


def encode_join_request(request: Mapping[str, Any]) -> str:
    """Encode a join request for both manual entry and QR rendering."""
    if not isinstance(request, Mapping):
        raise JoinContractError("join request is invalid", code="invalid_request")
    required = {
        "schema_version", "master_endpoint", "cluster_id", "target_node_id",
        "target_public_key", "capabilities", "requested_at",
        "request_expires_at", "request_nonce", "request_digest",
    }
    if not required.issubset(request) or request.get("schema_version") != SCHEMA_VERSION:
        raise JoinContractError("join request is incomplete", code="invalid_request")
    if request.get("target_node_id") == "master":
        raise JoinContractError("client join target cannot be master", code="role_escalation")
    _normalize_endpoint(request.get("master_endpoint"))
    _safe_id(request.get("cluster_id"), field="cluster_id")
    _safe_id(request.get("target_node_id"), field="target_node_id")
    _unb64(request.get("target_public_key"), field="target_public_key", expected_length=32)
    _normalize_capabilities(request.get("capabilities"))
    for field in ("requested_at", "request_expires_at"):
        if not isinstance(request.get(field), int) or request[field] < 0:
            raise JoinContractError(f"{field} is invalid", code="invalid_request")
    _unb64(request.get("request_nonce"), field="request_nonce", expected_length=18)
    expected_digest = hashlib.sha256(
        _canonical({key: value for key, value in dict(request).items() if key != "request_digest"})
    ).hexdigest()
    if request.get("request_digest") != expected_digest:
        raise JoinContractError("join request digest mismatch", code="request_tampered")
    encoded = f"{REQUEST_PREFIX}.{_b64(_canonical(dict(request)))}"
    if len(encoded) > MAX_REQUEST_LENGTH:
        raise JoinContractError("join request is too large", code="request_too_large")
    return encoded


def decode_join_request(value: str) -> dict[str, Any]:
    """Decode and validate a manual/QR join request."""
    if not isinstance(value, str) or len(value) > MAX_REQUEST_LENGTH or not value.startswith(f"{REQUEST_PREFIX}."):
        raise JoinContractError("join request prefix is invalid", code="invalid_request")
    parts = value.split(".")
    if len(parts) != 2:
        raise JoinContractError("join request encoding is invalid", code="invalid_request")
    try:
        request = json.loads(_unb64(parts[1], field="request").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JoinContractError("join request is not valid JSON", code="invalid_request") from exc
    if not isinstance(request, dict):
        raise JoinContractError("join request payload is invalid", code="invalid_request")
    encode_join_request(request)
    return request


def issue_join_grant(
    request: Mapping[str, Any],
    *,
    issuer_key_id: str,
    issuer_private_key: Ed25519PrivateKey,
    now: int | float | None = None,
    ttl_seconds: int = 300,
    issuer_public_key: str | None = None,
) -> dict[str, Any]:
    """Issue a short-lived client-only grant for an already-authorized request."""
    if not _SAFE_KEY_ID.fullmatch(str(issuer_key_id)):
        raise JoinContractError("issuer_key_id is invalid", code="invalid_field")
    _require_crypto()
    if not isinstance(issuer_private_key, Ed25519PrivateKey):
        raise JoinContractError("issuer private key is invalid", code="invalid_key")
    resolved_issuer_public_key = issuer_public_key or public_key_for_private(issuer_private_key)
    _unb64(resolved_issuer_public_key, field="issuer_public_key", expected_length=32)
    required = {
        "schema_version", "master_endpoint", "cluster_id", "target_node_id",
        "target_public_key", "capabilities", "request_expires_at", "request_digest",
    }
    if not required.issubset(request):
        raise JoinContractError("join request is incomplete", code="invalid_request")
    issued_at = _now_seconds(now)
    if issued_at >= int(request["request_expires_at"]):
        raise JoinContractError("join request has expired", code="request_expired")
    if isinstance(ttl_seconds, bool) or not 60 <= int(ttl_seconds) <= 900:
        raise JoinContractError("grant TTL is outside the allowed window", code="invalid_ttl")
    request_copy = dict(request)
    digest = hashlib.sha256(
        _canonical({key: value for key, value in request_copy.items() if key != "request_digest"})
    ).hexdigest()
    if digest != request_copy.get("request_digest"):
        raise JoinContractError("join request digest mismatch", code="request_tampered")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "grant_type": "client_only",
        "master_endpoint": _normalize_endpoint(request["master_endpoint"]),
        "cluster_id": _safe_id(request["cluster_id"], field="cluster_id"),
        "target_node_id": _safe_id(request["target_node_id"], field="target_node_id"),
        "target_public_key": request["target_public_key"],
        "role": "client",
        "capabilities": _normalize_capabilities(request["capabilities"]),
        "issued_at": issued_at,
        "expires_at": issued_at + int(ttl_seconds),
        "nonce": _b64(secrets.token_bytes(18)),
        "request_digest": request_copy["request_digest"],
        "issuer_key_id": str(issuer_key_id),
        "auth_method": "totp",
        "issuer_public_key": resolved_issuer_public_key,
    }
    signature = _b64(issuer_private_key.sign(_canonical(payload)))
    return {
        "schema_version": SCHEMA_VERSION,
        "payload": payload,
        "signature": signature,
    }


def encode_join_grant(grant: Mapping[str, Any]) -> str:
    """Return the same compact string for manual entry and QR encoding."""
    if not isinstance(grant, Mapping) or set(grant) != {"schema_version", "payload", "signature"}:
        raise JoinContractError("grant envelope is invalid", code="invalid_grant")
    if grant.get("schema_version") != SCHEMA_VERSION:
        raise JoinContractError("grant schema is unsupported", code="unsupported_schema")
    payload = grant.get("payload")
    signature = grant.get("signature")
    if not isinstance(payload, Mapping) or not isinstance(signature, str):
        raise JoinContractError("grant envelope is invalid", code="invalid_grant")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("grant_type") != "client_only":
        raise JoinContractError("grant type is unsupported", code="invalid_grant")
    if payload.get("role") != "client":
        raise JoinContractError("grant is not client-only", code="role_escalation")
    _normalize_endpoint(payload.get("master_endpoint"))
    _safe_id(payload.get("cluster_id"), field="cluster_id")
    _safe_id(payload.get("target_node_id"), field="target_node_id")
    _unb64(payload.get("target_public_key"), field="target_public_key", expected_length=32)
    _normalize_capabilities(payload.get("capabilities"))
    for field in ("issued_at", "expires_at"):
        if not isinstance(payload.get(field), int) or payload[field] < 0:
            raise JoinContractError(f"{field} is invalid", code="invalid_grant")
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("request_digest", ""))):
        raise JoinContractError("request_digest is invalid", code="invalid_grant")
    _safe_id(payload.get("issuer_key_id"), field="issuer_key_id", pattern=_SAFE_KEY_ID)
    _unb64(payload.get("issuer_public_key"), field="issuer_public_key", expected_length=32)
    _unb64(payload.get("nonce"), field="nonce", expected_length=18)
    _unb64(signature, field="signature", expected_length=64)
    encoded = f"{GRANT_PREFIX}.{_b64(_canonical(dict(payload)))}.{signature}"
    if len(encoded) > MAX_GRANT_LENGTH:
        raise JoinContractError("grant is too large for manual/QR transport", code="grant_too_large")
    return encoded


def decode_join_grant(value: str) -> dict[str, Any]:
    if not isinstance(value, str) or len(value) > MAX_GRANT_LENGTH or not value.startswith(f"{GRANT_PREFIX}."):
        raise JoinContractError("grant prefix is invalid", code="invalid_grant")
    parts = value.split(".")
    if len(parts) != 3:
        raise JoinContractError("grant encoding is invalid", code="invalid_grant")
    try:
        payload = json.loads(_unb64(parts[1], field="payload").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JoinContractError("grant payload is not valid JSON", code="invalid_grant") from exc
    if not isinstance(payload, dict):
        raise JoinContractError("grant payload is invalid", code="invalid_grant")
    _unb64(parts[2], field="signature", expected_length=64)
    return {"schema_version": SCHEMA_VERSION, "payload": payload, "signature": parts[2]}


class JoinGrantLedger:
    """Durable nonce ledger owned by the user's main node."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        database_path = Path(self.path)
        self._private_store = LocalSecretStore(
            database_path.with_name("cluster_join_secrets.json")
        )
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cluster_join_grant_nonces (
                  nonce TEXT PRIMARY KEY,
                  grant_digest TEXT NOT NULL,
                  consumed_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cluster_bootstrap_request_nonces (
                  nonce TEXT PRIMARY KEY,
                  request_digest TEXT NOT NULL,
                  consumed_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cluster_join_keys (
                  key_scope TEXT PRIMARY KEY,
                  key_id TEXT NOT NULL,
                  private_key TEXT NOT NULL,
                  public_key TEXT NOT NULL,
                  updated_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cluster_join_pending_requests (
                  request_digest TEXT PRIMARY KEY,
                  request_json TEXT NOT NULL,
                  private_key TEXT NOT NULL,
                  public_key TEXT NOT NULL,
                  created_at INTEGER NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @staticmethod
    def _secret_ref(name: str) -> str:
        return f"local-secret-store:{name}"

    def _store_private_key(self, name: str, private_key: str) -> str:
        _unb64(private_key, field="private_key", expected_length=32)
        self._private_store.set(name, private_key)
        return self._secret_ref(name)

    def _resolve_private_key(self, stored: str, *, name: str) -> tuple[str, bool]:
        prefix = "local-secret-store:"
        if stored.startswith(prefix):
            reference = stored[len(prefix):]
            if reference != name:
                raise JoinContractError("private key reference mismatch", code="invalid_key")
            try:
                value = self._private_store.get(reference)
            except LocalSecretStoreError as exc:
                raise JoinContractError("private key store is unavailable", code="invalid_key") from exc
            _unb64(value, field="private_key", expected_length=32)
            return value, False
        # One-time migration for pre-SEC-BOUNDARY ledgers.
        return stored, True

    def consume(self, *, nonce: str, grant_digest: str, consumed_at: int | float | None = None) -> None:
        _unb64(nonce, field="nonce", expected_length=18)
        if not re.fullmatch(r"[0-9a-f]{64}", grant_digest):
            raise JoinContractError("grant digest is invalid", code="invalid_grant")
        timestamp = _now_seconds(consumed_at)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT grant_digest FROM cluster_join_grant_nonces WHERE nonce = ?",
                (nonce,),
            ).fetchone()
            if existing is not None:
                raise JoinContractError("grant nonce was already consumed", code="nonce_replayed")
            connection.execute(
                "INSERT INTO cluster_join_grant_nonces(nonce, grant_digest, consumed_at) VALUES (?, ?, ?)",
                (nonce, grant_digest, timestamp),
            )

    def consume_bootstrap_request(
        self,
        *,
        nonce: str,
        request_digest: str,
        consumed_at: int | float | None = None,
    ) -> None:
        """Atomically fence one encrypted bootstrap credential request."""
        _unb64(nonce, field="credential_request_nonce", expected_length=18)
        if not re.fullmatch(r"[0-9a-f]{64}", str(request_digest)):
            raise JoinContractError("bootstrap request digest is invalid", code="invalid_request")
        timestamp = _now_seconds(consumed_at)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT request_digest FROM cluster_bootstrap_request_nonces WHERE nonce = ?",
                (nonce,),
            ).fetchone()
            if existing is not None:
                raise JoinContractError(
                    "bootstrap request nonce was already consumed", code="nonce_replayed"
                )
            connection.execute(
                "INSERT INTO cluster_bootstrap_request_nonces(nonce, request_digest, consumed_at) VALUES (?, ?, ?)",
                (nonce, request_digest, timestamp),
            )

    def get_or_create_issuer_keypair(self, key_id: str = "main") -> tuple[str, JoinKeyPair]:
        """Return the durable main-node signing key owned by this SQLite store."""
        _safe_id(key_id, field="key_id", pattern=_SAFE_KEY_ID)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT key_id, private_key, public_key FROM cluster_join_keys WHERE key_scope = 'issuer'"
            ).fetchone()
            if row is not None:
                key_id_value = str(row["key_id"])
                secret_name = f"join_issuer:{key_id_value}"
                private_key, migrate = self._resolve_private_key(
                    str(row["private_key"]), name=secret_name,
                )
                if migrate:
                    reference = self._store_private_key(secret_name, private_key)
                    connection.execute(
                        "UPDATE cluster_join_keys SET private_key = ?, updated_at = ? WHERE key_scope = 'issuer'",
                        (reference, _now_seconds(None)),
                    )
                return str(row["key_id"]), JoinKeyPair(
                    private_key=private_key, public_key=str(row["public_key"])
                )
            pair = generate_join_keypair()
            private_ref = self._store_private_key(
                f"join_issuer:{key_id}", pair.private_key,
            )
            connection.execute(
                "INSERT INTO cluster_join_keys(key_scope, key_id, private_key, public_key, updated_at) VALUES ('issuer', ?, ?, ?, ?)",
                (key_id, private_ref, pair.public_key, _now_seconds(None)),
            )
            return key_id, pair

    def save_pending_request(self, request: Mapping[str, Any], keypair: JoinKeyPair) -> None:
        encode_join_request(request)
        if not isinstance(keypair, JoinKeyPair) or keypair.public_key != request.get("target_public_key"):
            raise JoinContractError("pending key does not match request target", code="request_mismatch")
        _unb64(keypair.private_key, field="private_key", expected_length=32)
        _unb64(keypair.public_key, field="public_key", expected_length=32)
        secret_name = f"join_pending:{request['request_digest']}"
        private_ref = self._store_private_key(secret_name, keypair.private_key)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cluster_join_pending_requests(request_digest, request_json, private_key, public_key, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(request_digest) DO UPDATE SET request_json=excluded.request_json,
                  private_key=excluded.private_key, public_key=excluded.public_key,
                  created_at=excluded.created_at
                """,
                (
                    str(request["request_digest"]),
                    json.dumps(dict(request), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    private_ref,
                    keypair.public_key,
                    _now_seconds(None),
                ),
            )

    def load_pending_request(self, request_digest: str) -> tuple[dict[str, Any], JoinKeyPair] | None:
        if not re.fullmatch(r"[0-9a-f]{64}", str(request_digest)):
            return None
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT request_json, private_key, public_key FROM cluster_join_pending_requests WHERE request_digest = ?",
                (request_digest,),
            ).fetchone()
        if row is None:
            return None
        try:
            request = json.loads(str(row["request_json"]))
            encode_join_request(request)
            private_key, migrate = self._resolve_private_key(
                str(row["private_key"]), name=f"join_pending:{request_digest}",
            )
            if migrate:
                reference = self._store_private_key(
                    f"join_pending:{request_digest}", private_key,
                )
                with self._lock, self._connect() as connection:
                    connection.execute(
                        "UPDATE cluster_join_pending_requests SET private_key = ? WHERE request_digest = ?",
                        (reference, request_digest),
                    )
        except (JoinContractError, LocalSecretStoreError, json.JSONDecodeError, TypeError):
            return None
        return request, JoinKeyPair(private_key=private_key, public_key=str(row["public_key"]))

    def delete_pending_request(self, request_digest: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM cluster_join_pending_requests WHERE request_digest = ?",
                (request_digest,),
            )
        self._private_store.delete(f"join_pending:{request_digest}")


def verify_and_consume_join_grant(
    encoded_grant: str,
    *,
    issuer_public_key: str,
    expected_request: Mapping[str, Any],
    ledger: JoinGrantLedger,
    now: int | float | None = None,
) -> dict[str, Any]:
    """Verify signature, binding, TTL and role, then atomically consume nonce."""
    payload = verify_join_grant(
        encoded_grant,
        issuer_public_key=issuer_public_key,
        expected_request=expected_request,
        now=now,
    )
    nonce = payload.get("nonce")
    digest = hashlib.sha256(encoded_grant.encode("ascii")).hexdigest()
    ledger.consume(nonce=nonce, grant_digest=digest, consumed_at=_now_seconds(now))
    return payload


def verify_join_grant(
    encoded_grant: str,
    *,
    issuer_public_key: str,
    expected_request: Mapping[str, Any],
    now: int | float | None = None,
) -> dict[str, Any]:
    """Verify a grant without consuming its nonce (for pre-switch checks)."""
    _require_crypto()
    grant = decode_join_grant(encoded_grant)
    # Validate the full envelope before any nonce state is consumed.
    encode_join_grant(grant)
    payload = grant["payload"]
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("grant_type") != "client_only":
        raise JoinContractError("grant type is unsupported", code="invalid_grant")
    try:
        public = Ed25519PublicKey.from_public_bytes(
            _unb64(issuer_public_key, field="issuer_public_key", expected_length=32)
        )
        public.verify(
            _unb64(grant["signature"], field="signature", expected_length=64),
            _canonical(payload),
        )
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise JoinContractError("grant signature is invalid", code="signature_invalid") from exc
    if payload.get("issuer_public_key") != issuer_public_key:
        raise JoinContractError("issuer public key mismatch", code="issuer_mismatch")
    if payload.get("role") != "client":
        raise JoinContractError("grant is not client-only", code="role_escalation")
    current = _now_seconds(now)
    if current < int(payload.get("issued_at", 0)) or current >= int(payload.get("expires_at", 0)):
        raise JoinContractError("grant is expired or not yet valid", code="grant_expired")
    expected_digest = hashlib.sha256(
        _canonical({key: value for key, value in dict(expected_request).items() if key != "request_digest"})
    ).hexdigest()
    if payload.get("request_digest") != expected_digest:
        raise JoinContractError("grant does not match this join request", code="request_mismatch")
    for field in ("master_endpoint", "cluster_id", "target_node_id", "target_public_key"):
        expected = expected_request.get(field)
        if field == "master_endpoint":
            expected = _normalize_endpoint(expected)
        if payload.get(field) != expected:
            raise JoinContractError(f"grant {field} mismatch", code="request_mismatch")
    _unb64(payload.get("target_public_key"), field="target_public_key", expected_length=32)
    return dict(payload)
