"""Short-lived encrypted delivery for the cluster bootstrap credential.

The bootstrap HTTP route is intentionally usable over the Tailnet overlay, but
it must never place the long-lived cluster secret in an HTTP response. A client
therefore supplies an ephemeral RSA public key and a one-time request nonce.
The master returns an AES-GCM credential envelope whose content key is wrapped
with RSA-OAEP-SHA256. The RSA private key exists only for that request.
"""

from __future__ import annotations

import base64
import json
import math
import secrets
import time
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception as exc:  # pragma: no cover - packaging/runtime dependent
    hashes = serialization = padding = rsa = AESGCM = None  # type: ignore[assignment]
    _CRYPTO_IMPORT_ERROR = exc
else:
    _CRYPTO_IMPORT_ERROR = None


SCHEMA_VERSION = "qlh.cluster.bootstrap-credential.v1"
KEY_ALGORITHM = "RSA-OAEP-SHA256"
CONTENT_ALGORITHM = "AES-256-GCM"
REQUEST_NONCE_BYTES = 18
CONTENT_NONCE_BYTES = 12
DEFAULT_CREDENTIAL_TTL_SECONDS = 120
MAX_CLOCK_SKEW_SECONDS = 300


class BootstrapCredentialError(ValueError):
    """Fail-closed bootstrap envelope error with a stable reason code."""

    def __init__(self, message: str, *, code: str = "credential_invalid") -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class BootstrapCredentialRequest:
    """Ephemeral client material; ``private_key`` must never be serialized."""

    private_key: Any
    public_key: str
    request_nonce: str
    requested_at: int

    def fields(self) -> dict[str, Any]:
        return {
            "credential_public_key": self.public_key,
            "credential_request_nonce": self.request_nonce,
            "credential_requested_at": self.requested_at,
        }


def _require_crypto() -> None:
    if rsa is None or AESGCM is None:
        raise BootstrapCredentialError(
            f"cryptography is unavailable: {_CRYPTO_IMPORT_ERROR}",
            code="crypto_unavailable",
        )


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: Any, *, field: str, expected_length: int | None = None) -> bytes:
    if not isinstance(value, str) or not value:
        raise BootstrapCredentialError(f"{field} must be base64url text")
    try:
        raw = base64.urlsafe_b64decode(value + ("=" * (-len(value) % 4)))
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BootstrapCredentialError(f"{field} is not valid base64url") from exc
    if expected_length is not None and len(raw) != expected_length:
        raise BootstrapCredentialError(f"{field} has invalid length")
    return raw


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _timestamp(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BootstrapCredentialError(
            f"{field} must be a finite timestamp", code="invalid_time"
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise BootstrapCredentialError(
            f"{field} must be a finite timestamp", code="invalid_time"
        )
    return int(numeric)


def create_bootstrap_credential_request(
    *, now: int | float | None = None,
) -> BootstrapCredentialRequest:
    """Create an in-memory RSA keypair and one-time request nonce."""
    _require_crypto()
    requested_at = int(time.time() if now is None else now)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return BootstrapCredentialRequest(
        private_key=private_key,
        public_key=_b64(public_der),
        request_nonce=_b64(secrets.token_bytes(REQUEST_NONCE_BYTES)),
        requested_at=requested_at,
    )


def validate_bootstrap_credential_request(
    *,
    public_key: str,
    request_nonce: str,
    requested_at: int | float,
    now: int | float | None = None,
) -> tuple[Any, int]:
    """Validate request freshness and return the parsed RSA public key."""
    _require_crypto()
    current = int(time.time() if now is None else now)
    requested = _timestamp(requested_at, field="credential_requested_at")
    if abs(current - requested) > MAX_CLOCK_SKEW_SECONDS:
        raise BootstrapCredentialError(
            "bootstrap credential request is expired or not yet valid",
            code="request_expired",
        )
    _unb64(
        request_nonce,
        field="credential_request_nonce",
        expected_length=REQUEST_NONCE_BYTES,
    )
    try:
        parsed = serialization.load_der_public_key(
            _unb64(public_key, field="credential_public_key")
        )
    except (TypeError, ValueError) as exc:
        raise BootstrapCredentialError(
            "credential public key is invalid", code="invalid_public_key"
        ) from exc
    if not isinstance(parsed, rsa.RSAPublicKey) or parsed.key_size < 2048:
        raise BootstrapCredentialError(
            "credential public key must be RSA-2048 or stronger",
            code="invalid_public_key",
        )
    return parsed, requested


def seal_bootstrap_credential(
    *,
    public_key: str,
    request_nonce: str,
    requested_at: int | float,
    cluster_id: str,
    node_id: str,
    cluster_secret: str,
    secret_epoch: int,
    now: int | float | None = None,
    ttl_seconds: int = DEFAULT_CREDENTIAL_TTL_SECONDS,
) -> dict[str, Any]:
    """Encrypt a long-lived secret into a short-lived target-bound envelope."""
    current = int(time.time() if now is None else now)
    parsed_key, _ = validate_bootstrap_credential_request(
        public_key=public_key,
        request_nonce=request_nonce,
        requested_at=requested_at,
        now=current,
    )
    if not isinstance(cluster_secret, str) or len(cluster_secret) < 16:
        raise BootstrapCredentialError(
            "cluster secret is unavailable", code="secret_unavailable"
        )
    if (
        isinstance(secret_epoch, bool)
        or not isinstance(secret_epoch, int)
        or secret_epoch < 1
    ):
        raise BootstrapCredentialError("secret epoch is invalid", code="invalid_epoch")
    if isinstance(ttl_seconds, bool) or not 30 <= int(ttl_seconds) <= 300:
        raise BootstrapCredentialError("credential TTL is invalid", code="invalid_ttl")
    if not cluster_id or not node_id:
        raise BootstrapCredentialError(
            "credential binding is incomplete", code="binding_invalid"
        )

    aad = {
        "cluster_id": str(cluster_id),
        "expires_at": current + int(ttl_seconds),
        "issued_at": current,
        "node_id": str(node_id),
        "request_nonce": request_nonce,
        "secret_epoch": secret_epoch,
    }
    content_key = AESGCM.generate_key(bit_length=256)
    content_nonce = secrets.token_bytes(CONTENT_NONCE_BYTES)
    plaintext = _canonical({"cluster_secret": cluster_secret})
    ciphertext = AESGCM(content_key).encrypt(
        content_nonce, plaintext, _canonical(aad)
    )
    wrapped_key = parsed_key.encrypt(
        content_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return {
        "schema": SCHEMA_VERSION,
        "key_algorithm": KEY_ALGORITHM,
        "content_algorithm": CONTENT_ALGORITHM,
        "aad": aad,
        "wrapped_key": _b64(wrapped_key),
        "nonce": _b64(content_nonce),
        "ciphertext": _b64(ciphertext),
    }


def open_bootstrap_credential(
    envelope: Mapping[str, Any],
    *,
    private_key: Any,
    expected_request_nonce: str,
    expected_cluster_id: str | None = None,
    expected_node_id: str | None = None,
    now: int | float | None = None,
) -> dict[str, Any]:
    """Decrypt and verify an envelope before any value reaches local config."""
    _require_crypto()
    if not isinstance(envelope, Mapping):
        raise BootstrapCredentialError("credential envelope is missing")
    if envelope.get("schema") != SCHEMA_VERSION:
        raise BootstrapCredentialError(
            "credential schema is unsupported", code="unsupported_schema"
        )
    if (
        envelope.get("key_algorithm") != KEY_ALGORITHM
        or envelope.get("content_algorithm") != CONTENT_ALGORITHM
    ):
        raise BootstrapCredentialError(
            "credential algorithms are unsupported", code="unsupported_algorithm"
        )
    aad = envelope.get("aad")
    if not isinstance(aad, Mapping) or set(aad) != {
        "cluster_id",
        "expires_at",
        "issued_at",
        "node_id",
        "request_nonce",
        "secret_epoch",
    }:
        raise BootstrapCredentialError(
            "credential binding is invalid", code="binding_invalid"
        )
    issued_at = _timestamp(aad.get("issued_at"), field="issued_at")
    expires_at = _timestamp(aad.get("expires_at"), field="expires_at")
    current = int(time.time() if now is None else now)
    if current < issued_at - MAX_CLOCK_SKEW_SECONDS or current >= expires_at:
        raise BootstrapCredentialError(
            "credential is expired or not yet valid", code="credential_expired"
        )
    if expires_at <= issued_at or expires_at - issued_at > 300:
        raise BootstrapCredentialError(
            "credential validity is invalid", code="credential_expired"
        )
    if aad.get("request_nonce") != expected_request_nonce:
        raise BootstrapCredentialError(
            "credential request nonce mismatch", code="binding_mismatch"
        )
    if expected_cluster_id is not None and aad.get("cluster_id") != expected_cluster_id:
        raise BootstrapCredentialError(
            "credential cluster mismatch", code="binding_mismatch"
        )
    if expected_node_id is not None and aad.get("node_id") != expected_node_id:
        raise BootstrapCredentialError(
            "credential node mismatch", code="binding_mismatch"
        )
    epoch = aad.get("secret_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise BootstrapCredentialError(
            "credential epoch is invalid", code="invalid_epoch"
        )
    if not hasattr(private_key, "decrypt"):
        raise BootstrapCredentialError(
            "credential private key is invalid", code="invalid_private_key"
        )
    try:
        content_key = private_key.decrypt(
            _unb64(envelope.get("wrapped_key"), field="wrapped_key"),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        plaintext = AESGCM(content_key).decrypt(
            _unb64(
                envelope.get("nonce"),
                field="nonce",
                expected_length=CONTENT_NONCE_BYTES,
            ),
            _unb64(envelope.get("ciphertext"), field="ciphertext"),
            _canonical(aad),
        )
        content = json.loads(plaintext.decode("ascii"))
    except Exception as exc:
        raise BootstrapCredentialError(
            "credential decryption or authentication failed",
            code="credential_auth_failed",
        ) from exc
    if not isinstance(content, dict) or set(content) != {"cluster_secret"}:
        raise BootstrapCredentialError("credential content is invalid")
    secret = content.get("cluster_secret")
    if not isinstance(secret, str) or len(secret) < 16:
        raise BootstrapCredentialError("credential secret is invalid")
    return {
        "cluster_secret": secret,
        "cluster_secret_epoch": epoch,
        "issued_at": issued_at,
        "expires_at": expires_at,
    }


__all__ = [
    "BootstrapCredentialError",
    "BootstrapCredentialRequest",
    "CONTENT_ALGORITHM",
    "KEY_ALGORITHM",
    "SCHEMA_VERSION",
    "create_bootstrap_credential_request",
    "open_bootstrap_credential",
    "seal_bootstrap_credential",
    "validate_bootstrap_credential_request",
]
