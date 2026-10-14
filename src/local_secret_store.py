"""Small encrypted-at-rest store for user-owned node credentials.

The store is deliberately separate from ``node_config.json`` so configuration
backups, diagnostics and support bundles cannot disclose the cluster secret in
plain text.  A random local wrapping key protects AES-GCM records; both files
are written atomically and restricted to the current user where the platform
supports POSIX modes.  This boundary protects against accidental disclosure,
not against an administrator or a compromised local user account.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import threading
from pathlib import Path
from typing import Any

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception as exc:  # pragma: no cover - packaging/runtime dependent
    AESGCM = None  # type: ignore[assignment]
    _CRYPTO_IMPORT_ERROR = exc
else:
    _CRYPTO_IMPORT_ERROR = None


_SCHEMA = "qlh.local-secret-store.v1"
_KEY_BYTES = 32
_NONCE_BYTES = 12
_STORE_LOCKS_GUARD = threading.Lock()
_STORE_LOCKS: dict[str, threading.RLock] = {}


class LocalSecretStoreError(RuntimeError):
    pass


def _shared_store_lock(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path.expanduser().resolve()))
    with _STORE_LOCKS_GUARD:
        lock = _STORE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _STORE_LOCKS[key] = lock
        return lock


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: Any, *, field: str, length: int | None = None) -> bytes:
    if not isinstance(value, str) or not value:
        raise LocalSecretStoreError(f"{field} is invalid")
    try:
        raw = base64.urlsafe_b64decode(value + ("=" * (-len(value) % 4)))
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise LocalSecretStoreError(f"{field} is invalid") from exc
    if length is not None and len(raw) != length:
        raise LocalSecretStoreError(f"{field} has invalid length")
    return raw


def _restrict_file(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        # Windows ACLs are inherited from the per-user LocalAppData directory.
        # chmod is still attempted because Python maps it to the read-only bit.
        pass


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    _restrict_file(tmp)
    os.replace(tmp, path)
    _restrict_file(path)


class LocalSecretStore:
    """Thread-safe AES-GCM record store backed by two user-owned files."""

    def __init__(self, path: str | Path, *, key_path: str | Path | None = None) -> None:
        if AESGCM is None:
            raise LocalSecretStoreError(
                f"cryptography is unavailable: {_CRYPTO_IMPORT_ERROR}"
            )
        self.path = Path(path)
        self.key_path = Path(key_path) if key_path is not None else self.path.with_suffix(
            self.path.suffix + ".key"
        )
        self._lock = _shared_store_lock(self.path)

    def _load_or_create_key(self) -> bytes:
        if self.key_path.is_file():
            raw = self.key_path.read_bytes()
            _restrict_file(self.key_path)
            return _unb64(
                raw.decode("ascii").strip(),
                field="wrapping_key",
                length=_KEY_BYTES,
            )
        if self.path.is_file():
            raise LocalSecretStoreError(
                "local secret wrapping key is missing for an existing store"
            )
        key = secrets.token_bytes(_KEY_BYTES)
        _atomic_write(self.key_path, (_b64(key) + "\n").encode("ascii"))
        return key

    def _load_payload(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema": _SCHEMA, "records": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LocalSecretStoreError("local secret store is unreadable") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != _SCHEMA
            or not isinstance(payload.get("records"), dict)
        ):
            raise LocalSecretStoreError("local secret store schema is invalid")
        _restrict_file(self.path)
        return payload

    def _write_payload(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("ascii") + b"\n"
        _atomic_write(self.path, encoded)

    def set(self, name: str, value: str) -> None:
        if not isinstance(name, str) or not name or len(name) > 128:
            raise LocalSecretStoreError("secret name is invalid")
        if not isinstance(value, str) or not value:
            raise LocalSecretStoreError("secret value is invalid")
        with self._lock:
            key = self._load_or_create_key()
            payload = self._load_payload()
            nonce = secrets.token_bytes(_NONCE_BYTES)
            ciphertext = AESGCM(key).encrypt(nonce, value.encode("utf-8"), name.encode("utf-8"))
            payload["records"][name] = {
                "nonce": _b64(nonce),
                "ciphertext": _b64(ciphertext),
            }
            self._write_payload(payload)

    def get(self, name: str) -> str:
        if not isinstance(name, str) or not name:
            return ""
        with self._lock:
            if not self.path.is_file():
                return ""
            if not self.key_path.is_file():
                raise LocalSecretStoreError(
                    "local secret wrapping key is missing for an existing store"
                )
            payload = self._load_payload()
            record = payload["records"].get(name)
            if not isinstance(record, dict):
                return ""
            try:
                plaintext = AESGCM(self._load_or_create_key()).decrypt(
                    _unb64(record.get("nonce"), field="nonce", length=_NONCE_BYTES),
                    _unb64(record.get("ciphertext"), field="ciphertext"),
                    name.encode("utf-8"),
                )
                value = plaintext.decode("utf-8")
            except Exception as exc:
                raise LocalSecretStoreError("local secret authentication failed") from exc
            return value

    def delete(self, name: str) -> None:
        with self._lock:
            if not self.path.is_file():
                return
            payload = self._load_payload()
            if payload["records"].pop(name, None) is not None:
                self._write_payload(payload)


__all__ = ["LocalSecretStore", "LocalSecretStoreError"]
