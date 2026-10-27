"""Durable replay protection for encrypted bootstrap credential requests."""

from __future__ import annotations

import base64
import math
import re
import sqlite3
import threading
import time
from pathlib import Path


class BootstrapNonceError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_request") -> None:
        self.code = code
        super().__init__(message)


class BootstrapRequestNonceLedger:
    """Consume one-time bootstrap nonces in the node's local SQLite store."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cluster_bootstrap_request_nonces (
                  nonce TEXT PRIMARY KEY,
                  request_digest TEXT NOT NULL,
                  consumed_at INTEGER NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @staticmethod
    def _validate_nonce(nonce: str) -> None:
        if not isinstance(nonce, str) or not nonce:
            raise BootstrapNonceError("credential request nonce is invalid")
        try:
            raw = base64.b64decode(
                nonce + ("=" * (-len(nonce) % 4)), altchars=b"-_", validate=True,
            )
        except (ValueError, TypeError, UnicodeEncodeError) as exc:
            raise BootstrapNonceError("credential request nonce is invalid") from exc
        if len(raw) != 18:
            raise BootstrapNonceError("credential request nonce has invalid length")

    def consume(
        self,
        *,
        nonce: str,
        request_digest: str,
        consumed_at: int | float | None = None,
    ) -> None:
        self._validate_nonce(nonce)
        if not isinstance(request_digest, str) or re.fullmatch(r"[0-9a-f]{64}", request_digest) is None:
            raise BootstrapNonceError("bootstrap request digest is invalid")
        if consumed_at is None:
            timestamp = int(time.time())
        elif isinstance(consumed_at, bool) or not isinstance(consumed_at, (int, float)):
            raise BootstrapNonceError("bootstrap request timestamp is invalid")
        elif not math.isfinite(float(consumed_at)) or consumed_at < 0:
            raise BootstrapNonceError("bootstrap request timestamp is invalid")
        else:
            timestamp = int(consumed_at)

        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT 1 FROM cluster_bootstrap_request_nonces WHERE nonce = ?",
                    (nonce,),
                ).fetchone()
                if existing is not None:
                    raise BootstrapNonceError(
                        "bootstrap request nonce was already consumed",
                        code="nonce_replayed",
                    )
                connection.execute(
                    "INSERT INTO cluster_bootstrap_request_nonces"
                    "(nonce, request_digest, consumed_at) VALUES (?, ?, ?)",
                    (nonce, request_digest, timestamp),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()


__all__ = ["BootstrapNonceError", "BootstrapRequestNonceLedger"]
