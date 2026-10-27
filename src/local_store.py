"""Main-node local asset storage backed by SQLite.

The Python runtime shares the control-svc sessions, session_messages, and
cluster_settings tables.  This module never changes PRAGMA user_version;
schema ownership remains with control-svc.  Legacy chat_history JSON files are
imported once and retained as user-owned recovery material.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_init_lock = threading.Lock()
_initialized_paths: set[str] = set()
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_LEGACY_MIGRATION_KEY = "legacy_chat_history_json_v1"
_MASTER_IDENTITY_KEY = "master_identity_v1"
_runtime_path = ContextVar("qlh_local_store_path", default=None)


@contextmanager
def bind_local_store(path):
    """Bind legacy operations to one runtime without changing process settings."""
    token = _runtime_path.set(os.path.abspath(os.fspath(path)))
    try:
        yield
    finally:
        _runtime_path.reset(token)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_store_dir() -> str:
    try:
        from config import STATE_DIR

        return os.path.abspath(STATE_DIR)
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "state",
        )


def _store_dir() -> str:
    return _get_store_dir()


def _legacy_store_dir() -> str:
    try:
        from config import LOG_DIR

        return os.path.join(os.path.dirname(LOG_DIR), "chat_history")
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs",
            "chat_history",
        )


def _sqlite_path() -> str:
    bound = _runtime_path.get()
    if bound is not None:
        return bound
    override = os.environ.get("QLH_SQLITE_PATH", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    state_path = os.path.join(_store_dir(), "qlh-control.sqlite3")
    legacy_path = os.path.abspath("qlh-control.sqlite3")
    if os.path.isfile(legacy_path) and not os.path.exists(state_path):
        return legacy_path
    return state_path


def _connect_raw(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=5.0, isolation_level=None)
    _restrict_sqlite_files(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    # Control-plane writes are user-owned durable state.  Every connection
    # explicitly requests FULL synchronization rather than relying on the
    # SQLite build default for WAL mode.
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _restrict_sqlite_files(path: str) -> None:
    for candidate in (path, path + "-wal", path + "-shm"):
        if not os.path.exists(candidate):
            continue
        try:
            os.chmod(candidate, 0o600)
        except OSError:
            pass


def _read_legacy_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("旧 JSON 无法导入，保留原文件: %s: %s", path, exc)
        return default


def _valid_session_id(value: object) -> str | None:
    session_id = str(value or "").strip()
    if not _SAFE_SESSION_ID.fullmatch(session_id):
        return None
    return session_id


def _valid_operation_id(value: object) -> str | None:
    operation_id = str(value or "").strip()
    if not operation_id:
        return None
    if not _SAFE_OPERATION_ID.fullmatch(operation_id):
        raise ValueError("operation_id is invalid")
    return operation_id


def _nonnegative_int(value: object, default: int = 0) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return default


def _import_legacy_json(connection: sqlite3.Connection) -> None:
    marker = connection.execute(
        "SELECT value FROM python_local_store_meta WHERE key = ?",
        (_LEGACY_MIGRATION_KEY,),
    ).fetchone()
    if marker:
        return

    legacy_dir = _legacy_store_dir()
    sessions_path = os.path.join(legacy_dir, "_sessions.json")
    sessions = _read_legacy_json(sessions_path, [])
    imported_sessions: set[str] = set()
    if isinstance(sessions, list):
        for item in sessions:
            if not isinstance(item, dict):
                continue
            session_id = _valid_session_id(item.get("id"))
            if not session_id:
                continue
            created_at = str(item.get("created_at") or _now())
            updated_at = str(item.get("updated_at") or created_at)
            connection.execute(
                """
                INSERT OR IGNORE INTO sessions
                  (session_id, title, created_at, updated_at, message_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    str(item.get("title") or "新对话")[:256],
                    created_at,
                    updated_at,
                    _nonnegative_int(item.get("message_count")),
                ),
            )
            imported_sessions.add(session_id)

    if os.path.isdir(legacy_dir):
        for name in os.listdir(legacy_dir):
            if name == "_sessions.json" or not name.endswith(".json"):
                continue
            session_id = _valid_session_id(name[:-5])
            if not session_id:
                continue
            messages = _read_legacy_json(os.path.join(legacy_dir, name), [])
            if not isinstance(messages, list):
                continue
            now = _now()
            connection.execute(
                """
                INSERT OR IGNORE INTO sessions
                  (session_id, title, created_at, updated_at, message_count)
                VALUES (?, '新对话', ?, ?, 0)
                """,
                (session_id, now, now),
            )
            imported_sessions.add(session_id)
            existing = connection.execute(
                "SELECT COUNT(*) AS count FROM session_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing and int(existing["count"]) > 0:
                continue
            imported_count = 0
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "").strip()
                content = message.get("content")
                if not role or not isinstance(content, str):
                    continue
                metrics = message.get("metrics")
                connection.execute(
                    """
                    INSERT INTO session_messages
                      (session_id, role, content, created_at, metrics)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        role,
                        content,
                        str(message.get("created_at") or now),
                        json.dumps(metrics, ensure_ascii=False) if metrics is not None else None,
                    ),
                )
                imported_count += 1
            if imported_count:
                connection.execute(
                    """
                    UPDATE sessions SET
                      message_count = MAX(message_count, ?), updated_at = ?
                    WHERE session_id = ?
                    """,
                    (imported_count, now, session_id),
                )

    connection.execute(
        """
        INSERT INTO python_local_store_meta(key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        (_LEGACY_MIGRATION_KEY, str(len(imported_sessions)), _now()),
    )


def initialize_local_store(path: str | None = None, *, migrate_legacy: bool = True) -> str:
    path = _sqlite_path() if path is None else os.path.abspath(os.path.expanduser(path))
    if path in _initialized_paths and os.path.isfile(path):
        return path
    with _init_lock:
        if path in _initialized_paths and os.path.isfile(path):
            return path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        connection = _connect_raw(path)
        try:
            mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()
            if mode is None or str(mode[0]).lower() != "wal":
                raise RuntimeError("本地 SQLite 需要 WAL 模式")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cluster_settings (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                  session_id TEXT PRIMARY KEY,
                  title TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  message_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                  ON sessions(updated_at DESC);
                CREATE TABLE IF NOT EXISTS session_messages (
                  message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL,
                  created_at TEXT,
                  metrics TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_session_messages_session
                  ON session_messages(session_id, message_id);
                CREATE TABLE IF NOT EXISTS model_registry (
                  model_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL DEFAULT '',
                  model_path TEXT NOT NULL DEFAULT '',
                  gguf_path TEXT,
                  quantization TEXT,
                  sha256 TEXT,
                  payload TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_registry_versions (
                  model_id TEXT PRIMARY KEY,
                  revision INTEGER NOT NULL CHECK(revision >= 0)
                );
                INSERT OR IGNORE INTO model_registry_versions(model_id, revision)
                  SELECT model_id, 1 FROM model_registry;
                CREATE TABLE IF NOT EXISTS model_download_jobs (
                  job_id TEXT PRIMARY KEY,
                  status TEXT NOT NULL,
                  progress REAL NOT NULL DEFAULT 0,
                  source TEXT NOT NULL,
                  target TEXT NOT NULL DEFAULT '',
                  model_id TEXT NOT NULL DEFAULT '',
                  preset_id TEXT NOT NULL DEFAULT '',
                  engine TEXT NOT NULL DEFAULT 'auto',
                  quant TEXT NOT NULL DEFAULT '',
                  total_bytes INTEGER NOT NULL DEFAULT 0,
                  downloaded_bytes INTEGER NOT NULL DEFAULT 0,
                  error_code TEXT,
                  error TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_model_download_jobs_created
                  ON model_download_jobs(created_at DESC);
                CREATE TABLE IF NOT EXISTS python_local_store_meta (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS python_conversation_turn_receipts (
                  operation_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  turn_sha256 TEXT NOT NULL,
                  committed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_python_turn_receipts_session
                  ON python_conversation_turn_receipts(session_id, committed_at DESC);
                """
            )
            try:
                connection.execute("ALTER TABLE model_registry ADD COLUMN updated_at TEXT")
            except sqlite3.OperationalError:
                pass
            connection.execute("BEGIN IMMEDIATE")
            try:
                if migrate_legacy:
                    _import_legacy_json(connection)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO python_local_store_meta(key, value, updated_at)
                VALUES ('__write_probe__', '1', ?)
                ON CONFLICT(key) DO UPDATE SET value='1', updated_at=excluded.updated_at
                """,
                (_now(),),
            )
            connection.rollback()
        finally:
            connection.close()
            _restrict_sqlite_files(path)
        _initialized_paths.add(path)
    return path


def _connect() -> sqlite3.Connection:
    return _connect_raw(initialize_local_store())


@contextmanager
def _write_connection() -> Iterator[sqlite3.Connection]:
    with _lock:
        connection = _connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _session_record(row: sqlite3.Row | None) -> Optional[dict]:
    if row is None:
        return None
    return {
        "id": row["session_id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "message_count": int(row["message_count"]),
    }


def create_local_session(session_id: str, title: str = "新对话") -> dict:
    session_id = _valid_session_id(session_id) or ""
    if not session_id:
        raise ValueError("session_id is invalid")
    now = _now()
    with _write_connection() as connection:
        connection.execute(
            """
            INSERT INTO sessions(session_id, title, created_at, updated_at, message_count)
            VALUES (?, ?, ?, ?, 0)
            ON CONFLICT(session_id) DO UPDATE SET
              title=excluded.title, updated_at=excluded.updated_at
            """,
            (session_id, str(title or "新对话")[:256], now, now),
        )
    return get_local_session(session_id) or {
        "id": session_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "message_count": 0,
    }


def get_all_local_sessions(limit: int = 50, offset: int = 0) -> list[dict]:
    connection = _connect()
    try:
        rows = connection.execute(
            """
            SELECT
              sessions.session_id, sessions.title, sessions.created_at,
              sessions.updated_at,
              MAX(
                sessions.message_count,
                (SELECT COUNT(*) FROM session_messages
                 WHERE session_messages.session_id = sessions.session_id)
              ) AS message_count
            FROM sessions ORDER BY updated_at DESC, session_id LIMIT ? OFFSET ?
            """,
            (max(0, int(limit)), max(0, int(offset))),
        ).fetchall()
        return [_session_record(row) for row in rows if row is not None]
    finally:
        connection.close()


def get_local_session_count() -> int:
    connection = _connect()
    try:
        row = connection.execute("SELECT COUNT(*) AS count FROM sessions").fetchone()
        return int(row["count"] if row else 0)
    finally:
        connection.close()


def get_local_session(session_id: str) -> Optional[dict]:
    connection = _connect()
    try:
        row = connection.execute(
            """
            SELECT
              sessions.session_id, sessions.title, sessions.created_at,
              sessions.updated_at,
              MAX(
                sessions.message_count,
                (SELECT COUNT(*) FROM session_messages
                 WHERE session_messages.session_id = sessions.session_id)
              ) AS message_count
            FROM sessions WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        return _session_record(row)
    finally:
        connection.close()


def update_local_session_title(session_id: str, title: str) -> Optional[dict]:
    with _write_connection() as connection:
        result = connection.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
            (str(title)[:256], _now(), session_id),
        )
        if result.rowcount != 1:
            return None
    return get_local_session(session_id)


def delete_local_session(session_id: str) -> int:
    with _write_connection() as connection:
        connection.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
        result = connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        return int(result.rowcount)


def increment_local_session_message_count(session_id: str) -> None:
    with _write_connection() as connection:
        connection.execute(
            """
            UPDATE sessions SET message_count = message_count + 1, updated_at = ?
            WHERE session_id = ?
            """,
            (_now(), session_id),
        )


def decrement_local_session_message_count(session_id: str, count: int = 2) -> None:
    with _write_connection() as connection:
        connection.execute(
            """
            UPDATE sessions SET message_count = MAX(0, message_count - ?), updated_at = ?
            WHERE session_id = ?
            """,
            (max(0, int(count)), _now(), session_id),
        )


def save_local_message(
    session_id: str,
    role: str,
    content: str,
    metrics: dict | None = None,
) -> None:
    session_id = _valid_session_id(session_id) or ""
    if not session_id:
        raise ValueError("session_id is invalid")
    now = _now()
    with _write_connection() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO sessions
              (session_id, title, created_at, updated_at, message_count)
            VALUES (?, '新对话', ?, ?, 0)
            """,
            (session_id, now, now),
        )
        connection.execute(
            """
            INSERT INTO session_messages(session_id, role, content, created_at, metrics)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                str(role),
                str(content),
                now,
                json.dumps(metrics, ensure_ascii=False) if metrics is not None else None,
            ),
        )


def save_local_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
    metrics: dict | None = None,
    *,
    operation_id: str | None = None,
) -> bool:
    """Atomically persist one completed turn and optionally deduplicate a replay.

    A successful turn is one user message, one assistant message, and one
    session-turn count update.  Keeping them in a single transaction prevents
    a crash from exposing a half-written conversation.  ``operation_id`` is
    an opaque request/task identity; replaying it with identical content is a
    no-op, while reusing it for different content fails closed.
    """
    session_id = _valid_session_id(session_id) or ""
    if not session_id:
        raise ValueError("session_id is invalid")
    normalized_operation_id = _valid_operation_id(operation_id)
    if metrics is not None and not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    metrics_json = (
        json.dumps(metrics, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if metrics is not None else None
    )
    # Idempotency identifies the semantic turn, not volatile diagnostics.
    # Retries can legitimately receive a new generation_id, elapsed time or
    # routing trace while still representing the same user/assistant pair.
    turn_payload = json.dumps(
        {
            "session_id": session_id,
            "user_message": str(user_message),
            "assistant_message": str(assistant_message),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    turn_sha256 = hashlib.sha256(turn_payload.encode("utf-8")).hexdigest()
    now = _now()

    with _write_connection() as connection:
        if normalized_operation_id:
            receipt = connection.execute(
                """
                SELECT session_id, turn_sha256, committed_at
                FROM python_conversation_turn_receipts
                WHERE operation_id = ?
                """,
                (normalized_operation_id,),
            ).fetchone()
            if receipt is not None:
                if receipt["session_id"] != session_id:
                    raise ValueError("operation_id has conflicting conversation turn")
                if receipt["turn_sha256"] != turn_sha256:
                    # Compatibility for receipts written before the semantic
                    # hash stopped including volatile metrics. Verify against
                    # the durable adjacent pair, then upgrade the receipt.
                    matching_turn = connection.execute(
                        """
                        SELECT 1
                        FROM session_messages AS user_message
                        JOIN session_messages AS assistant_message
                          ON assistant_message.message_id = user_message.message_id + 1
                         AND assistant_message.session_id = user_message.session_id
                        WHERE user_message.session_id = ?
                          AND user_message.created_at = ?
                          AND user_message.role = 'user'
                          AND user_message.content = ?
                          AND assistant_message.role = 'assistant'
                          AND assistant_message.content = ?
                          AND assistant_message.created_at = ?
                        LIMIT 1
                        """,
                        (
                            session_id,
                            receipt["committed_at"],
                            str(user_message),
                            str(assistant_message),
                            receipt["committed_at"],
                        ),
                    ).fetchone()
                    if matching_turn is None:
                        raise ValueError(
                            "operation_id has conflicting conversation turn"
                        )
                    connection.execute(
                        """
                        UPDATE python_conversation_turn_receipts
                        SET turn_sha256 = ?
                        WHERE operation_id = ?
                        """,
                        (turn_sha256, normalized_operation_id),
                    )
                return False

        connection.execute(
            """
            INSERT OR IGNORE INTO sessions
              (session_id, title, created_at, updated_at, message_count)
            VALUES (?, '新对话', ?, ?, 0)
            """,
            (session_id, now, now),
        )
        connection.execute(
            """
            INSERT INTO session_messages(session_id, role, content, created_at, metrics)
            VALUES (?, 'user', ?, ?, NULL)
            """,
            (session_id, str(user_message), now),
        )
        connection.execute(
            """
            INSERT INTO session_messages(session_id, role, content, created_at, metrics)
            VALUES (?, 'assistant', ?, ?, ?)
            """,
            (session_id, str(assistant_message), now, metrics_json),
        )
        connection.execute(
            """
            UPDATE sessions SET
              message_count = message_count + 2, updated_at = ?
            WHERE session_id = ?
            """,
            (now, session_id),
        )
        if normalized_operation_id:
            connection.execute(
                """
                INSERT INTO python_conversation_turn_receipts(
                  operation_id, session_id, turn_sha256, committed_at
                ) VALUES (?, ?, ?, ?)
                """,
                (normalized_operation_id, session_id, turn_sha256, now),
            )
    return True


def _default_context_messages() -> int:
    """对话窗口条数 —— 与 `chat_context.default_context_messages()` 同源。

    ★ 2026-10-27（#122）：这里原本是签名里的硬编码 `limit: int = 200`。
    它和 `chat_context.DEFAULT_CONTEXT_MESSAGES` 是**两个各自独立的 200**，
    而 `api_server` 调本函数时又不传 `limit` ⇒ 改任何一处都不生效
    （正是 #122 记的"历史累积缺少独立性能归因"里"窗口调不动"的那一半）。

    现在统一读 `QLH_CHAT_CONTEXT_MESSAGES`；本模块已经依赖 `config` 的
    `STATE_DIR`/`LOG_DIR`，再加这一项不引入新的分层耦合。
    """
    try:
        from config import CHAT_CONTEXT_MESSAGES
    except Exception:
        return 200
    try:
        return max(2, int(CHAT_CONTEXT_MESSAGES))
    except (TypeError, ValueError):
        return 200


def load_local_conversation(session_id: str, limit: int | None = None) -> list[dict]:
    """按时间顺序读出某个会话最近 `limit` 条消息。

    `limit=None` ⇒ 用 `QLH_CHAT_CONTEXT_MESSAGES`（缺省 200，与既有行为一致）；
    显式传值仍以传入为准，便于测试钉死窗口。
    """
    if limit is None:
        limit = _default_context_messages()
    connection = _connect()
    try:
        rows = connection.execute(
            """
            SELECT role, content, created_at, metrics FROM session_messages
            WHERE message_id IN (
              SELECT message_id FROM session_messages WHERE session_id = ?
              ORDER BY message_id DESC LIMIT ?
            )
            ORDER BY message_id
            """,
            (session_id, max(0, int(limit))),
        ).fetchall()
        messages = []
        for row in rows:
            item = {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            if row["metrics"]:
                try:
                    item["metrics"] = json.loads(row["metrics"])
                except json.JSONDecodeError:
                    item["metrics"] = {}
            messages.append(item)
        return messages
    finally:
        connection.close()


def get_local_conversation_count(session_id: str) -> int:
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM session_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["count"] if row else 0)
    finally:
        connection.close()


def clear_local_conversation(session_id: str) -> int:
    with _write_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM session_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        count = int(row["count"] if row else 0)
        connection.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
        connection.execute(
            "UPDATE sessions SET message_count = 0, updated_at = ? WHERE session_id = ?",
            (_now(), session_id),
        )
        return count


def delete_local_message_range(session_id: str, turn_index: int) -> int:
    offset = max(0, int(turn_index)) * 2
    with _write_connection() as connection:
        rows = connection.execute(
            """
            SELECT message_id FROM session_messages WHERE session_id = ?
            ORDER BY message_id LIMIT 2 OFFSET ?
            """,
            (session_id, offset),
        ).fetchall()
        ids = [int(row["message_id"]) for row in rows]
        if len(ids) != 2:
            return 0
        connection.execute(
            "DELETE FROM session_messages WHERE message_id IN (?, ?)",
            (ids[0], ids[1]),
        )
        connection.execute(
            """
            UPDATE sessions SET
              message_count = MAX(0, message_count - 2), updated_at = ?
            WHERE session_id = ?
            """,
            (_now(), session_id),
        )
        return 2


def get_local_user_settings() -> dict:
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT value FROM cluster_settings WHERE key = 'user_settings'"
        ).fetchone()
        if not row or not row["value"]:
            return {}
        value = json.loads(row["value"])
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}
    finally:
        connection.close()


def set_local_user_settings(settings: dict) -> None:
    if not isinstance(settings, dict):
        raise ValueError("settings must be an object")
    now = _now()
    with _write_connection() as connection:
        values = {
            "user_settings": json.dumps(settings, ensure_ascii=False, separators=(",", ":")),
        }
        if "saveHistory" in settings:
            values["save_history"] = "true" if bool(settings["saveHistory"]) else "false"
        if "distributedInference" in settings:
            values["distributed_inference_enabled"] = (
                "true" if bool(settings["distributedInference"]) else "false"
            )
        for key, value in values.items():
            connection.execute(
                """
                INSERT INTO cluster_settings(key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, value, now),
            )


def get_local_setting(key: str, default=None):
    """Read a user-owned cluster setting, decoding JSON when possible."""
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT value FROM cluster_settings WHERE key = ?", (str(key),)
        ).fetchone()
        if not row:
            return default
        raw = row["value"]
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return raw
    finally:
        connection.close()


def set_local_setting(key: str, value) -> None:
    """Write a user-owned cluster setting as JSON."""
    with _write_connection() as connection:
        connection.execute(
            """
            INSERT INTO cluster_settings(key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (str(key), json.dumps(value, ensure_ascii=False), _now()),
        )


def get_local_master_identity() -> dict:
    """Read the main-node MAC identity from the user-owned SQLite store."""
    value = get_local_setting(_MASTER_IDENTITY_KEY, {})
    if not isinstance(value, dict):
        return {}
    macs = value.get("mac_addresses")
    if not isinstance(macs, list):
        return {}
    normalized = sorted({str(mac).strip().lower() for mac in macs if str(mac).strip()})
    if not normalized:
        return {}
    return {
        "version": int(value.get("version") or 1),
        "mac_addresses": normalized,
        "bound_at": str(value.get("bound_at") or ""),
    }


def set_local_master_identity(mac_addresses: list[str]) -> dict:
    """Persist the physical MAC set that owns this main-node SQLite store."""
    normalized = sorted({str(mac).strip().lower() for mac in mac_addresses if str(mac).strip()})
    if not normalized:
        raise ValueError("无法记录空的主节点 MAC 地址集合")
    value = {
        "version": 1,
        "mac_addresses": normalized,
        "bound_at": _now(),
    }
    set_local_setting(_MASTER_IDENTITY_KEY, value)
    return value


def _write_model_registry_row(connection: sqlite3.Connection, model_id: str,
                              config: dict, now: str) -> None:
    connection.execute(
        """
        INSERT INTO model_registry
          (model_id, name, model_path, gguf_path, quantization, sha256, payload, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_id) DO UPDATE SET
          name=excluded.name, model_path=excluded.model_path, gguf_path=excluded.gguf_path,
          quantization=excluded.quantization, sha256=excluded.sha256,
          payload=excluded.payload, updated_at=excluded.updated_at
        """,
        (
            str(model_id), str(config.get("name", "")), str(config.get("model_path", "")),
            str(config.get("gguf_path", "")), str(config.get("quantization", "")),
            str(config.get("sha256", "")), json.dumps(config, ensure_ascii=False), now, now,
        ),
    )


class _RegistryPayload(dict):
    """Legacy dictionary projection carrying a non-serialized CAS token."""
    def __init__(self, value, registry_revision):
        super().__init__(value)
        self.registry_revision = registry_revision


def _catalog_service(connection=None):
    from contextlib import nullcontext
    from qlh_core.adapters.catalog_repository import SqliteCatalogRepository
    from qlh_core.control.catalog import CatalogService
    transaction = _write_connection if connection is None else lambda: nullcontext(connection)
    return CatalogService(SqliteCatalogRepository(
        connect=_connect, transaction=transaction, write_row=_write_model_registry_row, now=_now,
    ))


def get_local_model_registry_snapshot(model_id: str):
    """Read the durable alias revision, including tombstones, in one snapshot."""
    return _catalog_service().snapshot(str(model_id))


def publish_local_model_registry(model_id: str, config_json: str | dict,
                                 *, expected_revision: int) -> bool:
    config = json.loads(config_json) if isinstance(config_json, str) else dict(config_json)
    return _catalog_service().publish(str(model_id), config, expected_revision=expected_revision)


def save_local_experimental_model(model_id: str, config_json: str | dict,
                                  *, expected_revision: int | None = None) -> bool:
    """Compatibility insert; replacements require an explicit observed version."""
    if expected_revision is None:
        snapshot = get_local_model_registry_snapshot(model_id)
        if snapshot.payload is not None:
            return False
        expected_revision = snapshot.registry_revision
    return publish_local_model_registry(model_id, config_json, expected_revision=expected_revision)


def save_local_experimental_model_cas(model_id: str, config_json: str | dict,
                                      *, expected: dict | None) -> bool:
    """Legacy payload CAS, serialized with the catalog version commit.

    Registry dictionary projections retain a hidden version token through
    deepcopy. Plain dictionaries remain a payload-only compatibility input;
    new writers use publish_local_model_registry with an explicit revision.
    """
    config = json.loads(config_json) if isinstance(config_json, str) else dict(config_json)
    with _write_connection() as connection:
        from qlh_core.adapters.catalog_repository import SqliteCatalogRepository
        try:
            current = SqliteCatalogRepository._read(connection, str(model_id))
        except (TypeError, ValueError):
            return False
        expected_payload = None if isinstance(expected, _RegistryPayload) and not expected else expected
        if current.as_payload() != expected_payload:
            return False
        expected_revision = getattr(expected, "registry_revision", current.registry_revision)
        return _catalog_service(connection).publish(str(model_id), config, expected_revision=expected_revision)


def get_local_experimental_models() -> list[dict]:
    connection = _connect()
    try:
        connection.execute("BEGIN")
        rows = connection.execute(
            "SELECT r.payload, COALESCE(v.revision, 1) AS revision FROM model_registry r "
            "LEFT JOIN model_registry_versions v ON v.model_id = r.model_id ORDER BY r.model_id",
        ).fetchall()
        result = []
        for row in rows:
            try:
                value = json.loads(row["payload"])
                if isinstance(value, dict):
                    result.append(_RegistryPayload(value, int(row["revision"])))
            except (TypeError, json.JSONDecodeError):
                continue
        return result
    finally:
        connection.rollback()
        connection.close()


def delete_local_experimental_model(model_id: str, *, expected_revision: int | None = None) -> bool:
    if expected_revision is None:
        snapshot = get_local_model_registry_snapshot(model_id)
        expected_revision = snapshot.registry_revision
    return _catalog_service().remove(str(model_id), expected_revision=expected_revision)


def get_local_save_history() -> bool:
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT value FROM cluster_settings WHERE key = 'save_history'"
        ).fetchone()
        if not row or not row["value"]:
            settings_row = connection.execute(
                "SELECT value FROM cluster_settings WHERE key = 'user_settings'"
            ).fetchone()
            try:
                settings = json.loads(settings_row["value"]) if settings_row else {}
            except (json.JSONDecodeError, TypeError):
                settings = {}
            if not isinstance(settings, dict):
                settings = {}
            return bool(settings.get("saveHistory", True))
        return str(row["value"]).strip().lower() in {"1", "true", "yes", "on"}
    finally:
        connection.close()


def local_store_health() -> dict:
    path = initialize_local_store()
    connection = _connect_raw(path)
    try:
        quick = connection.execute("PRAGMA quick_check").fetchone()
        mode = connection.execute("PRAGMA journal_mode").fetchone()
        synchronous = connection.execute("PRAGMA synchronous").fetchone()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO python_local_store_meta(key, value, updated_at)
            VALUES ('__health_probe__', '1', ?)
            ON CONFLICT(key) DO UPDATE SET value='1', updated_at=excluded.updated_at
            """,
            (_now(),),
        )
        connection.rollback()
        healthy = bool(
            quick and str(quick[0]).lower() == "ok"
            and mode and str(mode[0]).lower() == "wal"
            and synchronous and int(synchronous[0]) == 2
        )
        return {
            "status": "ok" if healthy else "unavailable",
            "backend": "sqlite",
            "writable": healthy,
            "path": path,
            "journal_mode": "" if mode is None else str(mode[0]).lower(),
            "synchronous": "full" if synchronous and int(synchronous[0]) == 2 else "other",
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "backend": "sqlite",
            "writable": False,
            "path": path,
            "error": str(exc),
        }
    finally:
        connection.close()


def local_store_stats() -> dict:
    path = initialize_local_store()
    connection = _connect_raw(path)
    try:
        sessions = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = connection.execute("SELECT COUNT(*) FROM session_messages").fetchone()[0]
        return {
            "backend": "sqlite",
            "sqlite_path": path,
            "store_dir": os.path.dirname(path),
            "session_count": int(sessions),
            "message_count": int(messages),
            "file_count": 1,
        }
    except Exception as exc:
        return {
            "backend": "sqlite",
            "sqlite_path": path,
            "store_dir": os.path.dirname(path),
            "error": str(exc),
        }
    finally:
        connection.close()
