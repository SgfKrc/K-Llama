"""Shared conversation-context lifecycle for sync and streaming chat paths.

The service owns conversation state only. Model and KV lifecycle remain with
the embedding host, which reacts to session_changed and reloaded.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib

current_client_id = ContextVar("conversation_client_id", default="default")
from dataclasses import dataclass
from enum import Enum
from threading import Lock, RLock
from typing import Callable, Iterable, Iterator, Mapping, MutableMapping, Optional

from request_deadline import RequestDeadline, RequestDeadlineExceeded


DEFAULT_SESSION_ID = "default"
DEFAULT_CONTEXT_MESSAGES = 200


def default_context_messages() -> int:
    """对话窗口条数 —— 统一到 `QLH_CHAT_CONTEXT_MESSAGES`（缺省 200）。

    ★ 2026-10-27（#122）：此前这个数字在**两处各自硬编码 200**
    （本模块的 `DEFAULT_CONTEXT_MESSAGES` 与
    `local_store.load_local_conversation(limit=200)`），且 `api_server`
    调后者时不传 limit ⇒ 现场想调窗口只能改源码。

    这里**延迟到调用时**读取而不是模块级常量：`config` 的 `.env` 提升
    发生在进程启动早期（`koakuma.bat` 的 `:promote_qlh`），模块常量会在
    `import` 那一刻固化下来，一旦加载顺序变化就会读到一个不生效的旧值 ——
    这个项目已经在 `LOCAL_CPU_EXECUTOR_ENABLED` 上踩过一次同样的坑
    （见 `config.local_cpu_executor_enabled()` 的注释）。

    窗口按**条数**计，不是 token，所以它只是上限；真正进入 prompt 的规模
    仍由 `event=execution_prompt` 的 `prompt_tokens` 观测（见
    `qlh_core/bootstrap/execution.py`），两者配合才能判断"是历史太长"
    还是"单条太肥"。
    """
    try:
        from config import CHAT_CONTEXT_MESSAGES
    except Exception:                       # 独立使用本模块时不应因缺 config 崩掉
        return DEFAULT_CONTEXT_MESSAGES
    try:
        return max(2, int(CHAT_CONTEXT_MESSAGES))
    except (TypeError, ValueError):
        return DEFAULT_CONTEXT_MESSAGES


@dataclass(frozen=True)
class PreparedChatContext:
    session_id: str
    history: list[dict]
    revision: int
    session_changed: bool
    reloaded: bool


class TurnPersistenceResult(str, Enum):
    """Storage-side outcome for one completed conversation turn."""

    COMMITTED = "committed"
    SKIPPED = "skipped"
    REPLAY = "replay"


class ConversationContextConflict(RuntimeError):
    """The transcript changed after a request prepared its input snapshot."""


class ConversationContextService:
    """Single lifecycle for loading, trimming and committing chat history."""

    def __init__(
        self,
        *,
        load_history: Callable[[str], Iterable[Mapping]],
        persist_turn: Callable[
            [str, str, str, Optional[dict]], bool | TurnPersistenceResult
        ],
        default_session_id: str = DEFAULT_SESSION_ID,
        max_messages: Optional[int] = None,
    ) -> None:
        self._load_history = load_history
        self._persist_turn = persist_turn
        self.default_session_id = self._normalize_session_id(default_session_id)
        # `None` ⇒ 调用时读 `QLH_CHAT_CONTEXT_MESSAGES`；显式传入仍以传入值为准
        # （测试与嵌入式用法可以钉死窗口，不受环境变量影响）。
        self.max_messages = max(
            2, int(max_messages) if max_messages is not None else default_context_messages())
        self._active_by_client: dict[str, str] = {}
        self._kv_session_id: Optional[str] = None
        self.histories: MutableMapping[str, list[dict]] = {}
        self._generation = 0
        self._history_generations: dict[str, int] = {}
        self._revisions: dict[str, int] = {}
        self._committed_operations: dict[
            str, tuple[str, str, str, bool]
        ] = {}
        self._lock = RLock()
        # The active session and the model KV cache are process-wide facades.
        # Keep a complete prepare -> infer -> commit transaction serialized.
        self._transaction_lock = Lock()

    @property
    def active_session_id(self):
        with self._lock:
            return self._active_by_client.get(current_client_id.get())

    @active_session_id.setter
    def active_session_id(self, value):
        with self._lock:
            key = current_client_id.get()
            if value:
                self._active_by_client[key] = self._normalize_session_id(value)
            else:
                self._active_by_client.pop(key, None)

    def resolve_session(self, requested=None, *, client_id=None):
        key = str(client_id or current_client_id.get())
        with self._lock:
            return self._normalize_session_id(requested or self._active_by_client.get(key)
                                              or (self.default_session_id if key == "default" else "client-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:40]))

    def cached_histories(self):
        with self._lock:
            return {key: [dict(m) for m in value] for key, value in self.histories.items()}

    @staticmethod
    def _normalize_session_id(value: object) -> str:
        normalized = str(value or "").strip()
        return normalized or DEFAULT_SESSION_ID

    @staticmethod
    def _normalize_messages(messages: Iterable[Mapping]) -> list[dict]:
        normalized: list[dict] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or "").strip()
            if not role:
                continue
            normalized.append({
                "role": role,
                "content": str(message.get("content") or ""),
            })
        return normalized

    def _trim_in_place(self, history: list[dict]) -> None:
        overflow = len(history) - self.max_messages
        if overflow > 0:
            del history[:overflow]
        # Persisted chat rows are user/assistant pairs. Never keep an orphaned
        # assistant message at the start of the active model context.
        while len(history) > 1 and history[0].get("role") == "assistant":
            del history[0]


    @contextmanager
    def transaction(
        self,
        *,
        timeout: Optional[float] = None,
        deadline: Optional[RequestDeadline] = None,
        serialize: bool = True,
    ) -> Iterator[None]:
        """Serialize a complete conversation transaction in this process.

        When a request deadline is supplied, time spent queued behind another
        conversation is part of the same end-to-end budget. The lock is never
        acquired after that budget has expired.
        """

        if not serialize:
            yield
            return
        self.acquire_transaction(timeout=timeout, deadline=deadline)
        try:
            yield
        finally:
            self.release_transaction()

    def acquire_transaction(
        self,
        *,
        timeout: Optional[float] = None,
        deadline: Optional[RequestDeadline] = None,
    ) -> None:
        if deadline is None and timeout is None:
            self._transaction_lock.acquire()
            return

        wait_timeout = float("inf") if timeout is None else max(0.0, float(timeout))
        if deadline is not None:
            wait_timeout = min(wait_timeout, deadline.require_remaining())
        acquired = self._transaction_lock.acquire(timeout=wait_timeout)
        if acquired:
            if deadline is not None and deadline.expired():
                self._transaction_lock.release()
                raise RequestDeadlineExceeded(
                    "request deadline exceeded while acquiring conversation transaction"
                )
            return
        if deadline is not None:
            raise RequestDeadlineExceeded(
                "request deadline exceeded while acquiring conversation transaction"
            )
        raise TimeoutError("conversation transaction lock acquisition timed out")

    def release_transaction(self) -> None:
        self._transaction_lock.release()

    def _ensure_loaded_locked(
        self, session_id: str, *, force_reload: bool = False,
    ) -> tuple[list[dict], bool]:
        current = self._history_generations.get(session_id) == self._generation
        if not force_reload and session_id in self.histories and current:
            history = self.histories[session_id]
            self._trim_in_place(history)
            return history, False

        loaded = self._normalize_messages(self._load_history(session_id))
        self._trim_in_place(loaded)
        self.histories[session_id] = loaded
        self._history_generations[session_id] = self._generation
        self._revisions.setdefault(session_id, 0)
        return loaded, True

    def prepare(
        self,
        requested_session_id: Optional[str] = None,
        *,
        force_reload: bool = False,
        client_id: Optional[str] = None,
        runtime_kv: bool = True,
    ) -> PreparedChatContext:
        with self._lock:
            session_id = self.resolve_session(requested_session_id, client_id=client_id)
            session_changed = self._kv_session_id != session_id
            history, reloaded = self._ensure_loaded_locked(
                session_id, force_reload=force_reload,
            )
            self._active_by_client[str(client_id or current_client_id.get())] = session_id
            if runtime_kv:
                self._kv_session_id = session_id
            return PreparedChatContext(
                session_id=session_id,
                history=list(history),
                revision=self._revisions.get(session_id, 0),
                session_changed=session_changed,
                reloaded=reloaded,
            )

    def active_history(self) -> list[dict]:
        with self._lock:
            if self.active_session_id is None:
                return []
            history, _ = self._ensure_loaded_locked(self.active_session_id)
            return history

    def history_for(self, session_id: str) -> list[dict]:
        with self._lock:
            normalized = self._normalize_session_id(session_id)
            history, _ = self._ensure_loaded_locked(normalized)
            return history

    def commit_turn(
        self,
        session_id: Optional[str],
        user_message: str,
        assistant_message: str,
        metrics: Optional[dict] = None,
        *,
        operation_id: Optional[str] = None,
        expected_revision: Optional[int] = None,
    ) -> bool:
        """Persist and append exactly one completed user/assistant turn."""

        with self._lock:
            normalized = self._normalize_session_id(
                session_id or self.active_session_id or self.default_session_id
            )
            normalized_operation_id = str(operation_id or "").strip()
            if normalized_operation_id:
                prior = self._committed_operations.get(normalized_operation_id)
                if prior is not None:
                    prior_session, prior_user, prior_assistant, prior_persisted = prior
                    if (
                        prior_session != normalized
                        or prior_user != str(user_message)
                        or prior_assistant != str(assistant_message)
                    ):
                        raise ValueError(
                            "operation_id has conflicting conversation turn"
                        )
                    return prior_persisted
            current_revision = self._revisions.get(normalized, 0)
            if (
                expected_revision is not None
                and int(expected_revision) != current_revision
            ):
                raise ConversationContextConflict(
                    f"conversation context changed: session={normalized!r} "
                    f"expected_revision={expected_revision} "
                    f"current_revision={current_revision}"
                )
            history, _ = self._ensure_loaded_locked(normalized)
            persistence_result = self._persist_turn(
                normalized,
                str(user_message),
                str(assistant_message),
                metrics,
            )
            if persistence_result is TurnPersistenceResult.REPLAY:
                # Durable idempotency can outlive this process. When SQLite
                # reports an already-committed operation, reload its canonical
                # rows instead of appending the same turn to the warm cache.
                history = self._normalize_messages(self._load_history(normalized))
                self._trim_in_place(history)
                self.histories[normalized] = history
                self._history_generations[normalized] = self._generation
                if normalized_operation_id:
                    self._committed_operations[normalized_operation_id] = (
                        normalized,
                        str(user_message),
                        str(assistant_message),
                        False,
                    )
                return False
            persisted = (
                persistence_result is TurnPersistenceResult.COMMITTED
                or (
                    not isinstance(persistence_result, TurnPersistenceResult)
                    and bool(persistence_result)
                )
            )
            history.extend([
                {"role": "user", "content": str(user_message)},
                {"role": "assistant", "content": str(assistant_message)},
            ])
            self._trim_in_place(history)
            self._history_generations[normalized] = self._generation
            self._revisions[normalized] = current_revision + 1
            if normalized_operation_id:
                self._committed_operations[normalized_operation_id] = (
                    normalized,
                    str(user_message),
                    str(assistant_message),
                    persisted,
                )
                while len(self._committed_operations) > 2048:
                    self._committed_operations.pop(next(iter(self._committed_operations)))
            return persisted

    def invalidate(self, *, clear_histories: bool = True) -> None:
        """Invalidate every cached history without changing the active ID."""

        with self._lock:
            self._generation += 1
            affected = set(self._revisions) | set(self.histories)
            if self.active_session_id:
                affected.add(self.active_session_id)
            for session_id in affected:
                self._revisions[session_id] = (
                    self._revisions.get(session_id, 0) + 1
                )
            self._history_generations.clear()
            if clear_histories:
                self.histories.clear()

    def mark_current(self, session_id: str) -> None:
        with self._lock:
            normalized = self._normalize_session_id(session_id)
            history = self.histories.get(normalized)
            if history is not None:
                self._trim_in_place(history)
                self._history_generations[normalized] = self._generation
                self._revisions[normalized] = (
                    self._revisions.get(normalized, 0) + 1
                )

    def set_history(
        self, session_id: str, messages: Iterable[Mapping],
    ) -> list[dict]:
        """Replace one cached session with a current canonical window."""

        with self._lock:
            normalized = self._normalize_session_id(session_id)
            history = self._normalize_messages(messages)
            self._trim_in_place(history)
            self.histories[normalized] = history
            self._history_generations[normalized] = self._generation
            self._revisions[normalized] = (
                self._revisions.get(normalized, 0) + 1
            )
            return history

    def clear_session(self, session_id: str) -> None:
        self.set_history(session_id, [])

    def invalidate_session(
        self, session_id: str, *, clear_history: bool = True,
    ) -> None:
        """Invalidate one cache entry while preserving the active session ID."""

        with self._lock:
            normalized = self._normalize_session_id(session_id)
            self._history_generations.pop(normalized, None)
            self._revisions[normalized] = (
                self._revisions.get(normalized, 0) + 1
            )
            if clear_history:
                self.histories.pop(normalized, None)

    def drop_session(self, session_id: str) -> bool:
        with self._lock:
            normalized = self._normalize_session_id(session_id)
            was_active = self.active_session_id == normalized
            self.histories.pop(normalized, None)
            self._history_generations.pop(normalized, None)
            self._revisions[normalized] = (
                self._revisions.get(normalized, 0) + 1
            )
            for client, selected in list(self._active_by_client.items()):
                if selected == normalized:
                    self._active_by_client.pop(client, None)
            if self._kv_session_id == normalized:
                self._kv_session_id = None
            return was_active
