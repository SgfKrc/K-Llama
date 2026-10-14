"""Shared conversation-context lifecycle for sync and streaming chat paths.

The service owns conversation state only. Model and KV lifecycle remain with
the embedding host, which reacts to session_changed and reloaded.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from threading import Lock, RLock
from typing import Callable, Iterable, Iterator, Mapping, MutableMapping, Optional

from request_deadline import RequestDeadline, RequestDeadlineExceeded


DEFAULT_SESSION_ID = "default"
DEFAULT_CONTEXT_MESSAGES = 200


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
        max_messages: int = DEFAULT_CONTEXT_MESSAGES,
    ) -> None:
        self._load_history = load_history
        self._persist_turn = persist_turn
        self.default_session_id = self._normalize_session_id(default_session_id)
        self.max_messages = max(2, int(max_messages))
        self.active_session_id: Optional[str] = None
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

    def adopt_facade_state(
        self,
        active_session_id: Optional[str],
        histories: MutableMapping[str, list[dict]],
    ) -> None:
        """Adopt compatibility globals exposed by the legacy API facade."""

        with self._lock:
            self.active_session_id = (
                self._normalize_session_id(active_session_id)
                if active_session_id
                else None
            )
            if histories is not self.histories:
                affected = set(self.histories) | {
                    self._normalize_session_id(session_id)
                    for session_id in histories
                }
                for session_id in affected:
                    self._revisions[session_id] = (
                        self._revisions.get(session_id, 0) + 1
                    )
                self.histories = histories
                self._history_generations = {
                    self._normalize_session_id(session_id): self._generation
                    for session_id in histories
                }

    @contextmanager
    def transaction(
        self,
        *,
        timeout: Optional[float] = None,
        deadline: Optional[RequestDeadline] = None,
    ) -> Iterator[None]:
        """Serialize a complete conversation transaction in this process.

        When a request deadline is supplied, time spent queued behind another
        conversation is part of the same end-to-end budget. The lock is never
        acquired after that budget has expired.
        """

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
    ) -> PreparedChatContext:
        with self._lock:
            session_id = self._normalize_session_id(
                requested_session_id
                or self.active_session_id
                or self.default_session_id
            )
            session_changed = self.active_session_id != session_id
            history, reloaded = self._ensure_loaded_locked(
                session_id, force_reload=force_reload,
            )
            self.active_session_id = session_id
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
            if was_active:
                self.active_session_id = None
            return was_active
