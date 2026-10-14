from __future__ import annotations

import os
import sys
import threading
import time

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chat_context import (
    ConversationContextConflict,
    ConversationContextService,
    TurnPersistenceResult,
)
from request_deadline import RequestDeadline, RequestDeadlineExceeded


def test_transaction_lock_wait_is_capped_by_request_deadline():
    service = ConversationContextService(
        load_history=lambda _session_id: [],
        persist_turn=lambda *_args: True,
    )
    acquired = threading.Event()
    release = threading.Event()

    def hold_transaction():
        with service.transaction():
            acquired.set()
            release.wait(1.0)

    thread = threading.Thread(target=hold_transaction)
    thread.start()
    assert acquired.wait(0.5)

    started = time.monotonic()
    with pytest.raises(RequestDeadlineExceeded):
        service.acquire_transaction(
            deadline=RequestDeadline.start(0.05),
        )
    assert time.monotonic() - started < 0.3

    release.set()
    thread.join(1.0)
    assert not thread.is_alive()


def test_same_session_reloads_after_invalidation():
    durable = {
        "s1": [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ],
    }
    loads: list[str] = []

    def load(session_id):
        loads.append(session_id)
        return list(durable.get(session_id, []))

    service = ConversationContextService(
        load_history=load,
        persist_turn=lambda *_args: True,
    )

    first = service.prepare("s1")
    service.invalidate(clear_histories=True)
    second = service.prepare("s1")

    assert first.session_id == second.session_id == "s1"
    assert second.session_changed is False
    assert second.reloaded is True
    assert second.history == durable["s1"]
    assert loads == ["s1", "s1"]


def test_save_disabled_still_commits_in_memory():
    service = ConversationContextService(
        load_history=lambda _session_id: [],
        persist_turn=lambda *_args: False,
    )

    persisted = service.commit_turn("s1", "question", "answer")

    assert persisted is False
    assert service.history_for("s1") == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]


def test_warm_and_reloaded_context_share_the_same_window():
    durable = {
        "s1": [
            {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
            for index in range(220)
        ],
    }

    def load(session_id):
        return durable.get(session_id, [])[-200:]

    def persist(session_id, user, assistant, _metrics):
        durable.setdefault(session_id, []).extend([
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ])
        return True

    service = ConversationContextService(
        load_history=load,
        persist_turn=persist,
        max_messages=200,
    )
    service.prepare("s1")
    service.commit_turn("s1", "next", "reply")
    warm = list(service.history_for("s1"))

    service.invalidate(clear_histories=True)
    reloaded = list(service.prepare("s1").history)

    assert len(warm) == len(reloaded) == 200
    assert warm == reloaded
    assert warm[-2:] == [
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "reply"},
    ]


def test_completed_turn_operation_is_idempotent_in_memory():
    writes = []
    service = ConversationContextService(
        load_history=lambda _session_id: [],
        persist_turn=lambda *args: writes.append(args) or True,
    )

    first = service.commit_turn(
        "s1", "question", "answer", operation_id="request-1",
    )
    replay = service.commit_turn(
        "s1", "question", "answer", operation_id="request-1",
    )

    assert first is replay is True
    assert len(writes) == 1
    assert service.history_for("s1") == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]


def test_durable_replay_reloads_without_duplicating_warm_history():
    durable = {"s1": []}
    receipts: dict[str, tuple[str, str, str]] = {}

    def load(session_id):
        return list(durable.get(session_id, []))

    def persist(session_id, user, assistant, _metrics):
        operation_id = "request-replay"
        turn = (session_id, user, assistant)
        if operation_id in receipts:
            assert receipts[operation_id] == turn
            return TurnPersistenceResult.REPLAY
        receipts[operation_id] = turn
        durable.setdefault(session_id, []).extend([
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ])
        return TurnPersistenceResult.COMMITTED

    first = ConversationContextService(load_history=load, persist_turn=persist)
    assert first.commit_turn(
        "s1", "question", "answer", operation_id="request-replay",
    ) is True

    restarted = ConversationContextService(load_history=load, persist_turn=persist)
    restarted.prepare("s1")
    assert restarted.commit_turn(
        "s1", "question", "answer", operation_id="request-replay",
    ) is False
    assert restarted.history_for("s1") == durable["s1"]
    assert len(restarted.history_for("s1")) == 2


def test_operation_conflict_is_not_converted_to_memory_only_success():
    service = ConversationContextService(
        load_history=lambda _session_id: [],
        persist_turn=lambda *_args: (_ for _ in ()).throw(
            ValueError("operation_id has conflicting conversation turn")
        ),
    )

    try:
        service.commit_turn(
            "s1", "question", "different answer", operation_id="request-1",
        )
    except ValueError as exc:
        assert "conflicting conversation turn" in str(exc)
    else:
        raise AssertionError("operation-id conflict must fail closed")

    assert service.history_for("s1") == []


def test_stale_revision_cannot_commit_after_crud_or_another_turn():
    service = ConversationContextService(
        load_history=lambda _session_id: [],
        persist_turn=lambda *_args: TurnPersistenceResult.SKIPPED,
    )
    prepared = service.prepare("s1")
    service.clear_session("s1")

    try:
        service.commit_turn(
            "s1",
            "question",
            "answer",
            expected_revision=prepared.revision,
        )
    except ConversationContextConflict as exc:
        assert "expected_revision" in str(exc)
    else:
        raise AssertionError("stale conversation snapshot must not commit")

    assert service.history_for("s1") == []
