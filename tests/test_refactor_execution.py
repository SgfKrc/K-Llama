"""R4 request ownership, frozen linear execution, stop boundaries and recovery."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from qlh_core.adapters.request_repository import SqliteRequestRepository
from qlh_core.contracts.deadline import RequestDeadline, DeadlineCancelEvent
from qlh_core.contracts.deployment import canonical
from qlh_core.contracts.execution import (ExecutionError, ExecutionPlan, Segment, SamplingPolicy,
                                         StageIdentity, StageReceipt, REQUEST_DOMAIN)
from qlh_core.execution.requests import RequestService
from qlh_core.execution.linear import LinearExecutor
from qlh_core.execution.stopping import StopPolicy

pytestmark = pytest.mark.refactor_contract


def plan(*, local_head=False, engine="llama_cpp"):
    return ExecutionPlan("deployment", "boot", canonical({"term": 1}), 4, (
        Segment("master", "prefix", "master:0", 0, 0, 2, True, local_head, canonical({"engine": engine})),
        Segment("worker", "tail", "worker:1", 1, 2, 4, False, not local_head, canonical({"engine": "llama_cpp"}))))


@pytest.fixture
def repository(tmp_path):
    path = tmp_path / "requests.sqlite3"
    def connect():
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        return connection
    @contextmanager
    def transaction():
        connection = connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
    return SqliteRequestRepository(connect=connect, transaction=transaction), connect


@pytest.fixture
def service(repository):
    owner = RequestService(repository[0], boot_id="boot", capacity=2)
    yield owner
    owner.close()


def wait_for(predicate):
    until = time.monotonic() + 4
    while time.monotonic() < until:
        if predicate():
            return
        threading.Event().wait(0.005)
    raise AssertionError("condition did not become true")


def launch(service, key, execute, *, cleanup=lambda _r: True, deadline=None, cancel=None):
    result = {}
    def run():
        try:
            result.update(service.run(plan=plan(), request_id=key,
                deadline=deadline or RequestDeadline.start(5), execute=execute, cleanup=cleanup, cancel_event=cancel))
        except BaseException as exc:
            result["exception"] = exc
    thread = threading.Thread(target=run)
    thread.start()
    return result, thread


def test_fifo_pause_capacity_and_no_bypass(service):
    entered, release = threading.Event(), threading.Event()
    order = []
    def execute(record):
        order.append(record["request_id"])
        if record["request_id"] == "one":
            entered.set()
            assert release.wait(4)
        return {"response": record["request_id"]}
    one, t1 = launch(service, "one", execute)
    assert entered.wait(4)
    two, t2 = launch(service, "two", execute)
    wait_for(lambda: service.snapshot()["queue"] == ["two"])
    three, t3 = launch(service, "three", execute)
    wait_for(lambda: service.snapshot()["queue"] == ["two", "three"])
    full = service.run(plan=plan(), deadline=RequestDeadline.start(1), execute=execute, cleanup=lambda _r: True)
    assert full["reason_code"] == "request_queue_full"
    service.pause()
    paused = service.run(plan=plan(), deadline=RequestDeadline.start(1), execute=execute, cleanup=lambda _r: True)
    assert paused["reason_code"] == "request_queue_paused"
    release.set()
    for thread in (t1, t2, t3):
        thread.join(4)
        assert not thread.is_alive()
    assert order == ["one", "two", "three"]
    assert [one["response"], two["response"], three["response"]] == order
    service.resume()
    result = service.run(plan=plan(), deadline=RequestDeadline.start(1), execute=lambda _r: {"response": "ok"}, cleanup=lambda _r: True)
    assert result["response"] == "ok"


def test_cancel_terminal_does_not_release_native_slot(service):
    entered, native_return, cleanup_entered, cleanup_return = (threading.Event() for _ in range(4))
    second_entered = threading.Event()
    def execute(_record):
        entered.set()
        assert native_return.wait(4)
        return {"response": "late"}
    def cleanup(_record):
        cleanup_entered.set()
        assert cleanup_return.wait(4)
        return True
    one, t1 = launch(service, "one", execute, cleanup=cleanup)
    assert entered.wait(4)
    assert service.cancel("one")
    t1.join(2)
    assert one["reason_code"] == "generation_cancelled"
    assert service.snapshot()["requests"]["one"]["execution_state"] == "running"
    two, t2 = launch(service, "two", lambda _r: (second_entered.set() or {"response": "ok"}))
    wait_for(lambda: service.snapshot()["queue"] == ["two"])
    assert not second_entered.is_set()
    native_return.set()
    assert cleanup_entered.wait(3)
    assert service.snapshot()["requests"]["one"]["execution_state"] == "draining"
    assert not second_entered.is_set()
    cleanup_return.set()
    t2.join(3)
    assert two["response"] == "ok"
    first = service.snapshot()["requests"]["one"]
    assert (first["terminal"], first["execution_state"]) == ("generation_cancelled", "stopped")


def test_queue_deadline_uses_original_budget_and_never_executes(service):
    entered, release = threading.Event(), threading.Event()
    one, t1 = launch(service, "one", lambda _r: (entered.set(), release.wait(4), {"response": "ok"})[-1])
    assert entered.wait(3)
    calls = []
    deadline = RequestDeadline.start(0.1)
    two, t2 = launch(service, "two", lambda r: calls.append(r), deadline=deadline)
    t2.join(2)
    assert two["reason_code"] == "request_deadline_exceeded"
    assert two["metrics"]["request"]["deadline_epoch_ms"] == deadline.expires_at_epoch_ms
    release.set()
    t1.join(3)
    wait_for(lambda: not service.snapshot()["queue"])
    assert not calls


@pytest.mark.parametrize("cleanup", [lambda _r: False, lambda _r: (_ for _ in ()).throw(RuntimeError("KV cleanup failed"))])
def test_failed_cleanup_quarantines_deployment_and_keeps_debt(service, cleanup):
    result = service.run(plan=plan(), deadline=RequestDeadline.start(3), execute=lambda _r: {"response": "partial"}, cleanup=cleanup)
    assert result["reason_code"] == "request_runtime_cleanup_pending"
    assert result["metrics"]["request"]["execution_state"] == "cleanup_pending"
    refused = service.run(plan=plan(), deadline=RequestDeadline.start(1), execute=lambda _r: pytest.fail("slot reused"), cleanup=cleanup)
    assert refused["reason_code"] == "request_runtime_cleanup_pending"


def test_restart_has_own_recovery_domain_and_does_not_resume_kv(repository):
    repo, connect = repository
    first = RequestService(repo, boot_id="old")
    first.start()
    repo.transact(lambda state: state["requests"].update({"unfinished": {"terminal": "", "execution_state": "running"}}))
    second = RequestService(repo, boot_id="new")
    second.start()
    record = repo.read()["requests"]["unfinished"]
    assert (record["terminal"], record["execution_state"]) == ("request_process_restarted", "recovery_required")
    with connect() as connection:
        assert {row[0] for row in connection.execute("SELECT DISTINCT domain_id FROM request_journal_events")} == {REQUEST_DOMAIN}
        assert not connection.execute("SELECT 1 FROM sqlite_master WHERE name='model_control_events'").fetchone()
    assert repo.recovery_status()["commit_scope"] == "local_sqlite"
    first.close()
    second.close()


def test_journal_corruption_refuses_start(repository):
    repo, connect = repository
    RequestService(repo, boot_id="old").start()
    with connect() as connection:
        connection.execute("UPDATE request_journal_events SET previous_digest='broken'")
    with pytest.raises(ExecutionError, match="journal_corrupt"):
        RequestService(repo, boot_id="new").start()


@pytest.mark.parametrize("parts,expected", [(["hello<", "hello</en", "hello</end>user"], "hello"),
                                          (["abcST", "abcSTOP", "abcSTOPtail"], "abc")])
def test_stop_text_across_tokens_never_leaks_suffix(parts, expected):
    policy = StopPolicy(max_tokens=10, eos_ids=[999], stops=["</end>", "STOP"])
    emitted = []
    for i, text in enumerate(parts):
        emitted.append(policy.push(i, text))
        if policy.reason:
            break
    assert "".join(emitted) == policy.text == expected
    assert policy.reason == "text_stop"


def test_eos_flushes_safe_suffix_once_and_limit_finishes():
    policy = StopPolicy(max_tokens=5, eos_ids=[999], stops=["STOP"])
    assert policy.push(1, "aST") == "a"
    assert policy.push(999, "ignored") == ""
    assert policy.text == "a"
    assert policy.reason == "eos_token"
    with pytest.raises(ExecutionError, match="already_finished"):
        policy.push(1, "a")
    policy = StopPolicy(max_tokens=1)
    assert policy.push(1, "A") == "A" and policy.reason == "max_new_tokens"


class Ports:
    def __init__(self, tokens=(1, 2, 999)):
        self.tokens = tokens
        self.calls = []
        self.valid = True
    def current(self, plan):
        return self.valid
    def prepare(self, plan, record):
        self.calls.append("prepare")
    def eos_ids(self):
        return (999,)
    def prefix(self, segment, record, step, token):
        self.calls.append(("prefix", step, token))
        return step
    def stage(self, segment, identity, hidden, record):
        self.calls.append(("stage", identity))
        return StageReceipt(identity, str(hidden), self.tokens[hidden] if segment.head else hidden)
    def head(self, hidden):
        self.calls.append("head")
        return self.tokens[hidden]
    def decode(self, tokens):
        return "".join({1: "A", 2: "B"}[t] for t in tokens)


@pytest.mark.parametrize("engine", ["llama_cpp", "pytorch"])
@pytest.mark.parametrize("local_head", [False, True])
def test_ll_and_dl_frozen_segments_single_head_no_graph(service, engine, local_head):
    ports = Ports()
    executor = LinearExecutor(ports)
    placement = plan(local_head=local_head, engine=engine)
    result = service.run(plan=placement, deadline=RequestDeadline.start(3),
        execute=lambda record: executor.run(placement, record, max_tokens=4, sampling=SamplingPolicy()), cleanup=lambda _r: True)
    assert result["response"] == "AB"
    assert result["metrics"]["stop_reason"] == "eos_token"
    assert result["metrics"]["fallback"] is False
    assert ports.calls.count("head") == (3 if local_head else 0)
    stages = [c[1] for c in ports.calls if isinstance(c, tuple) and c[0] == "stage"]
    assert [i.step for i in stages] == [0, 1, 2]
    assert len({i.attempt_id for i in stages}) == 1
    assert all(i.deployment_id == "deployment" and i.assignment_id == "tail" for i in stages)


@pytest.mark.parametrize("field,value", [("authority", "term2"), ("deployment_id", "old"), ("assignment_id", "old"),
     ("incarnation", "old"), ("connection", 2), ("attempt_id", "old"), ("request_id", "old"), ("step", 3)])
def test_each_stage_identity_field_is_fenced(field, value):
    placement = plan()
    identity = StageIdentity.for_segment(placement, placement.segments[1], "r", "a", 0)
    with pytest.raises(ExecutionError, match="identity_mismatch"):
        LinearExecutor(Ports()).accept(identity, StageReceipt(replace(identity, **{field: value}), "abc", 1))


def test_duplicate_result_is_idempotent_but_conflicting_digest_is_rejected():
    placement = plan()
    identity = StageIdentity.for_segment(placement, placement.segments[1], "r", "a", 0)
    executor = LinearExecutor(Ports())
    receipt = StageReceipt(identity, "abc", 1)
    assert executor.accept(identity, receipt) == executor.accept(identity, receipt) == 1
    with pytest.raises(ExecutionError, match="result_conflict"):
        executor.accept(identity, StageReceipt(identity, "xyz", 2))


@pytest.mark.parametrize("sampling", [SamplingPolicy(0.7, 1), SamplingPolicy(0.7, 0.2)])
def test_sampling_refused_before_any_native_work(sampling):
    ports = Ports()
    with pytest.raises(ExecutionError, match="sampling_unsupported"):
        LinearExecutor(ports).run(plan(), {}, max_tokens=2, sampling=sampling)
    assert ports.calls == []


@pytest.mark.parametrize("change", ["gap", "whole", "double_head", "double_embedding"])
def test_invalid_plan_cannot_materialize(change):
    placement = plan()
    first, last = placement.segments
    if change == "gap": last = replace(last, start=3)
    if change == "whole": first = replace(first, end=4)
    if change == "double_head": first = replace(first, head=True)
    if change == "double_embedding": last = replace(last, embedding=True)
    with pytest.raises(ExecutionError):
        replace(placement, segments=(first, last))


def test_cancelled_and_partial_output_never_commit_conversation(service):
    from chat_context import ConversationContextService, ConversationContextConflict
    rows = []
    context = ConversationContextService(load_history=lambda _s: [], persist_turn=lambda *args, **kw: rows.append(args) or True)
    prepared = context.prepare("s")
    ready, release = threading.Event(), threading.Event()
    response, thread = launch(service, "one", lambda _r: (ready.set(), release.wait(4), {"response": "partial"})[-1])
    assert ready.wait(3)
    assert service.cancel("one")
    thread.join(2)
    if response.get("metrics", {}).get("request", {}).get("outcome") == "completed":
        context.commit_turn("s", "q", response["response"], expected_revision=prepared.revision)
    assert not rows
    release.set()
    wait_for(lambda: service.snapshot()["active"] is None)
    context.commit_turn("s", "q", "ok", expected_revision=prepared.revision)
    with pytest.raises(ConversationContextConflict):
        context.commit_turn("s", "q2", "stale", expected_revision=prepared.revision)
    assert len(rows) == 1


def test_deadline_and_terminal_compatibility_imports_share_types():
    import request_deadline
    import request_outcome
    from qlh_core.contracts.outcome import TerminalOutcomeLatch
    assert request_deadline.RequestDeadline is RequestDeadline
    assert request_outcome.TerminalOutcomeLatch is TerminalOutcomeLatch
    assert request_outcome.REASON_REQUEST_DEADLINE_EXCEEDED is request_deadline.REQUEST_DEADLINE_EXCEEDED


def test_journal_write_failure_prevents_execution_and_future_admission(service, monkeypatch):
    service.start()
    monkeypatch.setattr(service.repository, "transact", lambda _mutation: (_ for _ in ()).throw(OSError("disk full")))
    result = service.run(plan=plan(), deadline=RequestDeadline.start(2),
                         execute=lambda _r: pytest.fail("native work without durability"), cleanup=lambda _r: True)
    assert result["reason_code"] == "request_journal_write_failed"
    assert service.snapshot()["active"] is None
    assert service.run(plan=plan(), deadline=RequestDeadline.start(2), execute=lambda _r: {},
                       cleanup=lambda _r: True)["reason_code"] == "request_journal_write_failed"


def test_journal_failure_after_native_return_still_cleans_and_wakes_waiter(service, monkeypatch):
    cleaned = []
    def execute(_record):
        monkeypatch.setattr(service.repository, "transact", lambda _mutation: (_ for _ in ()).throw(OSError("disk full")))
        return {"response": "unsafe completion"}
    result = service.run(plan=plan(), deadline=RequestDeadline.start(2), execute=execute,
                         cleanup=lambda _r: cleaned.append(True) or True)
    wait_for(lambda: service.snapshot()["active"] is None)
    assert cleaned == [True]
    assert result["reason_code"] == "request_journal_write_failed"


def test_retry_uses_new_request_attempt_and_does_not_resume_old_kv(service):
    attempts = []
    def execute(record):
        attempts.append(record["attempt_id"])
        return {"response": "ok"}
    for _ in range(2):
        assert service.run(plan=plan(), deadline=RequestDeadline.start(2), execute=execute,
                           cleanup=lambda _r: True)["response"] == "ok"
    assert len(set(attempts)) == 2


def test_request_domain_is_independent_from_workflow_and_floor_cannot_disappear(repository):
    from cluster_auto_role_runtime import JournalPrefixAuthority, AutoRoleRuntimeError
    repo, _ = repository
    service = RequestService(repo, boot_id="boot")
    service.start()
    prefix = JournalPrefixAuthority(SimpleNamespace(recovery_status=lambda _domain: {"verified": False}))
    prefix.requests = service
    held = prefix.held(REQUEST_DOMAIN)
    assert held is not None and held.workflow_id == REQUEST_DOMAIN
    prefix.floor_provider = lambda: (held,)
    assert prefix.propose() == (held,)
    prefix.enforce(SimpleNamespace(commit_watermark=(held,)))
    prefix.requests = None
    with pytest.raises(AutoRoleRuntimeError, match="required recovery floor"):
        prefix.propose()
    with pytest.raises(AutoRoleRuntimeError, match="owner recovery domain"):
        prefix.enforce(SimpleNamespace(commit_watermark=(held,)))
    service.close()


def test_cancelling_queued_request_immediately_releases_capacity(service):
    entered, release = threading.Event(), threading.Event()
    one, t1 = launch(service, "one", lambda _r: (entered.set(), release.wait(4), {"response": "ok"})[-1])
    assert entered.wait(3)
    two, t2 = launch(service, "two", lambda _r: pytest.fail("cancelled queued request executed"))
    wait_for(lambda: service.snapshot()["queue"] == ["two"])
    assert service.cancel("two")
    t2.join(2)
    assert service.snapshot()["queue"] == []
    assert two["reason_code"] == "generation_cancelled"
    assert service.snapshot()["requests"]["two"]["execution_state"] == "stopped"
    release.set()
    t1.join(3)


def test_later_journal_failure_preserves_durable_completed_terminal(service, monkeypatch):
    original = service.repository.transact
    committed = threading.Event()
    def write(mutation):
        if committed.is_set():
            raise OSError("diagnostic write failed after durable completion")
        state = original(mutation)
        if any(r.get("terminal") == "completed" for r in state["requests"].values()):
            committed.set()
        return state
    monkeypatch.setattr(service.repository, "transact", write)
    result = service.run(plan=plan(), deadline=RequestDeadline.start(2),
                         execute=lambda _r: {"response": "ok"}, cleanup=lambda _r: True)
    wait_for(lambda: service.snapshot()["active"] is None)
    assert committed.is_set()
    assert result["response"] == "ok"
    record = result["metrics"]["request"]
    assert (record["terminal"], record["outcome"]) == ("completed", "completed")
    persisted = next(iter(service.repository.read()["requests"].values()))
    assert (persisted["terminal"], persisted["outcome"]) == ("completed", "completed")
    refused = service.run(plan=plan(), deadline=RequestDeadline.start(2),
                          execute=lambda _r: pytest.fail("admitted after journal failure"), cleanup=lambda _r: True)
    assert refused["reason_code"] == "request_journal_write_failed"
