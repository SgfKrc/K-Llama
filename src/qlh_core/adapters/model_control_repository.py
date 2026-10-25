"""SQLite model control transactions: state, outbox, debts and audit commit together.

The complete model-control aggregate has its own recovery domain. Local commits
are explicitly local_sqlite; this adapter never upgrades them to quorum commits.
"""
from __future__ import annotations

import atexit
from copy import deepcopy
import json
import os
import threading
import time
from qlh_core.contracts.deployment import MODEL_CONTROL_DOMAIN, canonical, ModelControlError
from qlh_core.adapters.sqlite_journal import SnapshotJournal
from qlh_core.adapters.model_control_retention import (
    guard_retired_records, initialize_retention, retain_model_control,
)


# ★ 2026-10-25（实验探针，默认关闭）：PERF-CONTROL-READ-01 要测「#132 的完整读放大」
#   —— 每个普通 decode 步究竟读了几次完整控制面快照、每次多少字节。生产代码里没有
#   计数器，而票要的正是这两个数，所以在这里放一个**受环境变量控制、默认零开销**的探针：
#       QLH_PROFILE_CONTROL_READS=<输出文件路径>
#   设置后，每次 `read()`（新建连接 + sqlite_master 探测）与 `_read()`（读整行 payload
#   并 json.loads）各记一条 `monotonic 时间戳 / 类别 / payload 字节 / 线程名`；按 decode
#   step 的时间窗口聚合可得「每步 N 次、合计 M 字节」，按**线程名**聚合则直接分出
#   「查询读取」（`qlh-tui-backend`：TUI 轮询投影）与「执行读取」（`linear-request-executor`
#   / 请求线程：逐步校验）各自的份额 —— 这正是票要求「分开测试执行读取、查询读取及
#   两者组合」的那两个量，且无需重启进程或做外部 A/B。
#   不设置时 `_profile_control_read` 立即返回，判定代价只是一个字符串真值测试。
#   记录先在内存攒批（阈值取小，保证长会话也能连续落盘），或进程退出时写盘。
_PROFILE_PATH = os.environ.get("QLH_PROFILE_CONTROL_READS", "")
_PROFILE_RECORDS: list[tuple[float, str, int, str]] = []
_PROFILE_LOCK = threading.Lock()
_PROFILE_BATCH = 256


def _flush_profile_control_reads() -> None:
    if not _PROFILE_PATH or not _PROFILE_RECORDS:
        return
    try:
        with open(_PROFILE_PATH, "a", encoding="utf-8") as handle:
            handle.writelines(f"{stamp:.6f}\t{kind}\t{size}\t{thread}\n"
                              for stamp, kind, size, thread in _PROFILE_RECORDS)
        _PROFILE_RECORDS.clear()
    except OSError:
        pass


def _profile_control_read(kind: str, size: int) -> None:
    if not _PROFILE_PATH:
        return
    thread = threading.current_thread().name
    with _PROFILE_LOCK:
        _PROFILE_RECORDS.append((time.monotonic(), kind, size, thread))
        if len(_PROFILE_RECORDS) >= _PROFILE_BATCH:
            _flush_profile_control_reads()


atexit.register(_flush_profile_control_reads)


def empty_state():
    return {"schema_version": 1, "revision": 0, "boot_id": "", "selected_operation": "",
            "active_deployment": "", "operations": {}, "supplies": {}, "deployments": {},
            "outbox": {}, "cleanup": {}, "artifact_cleanup": {}, "lifecycle": None, "commit_scope": "local_sqlite"}


class SqliteModelControlRepository:
    def __init__(self, *, connect, transaction, max_events=64, max_event_bytes=8 * 1024 * 1024):
        self._connect = connect
        self._transaction = transaction
        self._journal = SnapshotJournal("model_control_events", domain=MODEL_CONTROL_DOMAIN,
            error=lambda: ModelControlError("model_control_journal_corrupt"),
            max_events=max_events, max_bytes=max_event_bytes)

    @staticmethod
    def _initialize(connection):
        connection.execute("CREATE TABLE IF NOT EXISTS model_control_state (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS model_control_events (sequence INTEGER PRIMARY KEY, domain_id TEXT NOT NULL, previous_digest TEXT NOT NULL, digest TEXT NOT NULL, payload TEXT NOT NULL)")

    @staticmethod
    def _read(connection):
        row = connection.execute("SELECT payload FROM model_control_state WHERE id=1").fetchone()
        payload = row["payload"] if row else ""
        _profile_control_read("row", len(payload))
        value = json.loads(payload) if row else empty_state()
        if value.get("schema_version") != 1:
            raise ModelControlError("model_control_schema_invalid")
        return value

    def read(self):
        # No schema creation or state repair during queries.
        _profile_control_read("connect", 0)
        connection = self._connect()
        try:
            present = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_control_state'").fetchone()
            return self._read(connection) if present else empty_state()
        finally:
            connection.close()

    def transact(self, mutation, *, domain=MODEL_CONTROL_DOMAIN):
        with self._transaction() as connection:
            self._initialize(connection)
            self._journal.initialize(connection)
            self._journal.compact(connection, self._read)
            initialize_retention(connection)
            state = self._read(connection)
            self._journal.verify_head(connection, state)
            before_state = deepcopy(state)
            before = canonical(state)
            mutation(state)
            guard_retired_records(connection, before_state, state)
            retain_model_control(connection, state)
            if canonical(state) == before:
                return deepcopy(state)
            state["revision"] += 1
            self._journal.append(connection, state, domain=domain)
            connection.execute("INSERT INTO model_control_state VALUES (1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (canonical(state),))
            self._journal.compact(connection, self._read)
        return deepcopy(state)

    def prepare_recovery(self):
        with self._transaction() as connection:
            status = self._journal.verify(connection, self._read)
            self._initialize(connection)
            self._journal.initialize(connection)
            self._journal.compact(connection, self._read, verified=True)
            return status

    def recovery_status(self):
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            return self._journal.verify(connection, self._read)
        finally:
            connection.rollback()
            connection.close()
