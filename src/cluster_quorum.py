"""Durable quorum term, vote and lease protocol for P4.5.

This module is deliberately transport-agnostic.  ``QuorumVoter`` represents
one authenticated voter and persists only its own control metadata.  A later
TCP/API adapter may call the same methods, but this ticket does not enable
automatic roles or wire the protocol into control writes.
"""

from __future__ import annotations

import base64
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from cluster_control_contract import (
    CommitWatermark,
    ControlContractError,
    QuorumCertificate,
    VoterSet,
    VoterSignature,
    validate_certificate,
    sign_certificate,
)


QUORUM_SCHEMA_VERSION = 1
QUORUM_ERROR_CODES = frozenset(
    {
        "quorum_unavailable",
        "quorum_requires_witness",
        "quorum_term_conflict",
        "quorum_term_stale",
        "quorum_active_lease",
        "quorum_vote_rejected",
        "quorum_duplicate_vote",
        "quorum_expiry_regressed",
        "quorum_certificate_conflict",
        "quorum_ledger_corrupt",
        "quorum_invalid_identity",
        "quorum_invalid_leader",
        "quorum_invalid_policy",
        # ★ #93：voter 在签名前必须确认**自己耐久持有**证书里提议的已提交前缀。
        "quorum_watermark_unheld",
        "quorum_watermark_unavailable",
    }
)


class QuorumError(RuntimeError):
    """Stable fail-closed quorum protocol error."""

    def __init__(self, code: str, message: str) -> None:
        if code not in QUORUM_ERROR_CODES:
            raise ValueError(f"unknown quorum error code: {code}")
        self.code = code
        super().__init__(message)


def _now_ms(value: int | None) -> int:
    if value is None:
        return int(time.time() * 1000)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QuorumError("quorum_invalid_policy", "now_ms must be a non-negative integer")
    return value


def _safe_duration(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QuorumError("quorum_invalid_policy", "lease duration must be positive")
    return value


@dataclass(frozen=True)
class QuorumPolicy:
    """Availability policy kept separate from certificate validity."""

    lease_duration_ms: int = 10_000

    def __post_init__(self) -> None:
        _safe_duration(self.lease_duration_ms)

    def election_allowed(self, voter_count: int) -> bool:
        """Require a third configured voter for automatic quorum operation."""
        return voter_count >= 3


@dataclass(frozen=True)
class QuorumOutcome:
    accepted: bool
    reason: str
    term: int | None = None
    certificate: QuorumCertificate | None = None
    contacted_voters: tuple[str, ...] = ()
    signed_voters: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "term": self.term,
            "certificate": self.certificate.to_dict() if self.certificate else None,
            "contacted_voters": list(self.contacted_voters),
            "signed_voters": list(self.signed_voters),
        }


@dataclass(frozen=True)
class VoterLedgerSnapshot:
    voter_id: str
    cluster_id: str
    voter_set_epoch: int
    promised_term: int
    promised_leader: str
    active_term: int
    active_leader: str
    active_lease_id: str
    active_expires_at_ms: int | None
    active_certificate_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "voter_id": self.voter_id,
            "cluster_id": self.cluster_id,
            "voter_set_epoch": self.voter_set_epoch,
            "promised_term": self.promised_term,
            "promised_leader": self.promised_leader,
            "active_term": self.active_term,
            "active_leader": self.active_leader,
            "active_lease_id": self.active_lease_id,
            "active_expires_at_ms": self.active_expires_at_ms,
            "active_certificate_digest": self.active_certificate_digest,
        }


class SQLiteVoterLedger:
    """One voter's durable term and vote ledger."""

    def __init__(
        self,
        path: str | Path,
        *,
        voter_id: str,
        cluster_id: str,
        voter_set_epoch: int,
    ) -> None:
        self.path = os.path.abspath(os.path.expanduser(str(path)))
        self.voter_id = str(voter_id)
        self.cluster_id = str(cluster_id)
        self.voter_set_epoch = int(voter_set_epoch)
        if not self.voter_id or not self.cluster_id or self.voter_set_epoch < 0:
            raise QuorumError("quorum_invalid_identity", "voter ledger identity is invalid")
        self._lock = threading.RLock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS quorum_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS quorum_term_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    promised_term INTEGER NOT NULL CHECK (promised_term >= 0),
                    promised_leader TEXT NOT NULL,
                    active_term INTEGER NOT NULL CHECK (active_term >= 0),
                    active_leader TEXT NOT NULL,
                    active_lease_id TEXT NOT NULL,
                    active_expires_at_ms INTEGER,
                    active_certificate_digest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS quorum_votes (
                    voter_set_epoch INTEGER NOT NULL,
                    term INTEGER NOT NULL CHECK (term > 0),
                    voter_id TEXT NOT NULL,
                    leader_id TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    expires_at_ms INTEGER NOT NULL,
                    signature TEXT NOT NULL,
                    PRIMARY KEY (voter_set_epoch, term, voter_id)
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO quorum_term_state "
                "(id, promised_term, promised_leader, active_term, active_leader, "
                "active_lease_id, active_expires_at_ms, active_certificate_digest) "
                "VALUES (1, 0, '', 0, '', '', NULL, '')"
            )
            metadata = {
                "schema_version": str(QUORUM_SCHEMA_VERSION),
                "cluster_id": self.cluster_id,
                "voter_id": self.voter_id,
                "voter_set_epoch": str(self.voter_set_epoch),
            }
            for key, value in metadata.items():
                row = connection.execute(
                    "SELECT value FROM quorum_metadata WHERE key = ?", (key,)
                ).fetchone()
                if row is not None and str(row["value"]) != value:
                    raise QuorumError(
                        "quorum_ledger_corrupt",
                        f"voter ledger metadata conflict for {key}",
                    )
                connection.execute(
                    "INSERT OR IGNORE INTO quorum_metadata(key, value) VALUES (?, ?)",
                    (key, value),
                )

    def _state(self, connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute(
            "SELECT promised_term, promised_leader, active_term, active_leader, "
            "active_lease_id, active_expires_at_ms, active_certificate_digest "
            "FROM quorum_term_state WHERE id = 1"
        ).fetchone()
        if row is None:
            raise QuorumError("quorum_ledger_corrupt", "voter term state is missing")
        return row

    def snapshot(self) -> VoterLedgerSnapshot:
        with self._lock, self._connect() as connection:
            row = self._state(connection)
            return VoterLedgerSnapshot(
                voter_id=self.voter_id,
                cluster_id=self.cluster_id,
                voter_set_epoch=self.voter_set_epoch,
                promised_term=int(row["promised_term"]),
                promised_leader=str(row["promised_leader"]),
                active_term=int(row["active_term"]),
                active_leader=str(row["active_leader"]),
                active_lease_id=str(row["active_lease_id"]),
                active_expires_at_ms=(
                    None if row["active_expires_at_ms"] is None
                    else int(row["active_expires_at_ms"])
                ),
                active_certificate_digest=str(row["active_certificate_digest"]),
            )

    def reserve_term(self, leader_id: str, *, now_ms: int | None = None) -> int:
        now = _now_ms(now_ms)
        leader_id = str(leader_id)
        if not leader_id:
            raise QuorumError("quorum_invalid_identity", "leader_id must not be empty")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._state(connection)
            active_expiry = row["active_expires_at_ms"]
            if active_expiry is not None and int(active_expiry) > now:
                raise QuorumError(
                    "quorum_active_lease",
                    "voter has an unexpired certificate lease",
                )
            promised_term = int(row["promised_term"])
            promised_leader = str(row["promised_leader"])
            term = max(promised_term, int(row["active_term"])) + 1
            connection.execute(
                "UPDATE quorum_term_state SET promised_term = ?, promised_leader = ? WHERE id = 1",
                (term, leader_id),
            )
            connection.execute("COMMIT")
            return term

    def prepare(self, leader_id: str, term: int, *, now_ms: int | None = None) -> None:
        now = _now_ms(now_ms)
        if isinstance(term, bool) or not isinstance(term, int) or term <= 0:
            raise QuorumError("quorum_term_stale", "term must be positive")
        leader_id = str(leader_id)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._state(connection)
            active_expiry = row["active_expires_at_ms"]
            active_term = int(row["active_term"])
            if active_expiry is not None and int(active_expiry) > now and term > active_term:
                raise QuorumError(
                    "quorum_active_lease",
                    "cannot prepare a new term while a lease is active",
                )
            promised_term = int(row["promised_term"])
            promised_leader = str(row["promised_leader"])
            if term < promised_term:
                raise QuorumError("quorum_term_stale", "term is below the durable promise")
            if term == promised_term and promised_leader not in {"", leader_id}:
                raise QuorumError(
                    "quorum_term_conflict",
                    "term is already promised to another leader",
                )
            if term > promised_term or promised_leader != leader_id:
                connection.execute(
                    "UPDATE quorum_term_state SET promised_term = ?, promised_leader = ? WHERE id = 1",
                    (term, leader_id),
                )
            connection.execute("COMMIT")

    def sign_vote(
        self,
        certificate: QuorumCertificate,
        *,
        private_key: Any,
        now_ms: int | None = None,
    ) -> VoterSignature:
        now = _now_ms(now_ms)
        if certificate.voter_set_epoch != self.voter_set_epoch or certificate.cluster_id != self.cluster_id:
            raise QuorumError("quorum_vote_rejected", "certificate is for another voter ledger")
        if certificate.expires_at_ms <= now:
            raise QuorumError("quorum_vote_rejected", "certificate lease is already expired")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._state(connection)
            if int(row["promised_term"]) != certificate.term or str(row["promised_leader"]) != certificate.leader_id:
                raise QuorumError("quorum_vote_rejected", "voter has not prepared this leader and term")
            active_expiry = row["active_expires_at_ms"]
            if active_expiry is not None and int(active_expiry) > now:
                if not (
                    int(row["active_term"]) == certificate.term
                    and str(row["active_leader"]) == certificate.leader_id
                    and str(row["active_lease_id"]) == certificate.lease_id
                ):
                    raise QuorumError("quorum_active_lease", "another lease is active")
            existing = connection.execute(
                "SELECT leader_id, lease_id, expires_at_ms FROM quorum_votes "
                "WHERE voter_set_epoch = ? AND term = ? AND voter_id = ?",
                (certificate.voter_set_epoch, certificate.term, self.voter_id),
            ).fetchone()
            if existing is not None:
                if str(existing["leader_id"]) != certificate.leader_id or str(existing["lease_id"]) != certificate.lease_id:
                    raise QuorumError("quorum_duplicate_vote", "voter already voted for another leader in this term")
                if int(existing["expires_at_ms"]) > certificate.expires_at_ms:
                    raise QuorumError("quorum_expiry_regressed", "lease expiry cannot move backwards")
            signature = sign_certificate(
                certificate, voter_id=self.voter_id, private_key=private_key
            )
            connection.execute(
                "INSERT INTO quorum_votes(voter_set_epoch, term, voter_id, leader_id, lease_id, expires_at_ms, signature) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(voter_set_epoch, term, voter_id) DO UPDATE SET "
                "leader_id = excluded.leader_id, lease_id = excluded.lease_id, "
                "expires_at_ms = excluded.expires_at_ms, signature = excluded.signature",
                (
                    certificate.voter_set_epoch, certificate.term, self.voter_id,
                    certificate.leader_id, certificate.lease_id,
                    certificate.expires_at_ms, signature.signature,
                ),
            )
            connection.execute("COMMIT")
            return signature

    def commit_certificate(self, certificate: QuorumCertificate, voter_set: VoterSet, *, now_ms: int | None = None) -> None:
        now = _now_ms(now_ms)
        validate_certificate(certificate, voter_set, now_ms=now)
        signature_ids = {signature.voter_id for signature in certificate.signatures}
        if self.voter_id not in signature_ids:
            raise QuorumError("quorum_vote_rejected", "certificate does not contain this voter")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._state(connection)
            local_vote = connection.execute(
                "SELECT leader_id, lease_id, expires_at_ms, signature FROM quorum_votes "
                "WHERE voter_set_epoch = ? AND term = ? AND voter_id = ?",
                (certificate.voter_set_epoch, certificate.term, self.voter_id),
            ).fetchone()
            local_signature = next(
                signature for signature in certificate.signatures
                if signature.voter_id == self.voter_id
            )
            if (
                local_vote is None
                or str(local_vote["leader_id"]) != certificate.leader_id
                or str(local_vote["lease_id"]) != certificate.lease_id
                or int(local_vote["expires_at_ms"]) != certificate.expires_at_ms
                or str(local_vote["signature"]) != local_signature.signature
            ):
                raise QuorumError(
                    "quorum_vote_rejected",
                    "certificate is not backed by this voter's durable vote",
                )
            active_expiry = row["active_expires_at_ms"]
            if active_expiry is not None and int(active_expiry) > now:
                same = (
                    int(row["active_term"]) == certificate.term
                    and str(row["active_leader"]) == certificate.leader_id
                    and str(row["active_lease_id"]) == certificate.lease_id
                )
                if not same:
                    raise QuorumError("quorum_active_lease", "cannot replace an active certificate")
            if certificate.term < int(row["active_term"]):
                raise QuorumError("quorum_term_stale", "certificate term is below active term")
            connection.execute(
                "UPDATE quorum_term_state SET active_term = ?, active_leader = ?, "
                "active_lease_id = ?, active_expires_at_ms = ?, active_certificate_digest = ? WHERE id = 1",
                (
                    certificate.term, certificate.leader_id, certificate.lease_id,
                    certificate.expires_at_ms, certificate.digest(),
                ),
            )
            connection.execute("COMMIT")


class QuorumVoter:
    """Authenticated voter facade used by the collector and future transport."""

    def __init__(
        self,
        *,
        voter_id: str,
        private_key: Any,
        voter_set: VoterSet,
        ledger: SQLiteVoterLedger,
        watermark_source: Callable[[str], "CommitWatermark | None"] | None = None,
    ) -> None:
        if (
            voter_id not in voter_set.voter_map
            or ledger.voter_id != voter_id
            or ledger.cluster_id != voter_set.cluster_id
            or ledger.voter_set_epoch != voter_set.voter_set_epoch
        ):
            raise QuorumError("quorum_invalid_identity", "voter identity does not match voter set")
        try:
            from cryptography.hazmat.primitives import serialization

            public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            encoded_public_key = base64.urlsafe_b64encode(public_key).decode("ascii").rstrip("=")
        except Exception as exc:
            raise QuorumError("quorum_invalid_identity", "voter private key is invalid") from exc
        if encoded_public_key != voter_set.voter_map[voter_id].public_key:
            raise QuorumError("quorum_invalid_identity", "voter private key does not match identity")
        self.voter_id = voter_id
        self.private_key = private_key
        self.voter_set = voter_set
        self.ledger = ledger
        #: #93：查询「本 voter **自己耐久持有**的某 workflow 已提交前缀」的钩子。缺失时，
        #  只要证书里带了 `commit_watermark` 就必须拒签（fail-closed，不给错误的一致性承诺）。
        self.watermark_source = watermark_source

    def reserve_term(self, leader_id: str, *, now_ms: int | None = None) -> int:
        if leader_id not in self.voter_set.voter_map:
            raise QuorumError("quorum_invalid_leader", "leader_id is not a voter")
        return self.ledger.reserve_term(leader_id, now_ms=now_ms)

    def prepare(self, leader_id: str, term: int, *, now_ms: int | None = None) -> None:
        if leader_id not in self.voter_set.voter_map:
            raise QuorumError("quorum_invalid_leader", "leader_id is not a voter")
        self.ledger.prepare(leader_id, term, now_ms=now_ms)

    def _require_held_watermark(self, certificate: QuorumCertificate) -> None:
        """#93：证书提议了「已提交前缀」时，voter 必须**自己耐久持有**它才允许签名。

        持有判据（链摘要保证同 sequence 只有唯一合法点）：
        - 本地已到**更高的** sequence ⇒ 前缀必然包含提议点 ⇒ 通过；
        - 恰为同一 sequence ⇒ digest 必须一致（不一致即分叉 ⇒ 拒绝）；
        - 落后、无记录、或压根没有耐久日志来源 ⇒ 拒签（fail-closed）。
        """
        watermarks = tuple(certificate.commit_watermark)
        if not watermarks:
            return
        if self.watermark_source is None:
            raise QuorumError(
                "quorum_watermark_unavailable",
                "certificate declares a commit watermark but this voter has no durable journal",
            )
        for watermark in watermarks:
            held = self.watermark_source(watermark.workflow_id)
            if held is None:
                raise QuorumError(
                    "quorum_watermark_unheld",
                    "voter does not durably hold the declared committed prefix",
                )
            if held.last_sequence < watermark.last_sequence:
                raise QuorumError(
                    "quorum_watermark_unheld",
                    "voter durable prefix is behind the declared commit watermark",
                )
            if (
                held.last_sequence == watermark.last_sequence
                and held.checkpoint_digest != watermark.checkpoint_digest
            ):
                raise QuorumError(
                    "quorum_watermark_unheld",
                    "voter durable prefix diverges from the declared commit watermark",
                )

    def sign_vote(self, certificate: QuorumCertificate, *, now_ms: int | None = None) -> VoterSignature:
        if certificate.leader_id not in self.voter_set.voter_map:
            raise QuorumError("quorum_invalid_leader", "leader_id is not a voter")
        self._require_held_watermark(certificate)
        return self.ledger.sign_vote(certificate, private_key=self.private_key, now_ms=now_ms)

    def commit_certificate(self, certificate: QuorumCertificate, *, now_ms: int | None = None) -> None:
        self.ledger.commit_certificate(certificate, self.voter_set, now_ms=now_ms)

    def snapshot(self) -> VoterLedgerSnapshot:
        return self.ledger.snapshot()


class QuorumCollector:
    """Collect a strict-majority certificate without enabling any role."""

    def __init__(
        self,
        *,
        voter_set: VoterSet,
        voters: Mapping[str, QuorumVoter],
        policy: QuorumPolicy | None = None,
    ) -> None:
        self.voter_set = voter_set
        self.voters = dict(voters)
        if set(self.voters) - set(voter_set.voter_map):
            raise QuorumError("quorum_invalid_identity", "collector contains an unknown voter")
        self.policy = policy or QuorumPolicy()

    def _available(self, voter_ids: Sequence[str] | None) -> tuple[str, ...]:
        ids = self.voters.keys() if voter_ids is None else voter_ids
        return tuple(sorted({str(voter_id) for voter_id in ids if str(voter_id) in self.voters}))

    def _blocked(self, available: tuple[str, ...]) -> QuorumOutcome | None:
        if not self.policy.election_allowed(len(self.voter_set.voters)):
            return QuorumOutcome(False, "quorum_requires_witness", contacted_voters=available)
        if len(available) < self.voter_set.quorum_size:
            return QuorumOutcome(False, "quorum_unavailable", contacted_voters=available)
        return None

    def acquire(
        self,
        leader_id: str,
        *,
        available_voter_ids: Sequence[str] | None = None,
        now_ms: int | None = None,
        lease_id: str | None = None,
        commit_watermark: Sequence[CommitWatermark] = (),
    ) -> QuorumOutcome:
        now = _now_ms(now_ms)
        if leader_id not in self.voter_set.voter_map:
            raise QuorumError("quorum_invalid_leader", "leader_id is not a voter")
        available = self._available(available_voter_ids)
        blocked = self._blocked(available)
        if blocked is not None:
            return blocked
        reserved: dict[str, int] = {}
        for voter_id in available:
            try:
                reserved[voter_id] = self.voters[voter_id].reserve_term(leader_id, now_ms=now)
            except QuorumError:
                continue
        if len(reserved) < self.voter_set.quorum_size:
            return QuorumOutcome(False, "quorum_unavailable", contacted_voters=available)
        term = max(reserved.values())
        prepared: list[str] = []
        for voter_id in available:
            try:
                self.voters[voter_id].prepare(leader_id, term, now_ms=now)
                prepared.append(voter_id)
            except QuorumError:
                continue
        if len(prepared) < self.voter_set.quorum_size:
            return QuorumOutcome(False, "quorum_unavailable", term=term, contacted_voters=available)
        selected_lease_id = lease_id or f"lease-{uuid.uuid4().hex}"
        # #93：`commit_watermark` 由 leader **提议**（它认为已被多数耐久持有的前缀），
        # voter 会在 `sign_vote` 里先校验自己是否真的耐久持有（否则拒签）⇒ 拿到多数签名
        # 就等于「多数一致前缀」，无需再事后聚合。
        unsigned = QuorumCertificate(
            cluster_id=self.voter_set.cluster_id,
            voter_set_epoch=self.voter_set.voter_set_epoch,
            term=term,
            leader_id=leader_id,
            lease_id=selected_lease_id,
            expires_at_ms=now + self.policy.lease_duration_ms,
            signatures=tuple(),
            commit_watermark=tuple(commit_watermark),
        )
        signatures: list[VoterSignature] = []
        watermark_error = ""
        for voter_id in prepared:
            try:
                signatures.append(self.voters[voter_id].sign_vote(unsigned, now_ms=now))
            except QuorumError as exc:
                # 区分「多数不可用」与「多数不持有被提议的已提交前缀」：后者必须让调用方
                # 看得见（否则会误判成容量/网络问题而重试，掩盖一致性风险）。
                code = str(getattr(exc, "code", ""))
                if code.startswith("quorum_watermark"):
                    watermark_error = code
                continue
        if len(signatures) < self.voter_set.quorum_size:
            return QuorumOutcome(
                False, watermark_error or "quorum_unavailable", term=term,
                contacted_voters=available,
                signed_voters=tuple(sorted(signature.voter_id for signature in signatures)),
            )
        certificate = unsigned.with_signatures(signatures)
        validate_certificate(certificate, self.voter_set, now_ms=now)
        try:
            for signature in signatures:
                self.voters[signature.voter_id].commit_certificate(certificate, now_ms=now)
        except QuorumError:
            return QuorumOutcome(
                False, "quorum_unavailable", term=term,
                contacted_voters=available,
                signed_voters=tuple(sorted(signature.voter_id for signature in signatures)),
            )
        return QuorumOutcome(
            True, "quorum_certificate_issued", term=term, certificate=certificate,
            contacted_voters=available,
            signed_voters=tuple(sorted(signature.voter_id for signature in signatures)),
        )

    def renew(
        self,
        certificate: QuorumCertificate,
        *,
        available_voter_ids: Sequence[str] | None = None,
        now_ms: int | None = None,
    ) -> QuorumOutcome:
        now = _now_ms(now_ms)
        available = self._available(available_voter_ids)
        blocked = self._blocked(available)
        if blocked is not None:
            return blocked
        try:
            validate_certificate(certificate, self.voter_set, now_ms=now)
        except (ControlContractError, TypeError, ValueError):
            return QuorumOutcome(False, "quorum_unavailable", term=certificate.term, contacted_voters=available)
        if now + self.policy.lease_duration_ms <= certificate.expires_at_ms:
            return QuorumOutcome(False, "quorum_unavailable", term=certificate.term, contacted_voters=available)
        prepared = []
        for voter_id in available:
            try:
                self.voters[voter_id].prepare(certificate.leader_id, certificate.term, now_ms=now)
                prepared.append(voter_id)
            except QuorumError:
                continue
        if len(prepared) < self.voter_set.quorum_size:
            return QuorumOutcome(False, "quorum_unavailable", term=certificate.term, contacted_voters=available)
        # #93：续期必须**保留**已提交前缀承诺（否则一次续期就把提交证明丢掉了）。
        renewed_unsigned = QuorumCertificate(
            cluster_id=certificate.cluster_id,
            voter_set_epoch=certificate.voter_set_epoch,
            term=certificate.term,
            leader_id=certificate.leader_id,
            lease_id=certificate.lease_id,
            expires_at_ms=now + self.policy.lease_duration_ms,
            signatures=tuple(),
            commit_watermark=tuple(certificate.commit_watermark),
        )
        signatures = []
        watermark_error = ""
        for voter_id in prepared:
            try:
                signatures.append(self.voters[voter_id].sign_vote(renewed_unsigned, now_ms=now))
            except QuorumError as exc:
                code = str(getattr(exc, "code", ""))
                if code.startswith("quorum_watermark"):
                    watermark_error = code
                continue
        if len(signatures) < self.voter_set.quorum_size:
            return QuorumOutcome(
                False, watermark_error or "quorum_unavailable",
                term=certificate.term, contacted_voters=available,
            )
        renewed = renewed_unsigned.with_signatures(signatures)
        validate_certificate(renewed, self.voter_set, now_ms=now)
        try:
            for signature in signatures:
                self.voters[signature.voter_id].commit_certificate(renewed, now_ms=now)
        except QuorumError:
            return QuorumOutcome(False, "quorum_unavailable", term=certificate.term, contacted_voters=available)
        return QuorumOutcome(
            True, "quorum_lease_renewed", term=renewed.term, certificate=renewed,
            contacted_voters=available,
            signed_voters=tuple(sorted(signature.voter_id for signature in signatures)),
        )


__all__ = [
    "QUORUM_ERROR_CODES",
    "QUORUM_SCHEMA_VERSION",
    "QuorumCollector",
    "QuorumError",
    "QuorumOutcome",
    "QuorumPolicy",
    "QuorumVoter",
    "SQLiteVoterLedger",
    "VoterLedgerSnapshot",
]
