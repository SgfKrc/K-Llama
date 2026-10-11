"""Versioned control-plane authority contract for P4.5.

This module freezes the data and admission boundary before quorum election is
implemented.  A static ``master`` or ``client`` role may still start, but no
role can turn a missing or non-current quorum certificate into a control
write.  Election, durable term allocation, and endpoint wiring belong to
later P4.5 tickets.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
except Exception as exc:  # pragma: no cover - depends on runtime packaging
    InvalidSignature = Exception  # type: ignore[assignment,misc]
    Ed25519PrivateKey = None  # type: ignore[assignment,misc]
    Ed25519PublicKey = None  # type: ignore[assignment,misc]
    _CRYPTO_IMPORT_ERROR = exc
else:
    _CRYPTO_IMPORT_ERROR = None


CONTROL_SCHEMA_VERSION = "qlh.cluster.control.v1"
CERTIFICATE_TYPE = "quorum_certificate"
VOTER_SET_TYPE = "voter_set"
SIGNATURE_ALGORITHM = "ed25519"
STATIC_ROLES = frozenset({"master", "client", "unknown"})
CONTROL_CONTRACT_ERROR_CODES = frozenset(
    {
        "control_certificate_missing",
        "control_certificate_invalid",
        "control_certificate_expired",
        "control_certificate_not_quorum",
        "control_certificate_stale",
        "control_certificate_conflict",
        "control_certificate_not_current",
        "control_certificate_cluster_mismatch",
        "control_certificate_epoch_mismatch",
        "control_certificate_leader_unknown",
        "control_certificate_signature_invalid",
        "control_certificate_duplicate_vote",
        "control_voter_unknown",
        "control_crypto_unavailable",
        "control_static_role_invalid",
        "control_cluster_invalid",
    }
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_B64URL = re.compile(r"^[A-Za-z0-9_-]+$")


class ControlContractError(ValueError):
    """Fail-closed control contract error with a stable public code."""

    def __init__(self, code: str, message: str) -> None:
        if code not in CONTROL_CONTRACT_ERROR_CODES:
            raise ValueError(f"unknown control contract error code: {code}")
        self.code = code
        super().__init__(message)


def _require_crypto() -> None:
    if Ed25519PrivateKey is None or Ed25519PublicKey is None:
        raise ControlContractError(
            "control_crypto_unavailable",
            f"Ed25519 support is unavailable: {_CRYPTO_IMPORT_ERROR}",
        )


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: Any, *, field: str, expected_length: int) -> bytes:
    if not isinstance(value, str) or not value or not _B64URL.fullmatch(value):
        raise ControlContractError(
            "control_certificate_invalid",
            f"{field} must be unpadded base64url text",
        )
    try:
        raw = base64.urlsafe_b64decode(value + ("=" * (-len(value) % 4)))
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise ControlContractError(
            "control_certificate_invalid",
            f"{field} is not valid base64url",
        ) from exc
    if len(raw) != expected_length:
        raise ControlContractError(
            "control_certificate_invalid",
            f"{field} has invalid length",
        )
    return raw


def _safe_id(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ControlContractError("control_certificate_invalid", f"{field} is invalid")
    return value


def _non_negative_int(value: Any, *, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControlContractError("control_certificate_invalid", f"{field} is invalid")
    if value < (1 if positive else 0):
        raise ControlContractError("control_certificate_invalid", f"{field} is invalid")
    return value


def _now_ms(value: int | None) -> int:
    if value is None:
        return int(time.time() * 1000)
    return _non_negative_int(value, field="now_ms")


@dataclass(frozen=True)
class VoterIdentity:
    """A trusted voter identity; the private key never enters this contract."""

    voter_id: str
    public_key: str

    def __post_init__(self) -> None:
        _safe_id(self.voter_id, field="voter_id")
        _unb64(self.public_key, field="public_key", expected_length=32)

    def to_dict(self) -> dict[str, str]:
        return {"voter_id": self.voter_id, "public_key": self.public_key}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VoterIdentity":
        if not isinstance(value, Mapping) or set(value) != {"voter_id", "public_key"}:
            raise ControlContractError("control_certificate_invalid", "voter identity fields are invalid")
        return cls(voter_id=value["voter_id"], public_key=value["public_key"])


@dataclass(frozen=True)
class VoterSet:
    """Versioned trust root used to validate a quorum certificate."""

    cluster_id: str
    voter_set_epoch: int
    voters: tuple[VoterIdentity, ...] | Mapping[str, str] | Sequence[VoterIdentity]
    schema_version: str = CONTROL_SCHEMA_VERSION
    document_type: str = VOTER_SET_TYPE

    def __post_init__(self) -> None:
        if self.schema_version != CONTROL_SCHEMA_VERSION or self.document_type != VOTER_SET_TYPE:
            raise ControlContractError("control_certificate_invalid", "voter set schema is unsupported")
        cluster_id = _safe_id(self.cluster_id, field="cluster_id")
        epoch = _non_negative_int(self.voter_set_epoch, field="voter_set_epoch")
        raw_voters: Sequence[VoterIdentity | Mapping[str, Any]]
        if isinstance(self.voters, Mapping):
            raw_voters = tuple(
                VoterIdentity(voter_id=str(voter_id), public_key=public_key)
                for voter_id, public_key in self.voters.items()
            )
        else:
            raw_voters = self.voters
        identities = tuple(
            voter if isinstance(voter, VoterIdentity) else VoterIdentity.from_dict(voter)
            for voter in raw_voters
        )
        if not identities:
            raise ControlContractError("control_certificate_invalid", "voter set cannot be empty")
        if len({voter.voter_id for voter in identities}) != len(identities):
            raise ControlContractError("control_certificate_invalid", "voter ids must be unique")
        object.__setattr__(self, "cluster_id", cluster_id)
        object.__setattr__(self, "voter_set_epoch", epoch)
        object.__setattr__(self, "voters", tuple(sorted(identities, key=lambda voter: voter.voter_id)))

    @property
    def quorum_size(self) -> int:
        """Return the strict-majority threshold for this voter set."""
        return len(self.voters) // 2 + 1

    @property
    def voter_map(self) -> dict[str, VoterIdentity]:
        return {voter.voter_id: voter for voter in self.voters}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "cluster_id": self.cluster_id,
            "voter_set_epoch": self.voter_set_epoch,
            "voters": [voter.to_dict() for voter in self.voters],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VoterSet":
        required = {"schema_version", "document_type", "cluster_id", "voter_set_epoch", "voters"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ControlContractError("control_certificate_invalid", "voter set fields are not canonical")
        if not isinstance(value["voters"], list):
            raise ControlContractError("control_certificate_invalid", "voters must be a list")
        return cls(
            cluster_id=value["cluster_id"],
            voter_set_epoch=value["voter_set_epoch"],
            voters=tuple(VoterIdentity.from_dict(item) for item in value["voters"]),
            schema_version=value["schema_version"],
            document_type=value["document_type"],
        )


@dataclass(frozen=True)
class VoterSignature:
    """One voter's signature over the certificate's canonical payload."""

    voter_id: str
    signature: str
    algorithm: str = SIGNATURE_ALGORITHM

    def __post_init__(self) -> None:
        _safe_id(self.voter_id, field="voter_id")
        if self.algorithm != SIGNATURE_ALGORITHM:
            raise ControlContractError("control_certificate_invalid", "signature algorithm is unsupported")
        _unb64(self.signature, field="signature", expected_length=64)

    def to_dict(self) -> dict[str, str]:
        return {
            "voter_id": self.voter_id,
            "algorithm": self.algorithm,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VoterSignature":
        required = {"voter_id", "algorithm", "signature"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ControlContractError("control_certificate_invalid", "signature fields are not canonical")
        return cls(
            voter_id=value["voter_id"],
            algorithm=value["algorithm"],
            signature=value["signature"],
        )


@dataclass(frozen=True)
class QuorumCertificate:
    """A quorum-certified writable lease for one cluster term."""

    cluster_id: str
    voter_set_epoch: int
    term: int
    leader_id: str
    lease_id: str
    expires_at_ms: int
    signatures: tuple[VoterSignature, ...] | Sequence[VoterSignature]
    schema_version: str = CONTROL_SCHEMA_VERSION
    document_type: str = CERTIFICATE_TYPE

    def __post_init__(self) -> None:
        if self.schema_version != CONTROL_SCHEMA_VERSION or self.document_type != CERTIFICATE_TYPE:
            raise ControlContractError("control_certificate_invalid", "certificate schema is unsupported")
        object.__setattr__(self, "cluster_id", _safe_id(self.cluster_id, field="cluster_id"))
        object.__setattr__(
            self,
            "voter_set_epoch",
            _non_negative_int(self.voter_set_epoch, field="voter_set_epoch"),
        )
        object.__setattr__(self, "term", _non_negative_int(self.term, field="term", positive=True))
        object.__setattr__(self, "leader_id", _safe_id(self.leader_id, field="leader_id"))
        object.__setattr__(self, "lease_id", _safe_id(self.lease_id, field="lease_id"))
        object.__setattr__(
            self,
            "expires_at_ms",
            _non_negative_int(self.expires_at_ms, field="expires_at_ms", positive=True),
        )
        signatures = tuple(self.signatures)
        if any(not isinstance(signature, VoterSignature) for signature in signatures):
            raise ControlContractError("control_certificate_invalid", "certificate signatures are invalid")
        object.__setattr__(
            self,
            "signatures",
            tuple(sorted(signatures, key=lambda signature: signature.voter_id)),
        )

    def signing_payload(self) -> dict[str, Any]:
        """Return the exact payload every voter signs; signatures are excluded."""
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "cluster_id": self.cluster_id,
            "voter_set_epoch": self.voter_set_epoch,
            "term": self.term,
            "leader_id": self.leader_id,
            "lease_id": self.lease_id,
            "expires_at_ms": self.expires_at_ms,
        }

    def signing_bytes(self) -> bytes:
        return _canonical(self.signing_payload())

    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict())).hexdigest()

    def with_signatures(self, signatures: Sequence[VoterSignature]) -> "QuorumCertificate":
        return QuorumCertificate(
            cluster_id=self.cluster_id,
            voter_set_epoch=self.voter_set_epoch,
            term=self.term,
            leader_id=self.leader_id,
            lease_id=self.lease_id,
            expires_at_ms=self.expires_at_ms,
            signatures=tuple(signatures),
            schema_version=self.schema_version,
            document_type=self.document_type,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.signing_payload(),
            "signatures": [signature.to_dict() for signature in self.signatures],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuorumCertificate":
        required = {
            "schema_version", "document_type", "cluster_id", "voter_set_epoch",
            "term", "leader_id", "lease_id", "expires_at_ms", "signatures",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ControlContractError("control_certificate_invalid", "certificate fields are not canonical")
        if not isinstance(value["signatures"], list):
            raise ControlContractError("control_certificate_invalid", "signatures must be a list")
        return cls(
            cluster_id=value["cluster_id"],
            voter_set_epoch=value["voter_set_epoch"],
            term=value["term"],
            leader_id=value["leader_id"],
            lease_id=value["lease_id"],
            expires_at_ms=value["expires_at_ms"],
            signatures=tuple(VoterSignature.from_dict(item) for item in value["signatures"]),
            schema_version=value["schema_version"],
            document_type=value["document_type"],
        )


def sign_certificate(
    certificate: QuorumCertificate,
    *,
    voter_id: str,
    private_key: Any,
) -> VoterSignature:
    """Sign a certificate payload for the later quorum collector."""
    _require_crypto()
    voter = _safe_id(voter_id, field="voter_id")
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ControlContractError("control_certificate_invalid", "private key is invalid")
    return VoterSignature(voter_id=voter, signature=_b64(private_key.sign(certificate.signing_bytes())))


@dataclass(frozen=True)
class CertificateValidation:
    certificate_digest: str
    valid_voter_ids: tuple[str, ...]
    quorum_required: int
    voter_set_epoch: int
    term: int


def validate_certificate(
    certificate: QuorumCertificate | Mapping[str, Any],
    voter_set: VoterSet,
    *,
    now_ms: int | None = None,
) -> CertificateValidation:
    """Validate identity, signature, epoch, expiry, and strict-majority rules."""
    if not isinstance(voter_set, VoterSet):
        raise ControlContractError("control_certificate_invalid", "voter set is invalid")
    cert = certificate if isinstance(certificate, QuorumCertificate) else QuorumCertificate.from_dict(certificate)
    if cert.cluster_id != voter_set.cluster_id:
        raise ControlContractError(
            "control_certificate_cluster_mismatch",
            "certificate cluster does not match the voter set",
        )
    if cert.voter_set_epoch != voter_set.voter_set_epoch:
        raise ControlContractError(
            "control_certificate_epoch_mismatch",
            "certificate voter-set epoch does not match the voter set",
        )
    voters = voter_set.voter_map
    if cert.leader_id not in voters:
        raise ControlContractError(
            "control_certificate_leader_unknown",
            "certificate leader is not a voter",
        )
    if cert.expires_at_ms <= _now_ms(now_ms):
        raise ControlContractError(
            "control_certificate_expired",
            "certificate lease has expired",
        )
    _require_crypto()
    valid_ids: list[str] = []
    seen: set[str] = set()
    for signature in cert.signatures:
        if signature.voter_id in seen:
            raise ControlContractError(
                "control_certificate_duplicate_vote",
                "a voter may sign a term at most once",
            )
        seen.add(signature.voter_id)
        identity = voters.get(signature.voter_id)
        if identity is None:
            raise ControlContractError(
                "control_voter_unknown",
                "certificate contains a signature from an unknown voter",
            )
        try:
            public_key = Ed25519PublicKey.from_public_bytes(
                _unb64(identity.public_key, field="public_key", expected_length=32)
            )
            public_key.verify(
                _unb64(signature.signature, field="signature", expected_length=64),
                cert.signing_bytes(),
            )
        except (InvalidSignature, ValueError, TypeError) as exc:
            raise ControlContractError(
                "control_certificate_signature_invalid",
                "certificate contains an invalid voter signature",
            ) from exc
        valid_ids.append(signature.voter_id)
    if len(valid_ids) < voter_set.quorum_size:
        raise ControlContractError(
            "control_certificate_not_quorum",
            "certificate does not have a strict voter majority",
        )
    return CertificateValidation(
        certificate_digest=cert.digest(),
        valid_voter_ids=tuple(sorted(valid_ids)),
        quorum_required=voter_set.quorum_size,
        voter_set_epoch=cert.voter_set_epoch,
        term=cert.term,
    )


@dataclass(frozen=True)
class ControlWritePermit:
    """Non-secret proof that a control write passed the current gate."""

    certificate_digest: str
    term: int
    lease_id: str


@dataclass(frozen=True)
class ControlPlaneSnapshot:
    schema_version: str
    cluster_id: str
    mode: str
    static_role: str
    voter_set_epoch: int
    committed_term: int
    leader_id: str
    lease_id: str
    certificate_digest: str
    expires_at_ms: int | None
    read_only_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cluster_id": self.cluster_id,
            "mode": self.mode,
            "static_role": self.static_role,
            "voter_set_epoch": self.voter_set_epoch,
            "committed_term": self.committed_term,
            "leader_id": self.leader_id,
            "lease_id": self.lease_id,
            "certificate_digest": self.certificate_digest,
            "expires_at_ms": self.expires_at_ms,
            "read_only_reason": self.read_only_reason,
        }


class ControlPlaneAuthority:
    """Fail-closed certificate gate; election and persistence are separate tickets."""

    def __init__(
        self,
        *,
        cluster_id: str,
        voter_set: VoterSet,
        static_role: str = "unknown",
    ) -> None:
        cluster = _safe_id(cluster_id, field="cluster_id")
        if cluster != voter_set.cluster_id:
            raise ControlContractError(
                "control_certificate_cluster_mismatch",
                "authority cluster does not match the voter set",
            )
        if static_role not in STATIC_ROLES:
            raise ControlContractError("control_static_role_invalid", "static role is unsupported")
        self._cluster_id = cluster
        self._voter_set = voter_set
        self._static_role = static_role
        self._committed_term = 0
        self._certificate: QuorumCertificate | None = None
        self._lock = threading.RLock()

    @property
    def voter_set(self) -> VoterSet:
        return self._voter_set

    def install_certificate(
        self,
        certificate: QuorumCertificate | Mapping[str, Any],
        *,
        now_ms: int | None = None,
    ) -> CertificateValidation:
        """Install a validated certificate; this never allocates a new term."""
        cert = certificate if isinstance(certificate, QuorumCertificate) else QuorumCertificate.from_dict(certificate)
        validation = validate_certificate(cert, self._voter_set, now_ms=now_ms)
        with self._lock:
            digest = cert.digest()
            if cert.term < self._committed_term:
                raise ControlContractError(
                    "control_certificate_stale",
                    "certificate term is older than the committed term",
                )
            if cert.term == self._committed_term and self._certificate is not None:
                previous = self._certificate
                if cert.leader_id != previous.leader_id or cert.lease_id != previous.lease_id:
                    raise ControlContractError(
                        "control_certificate_conflict",
                        "two different certificates claim the same term",
                    )
                if cert.expires_at_ms < previous.expires_at_ms or (
                    cert.expires_at_ms == previous.expires_at_ms and digest != previous.digest()
                ):
                    raise ControlContractError(
                        "control_certificate_conflict",
                        "certificate renewal must strictly extend the same lease",
                    )
                self._certificate = cert
                return validation
            if cert.term == self._committed_term:
                raise ControlContractError(
                    "control_certificate_stale",
                    "a revoked certificate cannot be installed again",
                )
            self._certificate = cert
            self._committed_term = cert.term
            return validation

    def revoke_certificate(self, certificate: QuorumCertificate) -> bool:
        """Fence this node's current lease without forgetting its committed term."""
        with self._lock:
            if self._certificate is None or self._certificate.digest() != certificate.digest():
                return False
            self._certificate = None
            return True

    def snapshot(self, *, now_ms: int | None = None) -> ControlPlaneSnapshot:
        current = _now_ms(now_ms)
        with self._lock:
            certificate = self._certificate
            expired = certificate is not None and certificate.expires_at_ms <= current
            writable = certificate is not None and not expired
            reason = "" if writable else "control_certificate_expired" if expired else "control_certificate_missing"
            return ControlPlaneSnapshot(
                schema_version=CONTROL_SCHEMA_VERSION,
                cluster_id=self._cluster_id,
                mode="writable" if writable else "read_only",
                static_role=self._static_role,
                voter_set_epoch=self._voter_set.voter_set_epoch,
                committed_term=self._committed_term,
                leader_id=certificate.leader_id if certificate else "",
                lease_id=certificate.lease_id if certificate else "",
                certificate_digest=certificate.digest() if certificate else "",
                expires_at_ms=certificate.expires_at_ms if certificate else None,
                read_only_reason=reason,
            )

    def current_certificate(self) -> QuorumCertificate | None:
        """Return the installed public certificate for transport decoration."""
        with self._lock:
            return self._certificate

    def admit_control_write(
        self,
        certificate: QuorumCertificate | Mapping[str, Any] | None,
        *,
        now_ms: int | None = None,
    ) -> ControlWritePermit:
        """Require the exact current, unexpired certificate for a control write."""
        if certificate is None:
            raise ControlContractError(
                "control_certificate_missing",
                "control write requires a quorum certificate",
            )
        cert = certificate if isinstance(certificate, QuorumCertificate) else QuorumCertificate.from_dict(certificate)
        validate_certificate(cert, self._voter_set, now_ms=now_ms)
        with self._lock:
            if self._certificate is None:
                raise ControlContractError(
                    "control_certificate_not_current",
                    "certificate has not been committed by this authority",
                )
            if cert.term < self._committed_term:
                raise ControlContractError(
                    "control_certificate_stale",
                    "certificate term is older than the committed term",
                )
            if cert.digest() != self._certificate.digest():
                raise ControlContractError(
                    "control_certificate_not_current",
                    "control write certificate is not the current certificate",
                )
            return ControlWritePermit(
                certificate_digest=cert.digest(),
                term=cert.term,
                lease_id=cert.lease_id,
            )

    require_control_write = admit_control_write


__all__ = [
    "CERTIFICATE_TYPE",
    "CONTROL_CONTRACT_ERROR_CODES",
    "CONTROL_SCHEMA_VERSION",
    "ControlContractError",
    "ControlPlaneAuthority",
    "ControlPlaneSnapshot",
    "ControlWritePermit",
    "CertificateValidation",
    "QuorumCertificate",
    "SIGNATURE_ALGORITHM",
    "VoterIdentity",
    "VoterSet",
    "VoterSignature",
    "sign_certificate",
    "validate_certificate",
]
