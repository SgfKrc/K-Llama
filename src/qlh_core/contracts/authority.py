"""Stable authority identity and changing, verified lease proof."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class AuthorityError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class AuthorityIdentity:
    cluster_id: str
    voter_set_epoch: int
    leader_id: str
    term: int
    lease_id: str


@dataclass(frozen=True)
class RecoveryFloor:
    domain_id: str
    sequence: int
    digest: str


@dataclass(frozen=True)
class AuthorityProof:
    identity: AuthorityIdentity
    certificate_digest: str
    expires_at_ms: int
    committed_floor: tuple[RecoveryFloor, ...] = ()


@dataclass(frozen=True)
class ValidatedAuthority:
    proof: AuthorityProof
    certificate: Any
    verification: Any


@dataclass(frozen=True)
class RecoveryPermit:
    """One synchronous recovery invocation, issued only after full validation."""
    proof: AuthorityProof
    revision: int
    certificate: Any
    check: Callable[[], None]
    mutate: Callable[[Callable[[], Any]], Any]


def require_preserved_floor(previous: tuple[RecoveryFloor, ...],
                            proposed: tuple[RecoveryFloor, ...]) -> None:
    values = {item.domain_id: item for item in proposed}
    for floor in previous:
        value = values.get(floor.domain_id)
        if (value is None or value.sequence < floor.sequence
                or (value.sequence == floor.sequence and value.digest != floor.digest)):
            raise AuthorityError("control_certificate_watermark_invalid",
                                 "certificate must preserve every committed recovery domain")
