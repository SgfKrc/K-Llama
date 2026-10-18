"""Versioned client observations and machine-readable causal diagnostics."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Mapping

CLIENT_CONTRACT_VERSION = 2

@dataclass(frozen=True)
class Reason:
    domain: str
    code: str
    source: str
    retryability: str = "unknown"
    details: Mapping = field(default_factory=dict)
    cause: Mapping | None = None

    def snapshot(self):
        return {"domain": self.domain, "code": self.code, "source": self.source,
                "retryability": self.retryability, "details": deepcopy(dict(self.details)),
                "cause": deepcopy(self.cause)}

def reason(domain, code, source, *, retryability="unknown", details=None, cause=None):
    return Reason(domain, code, source, retryability, details or {}, cause).snapshot() if code else None

def envelope(payload, *, revision, identity, observed_at, validity="valid", reason=None):
    return {**deepcopy(payload), "schema_version": CLIENT_CONTRACT_VERSION,
            "revision": revision, "identity": deepcopy(identity), "observed_at": observed_at,
            "validity": validity, "reason_detail": deepcopy(reason)}
