"""Worker-side runtime registry with explicit cross-operation ownership transfer.

The registry answers one question that a digest comparison cannot: *who is
allowed to close a native runtime*. A digest match proves the weights are the
same; it says nothing about whether the process that wants to reuse them also
owns the cleanup fence of the process that created them. Reusing on digest
alone lets the old owner's unload path `close()` a runtime the new owner is
already running forward passes on.

The protocol implemented here is the one described in
`docs/跨operation运行时所有权转移协议-2026-10-27.md`:

* An instance is created with `generation == 1` and one owner.
* `transfer()` swaps the owner atomically under the registry lock, bumps
  `generation`, and marks the previous fence entry `superseded`.
* `release()` from a stale owner is refused (`stale_owner`) and leaves the
  instance `live`; only a `refcount` of zero moves it to `closed`.
* Content never changes on a transfer. Anything that would require re-reading
  weights is a `no_match` outcome, not a partial refresh.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
import uuid

__all__ = [
    "RuntimeEntry",
    "RuntimeRegistry",
    "TransferReceipt",
    "TransferError",
]


# Outcome codes. Kept as plain strings so callers can log them without importing
# an enum, matching the reason-code style used elsewhere in this codebase.
NO_MATCH = "no_match"
OWNER_MISMATCH = "owner_mismatch"
GENERATION_MISMATCH = "generation_mismatch"
NOT_LIVE = "not_live"
LOAD_SPEC_MISMATCH = "load_spec_mismatch"
STALE_OWNER = "stale_owner"
UNKNOWN_RUNTIME = "unknown_runtime"
OVER_RELEASE = "over_release"


class TransferError(Exception):
    """A refused transfer or release. `code` is one of the outcome constants."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass
class TransferReceipt:
    runtime_id: str
    generation: int
    artifact_identity: dict


@dataclass
class RuntimeEntry:
    runtime_id: str
    artifact_identity: dict
    owner: dict
    load_spec_digest: str = ""
    generation: int = 1
    refcount: int = 1
    state: str = "live"          # live | retiring | closed
    superseded: list = field(default_factory=list)   # fence entries, oldest first
    payload: object = None       # the actual native handle; opaque to the registry


def _frozen(value):
    """Order-independent comparable form of a nested identity mapping."""
    if isinstance(value, dict):
        return tuple(sorted((key, _frozen(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


class RuntimeRegistry:
    """Process-local registry of live native runtimes keyed by artifact identity."""

    def __init__(self):
        self._lock = threading.RLock()
        self._entries: dict[str, RuntimeEntry] = {}
        self._events: list[dict] = []

    # -- introspection ----------------------------------------------------

    def entries(self) -> list[RuntimeEntry]:
        with self._lock:
            return list(self._entries.values())

    def events(self) -> list[dict]:
        """Audit trail of transfers and releases, oldest first."""
        with self._lock:
            return [dict(item) for item in self._events]

    def get(self, runtime_id: str) -> RuntimeEntry | None:
        with self._lock:
            return self._entries.get(runtime_id)

    def _record(self, **event) -> None:
        self._events.append(event)

    def _find(self, artifact_identity, load_spec_digest):
        target = _frozen(artifact_identity)
        for entry in self._entries.values():
            if entry.state != "live":
                continue
            if _frozen(entry.artifact_identity) != target:
                continue
            if entry.load_spec_digest != load_spec_digest:
                continue
            return entry
        return None

    # -- lifecycle --------------------------------------------------------

    def register(self, artifact_identity: dict, owner: dict, *, load_spec_digest: str = "",
                 payload: object = None) -> RuntimeEntry:
        """Create a new live runtime. Reuse goes through `transfer`, not here."""
        with self._lock:
            entry = RuntimeEntry(
                runtime_id="rt_" + uuid.uuid4().hex,
                artifact_identity=dict(artifact_identity),
                owner=dict(owner),
                load_spec_digest=load_spec_digest,
                generation=1,
                refcount=1,
                payload=payload,
            )
            self._entries[entry.runtime_id] = entry
            self._record(kind="register", runtime_id=entry.runtime_id,
                         owner=_frozen(owner), generation=entry.generation)
            return entry

    def transfer(self, from_owner: dict, to_owner: dict, artifact_identity: dict, *,
                 expected_generation: int, load_spec_digest: str = "") -> TransferReceipt:
        """Move ownership of a matching live runtime to `to_owner`.

        Raises `TransferError` with a concrete code when the runtime must be
        rebuilt instead. Callers treat any error as "close and reload".
        """
        with self._lock:
            entry = None
            target = _frozen(artifact_identity)
            for candidate in self._entries.values():
                if _frozen(candidate.artifact_identity) == target:
                    entry = candidate
                    break
            if entry is None:
                raise TransferError(NO_MATCH)
            if entry.state != "live":
                raise TransferError(NOT_LIVE)
            if entry.load_spec_digest != load_spec_digest:
                raise TransferError(LOAD_SPEC_MISMATCH)
            if _frozen(entry.owner) != _frozen(from_owner):
                raise TransferError(OWNER_MISMATCH)
            if entry.generation != expected_generation:
                raise TransferError(GENERATION_MISMATCH)
            entry.superseded.append({
                "owner": dict(entry.owner),
                "generation": entry.generation,
                "superseded_by": dict(to_owner),
                "state": "superseded",
            })
            entry.owner = dict(to_owner)
            entry.generation += 1
            entry.refcount += 1
            self._record(kind="transfer", runtime_id=entry.runtime_id,
                         owner=_frozen(to_owner), generation=entry.generation)
            return TransferReceipt(runtime_id=entry.runtime_id,
                                   generation=entry.generation,
                                   artifact_identity=dict(entry.artifact_identity))

    def release(self, runtime_id: str, owner: dict, *, expected_generation: int) -> str:
        """Drop one reference. Returns the resulting `state`.

        A release from an owner that no longer matches the entry's generation is
        refused with `stale_owner`: that is exactly the case where an unload
        path left over from the previous operation would otherwise free a
        runtime the new operation is still using.
        """
        with self._lock:
            entry = self._entries.get(runtime_id)
            if entry is None:
                raise TransferError(UNKNOWN_RUNTIME)
            if _frozen(entry.owner) != _frozen(owner) or entry.generation != expected_generation:
                self._record(kind="release_refused", runtime_id=runtime_id,
                             owner=_frozen(owner), generation=expected_generation,
                             reason=STALE_OWNER)
                raise TransferError(STALE_OWNER)
            if entry.refcount <= 0:
                raise TransferError(OVER_RELEASE)
            entry.refcount -= 1
            if entry.refcount == 0:
                entry.state = "closed"
            self._record(kind="release", runtime_id=runtime_id,
                         owner=_frozen(owner), generation=entry.generation,
                         refcount=entry.refcount, state=entry.state)
            return entry.state

    def retire(self, runtime_id: str) -> None:
        """Mark an instance retiring; releases then drain it to closed."""
        with self._lock:
            entry = self._entries.get(runtime_id)
            if entry is not None and entry.state == "live":
                entry.state = "retiring"

    def close_all(self) -> list[RuntimeEntry]:
        """Force every entry closed and return them so the caller can free them."""
        with self._lock:
            drained = [entry for entry in self._entries.values() if entry.state != "closed"]
            for entry in drained:
                entry.state = "closed"
                entry.refcount = 0
            self._entries.clear()
            return drained
