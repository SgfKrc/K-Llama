"""Value contract for a durable journal prefix."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class JournalWatermarkError(ValueError):
    """Invalid journal prefix watermark."""

    code = "journal_watermark_invalid"


@dataclass(frozen=True)
class CommitWatermark:
    """A verified workflow checkpoint at the end of a durable event prefix."""

    workflow_id: str
    last_sequence: int
    checkpoint_digest: str

    def __post_init__(self) -> None:
        workflow_id = self.workflow_id
        if not isinstance(workflow_id, str) or not _SAFE_ID.fullmatch(workflow_id):
            raise JournalWatermarkError("workflow_id is invalid")
        if isinstance(self.last_sequence, bool) or not isinstance(self.last_sequence, int):
            raise JournalWatermarkError("last_sequence is invalid")
        if self.last_sequence < 0:
            raise JournalWatermarkError("last_sequence is invalid")
        if not isinstance(self.checkpoint_digest, str) or not _SHA256_HEX.fullmatch(self.checkpoint_digest):
            raise JournalWatermarkError("checkpoint_digest must be a lowercase sha256 hex string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "last_sequence": self.last_sequence,
            "checkpoint_digest": self.checkpoint_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CommitWatermark":
        required = {"workflow_id", "last_sequence", "checkpoint_digest"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise JournalWatermarkError("commit watermark fields are not canonical")
        return cls(
            workflow_id=value["workflow_id"],
            last_sequence=value["last_sequence"],
            checkpoint_digest=value["checkpoint_digest"],
        )


__all__ = ["CommitWatermark", "JournalWatermarkError"]
