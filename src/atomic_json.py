"""Atomic JSON publication; callers serialize state changes through replacement."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


def atomic_write_json(path: Path, value: Any, *, ensure_ascii: bool = True,
                      indent: int | None = None, sort_keys: bool = False,
                      trailing_newline: bool = False) -> None:
    """Publish a closed, unique sibling file with bounded Windows lock retries.

    This does not order concurrent state snapshots. The owner's lock must cover
    mutation, serialization and every replacement attempt, including backoff.
    """
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}-", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys)
            if trailing_newline:
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Readers/scanners without FILE_SHARE_DELETE can deny a replacement
        # even after our own write handle closes. Never retry other I/O stages.
        for attempt in range(8):
            try:
                temporary.replace(path)
                return
            except OSError as exc:
                if (sys.platform != "win32" or getattr(exc, "winerror", None) not in {5, 32, 33}
                        or attempt == 7):
                    raise
                time.sleep(0.01 * 2 ** attempt)
    except BaseException:
        # Best-effort cleanup must preserve the original persistence failure or
        # cancellation, including when another Windows handle blocks deletion.
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise
