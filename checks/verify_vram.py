"""Verify the VRAM guard against the measurements it was calibrated on.

Three artifacts were measured on a 8188 MiB card. Two fit and one spilled; the guard must
refuse exactly the one that spilled, or it is either useless or in the way.

Also checks the property the post-load check relies on: a stage that fits occupies more
video memory than its file size (the compute buffer comes on top), while a spilled stage
occupies less (the driver backed the rest with system RAM). That inequality is what makes
the fallback independent of the estimate's constants.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"G:\C\PYT\qlh")

from chain_client import estimate_vram_bytes, vram_info  # noqa: E402
from topology import Stage  # noqa: E402

MIB = 1 << 20
GIB = 1 << 30

# (label, artifact, measured VRAM delta in MiB, did it fit?)
CASES = [
    ("stage-0-8  8L", r"F:\qlh-models\stage-split-3\stage-0-8.gguf", 5237, True),
    ("stage-24-32 8L", r"F:\qlh-models\stage-split-3\stage-24-32.gguf", 5139, True),
    ("stage-8-24 16L", r"F:\qlh-models\stage-split-3\stage-8-24.gguf", 7660, False),
]

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        ok = False


def main() -> int:
    info = vram_info()
    if info is None:
        print("  [SKIP] nvidia-smi unavailable; the guard degrades to a warning")
        return 0
    total, used = info
    free = total - used
    print(f"  card: {total / MIB:.0f} MiB total, {free / MIB:.0f} MiB free")
    print()

    for label, path, measured, fits in CASES:
        p = Path(path)
        if not p.exists():
            print(f"  {label}: artifact missing, skipped")
            continue
        stage = Stage(0, p, "127.0.0.1", 9701, 8, "gpu", 99, 0, "", 6)
        est = estimate_vram_bytes((stage,)) / MIB
        file_mib = p.stat().st_size / MIB
        verdict = est <= free / MIB
        print(f"  {label:<16} file {file_mib:5.0f}M  est {est:6.0f}M  "
              f"measured {measured}M  -> {'fits' if verdict else 'refused'}")
        check(f"{label}: guard agrees with measurement", verdict == fits,
              f"guard={verdict} measured={fits}")
        # The post-load fallback's premise.
        check(f"{label}: delta {'>' if fits else '<'} file size",
              (measured > file_mib) == fits,
              f"delta {measured}M vs file {file_mib:.0f}M")

    print()
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
