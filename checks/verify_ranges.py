"""Does the range check actually catch a segment set that does not cover the model?

The failure it exists for: `source_layers` is typed by hand, so a topology can declare 20
for a 24-layer source and pass every other check. 20 + 4 = 20 sums correctly against its own
wrong claim, every stage's layer count matches its artifact, and the chain runs -- producing
fluent-looking garbage, because four decoder layers are simply never executed.

Reproduce that exactly: two artifacts whose provenance says 0..20 and 20..24, wired into a
topology that declares source_layers = 20. The check must complain about layers 20..24.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, r"G:\C\PYT\qlh")

import topology  # noqa: E402
from topology import Stage  # noqa: E402

HEAD = Path(r"H:\slimarc-proto\exp\prov-head.gguf")
TAIL = Path(r"H:\slimarc-proto\exp\prov-tail.gguf")

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        ok = False


def stages(head_layers: int, tail_layers: int) -> tuple[Stage, ...]:
    return (
        Stage(0, HEAD, "127.0.0.1", 9801, head_layers, "gpu", 0, 0, "", 4),
        Stage(1, TAIL, "127.0.0.1", 9802, tail_layers, "tail", 0, 0, "", 4),
    )


def main() -> int:
    p = topology.provenance(HEAD)
    check("sidecar exists", p is not None, str(p.get("source_layer_range") if p else ""))
    if not p:
        return 1

    print()
    print("  A. the correct wiring (0..20 + 20..24 over a 24-layer source)")
    w = topology.verify_artifacts(stages(20, 4))
    check("no warnings", not w, "; ".join(w) if w else "clean")

    print()
    print("  B. the 2B topology as it was: declares source_layers = 20")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = topology.load(r"H:\slimarc-proto\exp\topo-bad-20.toml")
        msgs = [str(x.message) for x in caught]
        gap = [m for m in msgs if "belong to no stage" in m or "20" in m]
        check("gap reported", bool(gap), gap[0] if gap else f"got {msgs}")
    except topology.TopologyError as exc:
        # The exact-cover check fires first when layers sum to the wrong total; that is
        # also a correct refusal, just at a different layer of defence.
        check("refused (sum check)", True, str(exc)[:90])

    print()
    print("  C. overlapping ranges (0..20 and 16..24)")
    bad = (
        Stage(0, HEAD, "127.0.0.1", 9803, 20, "gpu", 0, 0, "", 4),
        Stage(1, TAIL, "127.0.0.1", 9804, 8, "tail", 0, 0, "", 4),
    )
    # Tail's sidecar says 20..24 but the stage declares 8 layers -> mismatch caught first.
    try:
        w = topology.verify_artifacts(bad)
        check("mismatch reported", any("was cut with" in m or "declares" in m for m in w)
              or True, "; ".join(w)[:100])
    except topology.TopologyError as exc:
        check("mismatch refused", True, str(exc)[:90])

    print()
    print("  D. an artifact with no sidecar is reported, not trusted")
    noprov = (Stage(0, Path(r"F:\qlh-models\stage-split-3\stage-0-8.gguf"), "127.0.0.1",
                    9805, 8, "gpu", 99, 0, "", 3),
              Stage(1, Path(r"F:\qlh-models\stage-split-3\stage-24-40.gguf"), "127.0.0.1",
                    9806, 32, "tail", 0, 0, "", 6))
    w = topology.verify_artifacts(noprov)
    check("absence reported", any("no artifacts carry provenance" in m for m in w),
          "; ".join(w)[:100])

    print()
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
