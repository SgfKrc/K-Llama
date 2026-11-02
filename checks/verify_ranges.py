"""Does the range check actually catch a segment set that does not cover the model?

The failure it exists for: `source_layers` is typed by hand, so a topology can declare 20
for a 24-layer source and pass every other check. 20 + 4 = 20 sums correctly against its own
wrong claim, every stage's layer count matches its artifact, and the chain runs -- producing
fluent-looking garbage, because four decoder layers are simply never executed.

Reproduce that exactly: two artifacts whose provenance says 0..20 and 20..24, wired into a
topology that declares source_layers = 20. The check must complain about layers 20..24.

Cases A/B/C need the two artifacts under `archive-q35-2b` (their ranges and byte sizes are
read from the sidecars). Case D does not touch the disk: it replaces the provenance lookup
itself, so it is the one case that still runs after those artifacts are archived. That is
what `docs/CUT-TOOL.md` §2 points at when it says the sidecar fallback is pinned by this
script's D case. Missing artifacts are reported as SKIP, never as a pass.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import topology  # noqa: E402
from topology import Stage  # noqa: E402

ARCHIVE = Path(r"F:\qlh-models\archive-q35-2b")
HEAD = ARCHIVE / "q35-2b-head20.gguf"
TAIL = ARCHIVE / "q35-2b-tail20-24.gguf"
BAD_TOPO = ARCHIVE / "topo-bad-20.toml"

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


def case_a() -> None:
    print()
    print("  A. the correct wiring (0..20 + 20..24 over a 24-layer source)")
    w = topology.verify_artifacts(stages(20, 4))
    check("no warnings", not w, "; ".join(w) if w else "clean")


def case_b() -> None:
    print()
    print("  B. the 2B topology as it was: declares source_layers = 20")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            topology.load(BAD_TOPO)
        msgs = [str(x.message) for x in caught]
        gap = [m for m in msgs if "belong to no stage" in m or "20" in m]
        check("gap reported", bool(gap), gap[0] if gap else f"got {msgs}")
    except topology.TopologyError as exc:
        msg = str(exc)
        if "does not point to a file" in msg or "does not exist" in msg:
            # load() checks that a local stage's model exists before any layer arithmetic.
            # That error looks like a refusal but says nothing about coverage, and an
            # earlier version of this file counted every TopologyError as "refused" -- so
            # after the artifacts were archived, B would still print PASS while verifying
            # nothing. Keep the two apart.
            check("B refused for the wrong reason (artifact missing)", False, msg[:90])
        else:
            # The exact-cover check fires first when layers sum to the wrong total; that is
            # also a correct refusal, just at a different layer of defence.
            check("refused (sum check)", True, msg[:90])


def case_c() -> None:
    print()
    print("  C. overlapping ranges (0..20 and 16..24)")
    bad = (
        Stage(0, HEAD, "127.0.0.1", 9803, 20, "gpu", 0, 0, "", 4),
        Stage(1, TAIL, "127.0.0.1", 9804, 8, "tail", 0, 0, "", 4),
    )
    # Tail's sidecar says 20..24 but the stage declares 8 layers -> mismatch caught first.
    try:
        w = topology.verify_artifacts(bad)
        # No `or True` here: that made the assertion vacuous, so a change that let the
        # overlap through would still print PASS.
        check("mismatch reported", any("was cut with" in m or "declares" in m for m in w),
              "; ".join(w)[:100] or "no warning: the overlap slipped through")
    except topology.TopologyError as exc:
        check("mismatch refused", True, str(exc)[:90])


def case_d() -> None:
    print()
    print("  D. an artifact with no sidecar is reported, not trusted")
    # Every artifact on disk now has a sidecar, so simulate the pre-provenance case by
    # hiding the sidecar lookup rather than keeping a stale duplicate around. Nothing on
    # disk is read here, which is why this case survives the artifacts being archived.
    real = topology.provenance
    topology.provenance = lambda _p: None
    try:
        w = topology.verify_artifacts(stages(20, 4))
    finally:
        topology.provenance = real
    check("absence reported", any("no artifacts carry provenance" in m for m in w),
          "; ".join(w)[:100])


def main() -> int:
    artifacts = HEAD.is_file() and TAIL.is_file()
    full = artifacts and BAD_TOPO.is_file()
    if full:
        p = topology.provenance(HEAD)
        check("sidecar exists", p is not None,
              str(p.get("source_layer_range")) if p else "no .qlh.json")
        if not p:
            return 1
        case_a()
        case_b()
        case_c()
    else:
        missing = [str(x) for x in (HEAD, TAIL, BAD_TOPO) if not x.is_file()]
        print("  [SKIP] A/B/C need the cut artifacts; missing:")
        for path in missing:
            print(f"         {path}")
        print("         (ranges and byte sizes come from those files; D reads no disk and "
              "still runs)")

    case_d()

    print()
    if full:
        print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    else:
        # Only D ran; saying ALL PASS would claim the range check was exercised.
        print("RESULT:", "D ONLY -- A/B/C skipped, the coverage check itself did not run")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
