"""List every model artifact, whether anything references it, and how much it wastes.

The previous repository accumulated tens of gigabytes of cut segments nobody could account
for, because nothing ever compared "files on disk" against "files in use". This is that
comparison, so the answer takes seconds instead of an afternoon of `dir /s`.

Deliberately read-only and dependency-free: it prints a table and an exit code of 0. It
does not delete anything -- deciding a segment is dead requires knowing which topologies
still matter, and that is a judgement, not a rule.

Usage:
    python checks/inspect_disk.py            # scan the default roots
    python checks/inspect_disk.py F:\\models  # scan a specific root
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GIB = 1 << 30

# Where artifacts live. Anything not under these roots is not scanned.
DEFAULT_ROOTS = [
    Path(r"F:\qlh-models"),
    Path(r"H:\qlh_models"),
    ROOT / "build",
]

MODEL_RE = re.compile(r'model\s*=\s*"([^"]+)"')


def referenced() -> dict[str, list[str]]:
    """Map artifact path -> topologies that mention it.

    Both sides are normalised to forward slashes and compared by suffix, because a topology
    may name a remote path (L:/...) for an artifact that also exists locally.
    """
    refs: dict[str, list[str]] = {}
    for topo in sorted(ROOT.glob("topology*.toml")):
        try:
            text = topo.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in MODEL_RE.finditer(text):
            for key in _keys(m.group(1)):
                refs.setdefault(key, [])
                if topo.name not in refs[key]:
                    refs[key].append(topo.name)
    return refs


def _keys(raw: str) -> list[str]:
    """The filename plus the last two path components, for suffix matching."""
    norm = raw.replace("\\", "/")
    name = norm.rsplit("/", 1)[-1]
    parts = norm.split("/")
    keys = [name]
    if len(parts) >= 2:
        keys.append("/".join(parts[-2:]))
    return keys


def is_referenced(path: Path, refs: dict[str, list[str]]) -> list[str]:
    norm = str(path).replace("\\", "/")
    out: list[str] = []
    for key, topos in refs.items():
        if norm.endswith(key) or path.name == key:
            out.extend(t for t in topos if t not in out)
    return out


def sources() -> dict[str, list[str]]:
    """Map source-model filename -> segments cut from it.

    Every segment's sidecar records the full model it came from, so the sources can be
    derived instead of guessed. Without this the scan labels the 20 GiB checkpoint that
    everything is cut from as "unreferenced", which is exactly backwards.
    """
    out: dict[str, list[str]] = {}
    for root in DEFAULT_ROOTS:
        if not root.exists():
            continue
        for side in root.rglob("*.gguf.qlh.json"):
            try:
                import json

                meta = json.loads(side.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            src = meta.get("source")
            if src:
                out.setdefault(Path(src).name, []).append(side.name)
    return out


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv[1:]] or DEFAULT_ROOTS
    refs = referenced()
    srcs = sources()

    print(f"  topologies on disk: {len(list(ROOT.glob('topology*.toml')))}")
    for key, topos in sorted(refs.items()):
        if key.count("/") == 1:  # print once per artifact, using the longer key
            continue
        print(f"    {key:<44} <- {', '.join(topos)}")
    print()

    rows = []
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob("*.gguf"):
            rows.append(f)
    rows.sort(key=lambda p: p.stat().st_size, reverse=True)

    total = sum(p.stat().st_size for p in rows)
    unused_total = 0
    print(f"  {'size':>9}  {'prov':<4} {'used':<26} artifact")
    for p in rows:
        size = p.stat().st_size
        side = p.with_suffix(p.suffix + ".qlh.json")
        prov = "yes" if side.exists() else "-"
        topos = is_referenced(p, refs)
        if topos:
            used = ", ".join(topos)
        elif p.name in srcs:
            used = f"SOURCE for {len(set(srcs[p.name]))} segments"
        else:
            used = "*** UNREFERENCED ***"
            unused_total += size
        print(f"  {size / GIB:8.2f}G  {prov:<4} {used:<26} {p}")

    # Same filename and size in two places is a duplicate copy, not two artifacts.
    by_name: dict[tuple[str, int], list[Path]] = {}
    for p in rows:
        by_name.setdefault((p.name, p.stat().st_size), []).append(p)
    dupes = {k: v for k, v in by_name.items() if len(v) > 1}

    print()
    print(f"  artifacts: {len(rows)}   total {total / GIB:.2f} GiB")
    if dupes:
        waste = sum(k[1] * (len(v) - 1) for k, v in dupes.items())
        print(f"  duplicate copies: {waste / GIB:.2f} GiB reclaimable")
        for (name, size), paths in dupes.items():
            print(f"    {name} ({size / GIB:.2f} GiB) x{len(paths)}")
            for p in paths:
                print(f"      {p}")
    print(f"  unreferenced (excluding sources): {unused_total / GIB:.2f} GiB")

    orphans = [p for p in rows if p.with_suffix(p.suffix + ".qlh.json").exists()
               and not p.exists()]
    for side in (p for r in roots if r.exists()
                 for p in r.rglob("*.gguf.qlh.json")):
        art = Path(str(side)[: -len(".qlh.json")])
        if not art.exists():
            orphans.append(side)
    if orphans:
        print()
        print("  sidecars whose artifact is gone (safe to delete):")
        for p in orphans:
            print(f"    {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
