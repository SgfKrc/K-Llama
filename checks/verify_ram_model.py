"""Check the weighted RAM model against the measurement that motivated it.

The old rule charged a stage its full file size whenever `gpu_layers` was non-zero. On the
40-layer 35B that turned a 9.10 GiB load into a 20.82 GiB estimate, so the budget refused a
configuration that actually runs -- and the workaround was splitting the model across
machines, which is slower than not splitting it.

Two properties matter and they pull in opposite directions:
  * existing topologies must be unchanged (they are all-GPU or all-CPU stages)
  * a mixed stage must be estimated near its measured cost, from the safe side
"""
from __future__ import annotations

import sys
import warnings

sys.path.insert(0, r"G:\C\PYT\qlh")

import topology  # noqa: E402
from chain_client import (CPU_RESIDENT_RATIO, GPU_RAM_RATIO, estimate_ram_bytes,  # noqa: E402
                          ram_ratio)
from topology import Stage  # noqa: E402

GIB = 1 << 30
BIG = r"F:\qlh-models\qwen3-6-35b-a3b-gguf\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MEASURED_MIXED_GIB = 9.10   # 40 layers, 14 on GPU, observed via psutil available

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        ok = False


def stage(ngl: int, layers: int = 40) -> Stage:
    from pathlib import Path

    return Stage(0, Path(BIG), "127.0.0.1", 9901, layers,
                 "gpu" if ngl else "cpu", ngl, 0, "", 6)


def main() -> int:
    from pathlib import Path

    if not Path(BIG).exists():
        print("  [SKIP] 35B artifact not present")
        return 0

    print("  ratio by offload split (40 layers):")
    for ngl in (0, 14, 20, 40, 99):
        s = stage(ngl)
        print(f"    ngl={ngl:<3} ratio={ram_ratio(s):.3f}  "
              f"est {estimate_ram_bytes((s,)) / GIB:6.2f} GiB")

    # Endpoints must keep the old behaviour: those two coefficients are unchanged.
    check("all-CPU keeps CPU_RESIDENT_RATIO",
          abs(ram_ratio(stage(0)) - CPU_RESIDENT_RATIO) < 1e-9)
    check("all-GPU keeps GPU_RAM_RATIO",
          abs(ram_ratio(stage(99)) - GPU_RAM_RATIO) < 1e-9, "ngl above layer count clamps")
    check("half offload is the midpoint",
          abs(ram_ratio(stage(20)) - (GPU_RAM_RATIO + CPU_RESIDENT_RATIO) / 2) < 1e-9)

    # The point of the change: mixed configs must no longer be charged the whole file.
    mixed = estimate_ram_bytes((stage(14),)) / GIB
    check("mixed estimated below file size", mixed < 20.82, f"{mixed:.2f} GiB")
    check("mixed within 40% of measured", abs(mixed / MEASURED_MIXED_GIB - 1) < 0.40,
          f"est {mixed:.2f} vs measured {MEASURED_MIXED_GIB:.2f} GiB")
    check("mixed erring on the safe side (over, not under)", mixed >= MEASURED_MIXED_GIB,
          f"est {mixed:.2f} >= {MEASURED_MIXED_GIB:.2f}")

    print()
    print("  existing topologies (must be unchanged):")
    warnings.simplefilter("ignore")
    for name in ("topology-rog4.toml", "topology.toml", "topology-2b.toml"):
        p = Path(r"G:\C\PYT\qlh") / name
        if not p.exists():
            continue
        stages = topology.load(p)
        est = estimate_ram_bytes(stages) / GIB
        split = [(s.layers, s.gpu_layers) for s in stages if s.is_local]
        print(f"    {name:<22} {est:6.2f} GiB   (layers, gpu_layers)={split}")
        # Every local stage here is all-or-nothing, so the weighted model must agree with
        # the old step function exactly.
        old = sum(s.model.stat().st_size * (GPU_RAM_RATIO if s.gpu_layers
                                            else CPU_RESIDENT_RATIO)
                  for s in stages if s.is_local) / GIB
        check(f"{name}: unchanged from the old rule", abs(est - old) < 1e-6,
              f"{est:.2f} vs {old:.2f}")

    print()
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
