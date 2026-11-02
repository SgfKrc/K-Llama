"""Is a MoE expert GEMM's cost the same thing at 4 MiB (L3) as at 136 MiB (DRAM)?

``bench_lm_head.py --mode experts`` measures ``MUL_MAT_ID`` with a *fixed* id draw,
so every round reads the same 8 experts: 8 x 0.53 MiB = 4.2 MiB for IQ4_XS, which
fits in this host's 24 MiB L3.  The real decoder does not work that way -- each token
routes to a fresh 8 of 256 experts per layer, so the working set is the whole bank
(136 MiB per layer for IQ4_XS) and every read starts a new stream inside it.

That difference cannot show up in a loop over one id vector, so this probe re-runs the
same ``MUL_MAT_ID`` under three id policies:

``l3``          one fixed draw, repeated        -> 4.2 MiB resident, kernel-bound
``dram_seq``    round r uses experts 8r..8r+7   -> 136 MiB, single ascending stream
``dram_rand``   round r draws 8 at random        -> 136 MiB, 8 scattered streams

The ``l3`` vs ``dram_*`` pair isolates cache residency.  ``dram_seq`` vs ``dram_rand``
isolates stream count (same bytes, same experts, different locality).  And because
``q4_K/plain`` and ``q4_K/repack`` are *the same bytes through two kernels*, that pair
answers the actual question -- what an x86 interleaved GEMM is worth once the weight
no longer fits in cache -- which is the regime the decoder runs in.

Reuses ``bench_lm_head``'s ctypes layer, ``ExpertBank`` and ``MmidRun`` rather than
re-implementing them (AGENTS.md §7).  Nothing is written back to the GGUF; re-quantised
banks live in RAM.

Usage:
    python checks/probe_moe_cache.py --threads 1,4,8 --types iq4_xs,q4_K,q6_K
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_lm_head import (  # noqa: E402
    DEFAULT_BIN, DEFAULT_GGUF, DEFAULT_MODEL, REPACK_GEMM_TYPES, ExpertBank, Ggml,
    MmidRun, _InitParams, convert_expert_bank, read_expert_bank, read_expert_census,
    resolve_type, tiled_mm_enabled,
)

VARIANTS = ("l3", "dram_seq", "dram_rand")


class TokenGraph:
    """N distinct expert banks driven by ONE graph -- one thread-pool wake per token.

    The single-op probe pays a wake-up per ``graph_compute``, which at 0.05--0.5 ms of
    work per op is the same order as the work itself; that is why its 8-thread numbers
    come out *slower* than its 1-thread ones.  The decoder does not do that: one token
    submits 41 layers x 3 expert ops (gate/up/down) back to back, so the pool wakes
    once and stays hot.  Putting N ops in one graph reproduces that, and N distinct
    banks (136 MiB each) put the working set past L3 the way the real 5.6 GiB expert
    set does.
    """

    def __init__(self, g: Ggml, banks: list[ExpertBank], n_used: int,
                 hidden: np.ndarray) -> None:
        self.g = g
        self.banks = banks
        self.n_used = n_used
        self.ctx_a = g.base.ggml_init(_InitParams(64 << 20, None, True))
        self.a = g.base.ggml_new_tensor_3d(self.ctx_a, 0, banks[0].ne0, 1, 1)
        self.ids, self.out, self.runs = [], [], []
        for bank in banks:
            ids = g.base.ggml_new_tensor_2d(self.ctx_a, 26, n_used, 1)
            out = g.base.ggml_mul_mat_id(self.ctx_a, bank.w, self.a, ids)
            self.ids.append(ids)
            self.out.append(out)
        self.buf_a = g.base.ggml_backend_alloc_ctx_tensors(self.ctx_a, g.backend)
        g.base.ggml_backend_tensor_set(self.a, hidden.ctypes.data, 0, hidden.nbytes)
        self.graph = g.base.ggml_new_graph(self.ctx_a)
        for out in self.out:
            g.base.ggml_build_forward_expand(self.graph, out)

    def set_ids(self, ids_list: list[np.ndarray]) -> None:
        for tensor, ids in zip(self.ids, ids_list):
            self.g.base.ggml_backend_tensor_set(tensor, ids.ctypes.data, 0, ids.nbytes)

    def compute(self) -> float:
        t0 = time.perf_counter()
        status = self.g.base.ggml_backend_graph_compute(self.g.backend, self.graph)
        el = (time.perf_counter() - t0) * 1e3
        if status != 0:
            raise SystemExit(f"graph compute failed ({status})")
        return el

    def close(self) -> None:
        if self.buf_a is not None:
            self.g.base.ggml_backend_buffer_free(self.buf_a)
            self.buf_a = None
        if self.ctx_a is not None:
            self.g.base.ggml_free(self.ctx_a)
            self.ctx_a = None


def id_policy(variant: str, rng: np.random.RandomState, round_no: int,
              n_expert: int, n_used: int) -> np.ndarray:
    """One ``ids`` tensor [n_used, 1] for this round, per the variant's policy."""
    if variant == "l3":
        pick = np.arange(n_used)          # same experts every round
    elif variant == "dram_seq":
        start = (round_no * n_used) % n_expert
        pick = (start + np.arange(n_used)) % n_expert
    elif variant == "dram_rand":
        pick = rng.choice(n_expert, n_used, replace=False)
    else:
        raise ValueError(variant)
    return pick.astype(np.int32).reshape(n_used, 1)


def run_token_graph(g, args, src_type, raw3, ne0, ne1, ne2, n_used, hidden,
                    results, wanted, threads_list) -> int:
    """Per-token expert cost: N banks per type, one graph, one wake.

    Every type and buffer mode is built first, then measured **interleaved** across
    thread counts and rounds.  On this host the harness and a browser hold ~3 cores,
    and a contiguous block per type lets one busy interval land on one type -- measured:
    the same IQ4_XS point came out 28.4 ms/token in one run and 42.9 in the next when
    types were measured in sequence.  Interleaving is what makes the type comparison
    meaningful; the per-type absolute number still carries that noise.
    """
    n = args.token_graph
    zero_ids = [np.zeros((n_used, 1), dtype=np.int32)] * n
    prepared: list[tuple[str, str, list, TokenGraph]] = []
    converted: dict[str, np.ndarray] = {}
    for name in wanted:
        resolved = resolve_type(name)
        if resolved is None:
            continue
        gt, type_name = resolved
        if name not in converted:
            converted[name] = (raw3 if (type_name == "iq4_xs" and src_type == 23)
                               else convert_expert_bank(g, raw3, src_type, gt,
                                                        ne0, ne1, ne2))
        bank_raw = converted[name]
        modes = ["plain"] + (["repack"] if type_name in REPACK_GEMM_TYPES else [])
        for mode in modes:
            repack = mode == "repack"
            banks = [ExpertBank(g, bank_raw, gt, ne0, ne1, ne2, repack=repack)
                     for _ in range(n)]
            run = TokenGraph(g, banks, n_used, hidden)
            # ids must be valid before the first compute: mul_mat_id indexes the bank
            # with them, and an uninitialised buffer is a wild read (measured:
            # access violation on the first warmup).
            run.set_ids(zero_ids)
            prepared.append((type_name, mode, banks, run))

    try:
        for threads in threads_list:
            g.cpu.ggml_backend_cpu_set_n_threads(g.backend, threads)
            rngs = [np.random.RandomState(7 + i) for i in range(len(prepared))]
            for _, _, _, run in prepared:
                for _ in range(args.warmup):
                    run.compute()
            times: list[list[float]] = [[] for _ in prepared]
            for r in range(args.reps):
                for i, (_, _, _, run) in enumerate(prepared):
                    run.set_ids([id_policy("dram_rand", rngs[i], r, ne2, n_used)
                                 for _ in range(n)])
                    times[i].append(run.compute())
            for i, (type_name, mode, _, _) in enumerate(prepared):
                med = statistics.median(times[i])
                per_op = med / n
                moved = raw3.nbytes * n_used / ne2 * n / 2**30
                gbs = moved / (med / 1e3)
                key = f"{type_name}/{mode}/token/n{n}/t{threads}"
                results[key] = {"median_ms": med, "per_op_ms": per_op,
                                "token_123ops_ms": per_op * 123,
                                "gib_s": gbs, "min_ms": min(times[i]),
                                "n": len(times[i])}
                print(f"   {type_name:>8}/{mode:<6} t={threads:<2} "
                      f"med={med:7.3f} min={min(times[i]):7.3f} "
                      f"({per_op:6.4f}/op) {gbs:6.2f} GiB/s -> "
                      f"{per_op * 123:6.1f} ms/token")
            print()
    finally:
        for _, _, banks, run in prepared:
            run.close()
            for bank in banks:
                bank.close()
    g.base.ggml_backend_free(g.backend)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_GGUF / DEFAULT_MODEL)
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--types", default="iq4_xs,q4_K,q6_K")
    ap.add_argument("--threads", default="1,4,8")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=40,
                    help="rounds per point; 40 covers the 256-expert bank 1.25x")
    ap.add_argument("--token-graph", type=int, default=0,
                    help="N distinct expert banks in ONE graph: one thread-pool "
                         "wake per token, the way the decoder runs. 0 = single op")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    threads_list = [int(v) for v in args.threads.split(",") if v.strip()]
    wanted = [v.strip().lower() for v in args.types.split(",") if v.strip()]

    g = Ggml(args.bin)
    g.backend = g.cpu.ggml_backend_cpu_init()
    if not g.backend:
        raise SystemExit("ggml_backend_cpu_init failed")
    g.cpu.ggml_backend_cpu_set_n_threads(g.backend, 1)
    if not g.find_repack_buft():
        raise SystemExit("no CPU_REPACK buffer type")

    census = read_expert_census(args.model)
    meta = census["meta"]
    gate = [e for e in census["entries"]
            if e["role"] == "ffn_gate_exps" and e["type"] == 23]
    src = gate[0] if gate else [e for e in census["entries"]
                                if e["role"] == "ffn_gate_exps"][0]
    raw3, src_type, ne0, ne1, ne2 = read_expert_bank(args.model, src["name"])
    n_used = int(meta["expert_used_count"])
    bank_mib = raw3.nbytes / 2**20
    print(f"source {src['name']} [{ne0},{ne1},{ne2}] type={src_type} "
          f"({bank_mib:.0f} MiB bank, {raw3.nbytes / 2**20 * n_used / ne2:.1f} MiB "
          f"per routed draw); n_used={n_used}; tiled_mm={tiled_mm_enabled()}")
    print(f"working set: l3={raw3.nbytes / 2**20 * n_used / ne2:.1f} MiB  "
          f"dram={bank_mib:.0f} MiB\n")

    hidden = np.random.RandomState(0).randn(1, ne0).astype(np.float32)
    results: dict = {}

    if args.token_graph:
        return run_token_graph(g, args, src_type, raw3, ne0, ne1, ne2, n_used,
                               hidden, results, wanted, threads_list)

    for name in wanted:
        resolved = resolve_type(name)
        if resolved is None:
            print(f"skip unknown type {name}", file=sys.stderr)
            continue
        gt, type_name = resolved
        t0 = time.perf_counter()
        if type_name == "iq4_xs" and src_type == 23:
            bank_raw = raw3
        else:
            bank_raw = convert_expert_bank(g, raw3, src_type, gt, ne0, ne1, ne2)
        print(f"-- {type_name} ({bank_raw.nbytes / 2**20:.0f} MiB) converted in "
              f"{time.perf_counter() - t0:.0f}s")

        modes = ["plain"] + (["repack"] if type_name in REPACK_GEMM_TYPES else [])
        for mode in modes:
            repack = mode == "repack"
            bank = ExpertBank(g, bank_raw, gt, ne0, ne1, ne2, repack=repack)
            run = MmidRun(g, bank, 1, hidden, id_policy("l3", np.random.RandomState(0),
                                                        0, ne2, n_used))
            try:
                for variant in VARIANTS:
                    for threads in threads_list:
                        g.cpu.ggml_backend_cpu_set_n_threads(g.backend, threads)
                        rng = np.random.RandomState(7)
                        for _ in range(args.warmup):
                            run.compute()
                        times = []
                        for r in range(args.reps):
                            ids = id_policy(variant, rng, r, ne2, n_used)
                            g.base.ggml_backend_tensor_set(
                                run.ids, ids.ctypes.data, 0, ids.nbytes)
                            times.append(run.compute())
                        med = statistics.median(times)
                        moved = raw3.nbytes * n_used / ne2 / 2**30
                        gbs = moved / (med / 1e3)
                        key = f"{type_name}/{mode}/{variant}/t{threads}"
                        results[key] = {"median_ms": med, "min_ms": min(times),
                                        "max_ms": max(times), "gib_s": gbs}
                        print(f"   {mode:>6} {variant:<10} t={threads:<2} "
                              f"med={med:7.4f} min={min(times):7.4f} "
                              f"max={max(times):7.4f} ms  {gbs:6.2f} GiB/s")
            finally:
                run.close()
                bank.close()
        print()

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    g.base.ggml_backend_free(g.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
