# K-Llama v2.1

[中文](../README.md) | **English**

> Status: **Current**
> Updated: 2026-11-09
> Scope: the entry point for the split chain — what it is, what it can do, how to run it, the hard
> constraints, the measurement baselines, and the documentation index. Mechanism lives in
> [ARCHITECTURE.md]; boundary costs in [SEGMENTED-LIMITS.md]; patch composition in [PATCHES.md].
> The Chinese README is the source of truth; this file mirrors it.

K-Llama runs llama.cpp across machines by layer. Every node maps the same GGUF but creates and reads
only the tensors of its own layers; a segment hands the next one nothing but the hidden state —
`n_embd` floats (`ctx_tail` in `src/llama-context.cpp`). The split lives inside `llama_context`, so
`llama-server` is the chain driver: `/completion`, `/chat/completions`, slots and the prompt cache
all take the stock code path, and clients never learn which machine holds which layers.

Two generations share the name. v2 is the Python-driven chain (`chain_client.py`,
`stage_worker.py`, `stream.py`; TUI entry `koakuma`) that spreads a model over several machines.
v2.1 pushes that chain into `llama_context` so the exposed surface matches a whole model. The v2
chain still works; its throughput numbers are in [MEASUREMENTS.md].

## Capabilities

| Capability | Result and criterion | Details |
| --- | --- | --- |
| Local split | `--split-layer-range` + `--split-tail-range` serve from one process; `-np 4` output is byte-identical to serial | `verify.sh` group 5; [SPLIT-MEASUREMENTS.md] §4 |
| Prefix reuse | A 3103-token history recomputes in 19847 ms; the second turn solves 38 new tokens in 714 ms (2B chain, `-c 8192 -np 1`) | [SPLIT-MEASUREMENTS.md] §1 |
| Remote chain | Built by `llama_split_attach_remote()`; KV broadcast, chain-wide checkpoints, 120 s read timeout, a dead chain with a live process | `verify_chain_death.py`; [SEGMENTED-LIMITS.md] §2.7 |
| Remote speculation | DFlash / DSpark `layer_inp` taps ride the decode reply across processes (protocol v9); acceptance matches the whole model prompt by prompt | table below; [SPLIT-MEASUREMENTS.md] §14 |
| MTP speculation | The tail's NextN head is the draft; k steps recurse inside the stage | `verify.sh` group 8; [SPECULATION.md] §4 |
| grammar / reasoning budget | Candidates self-check token by token, a rejection rolls back the whole frame; an exhausted budget only tightens | `common_sampler_grammar_rejects()`; [PATCHES.md] fourth generation |
| Segment memory | Logits and pp graph buffers are sized by the rows actually read: tail 1710.76 → 3.79 MiB (1806-token ubatch, 2B chain, `-c 8192 -np 4 -b 2048`), remote `20:24` segment 2052.10 → 240.09 MiB (2B, `-c 4096 -b 2048`, 248320-token vocab) | [SPLIT-MEASUREMENTS.md] §5 |
| `seq_rm` | Returns `bool`; five call sites degrade on their own; a stalled chain leaves the process alive | [SEGMENTED-LIMITS.md] §2.9 |

The 2B chain above is `qwen35-2b-Q4_K_M` (head `0:20` + tail `20:24`, CPU-only `build-split`); the
full baseline is in [SPLIT-MEASUREMENTS.md] §0.

## Quick start

```bash
# Local two-segment chain: the tail runs in the same process
llama-server -m model.gguf --split-layer-range 0:32 --split-tail-range 32:40 -c 8192 -np 4
```

```bash
# Remote chain: one split-stage per box, the head lists them with --split-remote in chain order
split-stage -m model.gguf --layer-range 24:32 --role relay --rs-seq 8 -c 8192 --host 0.0.0.0 --port 9331
split-stage -m model.gguf --layer-range 32:40 --role tail  --rs-seq 8 -c 8192 --host 0.0.0.0 --port 9332
llama-server -m model.gguf --split-layer-range 0:24 -c 8192 \
  --split-remote 10.0.0.2:9331 --split-remote 10.0.0.3:9332
```

```bash
# Remote chain with DFlash k=2: the product path is -md plus --spec-type
llama-server -m model.gguf --split-layer-range 0:32 -c 4096 \
  --split-remote 10.0.0.3:9332 -md dflash.gguf --spec-type draft-dflash --spec-draft-n-max 2
```

- `--split-layer-range` is the layer range this process loads, and the head is always one;
  `--split-tail-range` is a same-box tail and is mutually exclusive with `--split-remote`;
  `--split-remote` repeats in chain order.
- `--role` and `--rs-seq` are mandatory for `split-stage`. The role used to be inferred by the
  loader, and a wrong inference produced fluent but wrong output.
- Raise `--split-rs-seq` when speculating: one round rolls back `n_max` positions. The tail has to
  report `has_nextn` in `OP_HELLO`; otherwise the server logs a WARN and drops draft-MTP. On weak
  machines set `-t 8` explicitly — it feeds both `n_threads` and `n_threads_batch`, and both default
  to 4.
- The current spelling is `--spec-draft-n-max`. Upstream removed `--draft-max` / `--draft-n`, and the
  old names abort the process.

## Hard constraints

1. **Architecture.** A chain head has to export hidden state from before `output_norm`.
   `split_arch_exports_pre_norm()` (`src/llama-context.cpp`) accepts `qwen35` and `qwen35moe`; any
   other architecture aborts before context creation. The tail only produces logits, so its
   architecture is free. The four M-RoPE sections cross segments (`split_expand_pos()` in
   `src/llama-context.cpp`); the image path has no end-to-end run yet — see
   [SPLIT-MEASUREMENTS.md] §7.
2. **The remote chain is single-sequence.** `llama_split_attach_remote()` rejects `n_seq_max > 1`.
   Concurrency means several servers on the head box, each with its own chain; inside one box use the
   local chain. See [SEGMENTED-LIMITS.md] §2.4.
3. **Pick the draft source by channel.** DFlash / DSpark travel as `layer_inp` taps (protocol v9) and
   cost nothing on a remote chain. MTP travels the nextn channel and accepts less on a chain than on
   a whole model; the cause is in the measurements section. See [SEGMENTED-LIMITS.md] §2.2.
4. **`--split-layer-range` validity.** A range starts at 0; a range that ends mid-model needs
   `--split-tail-range` or `--split-remote`. Both checks live in `common_init_from_params()`
   (`common/common.cpp`) and read the range back from the model (`llama_model_layer_range()`), which
   also covers the `LLAMA_LAYER_RANGE` entry point. Residual boundary: [SEGMENTED-LIMITS.md] §2.6.
5. **iGPU: offload whole segments.** `-ngl 99` on every segment; intermediate splits are worse, the
   crossover moves with prompt length, and 35B does not fit the shared heap, so the mainline runs on
   CPU. See [IGPU-SLOWDOWN.md].
6. **Hybrid partial `seq_rm`.** Upstream fails at the default `n_rs_seq = 0` and deletes no attention
   KV rows (upstream #28007 / #31656); this repo returns `bool` and lets five call sites degrade. See
   [SEGMENTED-LIMITS.md] §2.9.

## Measurements: draft-source choice and the cross-machine run

Baseline: 35B `Qwen3.6-35B-A3B-UD-Q4_K_M` (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, 21.11 GiB), greedy,
k=2, 2026-11-09. The three whole-model rows run on CPU (`-c 4096 -b 512 -t 8 -ngl 0`) with
`n_predict 64` over three prompts (B-tree, code, Chinese). The three chain rows put the head `0:32`
on this box and the tail `32:40` on a Surface (i5-1035G4 4C8T / 16 GB, Tailscale at 9.5 MB/s,
RTT 10–12 ms), prompt 1102 tokens.

| Shape | accept (a/g) | mean_len | vs whole model |
| --- | ---: | ---: | --- |
| whole + MTP k=2 | 0.9098 | 2.820 | — |
| whole + DSpark-6L k=2 | 0.8357 | 2.671 | — |
| whole + DFlash k=2 | 0.8298 | 2.660 | — |
| chain + DFlash k=2 | 117/141 = 0.82979 | — | same prompt by prompt |
| chain + DSpark k=2 | 117/140 = 0.83571 | — | same prompt by prompt |
| chain + MTP k=2 | 108/157 = 0.68790 | — | −24% |

MTP is the best short-context choice on a whole model. On a chain, DFlash and DSpark lose nothing —
local and remote chains agree — and the outputs stay byte-identical 3/3 (250 / 244 / 249
characters). The MTP gap comes from the two chain-side changes recorded in
[SPLIT-MEASUREMENTS.md] §11.4: catch-up decode runs only when `!is_chain && !is_remote`, and
`draft()` clears the draft KV every round under `is_chain`. Under the current draft-KV strategy the
loop is self-consistent; an opt-in catch-up has to change both sites at once.

Prefill tap cost: A 32.1 s, B 32.8 s, C 29.7 s, so A−C ≈ +2.4 s — in line with 25 MB ÷ 9.5 MB/s
≈ 2.6 s. The taps cross the wire in 512-row ubatches; a per-token round trip would show a difference
an order of magnitude larger.

The markov head (DSpark-6L) amplifies copyability, not context length. At k=8 and `n_predict 256` it
beats DFlash by 15.3 points on mid (1146-token prompt), by 4.1 points on hard (1157 tokens, Chinese
rewrite), and by 2.9 points on short. Mid and hard are nearly the same length, so the gap tracks how
much of the prompt can be copied. Per-position numbers put the head's value from pos2 onward, which
k=2 never reaches. Choose a source by how copyable the workload is (JSON keys, code signatures, tool
calls), then by whether k and `n_predict` sit in the range the head covers.

Measurement traps:

- Acceptance rates do not compare across `n_predict`. The same pair of draft sources differs by
  +0.59 points at `n=64` and by −0.94 points at `n=256`; report `n_predict` and k with every
  acceptance figure.
- `mean_len = 1 + accept × k`, so mean_len at different k is not comparable.
- DSpark-6L is based on an NVFP4 quantisation rather than the official BF16, so the comparison holds
  only between those two files.

## Code and acceptance

- Fork: `H:\llamacpp-c`, branch `rebase-merge`, a single commit `0c5c3bdd4` on parent `f477f795c`
  (current upstream, 193 commits later), tree `982ccecd`; against upstream, 52 files /
  +17831 −230.
- [llamacpp-split.patch]: 42 files / +14287 −229 / 16166 lines / 840.6 KiB. The pure split chain:
  layer-range loading, local and remote segments, aggregated memory, checkpoints, protocol v7
  verification channel / v8 remote MTP draft / v9 remote `layer_inp` taps, grammar and rbudget
  paths, `seq_rm` failures made visible, device-buffer width predicate. No SLIM-ARC; it compiles and
  passes the full acceptance run on its own.
- [llamacpp-slimarc.patch]: 13 files / +3544 −1 / 3662 lines / 155.2 KiB. SLIM-ARC prefetch, off by
  default, installed only with `SLIM_ARC_PREFETCH=1`. Prefetch is the only piece borrowed; expert
  residency, KV eviction and the I/O scheduler stay out of scope — see [SLIM-ARC-接入计划.md].
- Composition check: `f477f795c` + main patch → tree `d28e10df`; adding the increment → tree
  `982ccecd`, identical to HEAD's tree (`git write-tree`).
- Against upstream: 16 files are byte-identical to `f477f795c`, against zero before the rebase. The
  changes removed from 17 files were lag rather than contribution — upstream already carries those
  commits, so our adaptation is no longer needed. `src/models/dflash.cpp` re-entered the change set
  with its device-buffer diagnostic. Inventory in the upstream-overlap section of [PATCHES.md].

Every acceptance check compares bit- and byte-identical logits, not generated text; the table below
records the 2026-11-09 run:

| Suite | Result | Criterion |
| --- | ---: | --- |
| `bash build-split/verify.sh` | 52 (eleven groups: 22 in 1–8, 2 in 9, 20 in 10, 8 in 11) | bit / byte identity; group 11 runs the product path `llama-server + chain + -md` |
| `verify_chain_death.py` | 36 | 120 s read timeout, dead chain with a live process, failures visible |
| `verify_seq_id.py` | 9 | `seq_id` and positions across segments |
| `verify_mtp.sh` / `verify_mtp_e2e.sh` | 6 / 6 | chain draft bit-identical to the whole model, end-to-end byte-identical |
| `verify_grammar_spec.sh` / `verify_rbudget_spec.sh` | 23 / 36 | grammar whole-frame rollback, budget tightens only |
| `verify_seq_rm_exact.py --expect-known-fail` | 1 pass / 5 known-fail | the five are upstream hybrid baselines kept as controls |
| Pure split build (main patch only) | 52 + 36 | built and run independently on tree `d28e10df` |

The v2 chain is checked by `python checks/verify_*.py` and
`python -m unittest test_topology test_contract`; per-script counts and the latest results are in
[SPLIT-MEASUREMENTS.md] §6 and §9.7.

## Documentation map

| File | Contents |
| --- | --- |
| [ARCHITECTURE.md] | The three loading mechanisms, the handoff point, hard constraints, what counts as evidence, the native-RPC boundary |
| [PATCHES.md] | Patch generations and their differences, the upstream rebase record, full patch text |
| [SEGMENTED-LIMITS.md] | Every boundary: root cause, blast radius, what fixing it would cost |
| [SPLIT-MEASUREMENTS.md] | v2.1 measurements: checkpoints, MTP, DFlash, protocol v9, the selection matrix, the zero-loss cross-machine run |
| [MEASUREMENTS.md] | All v2 Python-chain measurements |
| [SPECULATION.md] | Speculation criteria, cost model, k sweep, integration measurements |
| [DEPLOYMENT.md] | What goes where, how to change the split, how to start remote workers |
| [TESTING.md] | Test channels: what is measured, how to run it, what counts as a pass |
| [MEASUREMENT-PITFALLS.md] | Why per-segment timing lies, and how to measure instead |
| [MEMORY-LIMITS.md] | Memory and VRAM arithmetic, calibration constants, core-pinning discipline |
| [IGPU-SLOWDOWN.md] | Why the iGPU loses, the `-ngl` sweep and where the crossover sits |
| [PROGRESS.md] | Current capability snapshot and optimisations in place |
| [SAMPLING.md] · [TUI.md] · [RESPONSES-API.md] | Sampling mapping · TUI pages and commands · `/v1/responses` translation layer |
| [SLIM-ARC-接入计划.md] | When the remaining page-fault work pays off |

Further topics (`LILICORR-轻量头.md`, `ANDROID-SURFACE-段接入评估.md`, `ISSUE-*.md`) live in
`docs/`, with retired documents in `docs/archive/`. The same relay applied to diffusion models is
[K-sd.cpp](https://github.com/SgfKrc/K-sd.cpp): a segment boundary carries MB-scale tensors there,
both sides wait for every segment within one sequence, and the headroom sits in many requests.

## License

MIT — see [LICENSE](../LICENSE). Upstream llama.cpp is MIT as well and this repo's patches ship under
the same terms; model weights and other third-party artifacts keep their own licenses.

[ARCHITECTURE.md]: ARCHITECTURE.md
[PATCHES.md]: PATCHES.md
[SEGMENTED-LIMITS.md]: SEGMENTED-LIMITS.md
[SPLIT-MEASUREMENTS.md]: SPLIT-MEASUREMENTS.md
[MEASUREMENTS.md]: MEASUREMENTS.md
[SPECULATION.md]: SPECULATION.md
[DEPLOYMENT.md]: DEPLOYMENT.md
[TESTING.md]: TESTING.md
[MEASUREMENT-PITFALLS.md]: MEASUREMENT-PITFALLS.md
[MEMORY-LIMITS.md]: MEMORY-LIMITS.md
[IGPU-SLOWDOWN.md]: IGPU-SLOWDOWN.md
[PROGRESS.md]: PROGRESS.md
[SAMPLING.md]: SAMPLING.md
[TUI.md]: TUI.md
[RESPONSES-API.md]: RESPONSES-API.md
[SLIM-ARC-接入计划.md]: SLIM-ARC-接入计划.md
[llamacpp-split.patch]: llamacpp-split.patch
[llamacpp-slimarc.patch]: llamacpp-slimarc.patch
