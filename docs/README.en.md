# K-Llama v2

[中文](../README.md) | **English**

> Status: **Usable** — three shapes are working and have been soak-tested: two machines at
> 20.9 tok/s, two weak devices at 6.50, three-stage over the public internet at 5.00; the
> three-stage form ran 60 rounds of `gen 256` with zero anomalies. The portable bundle deploys
> from scratch and produces tokens in-bundle (verified by three independent walkthroughs). The
> one shape never verified is a **clean Windows install** — see `DELIVERY.md` §1.2.
> Last updated: 2026-11-04
> Applies to: the engine as a whole — what it is, what has been measured, and what is still
> missing; readers are integrators and reviewers.

A llama.cpp engine built around **non-whole-model loading**. The full checkpoint stays where it
is; each node reads only the layers it owns, and stages exchange nothing but a hidden state
(`n_embd` float32 values, 8 KB).

Qwen3.6-35B-A3B (MoE, 40 layers, 21.87 GiB) is split 32 + 8 across two machines: the laptop
holds 32 layers with all attention and dense weights on the GPU, the remote box holds 8 layers
on CPU. Decode phase reaches **20.9 tok/s** with DFlash at k=5, against 10.5 tok/s for the whole
model on the laptop's GPU alone.

**Why it works**: every node maps the same checkpoint, but only *creates* and *reads* the tensors
of its own layer range — three mechanisms: layer-count override (`LLAMA_LAYER_RANGE`), name
remapping (local `blk.i` → source `blk.lo+i`), and lazy global tensors. The relay vector is taken
*before* `output_norm`, and only 8 KB of hidden state crosses the wire.
See **[`ARCHITECTURE.md`](ARCHITECTURE.md)** for the full account.

---

## Relationship to SLIM-ARC

The layer-segment relay here is our own; it is not the same layer of the stack as
[SLIM-ARC](https://github.com/Nexa-Language/SLIM-ARC).

SLIM-ARC is a submission by Sun Yat-sen University to the 2026 National Collegiate Computer
System Capability Competition (OS Design track, Proj59). It is a **MoE inference runtime for
memory-constrained edge systems**: mmap-backed on-demand loading, phase-aware page advice,
MoE expert prediction and reclamation, pressure-aware residency, a unified I/O budget and KV
cache management. It targets the case where the model is much larger than physical memory —
its headline result fits Qwen3-Next-80B-A3B into an 8 GiB cgroup and moves decode from
0.08 to 0.43 tok/s.

Two paths at the same problem:

| | This project | SLIM-ARC |
| --- | --- | --- |
| Strategy | **Scale out**: give layers to more machines | **Compress down**: swap pages in and out on one box |
| Precondition | each node fits its own segment | no single box fits the model |
| Cross-boundary cost | 8 KB hidden state over TCP | page faults from disk |
| Shape | multiple processes + TCP | one process + page-level scheduling |

What we borrow is its **prefetch module** — layer-boundary readahead that turns random page
faults into sequential reads. It is compiled into `llama.dll` with a win32 port. Its expert
residency, KV eviction, unified I/O scheduler and on-demand loader are not integrated.
See [`SLIM-ARC-接入计划.md`](SLIM-ARC-接入计划.md) and [`PATCHES.md`](PATCHES.md).

---

## What the split buys

Splitting pays off when one machine cannot hold the model: two machines' resources add up
instead of sharing one GPU's memory.

The split itself is a **net loss** — the same GPU memory is now shared by two segments. What
makes the two-machine chain faster than one machine is speculative decoding, not the split:

| Stage | Mechanism | Decode-phase throughput |
| --- | --- | ---: |
| No speculation | one token per step, one chain round trip | 12.2 |
| MTP recursion | k=2 off the tail's NextN head | 1.13× |
| **DFlash parallel block** | k=5, six candidates per round, one verify pass | **20.9** |

Measured on the 32:40 chain with `def quicksort(arr):`, 64-token generations, decode phase only.
At 128 tokens both acceptance and throughput drop — see
[`SPECULATION.md`](SPECULATION.md) §11.

---

## Scope

- **Architectures**: Qwen 3.5 and newer (`qwen35` / `qwen35moe`). Other architectures need
  tensor naming, RoPE variants and MTP position handling first.
- **Quantisation**: below 8 bit (Q4_K and friends). This is also what makes float32 relay
  sufficient: Q4_K's relative error is around 1e-2, far above float32 transport error, so
  transport noise sits under the quantisation noise.
- **Segments = nodes**. Splitting one machine into two segments is a design error: every extra
  process duplicates global tensor reads, a compute buffer and a round of page contention.

### Transport stays float32

Compressing to 4 KB was tried, twice:

| Transport dtype | Per-hop max\|diff\| | Greedy tokens identical |
| --- | ---: | ---: |
| float32 | 0 | 5/5 (baseline) |
| float16 | 3.9e-3 | **4/5**, diverges at step 22 |
| int16 fixed point | 2.2e-4 | **4/5**, diverges at step 16 |

int16's quantisation step is 20× finer than float16 and still flips a token — the divergence
points sit where the output is nearly tied, where 1e-4 is enough. The criterion is **bit-identical
greedy tokens across every prompt**; "mostly identical" does not qualify.

---

## Quick start

```powershell
koakuma                                    # TUI, one command
koakuma --topology topology-cross.toml     # two-stage chain across machines

# drive the chain directly (--spec-k defaults to -1: pick by prompt shape)
python chain_client.py --topology topology-cross.toml --prompt-text "中国的首都是" --chat --gen 32

# add DFlash parallel drafting (k=5 is the measured knee; requires --greedy)
python chain_client.py --topology topology-cross.toml --greedy --dflash <dflash.gguf> \
       --dflash-block 6 --prompt-text "def quicksort(arr):" --gen 64
```

Dependencies: `pip install numpy psutil gguf textual` (`gguf` only for cutting and provenance
checks, `textual` only for the TUI).

Manual steps: build the shim (`shim\build_shim_msvc.ps1`) → put the same checkpoint on every
machine → `layer_range` in the topology decides who loads what. Remote workers are started by
the topology's `start_cmd`. Full procedure in [`DEPLOYMENT.md`](DEPLOYMENT.md).

---

## Layout

```
protocol.py           frame format and message codec
wire.py               transport dtype for hidden state (float32 by default)
topology.py           topology parsing + fail-closed validation
keep_head.py          in-process shim wrapper
keep_head_worker.py   subprocess wrapper (n_gpu_layers=0 needs a fresh process)
engine.py             per-segment load and forward
stage_worker.py       one segment as a TCP server
chain_client.py       spawns workers, drives decode, both draft sources, memory/VRAM budget
stream.py             streaming generation with UTF-8 reassembly
server.py             /v1/chat/completions, /v1/models
proto_responses.py    translation layer for /v1/responses
koakuma.py            TUI and boot screen
checks/               verification and diagnostic scripts (10 verify_*, the rest diagnostics/probes)
```

---

## Four constraints worth knowing

**① Segments must tile the source model exactly.** A missing layer does not raise — it produces
fluent but wrong Chinese. `topology.load()` validates contiguity, non-overlap and declared widths
on every run.

**② VRAM oversubscription is silent.** WDDM lets an allocation exceed dedicated VRAM and backs
it with system memory while llama.cpp still prints `offloaded N/N`. Budgets are checked before
load and measured after (`VRAM_PER_BYTE` in `chain_client.py`).

**③ Never pin cores.** A worker has 5+ live threads; binding them to the two hyperthreads of one
physical core makes them fight.

**④ Cut points and the tail's layer count must be multiples of `full_attention_interval` (=4).**
qwen35moe derives layer type from the *local* layer index while that index is mapped back to a
source index, so the two only agree modulo 4 when `lo` is a multiple of 4. A 6-layer split like
`32:38` is rejected at parse time.

---

## Measured

Qwen3.6-35B-A3B (MoE, 40 layers, 21.87 GiB of weights), single sequence, greedy:

| Configuration | Throughput | Basis |
| --- | ---: | --- |
| Whole model on the laptop, `-ngl 6` | 10.5 tok/s | end to end |
| Two machines, 32:40, no speculation | 12.2 tok/s | decode phase, 128-token runs |
| **Two machines + DFlash k=5** | **20.9 tok/s** | decode phase, 64-token runs |
| Two weak devices, 24:40 (Surface + Y700, both CPU-only) | 6.50 tok/s | decode phase, 24-token runs |
| Three-stage 0:24 / 24:32 / 32:40 over Tailscale | 5.00 tok/s | decode phase, 24-token runs |

Everything the two-machine chain gains over one machine comes from speculation; the split itself
remains a net loss. Longer runs decay: the same configuration at 128 tokens gives lower acceptance
and lower throughput — the full k=3/5/8/15 sweep and its two conclusions are in
[`SPECULATION.md`](SPECULATION.md) §11. Per-segment time is U-shaped in `n_past`; see
[`MEASUREMENT-PITFALLS.md`](MEASUREMENT-PITFALLS.md).

---

## Not done (ordered by what this machine can verify)

**1. Speculation with sampling.** The prerequisites are all in place — splitting introduces no error
(2B over 16 positions and 35B over 12, logits bit-identical, top-8 fully coincident, TV=0; see
`DELIVERY.md` §1.1), and both logit return and the probe (`checks/probe_logits_delta.py`) exist.
What is missing is the acceptance test itself: today it compares token equality, while sampling
needs rejection sampling, which compares two distributions.

**2. SLIM-ARC's on-demand loader.** Only prefetch is integrated. Verifying the loader means
manufacturing a segment that does not fit — the whole checkpoint sits in the page cache here, so
`issued` stays at 0. On the weak devices the memory headroom is thin enough that the loader *could*
fire, but the measured effect there is negative: 22 GB of model against 16 GB of RAM lands in the
thrashing band. It optimises the degraded mode, which is not today's bottleneck.

**3. Multi-sequence token sharing.** Needs `kv_unified`, and every segment has to track
per-sequence positions. The server serialises requests behind one global lock, so concurrency
demand is low for a single user. Doing it properly starts with how the 30 linear layers' recurrent
state would be isolated per sequence.

**Done**: cutting tool retired, both draft sources integrated, DFlash k sweep, SLIM-ARC prefetch
with its win32 port, speculation wired into the serving path, and a sampling switch for the TUI.

The serving path holds no decode loop of its own: `stream.py` keeps only UTF-8 reassembly, timing
and stop conditions, and delegates decoding to `chain_client`'s `forward` / `speculate` /
`DFlashDriver`. Both the server and the TUI expose drafting flags, honoured only at
`temperature=0` — measured: the same greedy request produces byte-identical output with drafting on
and off. Details in [`PROGRESS.md`](PROGRESS.md).

---

## Docs

The Chinese README is the source of truth; this file mirrors it.

| File | Contents |
| --- | --- |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | **Core mechanism**: why non-whole-model loading works, hard constraints, criteria, RPC boundary |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | What goes where, how to change the split |
| [`MEASUREMENTS.md`](MEASUREMENTS.md) | All measurements |
| [`MEASUREMENT-PITFALLS.md`](MEASUREMENT-PITFALLS.md) | Why per-segment timing lies |
| [`MEMORY-LIMITS.md`](MEMORY-LIMITS.md) | Memory and VRAM arithmetic |
| [`SPECULATION.md`](SPECULATION.md) | Speculative decoding: decisions, cost model, the k sweep (§11), measurements |
| [`PATCHES.md`](PATCHES.md) | Three generations of patches, the full patch, RPC boundary |
| [`SLIM-ARC-接入计划.md`](SLIM-ARC-接入计划.md) | When page-fault optimisation is worth enabling |
| [`LILICORR-轻量头.md`](LILICORR-轻量头.md) | Block-consistency reranking for DFlash, and what training it costs |
| [`ANDROID-SURFACE-段接入评估.md`](ANDROID-SURFACE-段接入评估.md) | Criteria for weak devices, Android constraints |
| [`SAMPLING.md`](SAMPLING.md) · [`TUI.md`](TUI.md) | Sampling mapping · TUI design |
| [`ISSUE-recurrent-snapshot-rollback.md`](ISSUE-recurrent-snapshot-rollback.md) | Hybrid-model rollback defect across ubatches |
| [`PROGRESS.md`](PROGRESS.md) | Current capability snapshot and optimisations applied |

Verification: `python checks/verify_*.py`, `python -m unittest test_topology test_contract`.
