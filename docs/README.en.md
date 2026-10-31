# K-Llama v2

[中文](../README.md) | **English**

> Status: two-stage cross-machine chain working; DFlash parallel drafting, SLIM-ARC prefetch,
> the TUI, the OpenAI-compatible endpoint, sampling and VRAM verification are all in place
> Last updated: 2026-10-31

A llama.cpp engine built around **non-whole-model loading**. The full checkpoint stays where it
is; each node reads only the layers it owns, and stages exchange nothing but a hidden state
(`n_embd` float32 values, 8 KB).

Qwen3.6-35B-A3B (MoE, 40 layers, 14.8 GiB) is split 32 + 8 across two machines: the laptop
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
At 128 tokens the acceptance rate drops from 96% to 70% and throughput to 13.4 — see
[`SPECULATION.md`](SPECULATION.md).

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
cut.py                offline layer cutting (now only for cross-checking)
checks/               20 verification scripts
```

Line counts are omitted on purpose: they expire with every refactor, responsibilities do not.

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

Qwen3.6-35B-A3B (MoE, 40 layers, 14.79 GiB of weights), single sequence, greedy:

| Configuration | Throughput | Basis |
| --- | ---: | --- |
| Whole model on the laptop, `-ngl 6` | 10.5 tok/s | end to end |
| Two machines, 32:40, no speculation | 12.2 tok/s | decode phase, 128-token runs |
| **Two machines + DFlash k=5** | **20.9 tok/s** | decode phase, 64-token runs |
| Two machines + DFlash k=5 | 13.4 tok/s | decode phase, 128-token runs |

Everything the two-machine chain gains over one machine comes from speculation; the split itself
remains a net loss. Per-segment time is U-shaped in `n_past`; see
[`MEASUREMENT-PITFALLS.md`](MEASUREMENT-PITFALLS.md).

---

## Not done (ordered by what this machine can verify)

**1. A greedy switch for the TUI.** `stream.py` now shares `chain_client`'s decode engine, so the
server can run speculation; koakuma still has no sampling flags and takes the default sampling
path, where drafting does not hold. A `--greedy` passthrough is all it needs.

**2. Speculation with sampling.** The verifier only exposes argmax; sampling needs rejection
sampling plus per-position logits from the shim. The chain uses quantised weights across
machines, so "the two distributions agree" is a weaker premise here than on one box.

**3. SLIM-ARC's on-demand loader.** Only prefetch is integrated. Verifying the loader means
manufacturing a segment that does not fit — the whole checkpoint is in the page cache here, so
`issued` stays at 0. Either wait for the weak device or squeeze available memory artificially.
It optimises the degraded mode, which is not today's bottleneck.

**4. Multi-sequence token sharing.** Needs `kv_unified`, and every segment has to track
per-sequence positions. The server serialises requests behind one global lock, so concurrency
demand is low for a single user. Doing it properly starts with how the 30 linear layers' recurrent
state would be isolated per sequence.

**Done**: cutting tool retired, both draft sources integrated, DFlash k sweep, SLIM-ARC prefetch
with its win32 port, and **speculation wired into the serving path** — `stream.py` used to carry
its own plain decode loop, which is why the server and TUI never saw DFlash. It now keeps only
UTF-8 reassembly, timing and stop conditions, and delegates decoding to `chain_client`'s
`forward` / `speculate` / `DFlashDriver`. The server gained `--spec-k` / `--dflash` /
`--dflash-block`, honoured only for `temperature=0` requests. Measured: the same greedy request
produces byte-identical output with drafting on and off. Details in [`PROGRESS.md`](PROGRESS.md).

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
| [`SPECULATION.md`](SPECULATION.md) | Speculative decoding: decisions, cost model, measurements |
| [`PATCHES.md`](PATCHES.md) | Three generations of patches, the full patch, RPC boundary |
| [`SLIM-ARC-接入计划.md`](SLIM-ARC-接入计划.md) | When page-fault optimisation is worth enabling |
| [`LILICORR-轻量头.md`](LILICORR-轻量头.md) | Block-consistency reranking for DFlash, and what training it costs |
| [`ANDROID-SURFACE-段接入评估.md`](ANDROID-SURFACE-段接入评估.md) | Criteria for weak devices, Android constraints |
| [`SAMPLING.md`](SAMPLING.md) · [`TUI.md`](TUI.md) | Sampling mapping · TUI design |
| [`ISSUE-recurrent-snapshot-rollback.md`](ISSUE-recurrent-snapshot-rollback.md) | Hybrid-model rollback defect across ubatches |
| [`PROGRESS.md`](PROGRESS.md) | Current capability snapshot and optimisations applied |

Verification: `python checks/verify_*.py`, `python -m unittest test_topology`.
