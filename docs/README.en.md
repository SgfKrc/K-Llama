# K-Llama

K-Llama (Llama for Koakuma) is a distributed inference core for heterogeneous edge devices. The mainline is the lightweight GGUF/llama.cpp engine; the repository also owns a PyTorch layered-distribution engine plus a **layer pipeline** (including cross-framework layer relay), and the user-facing entry point is a cross-platform Textual TUI.

> **Independent project · not official**: K-Llama is an independent student innovation project (Beijing Jiaotong University, 2026) with no affiliation, sponsorship or endorsement from the llama.cpp project, and it does not represent that project's position. It is built on llama.cpp (a descriptive reference only; no ownership of or trademark claim to "llama.cpp", "llama" or any upstream name is asserted). Upstream components keep their own licenses and version pins.

> Status: **current**
>
> Updated: 2026-10-14
>
> This README describes only the current boundary of the main repository and its reproducible entry points. Experiment logs, historical implementations and external sub-projects are not equivalent to production capability.
>
> 中文: [../README.md](../README.md) - this file: docs/README.en.md

## What the Main Repository Does

- Runs or coordinates GGUF inference on Windows/Linux PCs, devices without CUDA, and Android nodes.
- Lets models too large for a single machine be carried jointly by multiple nodes under a layer-segment contract; each node holds only the model part actually assigned to it.
- Lets an Edge node complete local inference with <=1B models by default, while retaining the ability to serve as an RPC worker for larger models.
- Manages three node kinds (local, remote RPC, cross-framework) under one layer-segment contract, and selects the engine by device profile and capability.
- Controls distributed admission and failure recovery through device profiling, capacity planning, model identity, layer-segment contracts, leases and epoch fencing.
- Uses a Textual TUI for chat, status, model assets, nodes, distributed layout, queue, device, logs and settings; read-only single commands use a standard-library thin layer.

The engine is **dual-track**, and both tracks live in the main repository:

| Track | Engine | Capabilities | Dependency |
| --- | --- | --- | --- |
| **L** | llama.cpp / GGUF | Single-machine inference, RPC partial residency, TUI chat, the optimization trio | No torch (Edge default) |
| **D** | PyTorch / Safetensors | Layer splitting and tensor placement, inter-layer pipeline, multi-node layer-segment hosting, the upstream side of cross-framework relay, and the same-card control experiments | torch |

The L track is itself split into two paths by capability, decided by `select_inference_path` in `src/inference_path_selector.py` and controlled by the `INFER_PATH` switch in `src/config.py` (default `auto`): with CUDA detected and a local `llama-server` binary present it uses the `llama_server` subprocess path, otherwise it uses the llama.cpp binding inside the QLH engine. Layer-segment relay is forced onto the binding path, and an explicit `llama_server` request with no binary fails closed. Both paths report `engine_type="llama_cpp"` externally (`src/llama_server_engine.py`); the difference surfaces through `inference_path` / `capability()` / `supports()`.

The D track is not part of the Edge default dependency set, but layer splitting, the layer pipeline and cross-framework relay are implemented by the PyTorch stack (`src/model_module.py`, `src/tcp_comm.py`, `src/qwen3_pipeline_*.py`). On the same card llama.cpp is faster for a single sequence (about 4.3x; for the measurement setup see the [cross-framework relay project report](跨框架层接力-项目报告.md)), so the default production path is the L track.

## Architecture Overview

K-Llama is **two layers in one process**: a control plane aimed at people, and an engine layer aimed at machines and protocols. There is exactly one boundary between them — the layer-range contract `(layer_range, engine, location)`.

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Control plane (aimed at people)                                            │
│ Textual TUI · read-only single commands · HTTP API (/api/cluster/*)        │
│ device profile · capacity plan · layer-range contract · lease / epoch      │
│ fencing · admission                                                        │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ layer-range contract (layer_range, engine, location)
┌──────────────────────────────┴─────────────────────────────────────────────┐
│ Engine layer (aimed at machines)                                           │
│ Tier L: llama.cpp / GGUF            Tier D: PyTorch / Safetensors          │
│ ├ single-machine inference          ├ layer splitting and tensor placement │
│ ├ ggml RPC worker (borrowed GPU)    ├ inter-layer pipeline (qwen3_pipeline)│
│ └ layer-segment forward (shim)      └ cross-framework upstream (model_mod) │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ segment channel: v3 stage (stage_offer_v3 / layer_forward)
┌──────────────────────────────┴─────────────────────────────────────────────┐
│ Nodes and transport (cross-host, heterogeneous)                            │
│ local loopback · SSH tunnel · Surface (x86_64, Windows) · y700 (ARM64)     │
│ hidden compression: f32 / f16 / bf16 / int8_block128 · weak-net budgets    │
└────────────────────────────────────────────────────────────────────────────┘
```

One four-stage chain that has been measured end-to-end:

```
prompt → torch(0..7) →hidden→ Surface(8..15) →hidden→ y700(16..19) →hidden→ llama(20..23+head) → token
           local CUDA          x86_64 Windows        ARM64 Android           local llama.cpp
```

The acceptance criterion is **per-token argmax equality with the same-precision monolithic model**; any divergence is reported as FAIL, with no approximate substitute. Per-segment records: [current relay baseline](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md).

The two layers carry different verification cost, so the change site determines the required evidence:

| Criterion | Control plane | Engine layer |
| --- | --- | --- |
| Primary user | People (end users / operators) | Other software (TUI, API orchestration, peer nodes) |
| Failure mode | Degraded experience, retryable | Numerical error, potentially silent |
| Interface stability | Evolvable (pages / commands may change) | Strong contract (layer-segment contract, protocol version, record schema) |
| Separately replaceable | Yes (swap the front end, keep the engine) | No (replacing the engine changes numerical semantics; re-run the per-token comparison) |

Therefore: control-plane changes need UI and contract tests; engine-layer changes need per-token comparison, record-schema validation and matrix measurements.

## Layer Pipeline and Cross-Framework Layer Relay

### Unified node abstraction

The layer pipeline reduces "who holds which layers, with which engine, where, and at what capacity" to one abstraction: `(layer_range, engine, location)`. The three node kinds are isomorphic and share one contract and validation set:

| `kind` | Meaning | Engine | Transport |
| --- | --- | --- | --- |
| `local` | Layers inside the local process | llama.cpp / pytorch | in-process |
| `remote_rpc` | Borrowed compute (the `ggml-rpc-server` on an Android/PC node) | llama.cpp | network (ggml RPC) |
| `cross_framework` | Relay across engines (torch upstream + llama.cpp downstream) | both | in-process or stdio |

Implementation and endpoints:

- `src/pipeline_node_contract.py`: the `PipelineNode` contract, mapping of existing artifacts, fail-closed layout validation;
- `src/pipeline_capacity.py` for capacity solving, `src/pipeline_assignment_manifest.py` for the assignment manifest;
- `src/pipeline_reshard.py`: capacity re-solve + artifact readiness gate + atomic epoch commit;
- `GET /api/cluster/layers`, `GET /api/cluster/pipeline-capacity`, `GET /api/cluster/pipeline-reshard`.

### Cross-framework layer relay (D→L)

The upstream PyTorch segment computes up to layer N and hands the hidden states to a downstream llama.cpp cut-layer model. The injection point is llama.cpp's `llama_batch.embd` field, so the PyPI `llama-cpp-python` binding is enough — no fork or recompilation.

It solves two things: letting a model that does not fit on one machine be carried by several devices (the layer pipeline shares the same layer-splitting and hidden-handoff mechanism), and letting nodes with different engine capabilities (`local` / `remote_rpc` / `cross_framework`) share one layer pipeline. Cut-point assignment, mixed precision, operator replacement and batch overlap all build on passing hidden states between layers.

**Current conclusions** (correctness, speed and capacity are recorded separately; full data and method: [current baseline](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md)):

| Item | Statement |
| --- | --- |
| Correctness | Main-repository dual-engine D→L matrix 27/27 per-token identical (qwen2.5-0.5B K=4/8/12/16/20, qwen3.5-2B K=8/12/16/20; loads prefill 32/128/512, decode 32/64/256; batch 2/4; mixed precision fp16·f32·NF4 × Q4_K_M) |
| Sample | qwen2.5-0.5B, 12+12 layers, gen=32: upstream `model_module.forward_layers` + downstream `llama_engine.forward_layers_from_hidden`, **47.501 ms/step**, per-token identical to the plain llama.cpp monolithic model |
| Capacity | Two-segment gain **1.568x** for qwen2.5-0.5B and **1.457x** for qwen3-2b (strict definition: `C` counts only tensors outside every `*.layers.<i>.*`, computed by `src/relay_capacity_budget.py`; counting the `model.visual.*` tower as splittable gives 1.613x); under the same controlled 3.0 GB CUDA budget the monolithic model is rejected while a 12-layer upstream fits (the budget is a reproducible experiment constraint, not physical OOM) |
| Upstream precision | Full-precision PyTorch upstream + quantized GGUF downstream is a deliberate "incomplete quantization" strategy and must be compared against a same-precision downstream monolith |
| L→L channel | The pip binding's `llama_get_embeddings_ith` returns `output_norm(H)` (measured cos 0.999998) and cannot serve as a relay upstream; the patched keep-head channel works — `--path l2l_keep_head` / `d2l2l_keep_head` measured 32/32 (including a three-stage "1 torch upstream + 2 llama downstream" chain) |
| Cut-point solver | `scripts/relay_cut_plan.py` + `src/relay_cut_objective.py`: fit segment profiles from measurements (fixed overhead + per-layer cost), then solve, emitting `capacity_feasible` / `latency_estimate` / `risk_penalty`; supports n segments and the 4-layer multiple hard constraint for Qwen3.5. The two-segment loop passes on qwen2.5 (r² 0.96/0.99) and qwen3.5 (0.79/0.96) |
| Production role | Correctness already meets the Relay contract admission; speed only shifts the default routing preference. Cross-host long-run (LL/DL three-segment `gen=256`, per-token identical) and multi-segment cut-point combinations (G2/G3 32/32) passed on 2026-09-26; artifact auto-distribution and fault-injection acceptance remain open |

**Key quantitative conclusions**: the two sides' computation accounts for 99.3% (communication + sync + batching only 0.34%); the single largest gain is removing upstream idling on 20 layers (10.7x); the pipeline upstream holds only `embed_tokens + L0-3` (1.47 GB vs 4.55 GB, 3.1x). Rejected hypotheses: process boundary (8% only), reusing `llama_batch` (0.09%), naive truncation of upstream layers (numerically wrong), and treating `--override-tensor` as a speed lever (it is a capacity knob). The bottleneck is computation on both sides (cut point, kernel, batching), not the transport layer.

**P0 cut-point sweep (2026-09-18, upstream on GPU)**: the optimum is N=20 at 129.1 ms/step, 1.45x faster than the 186.8 ms/step without relay; upstream GPU costs about 2.5–4.3 ms/layer, downstream CPU llama.cpp about 5.8–8.6 ms/layer. With a CPU upstream the result reverses: no relay is fastest and total time is nearly insensitive to the cut point. Two general constraints: correctness does not change with the cut point, and the cut point must be a multiple of `full_attention_interval` (Qwen3.5 = 4), otherwise the cut-layer GGUF has mismatched layer types and fails to load.

**Against "all llama.cpp + CUDA" (2026-09-18)**: llama.cpp + CUDA monolithic (build-cuda, `-ngl 24`, f16) runs at 26.6 ms/token versus 129.1 ms/token for the best cross-framework relay, 4.85x slower. With a CUDA node in the cluster, the best practice is to use it as a llama.cpp CUDA worker (RPC / split) rather than as a relay upstream. P2 overlap (upstream GPU torch interleaved with downstream CPU llama.cpp) lowers serial 101.11 to 78.31 ms/token (1.291x). All of the above are same-host numbers.

Reports: `local_docs/evidence/relay-xframe/CORE-RELAY-XFRAME-02-*.json`; ticket-level statements in the [acceptance ledger](验收清单与资源限制登记.md).

### torch.compile and the layer-loop switch

Serving segmented forward passes with `torch.compile` uses two switches in `src/config.py` (overridable by environment variables):

| Switch | Default | Effect |
| --- | --- | --- |
| `USE_COMPILE` | `True` | Enables compilation. If unavailable it warns and falls back to eager without blocking startup; on Windows without [`triton-windows`](../requirements-compile.txt) this path is taken, and it works once installed (native compilation measured since 2026-09-19) |
| `USE_MONOLITHIC_FORWARD` | `False` | Additionally compiles the layer loop (`_LayerLoop`) used by `forward_layers()` |

Only the layer loop is compiled because `Qwen2Model.forward()` returns through `self.norm`, while a segmented forward with `has_lm_head=False` must return raw hidden states before norm, so the loop is wrapped and the pre/post steps stay in `forward_layers()`. Measured gains (`USE_MONOLITHIC_FORWARD=True`): Qwen2.5-0.5B 12 layers 1.674x, Qwen3.5-2B 24 layers 1.269x (different model/layer/prefill setups; the ratios are not directly comparable).

Three boundaries: compile and eager are not bit-identical (hidden difference is 1 ULP of f16, traced to the attention implementation path, so compile cannot be called worse); scenarios with a per-token-equality acceptance criterion must not enable compile; Windows needs `PYTHONUTF8=1` and `triton-windows`, and without them only the gain is lost, nothing crashes. Reports: `local_docs/evidence/relay-xframe/CORE-RELAY-XFRAME-02-a4-layer-loop-2026-09-18.json`, `…-b14-hybrid-layer-loop-2026-09-18.json`, `…-compile-numerics-2026-09-18.json`.

### Top-level transparency

The TUI and API top level only need aggregate resources (GPU/CPU/memory) and whether the run is distributed; the engine is chosen from resources + capabilities + goal. See [layer-pipeline node kinds and top-level transparency](archive/relay/层流水线节点类型与顶层透明性-可行性确认-2026-09-17.md).

## Current Status

| Capability | Current statement |
| --- | --- |
| Textual TUI | Unified `qlh` entry; chat, 9 functional screens and 1 debug fallback screen share one process and can start the backend on demand locally; write operations (model download/search/preflight/registration, cluster config, node management, log filtering/stats/export, device config, user settings) go through confirmation gates |
| Edge <=1B single machine | Model profiling, the GGUF/llama.cpp path and edge preflight exist; torch is not loaded by default |
| Same-host two-process RPC | llama host + `ggml-rpc-server` simulation and contract tests exist; not equivalent to cross-host production admission |
| PC RPC | Device scoring, automatic layer planning, lease/disconnect fallback and artifact-sync contracts exist; throughput gains are not claimed, and capacity gains are verified separately from D→L |
| Layer-segment contract and automatic resharding | Contract, fail-closed layout validation, capacity re-solve and atomic epoch commit have passed the development gate; real PC/Android fault injection, long-run and performance acceptance remain |
| Cross-framework relay (D→L) | Correctness admitted (27/27 per-token identical); cross-host long-run (LL/DL three-segment `gen=256`, 256/256) and multi-segment cut-point combinations (32/32) passed; artifact auto-distribution and fault-injection acceptance remain. Default routing prefers the directly runnable L/RPC path |
| PyTorch D track | The implementation of layer splitting, the layer pipeline and multi-node layer-segment hosting, and the control experiments; not part of the Edge default dependency set |
| Relay R | L→L, D→L, the f32/sampling matrix and SSH cross-host evidence all passed correctness verification; capacity gains quantified; default routing does not replace directly runnable L/RPC |
| Android | `qlh-android` P0 cross-compilation/JNI done; P1 device run, RPC worker, disconnect, thermal/power and security evidence outstanding |
| Runtime | Version windows follow [`requirements.txt`](../requirements.txt) (`transformers>=5.17.0,<5.18.0`, `llama-cpp-python==0.3.35`); all PyTorch sidecars are on 5.17.0; `.venv-test` ships `triton-windows` so native Windows `torch.compile` is available |
| Model assets | Model files are not committed (locally `models/` is a junction pointing outside the repository); the asset catalog covers registered models such as Qwen2.5-0.5B, Qwen3-0.6B, Qwen3.5-2B, Qwen3VL-4B, MiniCPM4-0.5B and DistilQwen2.5-DS3-0324-7B; Qwen3.6-35B-A3B and its community distill/derivative builds (both `qwen35moe` and `qwen3next` architecture headers) have been verified to load on llama.cpp `b10964` on this machine |
| Model selection and supply intent | Unified entry `POST/GET/DELETE /api/models/intent` (with `/catalog`, `/current`, `/connect`): the user submits only a model ID, engine, quantization and auto strategy, and the master runs planning, transfer, verification, loading and commit along the existing supply and native-prepare transactions; the Web model page, the Textual TUI and the Android selection panel share one catalog and operation progress (`src/model_intent*.py`, `src/api/routes_model_intent.py`) |

Experiments without a real-device, cross-host or production acceptance record count only as development evidence or PoC.

## Main Repository Boundary

| Kept in the main repository | Out of tree or not flowing back |
| --- | --- |
| llama.cpp/GGUF engine adaptation, RPC/layer-segment contracts, scheduling and recovery | Android UI/JNI project: `qlh-android` |
| FastAPI control plane, model/node/capability contracts | Product shell and front end: `qlh-shell` |
| Cross-platform Textual TUI, read-only command thin layer, Edge entry and quality gates | Release/installer: `qlh-release` |
| Single-host, same-host two-process and PC/Android mainline experiment interfaces | Toolbox: `qlh-toolbox` |
| PyTorch D-track engine (layer splitting, layer pipeline) and the cross-framework relay implementation | Image generation, web product UI, mail, operations workbench |
| Model registration, download verification, device profiling and distributed observability | Koakumix harness custom experiments, image generation and sidecar capabilities |

The main project keeps no image-generation runtime or assets; image generation belongs solely to Koakumix. Multimodality is not a fixed mainline dependency: the model fleet picks a text or vision model by device capability.

## Directory Layout

| Path | Contents |
| --- | --- |
| `src/` | K-Llama main code: control plane, engines, layer-segment/layer-pipeline contracts, TUI (grouped below) |
| `tests/` | pytest suite (TUI, RPC/layer-segment, scheduling, contracts, doc gates) |
| `scripts/` | Verification, experiment, environment and documentation tools (`edge_preflight.py`, `android_validation.py`, `llama_rpc_*.py`, `relay_*.py`, `run_doc_checks.py`, …) |
| `docs/` | Current documents; history and migrated content under `docs/archive/` |
| `schemas/` | Cross-process/cross-repo JSON Schema contracts (artifact-manifest, cluster-profile, experiment-record, …) |
| `fixtures/` | Test fixtures (including offline chat event replay for `qlh chat --fixture`) |
| `local_docs/` | Local experiment and acceptance raw records; not a public source interface |
| `runtime/` | Runtime logs and the llama.cpp runtime directory |
| `qlh.py` / `qlh_edge.py` | Interactive TUI/CLI entry and Edge entry |
| `qlh.bat` / `qlh.sh` / `K-Llama.bat` / `K-Llama.sh` / `bjtu.*` / `koakuma.*` | Launchers; `K-Llama` is the recommended alias, `bjtu` and `koakuma` are compatibility aliases, and the single entry script is `qlh.py` |
| `start_tui.*` / `start_backend.bat` / `setup_all_envs.*` | One-shot startup and multi-environment install scripts |
| `requirements*.txt` / `pytest.ini` / `pyrightconfig.json` | Dependency manifests and tool config |
| `models/`, `chat_history/`, `dist/`, `build/`, `test-results/`, `logs/` | Local artifacts or archive areas, not committed |

**Submodules (4)** — registered in `.gitmodules`, fetched with `git submodule update --init`:

| Submodule path | Remote |
| --- | --- |
| `android/` | `qlh-android` |
| `frontend_cybergothic/` | `qlh-shell` |
| `packaging/` | `qlh-release` |
| `harness_workbench/` | `Koakumix` |

**Related repositories (development/experiment tools, not submodules)** — clone separately into the workspace; covered by `.gitignore` and not committed:

| Local path | Remote | Purpose |
| --- | --- | --- |
| `tools/docagent/` | `qlh-docagent` | Documentation maintenance scanner |
| `tools/toolbox/` | `qlh-toolbox` | Tool collection |
| `tools/reasonix-codex-bridge/` | `reasonix-codex-bridge` | Controlled Reasonix ↔ Codex bridge (MCP + ACP) |
| `tools/dsh-codex-bridge/` | `dsh-codex-bridge` | Twin project (DSH side, vendors the bridge runtime) |
| `packages/spawnledger/` | `spawnledger` | Process ownership ledger |

`src/` grouped by responsibility (per-module interfaces: [module interfaces](模块接口说明.md), [core principles](核心技术原理.md)):

| Group | Representative modules |
| --- | --- |
| Control plane and entry | `api_server.py`, `api_errors.py`, `config.py`, `bootstrap.py`, `model_api_access.py`, `review.py`, `local_store.py` |
| Scheduling and cluster | `scheduler.py`, `scheduler_svc_http.py`, `cluster_join.py`, `cluster_transport.py`, `edge_cluster.py`, `node_config.py`, `node_runtime.py`, `transport_runtime.py`, `transport_port.py`, `network_address.py`, `network_path.py`, `proxy_config.py`, `wss_loopback.py` |
| Engines | `llama_engine.py` (tier L), `model_module.py` (tier D layer splitting), `island_engine.py`, `koakuma_engine.py`, `tcp_comm.py`, `inference_client.py`, `inference_svc_main.py`, `inference_service/`, `paged_kv_cache.py`, `external_provider.py`, `speculative*.py` |
| RPC and layer relay | `llama_rpc_contract.py`, `llama_rpc_device.py`, `llama_rpc_planner.py`, `relay_contract.py`, `relay_planner.py`, `relay_transport.py` |
| Layer-segment/layer-pipeline contracts | `pipeline_node_contract.py`, `pipeline_capacity.py`, `pipeline_assignment_manifest.py`, `pipeline_model_descriptor.py`, `pipeline_reshard.py`, `cache_unit_layout.py` |
| Cross-framework/multimodal pipeline | `qwen3_pipeline_*.py`, `qwen3_multimodal_*.py`, `gemma4_pipeline_*.py`, `multimodal.py` |
| Models and assets | `model_config.py`, `model_host.py`, `model_sync.py`, `model_downloader.py`, `model_download_jobs.py`, `model_search.py`, `model_registry_validation.py`, `model_runtime_contracts.py`, `local_model_assets.py` |
| Task graph and workflows | `task_graph*.py`, `task_journal.py`, `task_provider.py`, `task_worker_*.py`, `graph_orchestrator.py` |
| TUI and interaction | `tui_textual.py`, `tui_api.py`, `tui_shared.py`, `tui_sse.py`, `tui_backend.py`, `tui_commands.py` |
| Devices and soak | `device_profiler.py`, `provider_soak.py`; RAG now lives in `harness_workbench`/Koakumix |

## Quick Start

### 1. Get the code and submodules

```bash
git clone https://github.com/SgfKrc/K-Llama.git
cd K-Llama
git submodule update --init --recursive     # fetches the 4 submodules
```

Submodule remotes are in `.gitmodules`. Related repositories (development/experiment tools) are cloned on demand into `tools/`, `packages/`:

- Submodules: `qlh-android` · `qlh-shell` · `qlh-release` · `Koakumix`
- Related repositories: `qlh-docagent` · `qlh-toolbox` · `reasonix-codex-bridge` · `dsh-codex-bridge` · `spawnledger`

### 2. Choose a runtime environment

The main environment carries torch and serves the D-track layer pipeline, cross-framework relay and the full API:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The Edge environment installs only GGUF/llama.cpp and control-plane dependencies, without torch, Transformers or bitsandbytes:

```powershell
python -m venv .venv-edge
.\.venv-edge\Scripts\python.exe -m pip install -r requirements-edge.txt
.\.venv-edge\Scripts\python.exe scripts/edge_preflight.py --python .venv-edge\Scripts\python.exe --json
python scripts/llama_dependency_contract.py --json
```

The managed Gemma 4 MTMD profile is separate: `.venv-gemma4-native` uses the frozen `llama-cpp-python==0.3.28` binding and does not reuse the ordinary CPU wheel; its ABI marker and lock are checked independently.

On Linux/macOS replace `Scripts\python.exe` with `bin/python`. The interactive TUI needs Textual; read-only commands, the protocol layer and CI checks do not, and `uvicorn/FastAPI` is only needed when the backend is auto-started locally — a remote TUI never starts a local backend on the remote host. To install only the TUI dependencies:

```powershell
python -m pip install -r requirements-tui.txt
```

### 3. Start the TUI

```bash
python qlh.py chat
python qlh.py chat --route distributed_preferred --thinking
python qlh.py chat --fixture fixtures/chat_interactive_fixture.sse
python qlh.py status
python qlh.py models
```

`qlh chat` starts the backend in a daemon thread of the current process when no local backend is running, showing the probe phase on the splash screen; backend logs go to the log files and the log screen. Single commands such as `status` and `models` do not auto-start the backend, and `--fixture` is the offline chat event replay path. Write operations start from the shell: the model screen uses `L` to select a model (auto supply and prepare) / `U` to unload / `C` to cancel the selection, the queue screen `P` pause-resume / `S` policy / `C` clear queued items, and the chat screen supports `/model`, `/queue`, `/new`, `/resume`, `/rename`, `/sessions`, `/delete-session`, `/reset`; destructive and long-running operations show a confirmation dialog first. Model control endpoints are allowed by default over loopback; controlling a remote master requires `QLH_MODEL_API_TRUSTED_CIDRS` on that master.

On Windows use `qlh.bat` (or `K-Llama.bat`), on Linux/macOS `qlh.sh` (or `K-Llama.sh`); all forward to the same entry script `qlh.py`. Screens and command semantics: [TUI guide](TUI使用指南.md) and [TUI command set](TUI指令集.md).

## Models and Distribution

Model artifacts, download caches and large GGUF files are not committed. The model catalog and capability profiles are managed by the main-repository API/TUI, and a model enters the usable list only after passing format, digest, architecture, template, thinking, device-budget and source validation.

Recommended verification order:

1. Load a <=1B GGUF on a single machine; check the template, the thinking switch and streaming output.
2. Start the host and `ggml-rpc-server` on the same machine; check partial residency, capacity merging, output comparison and worker disconnect.
3. Complete real RPC, artifact sync, lease and recovery acceptance on a PC node.
4. For cross-framework or inter-layer pipelines, first re-check D→L numerical consistency and performance boundaries locally.
5. Then move to ARM64/Android worker acceptance.

A "sharded model" claim requires more than duplicating the full model, running whole task-graph requests in parallel, or old PyTorch two-host results. Small-model detour under node failure is a scheduling policy, not a separate product form. Model acquisition and device-profile tiers: [QLH at a Glance](项目速览-QLH-at-a-Glance.en.md).

## Android Validation

The Android project lives in the `android/` submodule; without a real Android device on the build machine, use layered evidence:

```powershell
# JVM protocol/state-machine/capability contract tests
python scripts/android_validation.py

# also build the fullDebug APK
python scripts/android_validation.py --assemble

# after connecting an emulator or adb device, install and launch the control plane
python scripts/android_validation.py --assemble --install --launch --serial emulator-5554

# machine-readable evidence
python scripts/android_validation.py --assemble --json
```

An x86_64 emulator validates the APK, UI, permissions, network and lifecycle, but cannot prove the `arm64-v8a` JNI RPC worker; an ARM64 AVD/QEMU only adds ARM compatibility and cannot replace thermal/power, background-reclaim, weak-network and long-run testing on a real phone. The full Android P1 criteria and the AVD/QEMU/remote-adb plan are in [Android validation alternatives](../android/Android验证替代路径-2026-09-18.md).

## Testing

Keep the test environment separate from the main one:

```powershell
python -m venv .venv-test
.\.venv-test\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv-test\Scripts\python.exe -m pytest -q
```

Targeted checks on high-risk mainlines:

```powershell
.\.venv-test\Scripts\python.exe -m pytest -q tests/test_tui_textual.py tests/test_tui_write_ops.py tests/test_tui_shared.py tests/test_tui_sse.py
.\.venv-test\Scripts\python.exe -m pytest -q tests/test_llama_rpc_planner.py tests/test_llama_rpc_device.py
.\.venv-test\Scripts\python.exe -m pytest -q tests/test_pipeline_node_contract.py tests/test_pipeline_reshard.py tests/test_pipeline_capacity.py
```

Serial full baseline: **6055 passed / 33 skipped / 0 failed** (`-n 0`, 2026-10-14, HEAD `b84522f3`). Occasional flakiness appears under xdist; the serial run is authoritative.

Real hardware, cross-host networks, Android ARM64, performance and long-run soak must store the raw command, environment, model digest, topology, output and failure boundary; a green test suite is not a substitute. Test channel split and marker semantics: [test channel notes](测试通道运行说明.md); criteria and control groups: [testing and acceptance criteria](测试与评判标准.md).

**Documentation checks** (static, standard library only): `python scripts/run_doc_checks.py` runs two checks — relative-link rot and README bilingual structure. The same set runs in CI ([`.github/workflows/checks.yml`](../.github/workflows/checks.yml)) and in the local pre-push hook ([`.githooks/`](../.githooks/README.md), enabled with `git config core.hooksPath .githooks`) — defined once, reused twice.

## Documentation Index

- [Overall architecture](整体架构.md) · [Module interfaces](模块接口说明.md) · [Core principles](核心技术原理.md) · [Distributed resource scheduling](分布式资源调度系统.md)
- [Cross-framework relay — current baseline and follow-up plan](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md) (index for the relay direction) · [project report](跨框架层接力-项目报告.md) · [capacity-gain measurements](跨框架层接力-容量收益实测-2026-09-21.md)
- [TUI guide](TUI使用指南.md) · [TUI command set](TUI指令集.md)
- [Testing and acceptance criteria](测试与评判标准.md) · [test channel notes](测试通道运行说明.md) · [release artifact gate](发行物门-RELEASE-ARTIFACTS.md)
- [Mainline plan: distributed inference and edge optimization](主线开发计划-分布式推理与边缘优化-2026-09-14.md) · [P4.5: dynamic master election and distributed management](主节点动态选举与分布式管理-P4.5立项-2026-09-21.md)
- [Parallelism and cross-framework survey](分布式推理并行与跨框架路线调研汇总-2026-09-15.md) · [KTransformers migration survey](KTransformers优化迁移调研与算法数据层优化方向-2026-09-23.md)
- [Outstanding work ledger (current)](遗留清单-2026-10-07.md) · [Known issues](已知问题记录.md) · [Acceptance ledger and resource limits](验收清单与资源限制登记.md) · [Outstanding work notes (historical archive)](未完成工作备忘-2026-09-23.md) · [DIST follow-up core-path defect audit](DIST后续核心链路缺陷审计-2026-10-07.md)
- [Document status and cleanup list](文档状态与清理清单.md) · [Archive index](archive/README.md)
- [Android validation alternatives](../android/Android验证替代路径-2026-09-18.md)

Historical plans and migrated capabilities live under `docs/archive/`; local experiment artifacts and acceptance raw records live under `local_docs/` and are not a public source interface.

## License

The main repository is under the [MIT License](../LICENSE). Submodules and upstream `llama.cpp` keep their own licenses and version pins.
