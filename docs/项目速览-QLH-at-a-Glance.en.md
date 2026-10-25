# K-Llama at a Glance (Newcomers · Reviewers Quick Entry)

> Status: **current**
>
> Updated: 2026-10-19
>
> **Language**: [English](项目速览-QLH-at-a-Glance.en.md) · [简体中文](项目速览-QLH-at-a-Glance.md)
>
> A 2-minute tour. The project's external name is K-Llama; the in-repository codename is QLH (code symbols use `QLH_*`). Full capabilities, boundaries and evidence live in the [root README](../README.md) and the specialized documents.

---

## What is this?

K-Llama is a **lightweight distributed LLM inference system for heterogeneous edge devices** (a 2026 Beijing Jiaotong University student innovation project). It unifies "who holds which layers, with which engine, where, and at what capacity" into a single abstraction, `(layer_range, engine, location)`, and uses it to run models **layer-by-layer across machines** that cannot hold the whole model.

**Two engine tiers**, chosen by device profile (not "if there is a GPU, use torch"):

| Tier | Engine | Suited to |
|---|---|---|
| **L tier** | llama.cpp / GGUF (including cut GGUF + `embd` injection) | Lightweight, edge, torch-less nodes |
| **D tier** | PyTorch (layer splitting, inter-layer pipeline, multi-node segments) | CUDA PCs; also the platform for cut-point search and experiments |

For explicit local full-model inference, `select_inference_path` in `src/inference_path_selector.py` chooses the binding or llama-server. LL segments use keep-head/native forwarding; DL uses a PyTorch first segment followed by llama.cpp segments. Current paths and capability boundaries are in the [architecture](整体架构.md#2-当前执行路径).

In distributed mode the master coordinates and forwards every continuous segment; workers keep star connections. Explicitly leaving compute participation and retiring the segment permits local full-model loading. Insufficient distributed capacity produces a named wait or refusal. Versioned clients and migration are described in [R5 client cutover](重构客户端切换-R5-2026-10-19.md).

Design principles: **data stays in-cluster, offline autonomy, reproducible acceptance**.

## Verified capabilities

| Capability | Criterion / date |
|---|---|
| Dual-machine real-device layer pipeline (QW1.8B layers 0-21 / 21-24; this model has since been retired from the built-in candidate list) | 3× `distributed_required` all passed, RTT 6-12 ms (2026-08-20) |
| Historical task-chain restart / kill-and-recover + Tailnet IPv6 (the mechanism was retired in ticket 14 R5; historical evidence only) | `wf_330d0aa1…`, reassignment 0 (2026-08-21) |
| D→L cross-framework relay correctness | Main-repo dual-engine matrix **27/27 per-token identical** (2026-09-21) |
| D→L capacity gain | qwen2.5-0.5B **1.568×** / qwen3-2b **1.457×** (strict definition: `C` counts only tensors outside every `*.layers.<i>.*`; counting the visual tower gives 1.613×); under the same controlled 3.0 GB CUDA budget the whole model is rejected while a 12-layer upstream passes (2026-09-21) |
| L→L keep-head channel | `--path l2l_keep_head` / `d2l2l_keep_head` **32/32**, including the three-stage "1 torch upstream + 2 llama downstream" case (2026-09-21) |
| Pure L→L multi-hop (three stages, all on device) | y700 head8 → y700 mid8-16 → y700 cut-k16 tail, host only sends commands ⇒ **32/32** (2026-09-22) |
| Pure L→L cross-device (three stages) | y700 head8 → **Surface** mid8-16 → y700 tail ⇒ **32/32**, two real machines cooperating with the host doing no compute (2026-09-22) |
| Android layer-segment numerical validation | Real ARM64 (Snapdragon 8 Gen 3 / Android 15) is per-token identical to x86_64; dotprod/i8mm kernels confirmed (2026-09-21/22) |
| Cut-point solver closed loop | `scripts/relay_cut_plan.py` fits segment profiles from measurements, then solves; the two-segment loop passes on qwen2.5 (r² 0.96/0.99) and qwen3.5 (0.79/0.96) |
| Windows native compile path | `triton-windows==3.8.0.post28` verified working; `PYTHONUTF8=1` is a prerequisite |
| Judging-policy fix | multi-model 0/4 → loose+512 becomes discriminative; DS3-0324-7B replaces R1 (v2 policy 2/4×3, format 8/11×3) |
| Sub-project: small-model harness workbench (S1-S8) | context budget / STATE memory / RAG / MCP, local gates; the related implementation has moved out to `harness_workbench` (Koakumix) |
| Web search / lightweight Fetch (WEB-TOOL G1-G6) | local dev gates; the capability and its `production_network_enabled` switch have moved out to `harness_workbench` and are no longer in the main-repository `src/` |

Full matrix, setups and report paths: [cross-framework relay — current baseline](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md).

## Outstanding work

- The R6 product acceptance was carried out on 2026-10-19 and is documented in the [R6 report](重构产品验收-R6-2026-10-19.md) (actual package startup, production-data migration and multi-device 9B LL/DL are covered by evidence); the remaining device and gain gates are tracked in its §7. Historical device figures retain their table dates; criteria are in the [refactor design](单一事实源重构设计-2026-10-18.md#12-验收与退出标准).
- Cross-master majority replication, physical fault recovery and trusted signed releases retain separate external gates; current status is in the [ticket plan](彻查后修复与重构排票计划-2026-10-09.md).
- Distributed speculation still requires a local draft, multi-position output and verifiable state rollback. Standalone PoCs have been retired; native local n-gram remains. See the [capability audit](当前功能与事实归属审计-2026-10-18.md#5-推理优化与图依赖边界).
- Network latency directly affects stepwise relay. Deployment decisions use measured RTT and capacity; comparisons use the same model, quantization, build and sampling setup.

## First-time setup

Prerequisites:

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 (3.12 recommended; the Edge tier needs 3.12+, since `requirements-edge.txt` pins `numpy==2.5.3`) | Main runtime and tool scripts |
| Node.js + npm | Node ≥ 18 | Product-shell branch only (`qlh-shell`) |
| JDK 17 + Android SDK (`compileSdk 36` / `minSdk 26`) | — | Only when building Android |
| Tailscale | latest | Distributed mode; most stable when nodes share a LAN |
| Git | — | clone (with submodules) |
| NVIDIA driver + CUDA (optional) | — | The L-tier `llama_server` path (auto-selected when CUDA and a local `llama-server` are both present) and the D tier |

```bash
git clone --recurse-submodules https://github.com/SgfKrc/K-Llama
cd K-Llama
python scripts/setup_envs.py --all            # mainline Python environments (no product-shell Node; qlh-shell maintains its own)
python scripts/setup_envs.py --only test    # only the named environments (names below)
python scripts/setup_envs.py --check          # verify only, no side effects
```

`setup_envs.py` covers six environments — `main` / `test` (`.venv-test`) / `edge` / `gemma4-native` / `gemma4-pipeline` / `qwen3-sidecar`; platform-specific heavy packages such as torch are not installed automatically — the script filters them and prints each environment's install command (for example `--torch-index-url https://download.pytorch.org/whl/cu126`). Re-run `--check` afterwards. Other environments and startup paths: [README Quick Start](README.en.md#quick-start).

The default model is chosen by device profile (`DEFAULT_MODEL_BY_TIER` in `src/model_config.py`, consumed by `get_active_model_paths()` in `src/config.py`):

| Device tier | Default model |
|---|---|
| Mobile | `<1B`: `qwen2.5-0.5b` |
| Ultrabook / edge device (shared VRAM ≤ 2 GB) | `<1B`: `qwen3-0.6b` |
| PC with discrete GPU | `~2B`: `qwen3-5-2b` |

The 2B tier peaks at 2.24 GB VRAM in measurement, which does not fit an ultrabook with ≤ 2 GB shared VRAM. Model files are not committed (`models/` is gitignored); acquisition: [README §models and distribution](README.en.md#models-and-distribution).

## How to read the docs

- **Entry points**: [README](../README.md) (zh) · [README.en](README.en.md) (en)
- **Relay (the most active direction)**: [current baseline](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md) is the index, with the [project report](跨框架层接力-项目报告.md) and [capacity measurements](跨框架层接力-容量收益实测-2026-09-21.md)
- **Architecture and interfaces**: [Overall architecture](整体架构.md) · [Module interfaces](模块接口说明.md) · [Core principles](核心技术原理.md)
- **Plans**: [Mainline plan](主线开发计划-分布式推理与边缘优化-2026-09-14.md) · [P4.5](主节点动态选举与分布式管理-P4.5立项-2026-09-21.md)
- **TUI**: [guide](TUI使用指南.md) · [command set](TUI指令集.md)
- **Testing**: [criteria](测试与评判标准.md) · [test channels](测试通道运行说明.md)
- **Android**: [validation alternatives](../android/Android验证替代路径-2026-09-18.md)
- **The docs themselves**: [document status and cleanup list](文档状态与清理清单.md) · [archive index](archive/README.md)

## Engineering culture

- **Evidence first**: every "done" carries test/log/real-device evidence, with criterion and date.
- **Criterion discipline**: consistency is judged by per-token argmax only; divergence is reported as FAIL; records are split into `mainrepo_end_to_end` / `raw_binding_probe` / `capacity_only` and never mixed.
- **Mechanized, not greenwashed**: the main-repo test channels are `unit` (parallel) / `external` (serial) / `smoke` (real weights, explicit opt-in); the ticket-14 refactor adds the R0–R5 fixed stage contract sets (`scripts/run_refactor_contracts.py --stage`) and the static boundary gate (`scripts/check_refactor_boundaries.py`). Every fix ships a negative case.
- **Measure before explaining**: performance conclusions first locate the bottleneck (per-segment time / RTT / dispersion).
- **User sovereignty**: model artifacts, keys and knowledge bases belong to the user; offline recovery works.

## License

MIT (Copyright (c) 2026 SgfKrc); see the repository [LICENSE](../LICENSE).
