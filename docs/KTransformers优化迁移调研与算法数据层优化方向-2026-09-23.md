# KTransformers 算子级优化调研 + QLH 算法/数据层优化方向

> 状态：**历史参考**（2026-10-25 降级，原为「现行」：其两条结论已被取代 ——
> 「投机解码永久关闭」与「hidden 压缩已上 wire」，见[遗留清单-2026-10-07](遗留清单-2026-10-07.md)
> §2 与 §3 的复核行；§7 联合排期中的 4 条候选优化至今未排期。保留原位而非归档，
> 是因为那 4 条未排期方向仍是后续排期的参考输入。）
> 更新日期：2026-10-25（状态行复核）
> 适用范围：KTransformers 算子级优化的迁移调研，以及 QLH 算法层／数据层（PyTorch 上游）的优化方向与联合排期。
>
> 结论摘要见 §1；可行性建议见 §6；联合排期见 §7；**未决的口径冲突**见 §8。

---

## 1. 结论摘要

**为什么做这份调研**：QLH 引擎层的**工程优化已经踩到边际效应** ——

- 同机实测：`llama.cpp + CUDA`（全层上 GPU）**26.6 ms/token**，而跨框架接力最优（torch GPU 上游 N=20 + llama.cpp CPU 下游）
  **129.1 ms/token** ⇒ 慢 **4.85×**（`README.md` §「与「全 llama + CUDA」的公平对照」）；
- 切点扫描：CPU 上游条件下总时间对切点几乎不敏感（N∈{4,8,12,16} 落在 214–225 ms/步，**±2.6%**），不存在中间谷底；
- 分段剖析：**99.3% 是两侧实际计算**，通信 + 同步 + batch 管理仅 **0.34%** ⇒ 优化必须打计算，**不打传输**。

所以下一步要看**算法层 / 数据层**的空间，而 KTransformers（下称 KT）以「算子级优化」著称，需要把它到底做了什么、哪些能迁移过来搞清楚。

**三条核心结论**：

1. **KT 不是、也不会成为 QLH 的第三个正式引擎**。它只作为公开研究参照，用来指导 QLH 自己的 **PyTorch 上游引擎**做算子、数据布局和异构调度实验；llama.cpp/GGUF 仍是稳定主线和无 CUDA/Edge 回退。
2. **KT 的价值是"CPU 侧算子与设备分工仍有数量级空间"的证据，而不是可复制的软件** —— 它的主要收益来自 MoE 专家下放 CPU + hot/cold 放置 + 专家调度；QLH 当前正式登记模型以 **dense ≤2B** 为主，因此先研究可迁移的 dense PyTorch 机制，MoE 仅作为科研扩展。
3. **最该做的一件事反而最便宜**：KT 相对 llama.cpp 的 27.79× 是**它自研 AMX kernel vs llama.cpp CPU 实现**的差距 —— 而同一份 AMX/AVX-512 实现**已经在主仓 vendor 的 llama.cpp 里**（`ggml/src/ggml-cpu/amx/mmq.cpp`、`ggml/src/CMakeLists.txt:387-401`）。所以**不要重写 llama.cpp kernel，要核实 QLH 实际构建是否打开这些变体**。
4. **KT 的算法级近似与 QLH 的默认判据直接冲突**：Expert Deferral、选择性专家激活可以作为科研对照，但不能绕过 QLH 的**逐 token argmax 一致、fail-closed**生产门。

**本文的定位**：研究对象是 `model_module.py` 及其后续拆分后的 PyTorch 上游组件；研究结果必须能与未优化 PyTorch、整模 llama.cpp 和连续层分布式三组基线比较。研究代码受 feature gate 控制，不进入 Koakuma backend 枚举，不进入 Edge 最小发行，也不让 KTransformers 依赖进入主仓运行时。

---

## 2. KTransformers 的核心思路与架构分工

**核心命题**：MoE 模型「激活稀疏 + 权重巨大」⇒ 把**稀疏激活的专家**放在**大容量 DRAM 的 CPU** 上算，把**密集、计算密集的 attention / shared-expert / 路由**放在**高带宽的 GPU** 上算。论文摘要明确其适用域是 **low-concurrency** 场景。

| 层 | KT 用什么 |
| --- | --- |
| 模型加载 / 算子替换 | 基于 **Transformers/HF 权重**，用 YAML `optimize_rules` 做 **match/replace 注入**（如 `model.layers.*.mlp.experts` → `KTransformersExperts`），逐算子指定 `prefill_device: cuda` / `generate_device: cpu`；QLH 只借鉴这种声明式边界，不引入 KT loader |
| CPU 侧算子 | 自研 **AMX / AVX-512 / AVX2** MoE kernel；`LLAMAFILE` 后端直接吃 **GGUF** 权重 |
| GPU 侧算子 | PyTorch/CUDA + **FlashInfer**（含 variable-batch **CUDA Graph**）、Triton MLA、FP8/GPTQ/Marlin |
| 服务 / 调度 | 自研 `balance_serve`（C++ continuous batching + chunked prefill）→ **2025-10 起改为集成 SGLang**（`sglang-kt`） |
| 多 GPU 扩展 | 交给 **SGLang** 做 TP；KT 聚焦「单机 CPU+GPU 协作」 |
| Fine-tune | LLaMA-Factory 集成（`ktransformers[sft]`） |

**硬前提**：PyPI wheel 仅 **Linux x86-64**（AVX2 起）；CUDA 需 SM 8.0+；**AMX 需 Sapphire Rapids+**；
`kt-cpuinfer` = **物理核数（非超线程）**、`kt-threadpool-count` = NUMA 节点数。
**Windows 原生支持在 2026 Q2 roadmap 里仍是待做项**（issue #1921）。

---

## 3. 「算子级优化」逐项定性

把 KT 的手段按**层次**拆开很重要 —— 因为只有一部分与 QLH 的场景相容：

| 手段 | 内容 | 层次 |
| --- | --- | --- |
| AMX 专用 kernel | Tile 寄存器（TDPBSSD/TDPBF16PS）、**Tiling-aware 权重预处理**（按 Tile 形状重排、64B 对齐、group-wise 对称量化）、每元素只访 DRAM 一次的多级 cache 编排 | **算子级 + 数据布局** |
| AMX ↔ AVX-512 运行时切换 | decode 算术强度低时 AMX 的 tile 派发开销反而不划算 ⇒ 按算术强度动态选 kernel（同一内存布局） | **算子级**（+ 轻量自适应） |
| MoE 算子融合 + work-stealing | 一层所有专家的 gate/up/down 融合成 2 个大任务；再切细后线程间原子窃取，缓解 prefill 专家偏斜 | **算子级（融合）+ 工程级（调度）** |
| 量化 | CPU：INT4/INT8 对称 group-wise（**警告 FP8→INT4 会显著降精度，应从 BF16 量化**）；GPU：GPTQ/RTN W4A16/W8A16；CPU/GPU 共享同一份权重 | **数据层 / 量化** |
| NUMA-aware TP | 权重按 NUMA 节点切片落本地内存；双路服务器 decode 吞吐 **+63%** | **工程级（内存亲和）+ 数据布局** |
| CUDA Graph-backed 调度 | 把 CPU 任务/拷贝做成连续图，消除 graph breakpoint ⇒ kernel launch 开销 >20% → ≈0 | **工程级** |
| **Expert Deferral** | 跨层**重排**专家执行，与 GPU attention 重叠；靠残差容忍性，精度平均降 **≤0.5%**，吞吐 **+1.45×** | **算法级（引入近似）** |
| **选择性专家激活** | 基于 OOD 离线 profile 少激活专家（6 vs 8），宣称质量不变、速度提升 | **算法级（近似/剪枝）** |
| **专家放置策略** | `uniform/frequency/front-loading/random` + 动态更新（prefill 时采集真实路由分布重排 hot/cold） | **算法级（数据驱动调度）** |
| Dual prefill | 按 token 数阈值在 CPU-GPU hybrid prefill 与 Layerwise GPU prefill 间切换 | **工程级 / 调度** |
| 三层前缀缓存 | GPU-CPU-Disk KVCache 复用（`config.yaml` 的 `kvc2`） | **数据层（缓存）+ 工程级实现** |
| 与 SGLang 集成 | continuous batching / chunked prefill / mem-fraction-static / max-running-requests | **工程级** |

---

## 4. 公开收益数字**与前提条件**

⚠️ 这些"大倍数"全部是**低并发 + CPU 侧算力薄（无 GPU offload）场景下 vs llama.cpp/PyTorch 的对比**；
KT 论文自己在摘要里就把它限定为 low-concurrency。高并发下瓶颈很快转到 CPU 内存带宽或 GPU。

| 数字 | 前提 | 来源 |
| --- | --- | --- |
| prefill **4.62–19.74×**、decode **1.25–4.09×** | SOSP'25 摘要，baseline = "existing systems" | 论文页 |
| Expert Deferral **+1.45×**，精度平均降 **≤0.5%** | 论文摘要 + LMSYS blog（两条独立来源一致） | madsys / LMSYS |
| AMX kernel **21.3 TFLOPS** sustained（单 Xeon socket），比 PyTorch 原生快 **3.9×** | LMSYS blog | LMSYS |
| 双路 NUMA TP **+63% decode**；CUDA Graph **launch 开销 >20% → ~0** | LMSYS blog | LMSYS |
| DeepSeek-V3 Q4_K_M：14 GB VRAM + 382 GB DRAM；prefill 54.21→286.55 tok/s、decode 8.73→13.69 tok/s；对 **llama.cpp**（同机）最高 **27.79× prefill / 3.03× decode** | 单机、Xeon Gold 6454S、4090D、**测前充分预热**、**低并发** | `doc/en/DeepseekR1_V3_tutorial.md` |
| Qwen3-30B-A3B：**消费级** i9-14900KF + DDR5-4000 + 4090 可跑 | 显式提醒「内存降频到 4000MT」 | `doc/en/AMX.md` |

---

## 5. 为什么 KT **不能直接搬**过来（场景差异是结构性的）

| 维度 | KT | QLH | 后果 |
| --- | --- | --- | --- |
| 模型 | 大 **MoE**（DeepSeek-V3/R1、Kimi-K2、GLM-5、Qwen3-30B/235B） | 当前正式登记以 **dense ≤2B** 为主；MoE 只作为科研样本 | 专家卸载整套不能直接迁移；其热度统计、放置和预取思想可进入 PyTorch 科研线 |
| 拓扑 | **单机** CPU+GPU（多 GPU 交给 SGLang） | 跨机、异构、跨框架层接力 | KT **不含任何跨机/层接力语义** ⇒ 对分布式中枢零帮助 |
| 硬件 | 服务级 x86（AVX512 至少、AMX 最佳，Linux） | Windows x86_64（Surface）+ **ARM64 Android**（y700） | KT runtime 不在 Edge 支持面内；但 PyTorch 上游研究可在 PC/CUDA 节点做，结果必须保留无 Torch 回退 |
| 判据 | 允许精度换吞吐（≤0.5% 降质换 1.45×） | **逐 token argmax 一致、fail-closed** | 算法级近似**违反判据** |

**不直接引入清单**：KT runtime、`kt-kernel`、`sglang-kt`、AMX/AVX-512 私有 kernel、NUMA 专用执行器、MoE 专家卸载运行时、
Gate/Up 融合实现、GPTQ/Marlin/FP8 依赖栈。它们可以作为外部对照或科研原型，但不得成为 QLH 的第三个后端。

**可借鉴的是机制而不是依赖**：声明式算子替换、设备能力约束、prefill/decode 双计划、数据热度驱动的放置、KV 分层缓存和计算成本画像，见下节。

---

## 6. 对 QLH 的可行建议

### 6.1 PyTorch 上游算法 / 数据层

| 优先级 | 建议 | 理由与来源 | 预估工作量 |
| --- | --- | --- | --- |
| **P0** | **先裁决"上游 torch GPU 层是否真的更贵"的口径冲突**（见 §7），用已有的段画像 `fit_segment_profile`（固定开销 + 边际每层）替代「平均每层」做路由决策 | 两处结论**方向相反** | 0.5–1 人日（纯分析，可复用 `scripts/relay_cut_plan.py`） |
| **P1** | **prefix cache / 会话级 KV 复用落到 L 档主路径**（llama.cpp prompt cache / `src/paged_kv_cache.py`），量化"多轮对话重复 prefill"的节省 | KT 用三层 GPU-CPU-Disk 复用证明**数据层**收益；QLH 调研已把「本地 prefix cache」列为 ✅ 可行 | 2–4 人日 |
| **P1** | **量化资产管线纪律**：上游/下游量化源一律从 **BF16** 出发；核对 `quant_type="int4"` **静默回退 fp16** 的实际生效 dtype | KT 明确警告 FP8→INT4 明显掉精度；QLH 已记录该回退陷阱 | 0.5–1 人日（多为核对与文档） |
| **已完成** | **hidden 压缩已上 wire**（2026-10-07 核实；原条目「`RELAY_WIRE_VERSION=1` 仍固定 f32」作废） | **只对弱网/跨机有意义**（同机占比 <1%） | — |
| **已废止** | **投机解码**：2026-10-07 核实为**永久关闭**，原 P2 条目「从 PoC 走到真实链路」作废，不再排期 | — | — |
| **科研档** | Expert Deferral、选择性专家激活、专家放置策略 | 先在 PyTorch 上游和 MoE 样本中验证；默认不能绕过逐 token 一致门 | 仅产出实验报告 |

### 6.2 工程层

| 优先级 | 建议 | 理由与来源 | 预估工作量 |
| --- | --- | --- | --- |
| **P0** | **核查并锁定 llama.cpp 构建的 CPU 指令变体**：在支持的 x86 上启用 AVX512/AMX 变体（`GGML_CPU_ALL_VARIANTS` 或 `sapphirerapids`），做同模型 A/B（每层耗时、decode ms/token） | vendor 内**已有** AMX/AVX512 实现，但 QLH 构建脚本只设了 `GGML_NATIVE`/`GGML_RPC`，**未显式启用**；KT 的 21 TFLOPS / 3.9× 说明该档位差异是**数量级**的 | 1–2 人日/节点 |
| **P0** | **给下游 CPU 侧做线程/亲和性标定**：`-t` = **物理核（非超线程）**、NUMA 策略、避开与 torch 上游争核（P2 交叠场景），产出「每层 ms / 线程数 / 是否与上游并发」表 | KT 把 `cpuinfer=物理核`、`threadpool=NUMA 数`、NUMA 放置当作一等参数，并称 NUMA TP +63% | 1–2 人日 |
| **P1** | **多请求/多序列交叠**（continuous batching + chunked prefill 的最小版）：把 `PIPELINE_MAX_CONCURRENT=1` 变成可解释可配置，先做队列级 micro-batch 交错 | KT 的 `balance_serve` 在 4 路并发下吞吐 +130%；QLH 已把「多请求流式交叠」列为 ✅ 可行但**改造量大** | 8–15 人日 |
| **P1** | **上游 torch 段引入 CUDA Graph / 减少 kernel launch**（不改数值，需与逐 token 判据并行验证；与已落地的 `torch.compile` 分开评估） | KT 用 CUDA Graph 把 launch 开销 >20% → ~0；QLH 上游 torch 每层耗时明显高于下游 | 3–5 人日 |
| **P2** | **权重 mmap 冷启动/缺页专项**（模型放磁盘/网络盘时的首 token 抖动） | KT roadmap 把「AI SSD / 慢 mmap 读盘」列为已识别瓶颈 | 2–3 人日 |

**一句话总纲**：落地路径应是「**先完成大文件拆解和行为基线 → 在 PyTorch 上游建立算子成本画像/注册表 → 做异构放置与 prefill/decode 双计划 → 再做 MoE 热度/预取科研**」；llama.cpp 继续承担稳定路径，KT 不作为运行时依赖。

---

## 7. 与大文件拆解计划的联合排期

大文件拆解必须先于 PyTorch 算子研究。原因不是形式上的代码整洁，而是当前 `scheduler.py`、`api_server.py` 和 PyTorch 执行路径之间仍有较大的装配耦合；在拆解前引入算子放置会把实验依赖继续埋进上帝对象，后续无法区分性能收益来自算法还是结构变化。

### 7.1 阶段与依赖

| 阶段 | 票号 | 交付 | 依赖 | 状态 |
| ---: | --- | --- | --- | --- |
| 0 | `REFACTOR-LARGEFILE-01` | 固化 scheduler/API OpenAPI、公共符号、monkeypatch 面、锁身份、导入/冷启动和全量定向测试基线 | 无 | 已完成 |
| 1 | `REFACTOR-LARGEFILE-02` | 拆出 `scheduler_layer_plan` 与 `scheduler_sidecars`；保留兼容门面及侧车工厂 patch 点 | 01 | **已完成（2026-09-23）** |
| 2 | `REFACTOR-LARGEFILE-03` | 拆出 task-worker、cluster/HA、pipeline mixin；保持实例私有属性、锁和调用点不变 | 02 | **已完成（2026-09-23）** |
| 3 | `REFACTOR-LARGEFILE-04` | 以 APIRouter 按领域拆分全部 API handlers，端点和行为保持不变 | 01 | **已完成（2026-09-23）** |
| 4 | `REFACTOR-LARGEFILE-05` | 重构收口：门面契约、OpenAPI 路径/方法集合、冷启动和完整回归，确认无逻辑夹带 | 03、04 | **已完成（990 passed, 4 skipped）** |
| 5 | `TORCH-OP-PROFILE-01` | 对项目 PyTorch 上游建立按算子形状、dtype、设备、阶段的成本画像；修正“平均每层”口径 | 05 | **已完成（CUDA profile，2026-09-23）** |
| 6 | `TORCH-OP-REGISTRY-01` | 建立逻辑算子到 eager/compile/实验实现的注册、能力声明和 fail-closed 回退合同 | OP-PROFILE-01 | **已完成（离线合同与 23 项定向测试；未改运行时）** |
| 7 | `TORCH-HETERO-PLAN-01` | 离线算子放置 planner：设备画像、内存、带宽、边界传输和正确性门；与连续层 planner 对照 | OP-REGISTRY-01 | **真实 layer-level CPU/CUDA FP32 矩阵已接入；2026-09-26 采用 `inner_loops=8`、24 层、各 7 样本复测，身份与 shape 匹配，CPU 单侧稳定 prefill/decode=22/24、12/24，CUDA=10/24、9/24，两设备共同=10/24、2/24；不稳定成本未修补，完整 planner 仍 fail-closed，未接运行时** |
| 8 | `TORCH-PHASE-PLAN-01` | prefill/decode 双计划和受控状态切换；失败时回退单一 PyTorch 计划或 llama.cpp | HETERO-PLAN-01 | **离线合同完成（25 项合成测试；未做硬件准入/未接运行时）** |
| 9 | `TORCH-ACT-COMPRESS-01` | PyTorch 双层段激活压缩；整模对拍、长 prompt 和逐 token 门禁 | HETERO-PLAN-01、PHASE-PLAN-01 | **离线实验完成（RTX 4060：f16 精确；int8/int4 分歧；未接运行时）** |
| 10 | `TORCH-HW-ADMIT-01` | 同负载 CPU/CUDA、阶段成本、KV 身份与真实链路准入矩阵 | OP-PROFILE-01、HETERO-PLAN-01、PHASE-PLAN-01、ACT-COMPRESS-01 | **完整前向复测完成（2026-09-24，主运行时 `torch 2.13.0+cu126`，FP32，CPU/CUDA 各 2 prompt × 3 layer ranges × prefill/decode × 20 repeats；CPU 2 个 cell、CUDA 8 个 cell 超 CV 0.10；4 个同机异构方向均 exact，但 3 个时延不稳定，准入拒绝；跨机生产推理仍未准入）** |
| 11 | `TORCH-RUNTIME-ADMIT-01` | 可验证准入证据、资源/KV 生命周期及默认关闭的阶段调度 | HW-ADMIT-01、PHASE-PLAN-01 | **锁定（HW-ADMIT-01 完整准入后方可排期）** |
| 12 | `TORCH-MOE-PLACEMENT-01` | 以可运行 MoE 样本验证专家热度、复制、预取和故障回退；不进入默认 dense 路径 | OP-REGISTRY-01、HW-ADMIT-01、RUNTIME-ADMIT-01 | 排队 |

### 7.2 共同门禁

- `REFACTOR-LARGEFILE-*` 期间不改变推理语义，不新增 KTransformers 依赖，不改默认 backend 选择。
- `TORCH-*` 只在 CUDA/PC 研究环境启用；Edge、Android 和无 CUDA 发行继续使用 llama.cpp/GGUF，不 import torch。
- 每个研究实现必须能退回未优化 PyTorch；整个 PyTorch 上游仍能退回 llama.cpp 或已有连续层计划。
- 研究报告必须同时记录正确性、首 token、decode、峰值内存、通信量和回退边界；性能数字不得跨设备、跨模型或跨阶段直接比较。
- MoE 研究不改变 dense 模型主线，不提前把 MoE 专家调度写进 `PipelineNode` 的稳定合同。

---

## 8. ⚠️ 未决的口径冲突

**同一件事（"层该推给上游还是下游"）在两处文档里结论方向相反**：

| 出处 | 口径 | 结论 |
| --- | --- | --- |
| `README.md`（切点扫描节） | 上游 GPU **2.5–4.3 ms/层**，下游 CPU **5.8–8.6 ms/层** | **层应尽量推给 GPU 上游** |
| [跨框架接力-当前有效基线与后续优化计划](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md) | 上游 **1.54 ms/层**，下游 **0.91 ms/层** | **方向相反** |

**很可能的原因**：「平均每层」被**固定开销**污染 —— 同一份文档的多轮表里，下游 24 层摊 1.13 ms/层、4 层摊 2.57 ms/层，
说明这个指标随切点变化，不能直接当边际成本用。

**处理口径**：段级路由只能比较同一模型、设备对、精度、负载、compile 状态与 warmup 下的重复实验，再用
`scripts/relay_cut_plan.py` 的段画像（**固定开销 + 边际每层**）拟合；不得用「整段平均 ms/层」直接排序。
2026-09-23 的上游算子画像补充了 PyTorch 内部成本分布，但它不是与 llama.cpp 同负载的两侧对照，**历史方向冲突仍未裁决**；
在新的同条件切点分析报告通过数据门前，两处旧结论均只可作为各自实验口径的观察，不可作通用路由规则。

### 8.1 TORCH-OP-PROFILE-01 首份主仓算子画像（2026-09-23）

工具：`scripts/torch_operator_profile.py`，仅供研究/验收使用，不进入运行时导入链。直接调用主仓
`ModelManager.load_layer_range()` 与 `forward_layers()`，prefill、decode 分开采样；operator dispatch scope 记录
输入/输出 tensor shape、dtype、device、stride，并与其直接 `aten` profiler 子事件配对。性能基准另用关闭插桩的
`perf_counter` 重复采样，CUDA 前后同步。

实测配置：Qwen2.5-0.5B-Instruct（manifest SHA-256 `40133469bc80b60b3e00680e221998a16809ee176c1e1a12c951798d6125a2b9`），
主仓 PyTorch 层段 `[0,12)`，FP16，eager（compile 关闭），RTX 4060 Laptop 8 GB，Torch 2.13.0+cu126、Transformers 5.17.0；
prefill 64 tokens、decode 8 步；每 phase 至少 2 轮 shape warmup，并在相同负载下持续运行至少 3 秒后取 5 个未插桩重复样本。短 prompt 为固定长度而重复到 64 tokens；
decode 重复输入 prompt 最后一个 token，不含 LM head、采样或真实生成 token 回馈。

| phase | 未插桩中位数 | 样本范围 | 算子签名数 | dispatch 调用 / 未匹配 |
|---|---:|---:|---:|---:|
| prefill | 16.641 ms / call | 10.810–17.265 ms（5 次；总体标准差 2.631 ms） | 97 | 1,367 / 0 |
| decode | 12.310 ms / forward call | 11.001–14.142 ms（每样本 8 步，5 次；总体标准差 1.125 ms） | 250 | 10,456 / 0 |

热点（直接 `aten` CUDA self time，按单次 forward call 归一；不是端到端 wall time）：prefill 中 QKV `mm`
（`[64,896] × [896,4864]`，FP16）约 2.085 ms，MLP down `mm` 约 1.045 ms；decode 对应 QKV `mm`
（`[1,896] × [896,4864]`，FP16）约 1.657 ms，MLP down `mm` 约 0.672 ms。profile 也观察到 RMSNorm 相关
`mean/pow/rsqrt` 的输入为 FP32，而线性层主要为 FP16；这说明单看参数 dtype 会漏掉混合中间算子形态。

**测量边界**：metadata dispatch/profiler 插桩使总墙钟达到未插桩约 8.46×（prefill）/10.58×（decode）；因此绝不引用
插桩墙钟作为性能数据。算子 CUDA self time 用于同一 profile 内的热点排序，不能跨 GPU/版本外推；本文也不含 llama.cpp
对照，不能解释两份 D→L 文档中方向相反的边际层耗时。3 秒负载预热后 prefill 样本仍有约 15.8% 的变异系数，说明笔记本 GPU 时钟/运行状态仍会显著影响测量；该轮 wall time 只作为受控配置下的观察，不作为路由阈值或跨引擎成本结论。单机路由成本仍须用同一实验身份的重复切点记录，经
`fit_segment_profile` 拆成固定开销和边际每层，并遵守 `relay_cut_plan.py` 的多轮/来源门禁。

原始报告（本机忽略目录）：`local_docs/evidence/torch-op-profile/2026-09-23-qwen25-05b-rtx4060-cuda-steady.json`。CPU-only 方法 smoke：
`local_docs/evidence/torch-op-profile/2026-09-23-qwen25-05b-cpu-smoke.json`。定向单测 `tests/test_torch_operator_profile.py`。

### 8.2 TORCH-OP-REGISTRY-01 算子实现注册合同（2026-09-23）

新增纯标准库模块 `src/torch_operator_registry.py`，默认目录包含 `linear_projection`、`attention_core`、`normalization`、
`transformer_layer_loop` 四种逻辑算子。eager 条目表示当前 Transformers/PyTorch 参考路径（CPU 仅登记 FP32，CUDA 登记 FP32/FP16），不声称存在可独立替换的 eager kernel；
目前唯一 compile 候选是 `torch.compile` 层循环。它编译的是整段 layer loop，不是 profiler 观察到的单个 `aten::mm`/attention kernel，
故不把 compile 错登记成逐算子实现。当前没有可通过证据门的实验 kernel，experimental 类型由合同支持，但不预登记虚构实现。

候选解析必须同时满足 device、dtype、prefill/decode phase、运行能力、显式 feature gate、compile 候选的模型规模下限，以及与模型和 workload
fingerprint 完全匹配的通过证据（含 artifact、正样本数和数值误差）。compile 模型规模门与当前运行时一致，为至少 1.5B 参数。
默认质量门要求逐 token argmax 一致；非 exact 实验必须调用方显式给出
绝对/相对误差上限。候选任一条件缺失或失败时解析到兼容的 eager reference；连参考实现也不满足上下文时抛出
`NoSafeImplementationError`，不静默猜测或放宽约束。实验实现还要求 policy 明确允许 experimental。

本模块仅为后续离线 planner 提供目录与解析合同；未接入 `ModelManager`、`torch.compile` 初始化或 Koakuma backend 选择，
不改变当前运行时路径。定向验证：`tests/test_torch_operator_registry.py` 覆盖能力/阶段、feature gate、证据范围、精度门、实验准入、
兼容回退与无安全参考时拒绝。

### 8.3 TORCH-HETERO-PLAN-01 离线算子异构放置（2026-09-23）

新增 `src/torch_hetero_plan.py`，消费设备资源画像、带方向的链路成本、按模型/负载/阶段/形状匹配的算子成本、数值证据与有向算子图；
通过 `OperatorRegistry.compatible_implementations()` 仅枚举满足设备、dtype、能力、feature gate 和正确性门的实现。测量行必须至少 3 个样本、warmup 稳定、
未启用 profiler 插桩，身份及算子签名完全匹配。planner 对权重驻留、峰值 workspace、边界激活缓冲和 safety margin 做容量检查；搜索有明确状态上限，
超限即拒绝，不返回部分“最优”结果。

时延模型为 `topological_device_queue_and_link_fifo_v1`：各设备按给定拓扑顺序串行执行，跨设备依赖经过对应有向链路 FIFO；独立分支可在不同设备队列上重叠。
这是可复现的离线列表调度估算，不代表 PyTorch/CUDA 运行时的真实重叠、链路竞争或动态调度。报告区分 `compute_work_ms`、`transfer_work_ms` 和调度 `total_ms`（makespan），
不把两种 work 总和误报成 DAG 延迟。

同负载连续层基线复用 `plan_relay_cut_n_segments()` 和现有固定开销/边际层耗时拟合；调用前从可用内存扣除保守的峰值 workspace 与最大层边界缓冲，
避免把算子 planner 计入而连续层基线忽略的瞬时内存伪装成公平对照。比较仍受现有连续层模型精度约束，报告记录调整量及基线拒绝原因。

定向验证：`tests/test_torch_hetero_plan.py` **16 passed**，覆盖精确有界搜索、独立 DAG 分支并行、链路/内存、插桩与身份拒绝、实验候选正确性门、连续层同负载对照、JSON 往返、输入覆盖保护和无 Torch 导入。CLI：
`python -m src.torch_hetero_plan --input scenario.json --json-out report.json`，输入 schema 为 `qlh.torch_hetero_plan_input.v1`。当前证据全部为合成输入；本机只有单一 CUDA 画像，
没有相同模型/负载的跨设备算子成本矩阵，因此**不宣称存在硬件性能优势**。该模块不进入模型加载或推理调用链；`TORCH-PHASE-PLAN-01` 已实现离线切换合同，但真实硬件准入仍要求同负载执行与调度校准，当前证据不足，不能启用运行时双计划。

### 8.4 TORCH-PHASE-PLAN-01 离线阶段计划与受控切换（2026-09-23）

新增 `src/torch_phase_plan.py`，仅定义离线准入合同和元数据状态机，不执行推理、不连接运行时，也不导入 Torch。prefill/decode 计划分别绑定模型、阶段负载和计划摘要；执行证据要求请求会话及参考输出一致、至少 3 个正时延样本、warmup 稳定、无 profiler 插桩，并通过 correctness 与 exact argmax 门。运行波动阈值和计划估算误差阈值必须由调用者显式给出，不在模块中暗设通用硬件阈值。

阶段 KV 合同覆盖每一层的 owner device 与 layout fingerprint，并核对 attention placement；输入 owner 先归一化，避免一次性迭代器被重复消费。prefill 完成后只有 KV 合同完全一致才能进入 decode。阶段失败只能从请求起点切换到有匹配 correctness 证据的单一 PyTorch 或 llama.cpp fallback；首选 fallback 失败且尚无输出时可尝试下一候选，一旦已发布输出则 fail-closed 中止，禁止从半截响应重启并拼接。

`PhasePlanBundle` 自身也校验 admitted 状态所需的完整阶段校准和 fallback 证据，避免直接构造一个空的 admitted bundle 绕过工厂准入。这里的 artifact reference 是调用方提供的审计引用，不是签名或内容真实性验证；合成测试仅验证合同逻辑，不构成实测准入凭证。

定向验证：`tests/test_torch_phase_plan.py` **25 passed**，覆盖计划摘要/模型/阶段负载绑定、逐层 KV owner 与布局、attention 放置、样本稳定性与计划误差门、fallback 排序和重试、输出后禁止 fallback、伪造 admitted bundle 拒绝、一次性 owner 输入及无 Torch 导入。定向组合测试还覆盖 `TORCH-HETERO-PLAN-01`、算子注册表和算子画像。

实测边界：本地 CPU 算子报告是 Qwen2.5-0.5B FP32、prefill 8 tokens/decode 2 steps、每阶段仅 1 个样本；RTX 4060 算子报告是 FP16、prefill 64/decode 8、每阶段 5 个样本。ACT 票随后新增的整模/双层段对拍虽然使用 RTX 4060 长负载，但未输出阶段计划摘要、会话/KV 合同及可验证的 `PhaseExecutionEvidence`。因此**本票只完成离线软件合同，不宣称真实 phase pair 已准入、性能已改善或运行时切换已实现**。阶段准入另列 `TORCH-HW-ADMIT-01`。

### 8.5 TORCH-ACT-COMPRESS-01 PyTorch 激活压缩实验（2026-09-23）

新增 `scripts/torch_activation_compress.py`，只用于研究：以整段 QLH `ModelManager` 为参考，再用两个连续 layer-range `ModelManager` 执行同一模型；每次 prefill/decode 边界复用 `relay_hidden_quant` 的编码器，且 baseline 走当前 `serialize_tensor_fast`。比较所有 prefill 位置 argmax 和真实 greedy 自回归 token；记录 payload、原始 tensor 与旧序列化三种大小、本地 wall time 及误差。没有改 PyTorch peer/TCP 生产序列化，没有接入运行时压缩。

报告：`local_docs/evidence/torch-activation-compression/2026-09-23-qwen25-05b-rtx4060-final2.json`。Qwen2.5-0.5B（manifest `40133469…12a2b9`）、RTX 4060 Laptop GPU、PyTorch 2.13.0+cu126、FP16、24 层按 12/12 切分；有效 prompt 128 tokens（180-token 技术段落截断，记录 token SHA256）、greedy 32 token、每档 3 次 warmup + 3 次重复，显式稳定性阈值 CV≤0.1。

| 档位 | prefill argmax | 自回归 token | payload / 旧序列化 | payload / 原始激活 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| `none` | 128/128 | 32/32 | 1.000× | 1.178× | 整模/双段基线一致；当前小张量 `torch.save` 有封装开销 |
| `f16` | 128/128 | 32/32 | 0.849× | 1.000× | 精确；源激活本来就是 FP16，数据本身没有压缩 |
| `int8_block128` | 123/128，首分歧位置 34 | 21/32，首分歧 token 21 | 0.438× | 0.516× | 严格逐 token 门失败 |
| `int4_block128` | 109/128，首分歧位置 0 | 7/32，首分歧 token 5 | 0.226× | 0.266× | 严格逐 token 门失败 |

wall time 只作本机诊断，不作为准入结果：样本是单 GPU、单 prompt、单机两个 layer-range manager，未经过 socket/network；`none` 控制 CV 2.4%，`f16` CV 19.1%，未达到显式 10% 稳定性门，故不据此判断加速或减速。FP16 payload 对原始 tensor 是 1.0×，因此不能称为激活数据压缩。压缩 payload 字节也未包含 codec mode/envelope 和外层 transport framing。int8/int4 即使字节更少也已触发真实输出分歧，不接受为精度/吞吐折中。

专项回归：`tests/test_torch_activation_compress.py` **8 passed**。报告可重跑；真实弱网/跨机传输、双设备方向、多个模型/负载和更长 prompt 矩阵仍未测。故本票只完成单机 CUDA 探针与拒绝/候选结论，`candidate_for_cross_device_link_test=false`、`cross_device_hardware_admitted=false`、`production_runtime_enabled=false`。

### 8.6 硬件准入与运行时接入拆票

本地代码核实确认 CPU 与 RTX 4060 的旧 profile 不可组成阶段校准对：旧 CPU 是 `[0,2)`/FP32/8-token/单样本，CUDA 是 `[0,12)`/FP16/64-token/固定 token decode；旧报告也没有主机与输入身份。HW 票新增 `scripts/torch_hardware_admit.py`，显式绑定模型权重/manifest、tokenizer、输入 token SHA、主机、运行时、dtype、线程数、实际 KV tensor 结构及逐样本未插桩时延；decode 成本使用同设备整模参考生成的 token trace 重放，另以完整自回归分段请求作正确性门，二者不混称。CPU/CUDA 控制组统一 FP32，部署默认 CPU FP32/CUDA FP16 另行观察。没有第二张 CUDA 卡时，任何结论仅限实测设备对、方向、层段和运行时构建，不外推到多 CUDA。

**2026-09-23 实测**：本机 `DESKTOP-KL4JIK7` / Windows 10，同一 Qwen2.5-0.5B 工件（manifest `40133469…12a2b9`、权重 SHA-256 `fdf756fa…fb7fe`）、同 tokenizer、8 线程，CPU `.venv-test` (`torch 2.13.0+cpu`) 与 RTX 4060 Laptop GPU `.venv-qwen3-sidecar` (`torch 2.13.0+cu126`) 分开执行；Transformers 均为 5.17.0。FP32 控制组为 64/256-token prompt × `[0,4)`/`[0,12)`/`[0,24)` × prefill/decode，每格 1 次 warmup + 5 次未插桩样本；decode 每样本是 8 个从同设备整模 greedy trace 取值的 teacher-forced forward。另有真实 greedy 8-step 完整模型与 CPU↔CUDA 12/12 分段全链正确性/时延样本。CPU 与 CUDA 完整模型在两条 token 轨迹上逐 token 一致；12 个 CPU/CUDA KV 结构（shape/dtype/layout）指纹逐项一致，设备放置单独记录。CPU/CUDA wheel build 不同，因此这是受控 FP32 的设备-运行时组合对照，不声称隔离了 GPU 硬件的纯因果效应。

| workload | layer end | CPU FP32 prefill median | CUDA FP32 prefill median |
| --- | ---: | ---: | ---: |
| 64 tokens | 4 | 37.02 ms | 6.40 ms |
| 64 tokens | 12 | 129.44 ms | 17.63 ms |
| 64 tokens | 24 | 404.24 ms | 39.87 ms |
| 256 tokens | 4 | 126.12 ms | 16.38 ms |
| 256 tokens | 12 | 388.71 ms | 47.01 ms |
| 256 tokens | 24 | 919.42 ms | 117.03 ms |

**准入结论：拒绝，不启用运行时计划。** 预先固定的 CV 门为 `population_stddev / mean <= 0.10`；完整 5 次矩阵中 CPU 有 4/12、CUDA 有 3/12 的 phase cell 超限，且 CPU→CUDA 两输入长度全通过 exact greedy token 门，CUDA→CPU 两档的时延稳定门均失败。所有异构方向的 greedy token 正确性均通过 FP32 控制组，但不能抵消时延稳定门。完整 per-cell CV、decode 样本、真实 cache tensor 和链接拷贝数据见本机忽略证据：`local_docs/evidence/torch-hardware-admit/cpu-fp32-full-r5.json`、`cuda-fp32-full-r5.json`、`cpu-cuda-fp32-comparison-r5.json`。不得把这些诊断中位数输入 planner 作为已准入成本。

**控制复测（2026-09-23，当前有效稳定性结论）**：原 CV 门和模型/输入/8 线程均不变，完整矩阵改为每格 3 次预热 + 20 次计时；prefill/decode 成对交替，固定 seed `20260923` 打乱 workload 与 layer-range 顺序。CPU 仍有 **4/12** 格超限（`64:4 prefill=0.106`、`64:12 prefill=0.102`、`256:12 prefill=0.113`/`decode=0.107`）；CUDA 有 **7/12** 格超限（`64:4 prefill=0.142`；decode：`64:4=0.169`、`64:12=0.171`、`64:24=0.114`、`256:4=0.168`、`256:12=0.155`、`256:24=0.139`）。CPU 进程 CPU 时间旁证在超限格也呈相近 CV，但 Windows 多线程进程时钟较粗；未采集 ETW/CPU 频率/温度，因此只能说重排和增加样本未消除抖动，不能把根因断言为温控或调度。CPU↔CUDA 四个 workload-direction 的 greedy 结果均 exact，但仅 64-token CPU→CUDA 的 prefill/decode 两格同时满足 CV 门；新 comparison 仍为 `phase_cost_matrix_admitted=false`、`same_host_cpu_cuda_split_admitted=false`。20 次原始样本及诊断旁证见 `cpu-fp32-interleaved-w20-20260923.json`、`cuda-fp32-interleaved-w20-20260923.json`、`cpu-cuda-interleaved-w20-comparison-20260923.json`。不得放宽阈值或只挑稳定方向进入 planner。

**控制复测（2026-09-24，当前有效稳定性结论）**：并行弱网实验已结束后重新串行执行同一 Qwen2.5-0.5B 工件、FP32、8 线程、3 次预热/20 次计时和 seed `20260923`。CPU 仍有 **4/12** 格超限：`64:4` prefill/decode=`0.154/0.116`、`64:12` prefill/decode=`0.135/0.113`；CUDA 仍有 **6/12** 格超限：`64:24 prefill=0.189`、`64:12 decode=0.136`、`64:4 prefill/decode=0.171/0.144`、`256:4 decode=0.119`、`256:12 decode=0.130`。同机 CPU/CUDA 分段四个 workload-direction 的 greedy token 与 KV 结构仍 exact，但 `phase_cost_matrix_admitted=false`、`same_host_cpu_cuda_split_admitted=false`；增加样本和移除弱网并行干扰没有使时延门通过。新证据为 `cpu-fp32-interleaved-w20-post-weaknet-20260924.json`、`cuda-fp32-interleaved-w20-post-weaknet-20260924.json`、`cpu-cuda-interleaved-w20-post-weaknet-comparison-20260924.json`。

**CPU 时延抖动诊断（2026-09-24，矩阵诊断根因已定位，完整前向仍有残余）**：前两轮复测只能说明「重排与增加样本未消除抖动」，并因未采 ETW / 频率 / 温度而**不下根因断言**。新增诊断工具 `scripts/torch_cpu_jitter_diagnosis.py`（逐逻辑核画像 + 亲和性×线程数对照；**只诊断，不改 CV 门、不放宽阈值、不产出可进 planner 的成本**）补齐了这块证据：

- **本机 CPU 为 `i9-13900H`：14 物理核 / 20 逻辑核 ⇒ Intel 混合架构（P-core + E-core）**。逐逻辑核画像（同一 matmul 负载、单核绑定、20 次/核）给出**完美双峰**：**12 个核 ≈ 14.5–16.0 ms**、**8 个核 ≈ 38.3–40.0 ms** ⇒ **P-core 比 E-core 快 2.76×**（12 = 6 物理核 × 2 HT；8 = 8 个 E 物理核）。
- **线程数 × 亲和性矩阵**（8 线程即 HW-ADMIT 现状；CV = `population_stddev / mean`）：

  | 线程数 | 默认核集（跨 P/E） | 仅 P 域（前 12 逻辑核） | 仅 E 域（后 8 逻辑核） |
  | --- | --- | --- | --- |
  | 4 | 0.0925 | 0.1820 | **0.0315** |
  | 6 | **0.3220** | 0.1153 | **0.0390** |
  | 8 | 0.0891 | 0.0574 | **0.0410** |
  | 12 | 0.0753 | 0.2615 | **0.0270** |

- **机制结论**：抖动来源是「**跨异构核域 + 超订物理核**」—— 默认核集上 8 线程会被调度到 P 或 E（或被迁移），P 域超订再叠加 HT 争用；而**同构域内且线程数 ≤ 域内物理核数**时 CV 稳定在 **0.027–0.041**，**远低于 0.10 门**。⇒ **门阈值本身没有问题，问题在测量流程**：CPU 标定应在**固定亲和性 + 线程数 == 该域物理核数**下进行（或按 P/E 域**分别标定**，并把核域写进 `device_profile`），否则同一份负载会同时混入两种量级的单核成本。这与本文档 P0 的「`-t` = 物理核（非超线程）、亲和性/NUMA 策略」建议相互印证，且**不涉及放宽阈值或只挑稳定格**。
- 证据：`local_docs/evidence/torch-hardware-admit/cpu-jitter-diagnosis-cores-20260924T020328.json`、`cpu-jitter-diagnosis-threads-20260924T020420.json`、`cpu-jitter-diagnosis-threads-t{4,6,8,12}-20260924.json`（含逐核排序均值与每档 20 次原始样本）。
- **诚实边界**：诊断负载是**矩阵乘同族**（非完整层前向）；未采 ETW/温度/频率计数器（改用亲和性实验**直接定位**核域异构，比频率采样更直接，但"频率/温控可能叠加"未被排除）；期间后台负载 21–38%（各档同条件对比，未做进程隔离）。

**HW-ADMIT 本机控制复测（2026-09-24）**：为把上述诊断落实到真实层段探针，`scripts/torch_hardware_admit.py` 新增了 opt-in 的 `--cpu-affinity CPU[,CPU...]` 与 `--interop-threads N`，并将请求的逻辑核、mask、应用方式和实际 inter-op 线程数写入 `runtime`；比较器在两份新报告都提供这些字段时对亲和性/线程不一致 fail-closed。定向 `tests/test_torch_hardware_admit.py` 为 **16 passed**，CPU 诊断回归为 **3 passed**。

- P-core 单线程条件：Qwen2.5-0.5B、FP32、`CPU 0`、Torch intra-op=1/inter-op=1、3 warmup/20 repeats；CPU 仅 `64:24 decode`（CV `0.2451`）和 `256:12 decode`（CV `0.1106`）超门。对应 CUDA 报告 12/12 timing cell 均过门，但完整比较仍因 CPU 两格和 `256` 两个异构方向不稳而拒绝 `same_host_cpu_cuda_split_admitted`。
- P-core 6 线程条件：仅绑 `0,2,4,6,8,10`（每个 P-core 取一个逻辑核）且 inter-op=1 后，短 cell 抖动下降但仍有 `64:4 prefill`、`256:12 prefill/decode` 超门；不把线程池控制误写成充分修复。
- 证据：`cpu-fp32-pcore-t1-interop1-w20-20260924.json`、`cuda-fp32-pcore-t1-interop1-w20-20260924.json`、`cpu-cuda-pcore-t1-interop1-w20-comparison-20260924.json`，以及 6-thread 对照 `cpu-fp32-pcore-t6-w20-20260924.json`、`cpu-fp32-pcore-t6-interop1-w20-20260924.json`。
- 结论：亲和性与 inter-op 控制已成为可复现实验能力，但本机 Torch CPU profile **仍未准入**；不得删异常样本、放宽 CV=0.10、或把 1-thread 作为默认运行时配置。下一步仍是隔离 ETW/频率/温度/后台负载并确认完整前向的长尾来源；`TORCH-RUNTIME-ADMIT-01` 继续锁定。

**CUDA telemetry 隔离复测（2026-09-24）**：`torch_hardware_admit.py` 新增 opt-in `--telemetry {none,cuda,all}`。CUDA 侧由持久化 `nvidia-smi` 流采集 `clocks.sm`、温度、功耗、利用率、`clocks_event_reasons.active`/`clocks_throttle_reasons.active`；`all` 另由持久化 Windows `typeperf` 流采集 CPU utility/frequency/thermal zone。采样线程和子进程不进入 phase timing 调用，报告对每个 range/异构方向写入 UTC telemetry window；采样失败记录到 `errors`，不视为稳定。

- 最终证据：`cuda-fp32-pcore-t1-interop1-telemetry-v3-w20-20260924.json`、`cpu-cuda-pcore-t1-telemetry-v3-comparison-20260924.json`。共 423 个 telemetry 样本（GPU 254、系统 169），错误 0，所有 6 个 range 和 4 个异构方向均有窗口边界。
- CUDA timing：仅 `256:24 prefill` 超 CV 门（`0.1194`）；CPU 两格仍超门，异构 64/256 两组方向均有 timing 不稳，比较器结果 `matched_fp32_cpu_cuda_pair=true`、`phase_cost_matrix_admitted=false`、`same_host_cpu_cuda_split_admitted=false`、`production_runtime_enabled=false`。
- GPU 状态旁证：`clocks.sm=210–2490 MHz`，温度 `58–73°C`；GPU 利用率从 0% 到 99%；active reason 位出现 `0x0/0x1/0x4/0x24`。其中 `0x1` 与 GPU idle 语义一致，但仅凭同期变化不能断言某个 reason 是 timing 长尾的唯一根因；未取得管理员权限，未做 `nvidia-smi -lgc` 锁频对照。
- 工具回归：`tests/test_torch_hardware_admit.py` **18 passed**，CPU 诊断 **3 passed**；telemetry 只诊断，不改 CV 门、不解锁运行时。下一步限定为管理员锁频或提高持续 GPU 利用率的受控对照；`TORCH-RUNTIME-ADMIT-01` 继续锁定。

另跑部署默认精度观察（1 次 warmup、每格 3 次未插桩样本）：CUDA 整模 FP16 reference 下，CPU FP32→CUDA FP16 的 64-token prefill/decode exact，256-token prefill 不 exact；CUDA FP16→CPU FP32 的 64-token prefill 不 exact、256-token exact，所测 decode token 均 exact。边界张量分别为 `[1,64,896]`/`[1,256,896]`，CPU→CUDA FP32→FP16 转换最大绝对误差约 `0.115`。结果说明混合精度层段的 prefill 正确性受 prompt 影响，部署默认组合不准入；证据 `local_docs/evidence/torch-hardware-admit/cuda-deployment-default.json`。

本机 QLH `serialize_tensor_fast` 经 loopback TCP echo exact；payload 为 230,957 / 919,085 bytes，但该旧样本是单机回环且不含生产认证/控制封套。**Surface 环境与资产复核（2026-09-24）**：`tailscale ping` 双向均显示经 WLAN 直连 underlay（本机看到 `192.168.0.100:41641`，Surface 看到 `192.168.0.101:41641`），控制面约 5–11 ms；但新开 Tailnet TCP 端口未获生产可达性证据，不能把控制 ping 当数据面时延。Surface 的隔离 `.venv-qwen3-sidecar` 已与主仓锁定版本对齐：Python `3.12.10`、Torch `2.13.0+cpu`、Transformers `5.17.0`、tokenizers `0.23.2`、safetensors `0.8.0`、accelerate `1.14.0`，`pip check` 通过；常驻 keep-head 服务使用的 `.venv-test` 未修改。主仓 Qwen2.5-0.5B 原生 Safetensors 工件已同步到 Surface，新目录的权重 SHA-256 `fdf756fa…fb7fe`、manifest SHA-256 `40133469…12a2b9` 与本机一致；Surface config/tokenizer 轻量探针通过（Qwen2Config、24 层、hidden 896），随后真实 CPU smoke 也完成 16 token greedy 输出，证据为 `surface-qwen25-0.5b-cpu-smoke-20260924.json`。Qwen1.8B 已按主仓裁决退役，只保留为历史/待清理资产，不再追 remote-code 补丁，也不作为硬件对照。两端主仓 revision 与 `model_module.py` 仍不同，故共同工件和依赖已对齐，但真实跨机 PyTorch peer 仍未宣称完成。

跨机张量传输做过三类 echo。首版 `crosshost-tailscale-tensor-echo-20260923.json` 在计时窗口里让接收端执行 QLH 反序列化和 `torch.equal` 全张量比较，故其 836/345 ms 中位数**不是纯 TCP RTT**，保留原始数据但不用于链路结论。修正版 `crosshost-tailscale-raw-echo-v2-20260923.json` 由 Surface 发起 Tailnet TCP，服务端只读帧并原样回发；64-token（230,957 bytes）与 256-token（919,085 bytes）各 20 次全部字节往返，raw application echo RTT 为 **median 69.6/180.6 ms**，CV=`0.970/0.404`，仍非稳定链路。新建 `scripts/torch_lan_echo_probe.py` 后，直接 WLAN 同网段 TCP 回环（Surface `192.168.0.100` → 本机 `192.168.0.101`）同样 exact，但 median 为 **159.7/782.5 ms**、CV=`0.462/0.384`，证据 `crosshost-wlan-direct-echo-20260924.json`；经 SSH reverse tunnel 的 Tailnet 数据面 median 为 **283.7/1197.6 ms**、CV=`0.170/0.105`，证据 `crosshost-tailnet-ssh-echo-20260924.json`。这些都是无 TLS/生产认证控制封套/模型推理的原始字节探针；控制 ping 的 5–11 ms 不能代表大 payload 数据面，生产跨机准入仍拒绝。后续需定位 TCP 大帧吞吐/长尾，并用同一 revision/runtime 的真实 PyTorch peer 完成跨机推理矩阵；CPU 抖动另需 ETW/CPU 频率与热状态诊断。定向回归 `tests/test_torch_lan_echo_probe.py` + `tests/test_torch_hardware_admit.py` 为 **17 passed**；`TORCH-RUNTIME-ADMIT-01` 仍锁定；Edge/Android 仍不引入 Torch。

`TORCH-RUNTIME-ADMIT-01` 必须等硬件准入后：实现真实报告的来源/内容验证与 plan 新鲜度；将 device、dtype、shape、layer range、KV cache 分配和生命周期绑定到运行计划；加入动态显存/RAM 预留、并发/计划失效处理、完整整请求回退及端到端故障测试。仅有 Python `admitted=True` 对象或离线 KV owner/layout 摘要不能证明真实缓存已按合同分配。运行时默认关闭，只有完整端到端准入通过才允许显式 feature gate。MoE placement 后移到这两票之后；Edge/Android 不增加 Torch 依赖。

---

## 8.7 实验票与产品准入的边界（2026-10-05 登记）

§8.1–§8.6 与本文列的 torch 实验票、以及 `docs/分票规划-稳定分布式版本与架构泛用性后置-2026-10-04.md`
中新增的 **`XFRAME-1/2/3`**（跨框架数值对齐实验），共同构成**同一批实验票**。纪律一致：
**实验先行——未取得确定成果前，不接入产品实现，不写进稳定合同。**

`XFRAME-1/2/3` 排在 `GEN-*`、`KT-1` 与本文 torch 票**之前**执行。依据：它直接决定「跨引擎
混搭能否作为长期部署形态」，且 `XFRAME-1`（分歧自动检测与量化）是其余两票以及 `GEN-3` 的
**判据前置**——没有它，「某项改动是否让分歧推迟」无法判定。

**一处据此解释的观察**：§6.2 / §8 提到「KT 推的 AMX / AVX-512 MoE kernel **已经在主仓 vendor
的 llama.cpp 里**（`ggml/src/ggml-cpu/amx/mmq.cpp`），但实际用的是 pip 预编译 wheel，其
`ggml-config.cmake` 里 `GGML_AMX_BF16/INT8/TILE` 与 `GGML_AVX512*` 全为 `OFF`」。核实结论：

- **源码在**（`android/app/src/main/cpp/llama.cpp/ggml/src/ggml-cpu/amx/mmq.cpp` 确实存在）；
- **未启用的原因不是我们的构建脚本刻意关闭** —— `GGML_AMX_*=OFF` 出自 **pip wheel 自带的
  `ggml-config.cmake`**（`.venv-*/Lib/site-packages/lib/cmake/ggml/`），即**预编译产物如此**；
  QLH 自有的构建脚本（`build_keep_head_shim.ps1`、`edge_sim_setup.sh`）也**未显式启用**它。

⇒ 因此这是一个**待评估的构建项**，归属 §8.6 的硬件准入实验票：需要先判定「目标机是否具备 AMX
（Sapphire Rapids+）」以及启用后对**数值一致性**的影响，再决定是否进入发布构建。**实验先行，
未取得确定成果前不改发布构建。**

## 9. 明确标注为「未核实」的条目

1. KT 与 **vLLM** 是否另有官方整合（只找到它用 vLLM 项目的 `llmcompressor` 做 GPU 量化）。
2. ~~KT 的 AMX kernel 是否已回流 llama.cpp（旧文档只写"考虑贡献"，未查到对应 PR）。~~
   **已核实（2026-10-05）**：AMX / AVX-512 的 CPU MoE kernel **已在主仓 vendor 的 llama.cpp 源码树内**
   （`android/app/src/main/cpp/llama.cpp/ggml/src/ggml-cpu/amx/mmq.cpp`），无需再从 KT 侧搬运。
   但**实际运行用的是 pip 预编译 wheel**，其 `ggml-config.cmake` 把 `GGML_AMX_BF16/INT8/TILE`
   与 `GGML_AVX512*` 全部置 `OFF` ⇒ **该 kernel 当前不参与计算**。是否启用归 §8.6 硬件准入票。
3. KT 的 **ARM64(KML) / Windows** 后端成熟度与实际性能（README 只列为可选构建，roadmap 仍在探索）。
4. §6 里所有**收益数字在 QLH 设备/模型上的外推**都属估计，**非实测**。
5. §7 的方向冲突**未做新实测裁决**。

---

## 附：主要来源

- KT 官方仓库文档：`doc/en/AMX.md`、`doc/en/balance-serve.md`、`doc/en/prefix_cache.md`、
  `doc/en/kt-kernel/experts-sched-Tutorial.md`、`doc/en/kt-kernel/Native-Precision-Tutorial.md`、
  `doc/en/DeepseekR1_V3_tutorial.md`、`kt-kernel/README.md`、`kt-kernel/scripts/README.md`、issue #1921
- LMSYS blog（AMX kernel 21.3 TFLOPS / NUMA +63% / CUDA Graph / Expert Deferral +1.45×）
- SOSP'25 论文摘要与 madsys 论文页
- QLH 侧：[跨框架接力-当前有效基线与后续优化计划](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md)、
  [分布式推理并行与跨框架路线调研汇总](archive/relay/分布式推理并行与跨框架路线调研汇总-2026-09-15.md)、`README.md`
