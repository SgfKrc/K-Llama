# MoE 的搬运与计算：head 段两处提速

> 状态：**方案**（§6 的七条前提已逐条判定，见 `MOE-搬运与计算优化-验证.md`：方案 A 可做，方案 B 的核心论据不成立；本文不含任何新实测，所有数字都标了口径与出处）
> 更新日期：2026-11-15
> 适用范围：Qwen3.6-35B-A3B Q4 下 **head 段**（本机 GPU+CPU 混合，29 层专家留 CPU）的 MoE 路径；读者为实现者与验收者。分段结构、协议、网络、量化档位不在本文范围。
> 相关：`DESIGN.md` §1（专家上 GPU 的实测与显存边界）、`IGPU-SLOWDOWN.md` §⑤/§⑦（`op_offload` 与带宽结论）、`SEGMENT-OFFLOAD.md`、`MEASUREMENTS.md` §3、`THREE-SEGMENT.md` §4（段耗时的成本结构）。
>
> **引用约定**：本仓库只有文档；代码路径分属三处——`H:\llamacpp-c`（llama.cpp fork，本文的 `repack.cpp` / `ggml-cpu.c` / `llama-moe-cache.cpp` / `common.h` / `split-stage/main.cpp` 都在这里）、
> `H:\sgfd9\Strata`（本文的 `include/strata/kernels/cpu/pool.hpp`、`Strata:docs/QWEN36.md`、`Strata:docs/INTEL.md`）、上游 `ggml-org/llama.cpp`（PR 编号）。

---

## 0. 一句话

head 段的两个瓶颈不是「算力不够」，是**搬得慢**和**喂不饱**：MoE 专家从 host 到 GPU 走 pageable memcpy（PCIe x8 上约减半），CPU 侧的 8 专家 gather 只有 6.5 GB/s 而同一后端的 dense GEMV 能到 41.7 GB/s。两处各有一个现成解法——上游被关闭的 PR #31122 与 Strata 的专家池批处理——合计改动 **400–800 行**，不碰分段结构。

## 1. 目标与基线

**目标**：Q4 质量下，速度超过三段链的 **5.85 t/s**。

**基线（每行带口径，禁止跨口径相减）**：

| 形态 | 口径 | t/s | 出处 |
| --- | --- | ---: | --- |
| 三段链（本机 head + Surface relay + Y700 tail） | 941 prompt / 64 gen / `-t 8` / `-ngl 0` | **5.85** | `INDEX.md` §一（原始记录 `20261112-D1`） |
| 两段链（本机 `0:32` + Y700 `32:40`） | 同上 | 7.46 | `20261112-D2` |
| 弱设备链（Surface `0:24` + Y700 `24:40`） | n=24 / IQ4_XS 蒸馏版 / 21.87 GiB | 6.50 | `ANDROID-SURFACE-段接入评估.md` §0 |
| 单机整模纯 CPU | 941 + 64 | 8.23 | `DESIGN.md:25` |
| 单机整模 GPU+CPU | **pp512 / tg256** | **30.45 ± 0.27** | `DESIGN.md:26`（`-ngl 99 -ncmoe 29`） |
| 弱设备每层 | — | Surface 3.76 ms / Y700 2.84 ms | 同上 §2.1 |

> **口径警告**：30.45 是 `pp512/tg256`，与 5.85 的 `941+64` 不是同一口径，**不能相减**。补这个口径是 §6 第 1 条。

**三段 170.9 ms/token 的成本拆解**（按 `THREE-SEGMENT.md` §4 的公式 `stage_ms[i] = RTT + 段内计算 + IPC/唤醒 + (仅尾段) lm_head`，配本机实测系数）：

| 项 | 值 | 占比 | 依据 |
| --- | ---: | ---: | --- |
| 每段固定开销 × 3 | 3 × 14.78 = **44.3 ms** | 26% | 2026-11-13 单机 decode 截距实测 |
| 往返 × 4（每段去+回） | 2×9 + 2×27.5 = **73 ms** | 43% | 本机↔Surface 9 ms、本机↔Y700 25–30 ms |
| 段内计算（40 层） | **54 ms** | 31% | 170.9 − 44.3 − 73 |
| 8 KB hidden 传输 | **2 ms** | 1% | 100 Mbps 链路 |

**结论：通信数据量不是瓶颈（1%）。RTT 与每段固定开销占 69%，段内计算占 31%。** 本文只处理那 31% 里 head 段的部分——因为 RTT 是网络物理属性，每段固定开销在 `ARCHITECTURE.md` §11（串行叠加）与 `INDEX.md` §四（10–20 ms）已单独讨论。

## 2. 两处瓶颈的定位

### 2.1 搬运：pageable → GPU 的 memcpy

默认 mmap 加载时，loader **刻意**把 pinned host buffer 降级成 plain CPU（上游 PR #31122 的 Motivation 原文：`the loader deliberately downgrades the pinned host buffer type to plain CPU`）。于是 MoE 专家缓存上传的**每一个字节**都是 `cudaMemcpyAsync` from **pageable**——decode 路径的 cache 填充、大 batch 的 used-expert 上传。

上游实测（Qwen3.6-35B-A3B Q4_K_XL 22.3 GB、RTX 4060 Ti 8 GB、PCIe 4.0 **x8**、`--cpu-moe`）：

| 配置 | prompt t/s | gen t/s |
| --- | ---: | ---: |
| `--moe-cache-mib 3072`，mmap（pageable 上传） | 62.9 | **18.8** |
| 同配置 **cache-off** | — | **22.3** |
| `--moe-cache-mib 3072 --load-mode none`（全 pinned，21 GiB 锁定，慢加载） | 279.1 | 32.3 |

**pageable 下 cache 是负收益（18.8 < 22.3）**——这是本文最重要的一个事实：只打开 `--moe-cache-mib` 而不解决 pinning，会让 head 段变慢。

### 2.2 计算：8 专家 gather 只有 6.5 GB/s

llama.cpp 的 MoE CPU 路径在 `ggml/src/ggml-cpu/repack.cpp:4783` 的 `forward_mul_mat_id`：

```cpp
for (int cur_a = 0; cur_a < n_as; ++cur_a) {          // 逐 expert
    ...
    for (int ir1 = 0; ir1 < nr1; ir1++) {             // decode 时 nr1 = 1
        gemv<...>(ne00, ..., src0_cur + src0_cur_start * nb01, ...);   // 一次单行 GEMV
    }
}
```

decode 时每个 expert 只有 1 行，于是变成 **8 次独立的 1.72 MB GEMV**，每次线程只在 `src0_cur_start..end`（按输出行切）上分块，调用之间有同步空隙。

实测对照（`IGPU-SLOWDOWN.md` §⑦）：

| 路径 | 有效带宽 |
| --- | ---: |
| `MUL_MAT_ID_VEC`（**8 专家 gather**） | **6.5 GB/s** |
| `MUL_MAT_VEC`（连续权重，dense） | 6.6 GB/s |
| 9B dense 纯 CPU（`-t 8`） | **41.7 GB/s** |

前两行证明 **gather 的地址惩罚≈0**（6.5 vs 6.6），所以 6.4 倍的差距不是「地址不连续」，是**每层只有 8 个专家、并行度喂不饱所有核**。

## 3. 方案 A：host 专家页 pinning（移植上游 #31122）

### 3.1 上游的状态

| 项 | 内容 |
| --- | --- |
| PR | `ggml-org/llama.cpp#31122`「llama : opt-in host expert page pinning for the MoE cache」，作者 `S95F`，**1 个 commit**（`25f222c`），**Closed** |
| 关闭原因 | **流程，不是技术**：PR 模板未填 / 新贡献者限 1 个 open PR（他有 2 个）/ PR 描述是 AI 生成的（代码允许 AI，描述必须自己写） |
| 现成基础 | `ggml_backend_cuda_register_host_buffer`（`cudaHostRegister`，env-gated by `GGML_CUDA_REGISTER_HOST`）**已存在并导出**，只是**没有 in-tree 调用者** |

### 3.2 做法

`GGML_CUDA_REGISTER_HOST=1` 时，在 **MoE cache 构造**阶段：

1. 把 cache 要管的那些层的 **host 专家 range 按真实 OS 页大小对齐并合并**；
2. 用 `cudaHostRegister`（read-only、portable）注册给驱动；
3. **保留 mmap 的快速加载与 file-backed pages**——这是它相对 `--load-mode none` 的核心优势；
4. 驱动拒绝的 range 保持 pageable（现有 wrapper 已经 fail soft）；
5. 已在 pinned host buffer 里的专家（如 `--load-mode none`）跳过；
6. teardown 时全部 unregister；
7. env 在 cache 内也检查一次，未设（默认）时这段代码**完全不运行**。

### 3.3 cap 与 WDDM（这一条是 load-bearing）

- 默认上限 = **物理 RAM 的 1/4**（通过 `ggml_backend_dev_memory` 拿 CPU device；拿不到则 8 GiB）；`LLAMA_MOE_PIN_MIB` 覆盖；≤ 0 禁用；
- **Windows 上 pinned 页计入 WDDM shared budget（约 RAM 的一半）**：上游实测「无限注册到 **~15.3 / 31 GiB** 后系统不稳定——generation variance 爆炸，一次运行以 `ggml_backend_cuda_buffer_set_tensor` 的 VRAM OOM 失败」。

这与 `MEMORY-LIMITS.md` §5「WDDM 超配不报错只掉速」是同一件事的两个后果。**cap 必须设，不能贴边。**

### 3.4 收益（上游实测，同模型同档显存）

`-ngl 99 -ncmoe 40 --moe-cache-mib 2048`，**mmap 全程**：

| 配置 | pp512 | tg128 |
| --- | ---: | ---: |
| pageable（master） | 234.7 ± 14.6 | 27.65 ± 1.41 |
| pinned，cap 4 GiB（22% 专家字节） | 296.9（**+27%**） | 30.59（**+11%**） |
| pinned，cap 8 GiB（44%） | 337.8（**+44%**） | 33.00（**+19%**） |

实际运行（600 token prompt + 256 gen、4k ctx、`--moe-cache-mib 3072`、默认 cap）：

| | prompt | gen |
| --- | ---: | ---: |
| pageable | 62.9 | 22.3 |
| pinned | **197.3** | **28.9** |

**prompt 侧的提升（+214%）远大于 gen（+30%）**——对 941-token 的链上场景，prefill 是 `prompt_ms 28.7 s` 的大头，这一项的绝对收益比 gen 更值钱。

### 3.5 正确性与已知限制

- **逐位一致**：temp 0、固定 seed 下，pinning 开/关生成文本**完全相同**——注册只改驱动怎么拷字节，不改字节本身；
- **已知限制**：一个 model 上两个 context 时，第二个 context 的注册会 fail soft（already registered），而第一个 context 的 teardown 会把共享 range unregister——**性能问题，不是正确性问题**。

### 3.6 实现点

| 项 | 位置 |
| --- | --- |
| cache 构造/析构 | `src/llama-moe-cache.cpp`（`llama_moe_cache` 的 ctor/dtor） |
| cap 来源 | `ggml_backend_dev_memory(dev, &free, &total)` |
| 前置条件 | `moe_cache_mib > 0`（**K-Llama 现在默认 0**，`common.h:735`；`split-stage` 已接线但默认 0） |
| 估行数 | 100–200 |

## 4. 方案 B：跨 expert 批处理（Strata 风味内核）

### 4.1 靶子

`ggml/src/ggml-cpu/repack.cpp:4783` 的 `forward_mul_mat_id`（§2.2 已贴结构）。同一套「逐 expert」逻辑在非 repack 路径也有一份：`ggml-cpu.c:1691-1755`。

### 4.2 Strata 的做法（只借设计，不借代码）

`include/strata/kernels/cpu/pool.hpp:3-30`：

> A layer runs **TEN experts against ONE activation**, and the expert kernel is **DRAM-bound**（42.55 GB/s on 6 cores, tracking core count almost exactly）。So the pool's job is not to be clever —
> it is **to keep every physical core reading expert bytes for the whole layer**, and to be cheap enough that **ten dispatches per layer cost less than one expert**。

机制：一层 N 个专家组成一个 `ExpertJob` 批 → `head`/`done` 两个 64-bit 原子计数 → 每核一次 CAS claim（claim 带 epoch，防止迟到 worker 抢下一批）→ 全部完成后 barrier → host 求和。

### 4.3 改法

把「逐 expert 调 gemv」改成**跨 expert 的 job 队列**：

1. 先按 `ids` 把 (expert, 行块) 展平成 job 列表（现有 `matrix_rows` / `matrix_row_counts` 已经做了分组）；
2. 所有线程从一个原子计数器上 claim job（一次 claim 一个 (expert, 行块)）；
3. 每个 job 内部仍是现有的 `gemv`，只是不再由「当前 expert 的线程子集」独占；
4. 结束时 barrier。

**关键收益点**：decode 时每 expert 只有 1 行、`ne00 = 2048`，每线程一次 gemv 只读 64 行 × 2048 × 0.5 B ≈ **65 KB**——数据量小、调用间有同步空隙。批处理把这些 65 KB 的碎片合成一次连续的 claim 流，让所有核在整个 MoE 段保持满载。

| 改动 | 位置 | 估行数 |
| --- | --- | ---: |
| repack 路径的 job 队列化 | `repack.cpp:4783-4919` | 150–300 |
| 非 repack 路径同步改，或让它强制走 repack | `ggml-cpu.c:1691-1755` | 100–200 |
| **合计** | | **250–500** |

### 4.4 预期（**推算，未实测**）

| 假设 | 6.5 → | 每层 ms（Surface 基准 3.76） | 弱设备链 t/s |
| --- | ---: | ---: | ---: |
| 保守 | 10 GB/s | 2.44 | ~8.5 |
| 中 | 15 GB/s | 1.63 | ~11.8 |
| 上限（dense 的 41.7） | 30 GB/s | 0.82 | ~20.0 |

**这些是算术外推，不是实测**；§6 第 5 条要求先验证「6.5 与 42.55 是否同口径」。

## 5. 两方案的叠加关系

| | 方案 A（pinning） | 方案 B（批处理） |
| --- | --- | --- |
| 作用路径 | host → GPU 的 **H2D 拷贝** | CPU 侧的 **专家 GEMV** |
| 受益条件 | GPU cache 有命中、且命中率 > 0 | 段内有 host 专家（head 段 29 层即满足） |
| 失效条件 | 全 offload 的段（无 host 专家） | 全 offload 的段（无 CPU 专家计算） |
| 风险 | 低（不改字节） | 中（改调度，需逐位对照） |
| 成本 | 100–200 行 | 250–500 行 |

**两者不冲突，作用在不同资源上**：A 吃 PCIe，B 吃 DDR。head 段有 **29 层 host 专家**（`-ncmoe 29`，所以 B 有效）；`moe_cache` 打开之后才有 GPU cache（所以 A 届时才有效）。

**但顺序有讲究**：A 必须先做——不先修搬运就打开 cache 等于主动降速（§2.1 的 18.8 < 22.3）；这条是否在本机成立，正是 §6 第 3 条。

## 6. 前提待实证（七条，本文的交付重点）

以下每一条都是**本文的结论所依赖、但目前没有实测**的假设。任何一条不成立，对应的方案收益就要打折或作废。

| # | 前提 | 现状 | 怎么验 | 不成立时 |
| ---: | --- | --- | --- | --- |
| 1 | **head 段同口径（941+64）速度** | 只有 `pp512/tg256` 的 30.45 | `-ngl 99 -ncmoe {9,10,11,12}`，941 prompt + 64 gen，`-r 6` 丢首个 | 基线不明，所有百分比收益无法换算成绝对 t/s |
| 2 | **#31122 在 Q4_K_M + 4060 Laptop 上复现** | 上游是 Q4_K_XL 22.3 GB + 4060 Ti 8 GB；本机是 Q4_K_M 22.66 GB + 4060 Laptop 8 GB | cherry-pick 后同口径 A/B | +11%~+30% 不成立，方案 A 作废 |
| 3 | **打开 `moe_cache` 后的 pageable 基线** | K-Llama 默认 `moe_cache_mib = 0`；上游数据说 pageable 下 cache 是 regression（18.8 vs 22.3） | 打开后同口径测，与 cache-off 对照 | 可能「开了更慢」，必须先修 A 再开 |
| 4 | **6.5 GB/s 与 42.55 GB/s 是否同口径** | 前者是 8 专家 gather（llama），后者是 10 专家批（Strata，6 核） | 读 Strata `Memory/LEDGER.md` L9 的 bench 条件；本机用 `perf` 看 MoE 段的 IPC 与 DRAM 计数 | 批处理上限被高估，§4.4 的表要下修 |
| 5 | **批处理改动的收益** | 纯推算（§4.4） | 实现后 A/B，逐位对照 + tg/pp | 白做，退回原调度 |
| 6 | **WDDM cap 在 32 GB RAM 上的可用值** | 上游：无限注册到 15.3/31 GiB 后不稳；默认 cap = RAM/4 = 8 GiB | 从 2 GiB 起逐步加，每档跑 `-r 10` 看 generation variance | cap 贴边会让系统不稳（variance 爆炸 / VRAM OOM） |
| 7 | **分段链上 `moe_cache` 是否真的生效** | `split-stage` 已接线（`main.cpp:534-535, 1073-1074`），head 是 `llama-server` | 链上跑并抓日志，确认 cache 构造与命中统计 | 只在单机有效，链上收益为零 |

**第 1 条是所有其他条的前提**——没有同口径基线，后面每一条都无法判断「+11%」到底值多少 t/s。

## 7. 实施顺序与验收

| 步 | 动作 | 验收 | 预计 |
| ---: | --- | --- | --- |
| 0 | 补同口径基线（§6 第 1 条） | 941+64 下 `-ncmoe` 扫描的 tg 曲线 | 半天 |
| 1 | 打开 `--moe-cache-mib`（零代码） | 与 cache-off 对照；若变慢则停在此步，先做 2 | 半天 |
| 2 | cherry-pick #31122（或按 §3.2 重写） | temp 0 逐位一致；tg/pp 对照；cap 扫描（§6 第 6 条） | 1–2 天 |
| 3 | 跨 expert 批处理（§4.3） | 逐位一致；MoE 段的有效 GB/s（`perf` 或计时反算） | 1 周 |
| 4 | 链上端到端（§6 第 7 条） | 三段链 941+64 的 t/s，与 5.85 对照 | 1 天 |

**每一步的判据都是「同口径 A/B + 逐位一致」**，不接受「看起来快了」。

## 8. 不做的事

| 不做 | 理由 |
| --- | --- |
| **改造 Strata 成分段引擎** | Strata 的 Qwen3.6 路径（`src/qwen35`，5,378 行）**只在 CUDA/HIP**（`Strata:docs/QWEN36.md:8`）；Surface（Iris Xe）与 Y700（Adreno）是 Vulkan/OpenCL，跑不了 |
| **移植 SYCL/Vulkan** | `sycl/` 有 114,290 行基础设施但缺 `qwen35`；且 `Strata:docs/INTEL.md:9` 说 Intel Arc 上 llama.cpp SYCL 只有 Strata CUDA 的 1/3 |
| **纯 CPU 跑 Strata** | `src/kernels/cpu/` 只有专家与量化 kernel，attention/KV/GDN/QSA（~15,149 行）没有 CPU 版 |
| **改分段结构 / 协议 / 网络** | 通信数据量只占 1%（§1）；RTT 是物理属性 |
| **换量化** | Q4 是质量目标；Q2/Q3 的容量优势是另一个取舍 |
| **动 `op_offload` 默认值** | `IGPU-SLOWDOWN.md` §⑤ 已归因，且分段链默认 `no_op_offload = true`，改了也不生效 |

## 9. 与 K-Strata 立项的关系

本文**不构成 K-Strata 立项**：它做的是「在 K-Llama 里把 MoE 的搬运与计算修好」——这两处修完，head 段与整模都会受益，与是否分段无关。K-Strata 在当前硬件上**不具备前提**（唯一能跑 Qwen3.6 的后端是 CUDA/HIP，三台机器里只有本机是 CUDA；要立项先得有一台 CUDA/HIP 的 relay 机器）；将来硬件到位，Strata 值得抄的也只有两处，而这两处本文已经覆盖——CPU 专家池的批处理调度（§4.2）与 host 专家的 pinned 直传（§3.2）。
