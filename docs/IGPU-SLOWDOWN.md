# 核显能不能用：三轮实测的结论与修正

> 状态：**现行**
> 更新日期：**2026-11-10**（本轮用当前代码重测核显。此前 2026-11-09 版的核显数据全部来自 v2（2026-11-06 前后）的
> `llama-cli` 实测、标注为「来自 v2、未复测」；本轮逐个复测并改写，方法与原始数据见 §⑥）
> 适用范围：核显为什么慢（`-ngl` 扫描、prefill 交叉点、host buffer 抢占、`op_offload`），以及 CUDA 侧同类的
> `op_offload` 默认值问题。段级 offload 的语义与显存花法归
> [`SEGMENT-OFFLOAD.md`](SEGMENT-OFFLOAD.md)。

**结论（2026-11-10 复测；9B Q4_K_M，`-t 8`，单卡隔离 `GGML_VK_VISIBLE_DEVICES=0`，llama-bench `-r 3` + 热身 1 轮）**：
全量 offload 的 prefill 仍大赢、decode 基本持平，但 v2 记的「+5.4% / 3.1×」在**同构建 CPU 基线**下缩水为
**−4.8% / 1.90×**；缩水的主因是 v2 的 CPU 基线用错了构建，不是核显变差：

| 指标（9B Q4_K_M） | 同构建 CPU-only | Iris `-ngl 99` | 比 | v2 记的（旧口径） |
|---|---:|---:|---:|---|
| decode（tg64） | 7.35 | 7.00 | **−4.8%** | 6.72 → 7.08（+5.4%） |
| prefill（pp256） | 42.70 | **80.98** | **1.90×** | 29.20 → 90.80（3.1×） |
| prefill（pp32） | 45.18 | 30.94 | 0.69×（输） | 0.93× |

**基线**：v2 用的是 `build-cpu`（`GGML_CPU_ALL_VARIANTS=OFF` 的通用变体，pp32 30.40 / pp256 33.53）；同构建 CPU-only
（`build-vulkan-new/bin` 移走 `ggml-vulkan.dll`，加载 haswell 变体）是 45.18 / 42.70 —— **通用变体慢 33% / 21%**。v2 的
「3.1×」里约 1.2× 是这条基线差异，真实值 1.90×；v2 的 §③ 自己记过这个口径坑却没从结论里扣掉——**这是方法论教训，
不只是数字修正**。

**`-ngl 0` 的 −41% 不复现**：当前代码（571a9d0 / 19776b74e）实测 6.19–6.78 t/s（−16%/−8%）；v2 时代的 e0f142d 构建
（`H:\llamacpp-clean\build-vulkan`）单卡 6.5（−7%）、双设备 6.1（−13%），都到不了 v2 记的 4.1。原因：571a9d0/19776b74e
已含 `src/llama-model-loader.cpp` 的 `(use_mmap || IGPU)` host-buffer 绕过，「未 offload 的层进 Vulkan host buffer」
在当前代码上已不成立；`-ngl 0` 剩下的亏损来自 `op_offload` 把 prefill 算子搬上 GPU，不是 host buffer。

**中间值仍然更差**（`-ngl 16` tg 5.92 = −19%；`-ngl 0` 默认 6.19–6.78 = −16%/−8%），「全量或不用」不变；但「中间值慢的
主因」从 v2 的 host buffer 抢占改成了 `op_offload`（见 §⑤/§⑥）。**第一轮为什么测反了**：① 默认双设备枚举（Iris + 4060）
走 `multi_device` 慢路径；② `llama-cli` 的短 prompt（~50 token）正好落在交叉点左侧；③ CPU 后端不同级（`GGML_CPU_ALL_VARIANTS`）。

**35B 装不进**：20.82 GiB > Iris 共享堆 ~15.4 GiB，部分 offload 会落进 OOM 路径。**方法学**： `llama-bench` 的 `-ngl`
不要一次传多个值（多值模式与单值能差 3 倍）；避免高负载（同一配置重复测能差 25%）；Vulkan 首轮要热身；
**基线与测试构建必须同源** —— v2 踩的就是这条。

三个问题：① v2 解释过，结论齐全，只是分散在 5 份文档里；② 中间值慢的主因 v2 记的是 host buffer 抢占，2026-11-10
复测后应改为 `op_offload`； ③ v2 的 gen 中位（6.9 → 4.1 → 5.8 → 6.7）是本轮唯一复现不出来的一组：同口径实测
7.0 → 6.6 → — → 7.2。

## ① v2 到底有没有解释过：有

### 1.1 最直接的一条：Vulkan 加载即污染 CPU 层

`slimarc/prototype:docs/ANDROID-SURFACE-段接入评估.md` 90–99 行：**Vulkan 后端只要被加载、即使 `-ngl 0`，CPU 侧
权重也会进 Vulkan host buffer**（`make_cpu_buft_list` 把 host buft 插在 CPU buft 之前）；禁掉 `ggml-vulkan.dll` 后
Surface 从 245 ms 降到 112 ms。同节还有弱设备结论：Iris Plus 慢 30%、Adreno 750 慢 5–25%，归因是「共享同一块
LPDDR，只增加一次拷贝，不增加带宽」——只适用于弱设备；[`PROGRESS.md`](PROGRESS.md) §1.1 复述了同一结论。

**2026-11-10 复测：这条机制在当前代码上已被修掉。** 571a9d0 / 19776b74e 的 `src/llama-model-loader.cpp` 在
`select_weight_buft` 末尾加了 `(use_mmap || IGPU)` 分支，host buft 被换回 CPU buft；9B `-ngl 0` 单卡实测 tg64
6.19–6.78，只比同构建纯 CPU（7.35）低 8–16%，不是 v2 的 −41%（§⑥）。

### 1.2 核显的算子级实测（唯一直接测核显算力的一条）

[`DEPLOYMENT.md`](DEPLOYMENT.md) §6.8：IQ4_XS 在**所有后端都只有 MMV、没有 MMQ**（OpenCL/OpenVINO 不支持），
`CPU_REPACK` 全部回退；head 段约 15.9 GiB 里 8.37 是 IQ4_XS，塞进 15.82 GiB 共享堆会连系统挤死，tail 四种后端全覆盖、
约 6.0 GiB 可整段常驻，所以核显只可能救 tail。本机 Iris Xe 用 ctypes 直调 OpenCL：流式读 780 MiB **13.1 ms =
58.2 GB/s**（CPU 6 线程 23.7 GB/s），`lm_head` 形状 GEMV **20.5 ms = 23.6 GB/s**（三版实现 22.6–24.2）——**低算术
强度的 GEMV 只吃约四成 DRAM 带宽**，这条路上没有数量级；`--list-devices` 报 16194 MiB 正是那 15.82 GiB 共享堆。

### 1.3 offload 粒度的实测表

| 位置 | 内容 |
|---|---|
| [`MEASUREMENTS.md`](MEASUREMENTS.md) §3.1 | 专家上卡 11 层 **30.45** vs 整层上卡 11 层 **10.11**（3.0×）：专家占权重 88%，每 token 只用 8/256 |
| [`MEASUREMENTS.md`](MEASUREMENTS.md) §3 | `-ncmoe` 扫描：0→25.94、8→28.99、10→32.01、12→**17.31**、14→**11.23**、16→**9.75** |
| [`DESIGN.md`](DESIGN.md) | 11 层 30.45 最优，12 层 **15.29** 断崖；WDDM 超显存不报错，静默用系统内存，只表现为掉速 |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) §6.5/6.6 | `gpu_layers` 4/5/6/7 → 7.2 / 7.9 / **8.7** / 8.1；32/32 陷阱：stage 0 最快（135.4）但 tg 更慢，原因是**内存**换页 |

`-ngl` 与 `-ncmoe` 两条正交路径见 [`ARCHITECTURE.md`](ARCHITECTURE.md)；WDDM 超额分配静默、照旧打印 `offloaded N/N`
见 `README.md`。

## ② 机制分析：哪几条站得住

**A. host buffer 抢占（v2 的第一因，2026-11-10 已被修掉）。** `make_cpu_buft_list()` 顺序是 `CPU: ACCEL -> GPU host ->
CPU extra -> CPU`，只要 `!no_host`，host buft 就排在 CPU buft **之前**；v2 的后果是「加载 `ggml-vulkan.dll` 后
`-ngl 0` 的纯 CPU 就不存在」。**当前代码在 loader 层（`llama-model-loader.cpp` 的 `(use_mmap || IGPU)` 分支）把 host
buft 换回 CPU buft，这条后果已不成立**；`--no-host` 仍负优化（`-ngl 0` pp256 53.66→27.35），原因换成了 B。

**B. GPU 接管算子，代价是往返拷贝（上游 `#31538` 的 `0cc4m` 确认）。** 用 GPU host buffer 是刻意的：GPU 仍会接管大
算子、且从 host buffer 拷贝快得多。实测两个方向都不如纯 CPU：9B `-ngl 16` 5.8 → `--no-host` **3.2**；`-ngl 0`
4.1 → `--no-host` 5.3。**2026-11-10：当前代码下 `op_offload` 取代 host buffer 成为「`-ngl 0` 慢」的主因**（§⑥）。

**C. 显存溢出静默（独显场景）**：WDDM 不拒绝超额分配，只掉速（12 层专家 33→17，14 层→11）。**核显上不成立**：核显的
「显存」就是系统内存，`#31538` 的单 heap OOM（错误指向 kv cache）已修，不是「慢」的原因。

**D. 量化类型缺 MMQ**：IQ4_XS 在所有后端只有 MMV 没有 MMQ；v2 的 35B 权重 8.37 GiB 是 IQ4_XS。这解释 prefill 上不去，
不解释 decode 的损失。

**E. 共享内存总线是 decode 的上限。** DDR5-4800 双通道理论 76.8 GB/s；9B Q4_K_M（5.68 GB）每 token 要过一遍权重：

| 配置（2026-11-10，llama-bench tg64） | gen | 推算有效带宽 |
|---|---:|---:|
| 同构建 CPU-only | 7.35 t/s | 41.7 GB/s（54% 理论） |
| 核显全量 `-ngl 99` | 7.00 | 39.8 |
| 核显 `-ngl 16` | 5.92 | 33.6 |
| 核显 `-ngl 0` | 6.19 | 35.2 |

核显与 CPU 吃同一根总线，上限就是 CPU 的上限；搬到另一个执行单元拿不到增量带宽。`-ngl 0 --no-op-offload` 的 7.44
（42.3 GB/s）略高于纯 CPU，属噪声带内的「没有额外开销」，不是增量带宽。

**判定**：Iris Xe 纸面 96EU ~2.3 TFLOPS fp32 高于 i9-13900H `-t 8` 的 1.0–1.1，但实测更慢，瓶颈在带宽与拷贝（*TFLOPS
是推算，未实测*）；「带宽被核显征用」不成立（同一条总线，只多一次往返）；「撑爆共享显存换页」在核显上不成立（已修）。

**推测（未验证）**：① 2B 的 decode 在 `-ngl 0` 上比纯 CPU 还快 10%，可能因权重更易命中 L3/页缓存；② pp32→pp256 的曲线翻转没做 perf 对比；③ `#31538` 的修复是否只改了分配位置。

## ③ 第一轮实测：现成的 Vulkan 构建，口径有误

`H:\llamacpp-c` 下现成四份构建，没编译任何东西：

| 构建 | 后端 | 时间戳 | 用途 |
|---|---|---|---|
| `build-vulkan` | `GGML_VULKAN=ON` | **11-06 编，commit 571a9d0**（已含 IGPU host-buffer 修复；不是 v2 的 e0f142d） | 中间构建 |
| `H:\llamacpp-clean\build-vulkan` | `GGML_VULKAN=ON` | 11-02，commit **e0f142d**（v2 原始构建，只有 `llama-cli`） | 复现 v2 |
| `build-cpu` | 全 OFF（`GGML_CPU_ALL_VARIANTS=OFF`） | 571a9d0 | 通用变体基线（**不是公平基线**，见 §⑥） |
| `build-cuda` / `build-split` | CUDA / 全 OFF | 10-30 / — | 4060 对照 / v2.1 纯 CPU（跑不了 GPU） |

数据（`llama-cli`，`-t 8`，`-n 48`，每组 3 次取中位）。**这是第一轮口径，标签不可按字面读**：当时没设
`GGML_VK_VISIBLE_DEVICES`，Vulkan 默认枚举 Iris Xe + RTX 4060，走 `multi_device` 慢路径。这一轮的口径问题
（双设备、基线与测试构建不同源）**2026-11-10 已修正，见 §⑥**，本表只作历史对照；仍然有效的是「中间值与 `-ngl 0` 都比纯
CPU 差」与「部分 offload 在独显上也要付一半代价」，**「`-ngl 0` −41%」不再成立**。

**Qwen3.5-9B Q4_K_M（32 层，5.68 GB，dense）**

| 配置 | Prompt t/s | Gen t/s | Gen 相对纯 CPU |
|---|---:|---:|---:|
| `build-cpu`，`-ngl 0`（不加载 GPU 后端） | 21.4 | **6.9** | 1.00 |
| `build-vulkan`，`-ngl 99`（双设备） | 12.6 | 6.7 | −3% |
| `build-vulkan`，`-ngl 16` | 15.1 | 5.8 | −16% |
| `build-vulkan`，`-ngl 0`（只加载 Vulkan） | 7.9 | **4.1** | **−41%** |
| `build-vulkan`，`-ngl 0 --no-host` | 14.5 | 5.3 | −23% |
| `build-cuda`，RTX 4060 `-ngl 99` | 117.8 | 29.3 | **4.2×** |

**Qwen3.5-2B Q4_K_M（25 层，1.31 GB，Prompt / Gen）**：`build-cpu` `-ngl 0` 105.3 / 24.6；`build-vulkan` 核显
`-ngl 0` 74.7 / 26.2、`-ngl 0 --no-host` 92.8 / 24.5、`-ngl 12` 59.0 / 23.5、`-ngl 12 --no-host` 60.9 / 16.0、
`-ngl 99` 55.2 / 26.5；4060 `-ngl 12` 87.7 / 39.3、`-ngl 12 --no-host` 120.6 / 39.6、`-ngl 99` 199.1 / 74.3；
`build-cuda` 4060 `-ngl 99` 226.1 / 74.8。

读数：① 9B 上 `-ngl 0` 与中间值都输，v2 归因于 host buffer 里的层越多；2026-11-10 复测否掉了这个量级与归因
（−41% → −16%，原因是 `op_offload`，见 §⑥）。② 部分 offload 在独显上也要付一半代价（2B 上 4060 `-ngl 12` 39.3
vs `-ngl 99` 74.3，−47%），「不要用中间值」对独显同样成立，绝对值仍是正收益。③ prefill 两轮相反：第一轮 9B 核显
12.6 / 15.1 vs CPU 21.4 像绝对短板；第二轮 pp256 是 90.80 vs 29.20（3.1×）、只有 pp32 输（0.93×）；2026-11-10
同构建基线复测为 1.90×、交叉点 pp≈32–48（§⑥）。④ 2B 核显 gen +7% / +9.2%、9B +5.4% 这三条 2026-11-10 都缩水为
−2.1% / −4.8%，**只剩「装不进共享堆」这条硬边界成立**。⑤ Vulkan 首次运行要编译 pipeline，第一次采样能差一个量级
（第一轮 4060 `-ngl 99` 只有 9.5 t/s，热身两轮后 168.4）；§⑥ 的每组配置都跑了两轮、只取第二轮。

**35B 没测**：35B-A3B（20.82 GiB）CPU 上约 1 t/s，48 token 要 40 秒以上，且 15.82 GiB 共享堆装不下，会落进 `#31538` 的
OOM 路径而不是「慢」；2026-11-10 复测同样跳过（体积不变）。

## ④ 部署规则（2026-11-10 修订）

**不要用部分 offload；纯 CPU 机器别加载 GPU 后端。** `571a9d0`/`19776b74e` 的 loader 层已经不让未 offload 的层进
host buffer，所以 `-ngl 0` 不再像 v2 说的掉 41%，实测 9B dense 生成 7.35 → 6.19（`-ngl 0`）/ 5.92（`-ngl 16`）。
「要纯 CPU 就别链接 GPU 后端」仍成立，理由从「host buffer 抢占」换成了「`op_offload` 把 prefill 算子搬上 GPU + sched
中间 buffer 走 Vulkan host buft」。

- **核显值不值得用**：**decode 基本持平（−4.8%），prefill pp≥48 大赢（1.9–2.4×），pp≤32 输（0.69×）**。值得用的场景是
  「长 prompt、decode 只占一小部分」：长文档 prefill、一次性大上下文、agent 读长文那一步。短 prompt 对话、长生成
  （2k token 输出）用纯 CPU 更好。硬边界仍是体积：35B（20.82 GiB）装不进 ~15.4 GiB 共享堆。
- **「全量或不用」仍成立**：`-ngl 16` tg 5.92（−19%）、`-ngl 0` 默认 6.19（−16%），中间态是负的。
- **`--no-op-offload` 不是万能开关**：只在 `-ngl < 全部` 时有作用，且 pp256 上是负的（−42%）；只有 pp32 附近的小
  batch 是纯赚（+75%）。它能修的是「小 batch 被搬上 GPU」的浪费，治不了「全量 offload 的 decode 慢」。
- 本项目仍选纯 CPU 的两条理由不变：35B 装不进共享堆；核显与 CPU 共享同一条内存总线，全量 offload 的 decode 拿不到
  增量带宽（§② 的 E 条）。
## ⑤ 同一个坑在 CUDA 上：`op_offload` 的默认值

与核显无关，但属于同一类问题：§1.1 说 Vulkan「加载即污染 CPU 层」，CUDA 上换个名字又出现一次。测量装置是段级
probe（head `0:32`，35B MoE，CUDA 4060），与 `llama-cli` 装置不同，绝对值不能互比，能比的是机制与方向。

**现象**：CUDA 构建默认 `op_offload = true`（上游默认）。它让 GPU 接管落在 `CUDA_Host` buffer 里的 host
权重的算子——权重没有搬家，**算子在每个 token 被搬到 GPU 执行一次再回来**。段级实测（`-c 1024`，`-ngl 0` 与 `-ngl 8`
两档）：
| head `0:32` | 纯 CPU | `op_offload` 开（默认） | `--no-op-offload` |
|---|---:|---:|---:|
| tg（`-ngl 0`） | 16.37 | 11.25（**−31%**） | 15.69 |
| tg（`-ngl 8`） | — | 12.47 | **21.00（+68%）** |

| | Vulkan（核显 / 独显） | CUDA |
|---|---|---|
| 触发条件 | 加载 `ggml-vulkan.dll` 后，`-ngl < 全部` 时 `op_offload` 把大 batch 算子搬上 GPU（`op_batch_size >= 32`） | 构建默认开，`-ngl 0` 也躲不掉 |
| 机制（2026-11-10 修） | v2 记的「未 offload 层进 host buffer」已被 `llama-model-loader.cpp` 的 `(use_mmap \|\| IGPU)` 分支修掉；当前只剩 sched 中间 buffer 走 Vulkan host buft + `op_offload` 搬运 | 权重留在 `CUDA_Host`（pinned），算子被搬上 GPU 执行 |
| 代价 | `-ngl 0` 比纯 CPU 慢 16%（6.19 vs 7.35）；prefill pp256 反被 `op_offload` 抬到 53.66 | `-ngl 0` 比纯 CPU 慢 31% |
| 摆正方式 | 要纯 CPU 就别加载后端（不是 `-ngl 0`）；`--no-op-offload` 只对小 batch 有用，pp256 是负的 | `--no-op-offload`：CPU 层的算子回自己的 kernel |
| 错误方向 | `--no-host`：GPU 从 CPU buffer 跨总线拷（实测 `-ngl 0` 53.66→27.35） | `--no-host`：跨 PCIe 读 pageable 内存 |

**Vulkan 上同名但不同义**：`--no-op-offload` 在 Vulkan 是 **prefill 搬运开关**（只对 `-ngl < 全部` 且 batch≥32 的
op 生效，pp256 上关掉是负的），在 CUDA 是 **decode 算子开关**（`-ngl 0`/`-ngl 8` 都正收益）。两边的收益方向与适用
条件不同，**不能共用一条规则**；Vulkan 侧完整数据见 §⑥。

共同判据：后端被加载 / 运算被接管，不等于算力被用上；只要本该在 CPU 上算的东西离开 CPU 本地路径，代价就是每 token
一次往返。两处修法相同——**让 CPU 层的算子留在 CPU 上**。

**部署含义**：这就是「独显上 head 段 offload 负收益」的完整归因——不是设备不行，是一个默认值。`split-stage` 与
`llama-server` 的 split 角色现在都默认关它（`QLH_SPLIT_DEFAULT_NO_OP_OFFLOAD`，显式 `--op-offload` 仍可覆盖）；同一个
head 在 probe 上是正收益、经 server 一度只剩 −10%。语义、显存花法与验收见 [`SEGMENT-OFFLOAD.md`](SEGMENT-OFFLOAD.md) §4。

## ⑥ 2026-11-10 复测：当前代码上的核显

**装置**：Iris Xe（`uma: 1`，16194 MiB / 15426 free），单卡隔离 `GGML_VK_VISIBLE_DEVICES=0`；构建三份：
`build-vulkan`（**571a9d0**，11-06）、`build-vulkan-new`（**19776b74e**，本轮现编，参数同 `build-vulkan`）、
`H:\llamacpp-clean\build-vulkan`（**e0f142d**，v2 原始构建）。9B Q4_K_M，`-t 8`，llama-bench `-r 3` + 热身 1 轮、
只取第二轮，llama-cli `-n 48 -st` 交叉验证；2B 同装置。

**基线先修**：`build-cpu` 是 `GGML_CPU_ALL_VARIANTS=OFF` 的通用变体，与带 Vulkan 的构建不同级。公平基线是同一构建
移走 `ggml-vulkan.dll`（`_bin_cpuonly`，实际加载 `ggml-cpu-haswell`）：

| 9B，llama-bench | pp32 | pp256 | tg64 |
|---|---:|---:|---:|
| 同构建 CPU-only（haswell） | **45.18** | **42.70** | **7.35** |
| `build-cpu`（通用变体） | 30.40 | 33.53 | 7.17 |
| 差 | −33% | −21% | −2.4% |

**① `--no-op-offload` 在 Vulkan 上有作用，但只在 `-ngl < 全部` 时**（19776b74e，单卡）：

| 配置 | pp32 | pp256 | tg64 |
|---|---:|---:|---:|
| `-ngl 0`（默认 `op_offload` 开） | 16.21 | 53.66 | 6.19 |
| `-ngl 0 --no-op-offload` | **28.43**（+75%） | 31.23（**−42%**） | **7.44**（+20%） |
| `-ngl 99` / `-ngl 99 --no-op-offload` | 30.94 / 30.82 | **80.98** / 80.36 | 7.00 / 7.19（噪声内） |

语义：`ggml-backend.cpp:1386` 的 `op_offload` 分支只在「权重落在 host/CPU buffer」时触发；Vulkan 判据是
`op_batch_size >= 32`（`GGML_OP_OFFLOAD_MIN_BATCH`），decode 的 MUL_MAT `ne[1]=1` 不触发、prefill `batch>=32` 才触发。
所以它是 prefill 搬运开关、不是单调正收益：pp32 关掉纯赚、pp256 关掉反而慢 42%。边界：只对「部分 offload」有效，
治不了全量 offload 的 decode。

**② 全量 offload 与中间值**：prefill 42.70 → 80.98（**1.90×**）；tg64 7.35 → 7.00（−4.8%），llama-cli 口径 7.0 → 7.2
（+2.9%）。中间值仍差：`-ngl 16` 22.12 / 65.92 / **5.92**（tg −19%）；`-ngl 0` 16.21 / 53.66 / **6.19**（−16%）；
`--no-host` 仍负优化（`-ngl 0` 13.38 / 27.35 / 5.95），`-ngl 99` 无影响。

**③ prefill 交叉点**：pp32 0.69×（输）→ pp48 1.28× → pp64 1.96× → pp96 2.05× → pp128 2.44× → pp256 1.90×，
交叉点 **pp≈32–48**（v2 记 32–64，方向一致）。

**④ 2B**（同构建 CPU-only 203.02 / 186.63 / 26.05）：`-ngl 99` 162.95 / 386.77 / 25.49（prefill **2.07×**、
decode −2.1%）；`-ngl 0` 62.18 / 173.33 / 28.68（decode +10%，未重复确认，存疑）；`-ngl 0 --no-op-offload`
128.32 / 161.49 / 28.92。v2 的 2B +9.2% / 2.5× 同样缩水。

**⑤ `-ngl 0` 的 −41% 为何复现不出来**：e0f142d 构建实测单卡 tg 6.5（−7%）、双设备 6.1（−13%）；571a9d0 / 19776b74e 是
6.78 / 6.19，都到不了 v2 记的 4.1。−41% 只在第一轮表里出现过（双设备口径），「第二轮单卡隔离复现了 −41%」本轮无法复现。