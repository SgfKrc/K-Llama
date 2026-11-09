# 核显能不能用：两轮实测的结论与修正

> 状态：**现行**
> 更新日期：2026-11-09
> 适用范围：核显为什么慢（`-ngl` 扫描、prefill 交叉点、host buffer 抢占），以及 CUDA 侧同类的
> `op_offload` 默认值问题。核显数据全部来自 v2（2026-11-06 前后）两轮 `llama-cli` 实测；
> 2026-11-09 未重测核显，本轮只补 §⑤。段级 offload 的语义与显存花法归
> [`SEGMENT-OFFLOAD.md`](SEGMENT-OFFLOAD.md)。

**结论**：第一轮「核显在任何配置下都不如纯 CPU」是测量口径造成的错误结论。第二轮把 Iris Xe
单卡隔离（`GGML_VK_VISIBLE_DEVICES=0`）后，9B/2B 上**全量 offload 是赢的**；中间值与 `-ngl 0` 仍然一律更差，因为只要 GPU
后端被加载，未 offload 的层就进后端 host buffer。

| 指标（9B Q4_K_M，`-t 8`） | 纯 CPU | Iris `-ngl 99` | 比 |
|---|---:|---:|---:|
| decode（tg64） | 6.72 | **7.08** | **+5.4%** |
| prefill（pp256） | 29.20 | **90.80** | **3.1×** |

第二轮的完整数据：2B decode **+9.2%**、prefill **2.5×**；0.5B decode 持平、prefill **7.0×**。 prefill 的交叉点在
**pp≈32–64**（pp32 是唯一输的点，0.93×），再长是 2.6–3.2×。

**第一轮为什么测反了**：① 默认双设备枚举（Iris + 4060）走 `multi_device` 慢路径；② `llama-cli` 的短 prompt（~50
token）正好落在交叉点左侧；③ `build-cpu` 与 `build-vulkan` 的 CPU 后端不同级（`GGML_CPU_ALL_VARIANTS`）。

**中间值仍然更差**：`-ngl 0`（加载了后端却一层不上）比纯 CPU 慢 41%，各模型的中间层数也都低于全量，原因是 host buffer
抢占。`571a9d0` 已包含 `#31538` 的修复（`src/llama-model-loader.cpp` 的 `(use_mmap || device_type == IGPU)` 分支），「未
offload 层不进 host buffer」在上游层面已处理，试图在 `make_cpu_buft_list` 再改一次是负优化（2B `-ngl 0` tg
30.53→22.18、4060 `-ngl 12` pp 1226→263，已回滚）。

**35B 装不进**：20.82 GiB > Iris 共享堆 ~15.4 GiB，部分 offload 会落进 OOM 路径。**方法学**： `llama-bench` 的 `-ngl`
不要一次传多个值（多值模式与单值能差 3 倍）；测量时避免高负载，同一配置重复测能差 25%。

三个问题：① v2 解释过，结论齐全，只是分散在 5 份文档里；② 中间值慢的主因是 host buffer 抢占； ③ 第一轮本机 9B dense 的
gen 中位：纯 CPU **6.9 t/s** → `-ngl 0` **4.1**（−41%）→ `-ngl 16` **5.8** → `-ngl 99` **6.7**（双设备口径；单卡隔离后是
7.08，见上表）。

## ① v2 到底有没有解释过：有

### 1.1 最直接的一条：Vulkan 加载即污染 CPU 层

`slimarc/prototype:docs/ANDROID-SURFACE-段接入评估.md` 的 90–99 行记 Surface 单机 8 层 `112 ms`（14 ms/层）与「Vulkan
未禁用时」的对照，正文结论是：**Vulkan 后端只要被加载，即使 `-ngl 0`，CPU 侧权重也会进 Vulkan host
buffer**（`make_cpu_buft_list` 把 host buft 插在 CPU buft 之前）。禁掉 `ggml-vulkan.dll` 后 Surface 从 245 ms 降到 112
ms，页缓存热起来后 24 层只要 90 ms；表里 112 ms 那行标成「Vulkan 未禁用时」是笔误，按正文读法是未禁用 245 ms、禁用 112
ms。 [`PROGRESS.md`](PROGRESS.md) §1.1 复述了同一结论。

### 1.2 弱设备侧的并列结论（不含本机 Iris Xe）

同一份评估的第四节：Iris Plus **比 CPU 慢 30%**、Adreno 750 **慢 5–25%**，都不可用。Surface 关掉 Vulkan 后 head 245 →
112 ms；Y700 上 `-ngl 99` 是 29.8 t/s，纯 CPU 37.6 t/s。原文的归因是 **「共享同一块
LPDDR，把层搬到另一组执行单元只增加一次拷贝，不增加带宽」**——这正是「带宽征用」
疑问的答案：没有增量带宽，只有增量拷贝。只适用于弱设备；本机 Iris Xe 单卡隔离后全量 offload 是赢的，两类设备各自实测。

### 1.3 核显的算子级实测（唯一直接测核显算力的一条）

[`DEPLOYMENT.md`](DEPLOYMENT.md) §6.8 的三条：

- IQ4_XS 在**所有后端都只有 MMV（向量乘），没有 MMQ（矩阵乘）**，Vulkan 与 SYCL 一样，OpenCL
  与 OpenVINO 干脆不支持。所以 `CPU_REPACK` 全部回退。
- head 的 15.86 GiB 里 8.37 是 IQ4_XS，塞进核显 15.82 GiB 共享堆会连系统一起挤死；tail 全是
  Q5_K/Q6_K/Q8_0，四种后端全覆盖，4.88 GiB 能整段常驻。核显只可能救 tail。
- 本机 Iris Xe，ctypes 直调 OpenCL：流式读 780 MiB **13.1 ms = 58.2 GB/s**（CPU 6 线程
  23.7 GB/s）；`lm_head` 形状 GEMV **20.5 ms = 23.6 GB/s**（朴素/协作/向量化三版都在 22.6–24.2）。
  两者差 2.5 倍——**低算术强度的 GEMV 只能吃到约四成 DRAM 带宽**，这条路上没有数量级。

`--list-devices` 报 16194 MiB，正是那 15.82 GiB 共享堆，说明这组数据就在本机测的。

### 1.4 offload 粒度的实测表

| 位置 | 内容 |
|---|---|
| [`MEASUREMENTS.md`](MEASUREMENTS.md) §3.1 | 专家上卡 11 层 **30.45** vs 整层上卡 11 层 **10.11**（3.0×）：专家占权重 88%，每 token 只用 8/256 |
| [`MEASUREMENTS.md`](MEASUREMENTS.md) §3 | `-ncmoe` 扫描：0→25.94、8→28.99、10→32.01、12→**17.31**、14→**11.23**、16→**9.75** |
| [`DESIGN.md`](DESIGN.md) | 11 层 30.45 最优，12 层 **15.29** 断崖；WDDM 超显存不报错，静默用系统内存，只表现为掉速 |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) §6.5 | `gpu_layers` 4/5/6/7 → 7.2 / 7.9 / **8.7** / 8.1；7 层 stage 0 更快（346.1）但 tg 变慢 |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) §6.6 | 32/32 陷阱：stage 0 最快（135.4）但 tg 更慢，原因是**内存**换页，不是显存 |

v2 有「开 N 层后 tok/s 下降」的实测表，而且不止一张；缺的只是把它们归因成一条机制说明。 `-ngl` 与 `-ncmoe`
两条正交路径的定义见 [`ARCHITECTURE.md`](ARCHITECTURE.md)，WDDM 超额分配静默、照旧打印 `offloaded N/N` 见 `README.md`。

## ② 机制分析：哪几条站得住

**A. host buffer 抢占是「开核显反而慢」的第一因，与算力无关。** 代码在 `H:\llamacpp-c\src\llama-model.cpp` 的
`make_cpu_buft_list()`：顺序是 `CPU: ACCEL -> GPU host -> CPU extra -> CPU`，只要
`!no_host`，`ggml_backend_dev_host_buffer_type()` 拿到的 host buft 就排在 CPU buft **之前**（`emplace_back` 后
`break`）；`--no-host` 是开关（`common/arg.cpp` 的 `--no-host` 选项、`common/common.h` 的 `no_host` 字段）。后果：**只要
`ggml-vulkan.dll` 被加载， `-ngl 0` 的「纯 CPU」就不存在了**，所有层都进 host buffer。

**B. GPU 接管大算子，代价是往返拷贝（上游维护者确认，实测印证）。** llama.cpp issue `#31538` 里 Vulkan 后端维护者
`0cc4m` 的原话是：用 GPU host buffer 是刻意的，因为**GPU 仍会接管大算子**，而且**从 host buffer
拷贝快得多**。实测两个方向都不如纯 CPU（第一轮口径，第二轮没有重测 `--no-host`）：9B `-ngl 16` 5.8 → `--no-host`
**3.2**（GPU 要从 CPU buffer 跨总线拷）； `-ngl 0` 4.1 → `--no-host` 5.3（好一点，仍到不了纯 CPU 的 6.9）。host buffer
是「GPU 可直接读」，代价是 CPU 层失去本地性；关掉它 GPU 就得跨总线拷。

**C. 显存溢出静默（独显场景）**：WDDM 不拒绝超额分配，只掉速。12 层专家 33→17，14 层→11
（[`DESIGN.md`](DESIGN.md)、[`MEASUREMENTS.md`](MEASUREMENTS.md)、`README.md` 口径相同）。
**这条在核显上不成立**：核显的「显存」就是系统内存，没有独立显存可撑爆；核显的问题是 `#31538` 描述的单 heap `budgeted
heap` 分配，报 `ErrorOutOfDeviceMemory`，错误指向 kv cache——该 OOM 已被修掉，所以它不是「慢」的原因。

**D. 量化类型缺 MMQ**（prefill 慢的来源）：IQ4_XS 在所有后端只有 MMV 没有 MMQ；v2 的 35B 权重 8.37 GiB 是 IQ4_XS。这解释
prefill 上不去，不解释 decode 的损失。

**E. 共享内存总线是 decode 的上限。** 本机 DDR5-4800 双通道，理论 76.8 GB/s；9B Q4_K_M （5.68 GB）decode 每 token
要过一遍全部权重：

| 配置 | gen | 推算有效带宽 |
|---|---:|---:|
| 纯 CPU | 6.9 t/s | 41.5 GB/s（54% 理论） |
| 核显全量 | 6.7 | 38.1 |
| 核显部分 | 5.8 | 32.9 |
| 核显 `-ngl 0` | 4.1 | 23.3 |

核显与 CPU 吃同一根总线，核显能拿到的上限就是 CPU 的上限；搬到另一个执行单元拿不到增量带宽。

| 说法 | 判定 | 依据 |
|---|---|---|
| 「老核显自己算力太菜」 | 纸面上不成立 | Iris Xe 96EU 理论 ~2.3 TFLOPS fp32（96 EU × 8 lane × 2 × 1.5 GHz）；i9-13900H `-t 8` 约 1.0–1.1 TFLOPS，14 核全开约 1.6。核显纸面算力更高而实测更慢，瓶颈在带宽与拷贝。*TFLOPS 是推算，未实测* |
| 「带宽被核显征用」 | 不成立 | 本身就是一条总线；offload 不会让 CPU 少掉带宽，只让被 offload 的层多一次往返 |
| 「撑爆共享显存导致换页」 | 核显上不成立 | 撑爆的是单 heap 的 budgeted heap（`#31538`），且已修；修完仍慢 |

**推测（未验证）**：① 2B gen +9.2% 高于 9B +5.4%，可能因为 2B 权重 1.31 GB 更容易命中 L3/ 页缓存，9B 5.68 GB
更接近带宽受限；② pp32→pp256 的曲线翻转没做 perf 对比；③ `#31538` 的修复是否只改了分配位置（本机 32 GiB 内存 + 16 GiB
共享堆装 5.68 GB 不触发 OOM，无法复现）。

## ③ 第一轮实测：现成的 Vulkan 构建，口径有误

`H:\llamacpp-c` 下现成四份构建，没编译任何东西：

| 构建 | 后端 | 时间戳 | 用途 |
|---|---|---|---|
| `build-vulkan` | `GGML_VULKAN=ON` | **11-02 21:09**，commit `e0f142d` | 用户提 issue 当天编译的那份 |
| `build-cpu` | 全 OFF | 同 commit | 真正的纯 CPU 基线 |
| `build-cuda` | `GGML_CUDA=ON` | 10-30 | 4060 对照 |
| `build-split` | 全 OFF（CUDA/VULKAN/SYCL/HIP/OPENCL/OPENVINO/BLAS） | — | v2.1 的纯 CPU 构建，跑不了 GPU |

v2 时代的 GPU 走另一份资产：`H:\slimarc-proto\shim\bin\ggml-cuda.dll`（54 MB，10-28）+ `cublas64_13.dll` /
`cublasLt64_13.dll`，配 `build_cuda_shim.ps1` 编的 shim（`DEPLOYMENT.md` §1： shim CPU 版约 8.1 MB、CUDA 版约 553
MB）。v2 的 GPU 实验全是 CUDA（RTX 4060），Vulkan/核显是独立对照线，与 `build-split` 无关。

数据（`llama-cli`，`-t 8`，`-n 48`，每组 3 次取中位）。**这是第一轮口径，标签不可按字面读**：当时没设
`GGML_VK_VISIBLE_DEVICES`，Vulkan 默认枚举 Iris Xe + RTX 4060，走 `multi_device` 慢路径。仍然有效的是「`-ngl 0`
与中间值都比纯 CPU 差」（第二轮复现了 −41%）与部分 offload 在独显上也要付一半代价。

**Qwen3.5-9B Q4_K_M（32 层，5.68 GB，dense）**

| 配置 | Prompt t/s | Gen t/s | Gen 相对纯 CPU |
|---|---:|---:|---:|
| `build-cpu`，`-ngl 0`（不加载 GPU 后端） | 21.4 | **6.9** | 1.00 |
| `build-vulkan`，`-ngl 99`（双设备） | 12.6 | 6.7 | −3% |
| `build-vulkan`，`-ngl 16` | 15.1 | 5.8 | −16% |
| `build-vulkan`，`-ngl 0`（只加载 Vulkan） | 7.9 | **4.1** | **−41%** |
| `build-vulkan`，`-ngl 0 --no-host` | 14.5 | 5.3 | −23% |
| `build-vulkan`，`-ngl 16 --no-host` | 8.5 | 3.2 | −54% |
| `build-cuda`，RTX 4060 `-ngl 99` | 117.8 | 29.3 | **4.2×** |

**Qwen3.5-2B Q4_K_M（25 层，1.31 GB）**

| 配置 | Prompt t/s | Gen t/s |
|---|---:|---:|
| `build-cpu`，`-ngl 0` | 105.3 | 24.6 |
| `build-vulkan`，`-ngl 0` | 74.7 | 26.2 |
| `build-vulkan`，`-ngl 0 --no-host` | 92.8 | 24.5 |
| `build-vulkan`，核显 `-ngl 12` | 59.0 | 23.5 |
| `build-vulkan`，核显 `-ngl 12 --no-host` | 60.9 | 16.0 |
| `build-vulkan`，核显 `-ngl 99` | 55.2 | 26.5 |
| `build-vulkan`，4060 `-ngl 12` | 87.7 | 39.3 |
| `build-vulkan`，4060 `-ngl 12 --no-host` | 120.6 | 39.6 |
| `build-vulkan`，4060 `-ngl 99` | 199.1 | 74.3 |
| `build-cuda`，4060 `-ngl 99` | 226.1 | 74.8 |

读数：

1. 9B 上 `-ngl 0` 与中间值都输，最差的是 `-ngl 0`（−41%）：**`-ngl` 越接近 0 越差**，因为
   host buffer 里的层越多。`-ngl 99` 第一轮 −3%，单卡隔离后 **+5.4%**。
2. 部分 offload 在独显上也要付一半代价：2B 上 4060 `-ngl 12` 39.3 vs `-ngl 99` 74.3（−47%）；
   「不要用中间值」对独显同样成立，绝对值仍是正收益。
3. prefill 两轮相反：第一轮 9B 核显 12.6 / 15.1 vs CPU 21.4 像绝对短板；第二轮 pp256 是
   90.80 vs 29.20（3.1×），只有 pp32 输（0.93×），交叉点 pp≈32–64。
4. 2B 核显 gen 第一轮 +7%、第二轮 +9.2%，方向一致；9B 第二轮 +5.4%——**装不进共享堆才是硬
   边界**（35B 20.82 GiB vs ~15.4 GiB），「优势随模型变大立刻消失」不成立。
5. Vulkan 首次运行要编译 pipeline，第一次采样能差一个量级（第一轮 4060 `-ngl 99` 只有 9.5 t/s，
   热身两轮后 168.4），与 [`MEASUREMENT-PITFALLS.md`](MEASUREMENT-PITFALLS.md) 的「同一个段能差
   19 倍」同类。

**35B 没测**：35B-A3B（20.82 GiB）CPU 上约 1 t/s，48 token 要 40 秒以上，且 15.82 GiB 共享堆装不下，会落进 `#31538` 的
OOM 路径而不是「慢」；2B/9B 已把机制分离干净。

## ④ 部署规则

**不要用部分 offload，也不要在纯 CPU 机器上留着 GPU 后端。** 只要 `ggml-vulkan` / `ggml-cuda` 被加载，未 offload
的层（含 `-ngl 0`）就进后端 host buffer，GPU 仍接管大算子——等于给这些层加一次往返拷贝，实测 9B dense 生成从 6.9 t/s
掉到 4.1（`-ngl 0`）/ 5.8（部分）。**要纯 CPU 就别链接 GPU 后端；要 GPU 就全量
offload**（全量是正收益，数字见上）。另有单 heap iGPU 上 host buffer
从显存池分配的问题（`#31538`，上游称「没有简单的解」），`571a9d0` 已含修复。

- **「全 CPU」= 不加载 GPU 后端**：`-ngl 0` 与前者差 41%。**核显全 offload 在装得进共享堆的
  模型上值得**（9B/2B 见上），硬边界是体积。**中间态是负的**：`-ngl 0` / 部分 offload 第二轮仍
  输（−41% / −16%），「全量或不用」不变。本项目选纯 CPU 的两条理由：35B（20.82 GiB）装不进
  ~15.4 GiB 共享堆；核显与 CPU 共享同一条内存总线，中间态必然多一次往返。

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
| 触发条件 | 加载 `ggml-vulkan.dll` 即发生，`-ngl 0` 也躲不掉 | 构建默认开，`-ngl 0` 也躲不掉 |
| 机制 | `make_cpu_buft_list` 把后端 host buft 排在 CPU buft 之前，未 offload 的层权重搬进 host buffer | 权重留在 `CUDA_Host`（pinned），算子被搬上 GPU 执行 |
| 代价 | `-ngl 0` 比纯 CPU 慢 41% | `-ngl 0` 比纯 CPU 慢 31% |
| 摆正方式 | 要纯 CPU 就别加载后端（不是 `-ngl 0`） | `--no-op-offload`：CPU 层的算子回自己的 kernel |
| 错误方向 | `--no-host`：GPU 从 CPU buffer 跨总线拷 | `--no-host`：跨 PCIe 读 pageable 内存 |

共同判据：后端被加载 / 运算被接管，不等于算力被用上；只要本该在 CPU 上算的东西离开 CPU 本地路径，代价就是每 token
一次往返。两处修法相同——**让 CPU 层的算子留在 CPU 上**。

**部署含义**：这就是「独显上 head 段 offload 负收益」的完整归因——不是设备不行，是一个默认值。 `split-stage` 与
`llama-server` 的 split 角色现在都默认关它（`QLH_SPLIT_DEFAULT_NO_OP_OFFLOAD`，显式 `--op-offload` 仍可覆盖）；同一个
head 在 probe 上是正收益、经 server 一度只剩 −10%。语义、显存花法与验收见 [`SEGMENT-OFFLOAD.md`](SEGMENT-OFFLOAD.md) §4。
