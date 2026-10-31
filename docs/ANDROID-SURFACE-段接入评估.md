# 弱 CPU + 16 GiB 设备能否作为推理链的一段

> 状态：**调研结论（未在目标设备实测）**，切点实验已实测
> 更新日期：2026-10-31
>
> 候选：Y700 安卓平板（骁龙 8 Gen 3，16 GiB）、Surface Pro 7（i5-1035G4，4C/8T，16 GiB）
> 基准：本机 head 0:32（RTX 4060 Laptop）+ ROG tail 32:40
>
> 适用范围：什么条件下该把第三台弱设备接进链、接进来要付多少成本、安卓侧的硬约束。
> 不含部署步骤——那部分等设备到手后再落到 `docs/DEPLOYMENT.md`。

---

## 0. 结论

**当前 35B 不需要第三段。** 它是串行链的一段，段耗时直接加进 TPOT，弱设备慢 3 倍就是慢 3 倍，
没有摊薄机制：

| 接法 | 每轮 | 吞吐 | 相对现状 |
| --- | ---: | ---: | ---: |
| 现状（ROG 跑 32:40，实测） | 236 ms | 16.7 t/s | 基线 |
| 弱设备替代 ROG 跑 8 层 | ~375 ms | ~10.7 t/s | **−36%** |
| 三段（本机 0:24 / 弱设备 24:32 / ROG 32:40） | ~442 ms | ~9.0 t/s | **−46%** |
| 弱设备跑 36:40（本机 0:36） | ~281 ms | ~14.2 t/s | **−15%** |

弱设备 8 层取 220 ms（190–260 ms 区间的中值，§2.2），4 层按半量取 110 ms。

**它变成必需品的条件是唯一的**：模型大到两个节点装不下，第三段是"容量的必要成本"
而不是"算力的增益"。当前 35B（14.79 GiB 权重）分两段已经贴边（ROG 可用只剩 0.82 GiB）；
换到 40–50 GiB 的 80B-A3B 级模型时，两节点必然装不下，那时才轮到它。

**若必须二选一：Y700 优于 Surface**，主因是可用内存（10–12 vs 8–9.5 GiB）和散热余量，
不是算力——两者算力基本打平（§4）。

---

## 1. 安卓侧怎么跑

### 1.1 三条路，只有两条能用

| 路径 | 官方状态 | 产物 | 能被 Python 调 |
| --- | --- | --- | --- |
| `examples/llama.android` + Android Studio | 支持，Kotlin binding | `.aar` + 示例 App | 否（JNI，不是 C ABI） |
| Termux 自编译 | 官方推荐，`apt install libandroid-spawn` | ELF + `.so` | **是** |
| NDK 交叉编译 | 支持，文档给了完整 cmake 命令行 | `libllama.so` + `libggml-*.so` | **是** |

现成 APK（PocketPal、ChatterUI、MLC Chat）全是 UI 应用，没有对外接口。`llama-server` 也不行：
本协议是 **hidden 进 / hidden 出**，server 是 token 进 / token 出，入向的
`llama_batch_ext_set_embd_state` 只在 shim 层存在。**必须自己编 .so + 自己的 TCP 服务。**

好消息是 `stage_worker.py` 只依赖 `argparse / os / socket / sys / pathlib / numpy`
加本地 `protocol.py`、`wire.py`、`engine.py`——纯 stdlib，零移植成本。

### 1.2 两条可行路径

**路径 A：Termux 内编译（推荐）**

```bash
pkg install clang cmake make git python && apt install libandroid-spawn
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
      -DGGML_LLAMAFILE=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON
cmake --build build -j6
clang -shared -fPIC -O3 qlh_keep_head.c -Iinclude -Iggml/include \
      -Lbuild/bin -lllama -lggml -o libqlh_keep_head.so
```

产物天然链 Termux 的 libc++，`ctypes.CDLL` 零摩擦；`GGML_NATIVE=ON` 自动检测 i8mm/dotprod。
耗时：编译 20–40 分钟（8 核平板，会发热降频），shim 移植 1–2 小时，联调 2–4 小时。
风险：编译峰值内存 2–3 GiB 可能 OOM；切后台会中断。

**路径 B：PC 上 NDK 交叉编译 + adb push** —— 多两个坑：
`-DANDROID_STL=c++_static`（否则 `libc++_shared.so` 与 Termux 的冲突）、
`-DGGML_OPENMP=OFF`（NDK 不支持 CMake 的 OpenMP 查找，`docs/android.md:74`）。
编译本身 10 分钟，其余 2–4 小时全在排 ABI/STL。

### 1.3 shim 的移植面只有 4 处

`qlh_keep_head_gpu.c` 里平台相关的代码：`#include <windows.h>`（:44）、
两处 `GetProcAddress`（:96/:98）、`GetModuleFileNameA`（:159）。

关键：`llama_set_embeddings_layer_inp` / `llama_get_embeddings_layer_inp` 在 fork 里是
`extern "C"`（`src/llama-context.cpp:4331`、`:4359`），**Linux/Android 直接链接就能解析**。
那两处 `GetProcAddress` 是为 MSVC 的 mangled 名准备的兼容层，条件编译掉即可。整片约 20 行预处理。

### 1.4 五个硬坑

1. **CPU variant 别指望 armv9.2。** 8 Gen 3 标称 Armv9.2-A，但高通这代没开 SVE/SVE2，
   实际生效的是 `android_armv8.6_1`（DOTPROD + FP16 + i8mm）。手动指定就是
   `-DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16`。
2. **KleidiAI 对本项目的量化无效。** 它覆盖 Q4_0/Q8_0 系；主力 Q6_K 走的是
   `ggml-cpu/arch/arm/repack.cpp` 的专职 kernel（`ggml_gemv_q6_K_8x4_q8_K`:1381、
   `_8x8_q8_K`:1570、`ggml_gemm_q6_K_8x4_q8_K`:4782），i16 加宽 + vdotq，不是标量。不需要它。
3. **模型必须放 Termux 私有目录**，不能放 `/sdcard`——FUSE 上 mmap 极慢甚至失败，
   而本方案靠 mmap 做按需换页。最容易踩、最难诊断。
4. **整模 14.79 GiB 要传到设备上**（`layer_range` 要求每节点持有完整文件）。
   `adb push` 约 100 MB/s → 2.5 分钟。
5. **Android LMK**：4.4 GiB 权重在 16 GiB 设备上安全，但长时间无交互会被
   `PhantomProcessKiller` 冻结，需要 wakelock。

---

## 2. 算力与带宽的现实预期

### 2.1 带宽

| 设备 | 内存规格 | 理论 | decode 有效 | 来源 |
| --- | --- | ---: | ---: | --- |
| ROG 9600X | DDR5 双通道 | ~90 GB/s | ~50–55 | 推算 |
| Surface Pro 7 | LPDDR4x-3733 双通道 | 59.7 GB/s | ~30–35 | notebookcheck |
| Y700 (8 Gen 3) | LPDDR5X-9600，64-bit | 76.8 GB/s | ~26–30 | 规格 + 论文推断 |

安卓侧硬锚点：COTS 移动设备测量（arXiv 2410.03613）里 Vivo Pad3 Pro（天玑 9300，16 GB /
可用 10 GB）跑 Llama-2-7B Q4 得 **8.22 tok/s**；8 Gen 3 吞吐约为其 80% → 约 6.6 tok/s。
7B Q4 权重 ≈ 4.1 GiB → 有效权重带宽 ≈ 26 GB/s。

同一篇给出了线程数的结论：**大核数即最优线程数，加小核会劣化**。
8 Gen 3 = 1×X4 + 3×A720@3.15 + 2×A720@2.96 + 2×A520 → 大核 6 个，所以 `-t 5` 或 `-t 6`，
实测扫 4/5/6。**不要绑核**。

### 2.2 每层耗时

ROG 的 76 ms 要拆开才可比。`README.md:173-175` 记的关键事实：

- 同一段**独占内存**时 30.85 ms
- worker 常驻后 53.55 ms
- 差额 **22.7 ms 是内存压力**（8 层 4.16 GiB 把 15.2 GiB 内存压到只剩 0.82 GiB 可用）
- 76 ms 是 4 行 token 的实测（专家路由在 4 行间不完全重合，约 1.4× 单行）

换到 16 GiB 设备，那 22.7 ms 消失，纯算力基准应该是 **53 ms**。再按 Amdahl 缩放
（4C Ice Lake-U 持续算力约为 9600X 的 1/6，带宽 0.6）：**190–260 ms**，标注为推测。

### 2.3 这段有没有价值

三条理由，按权重排：

1. **它是串行链的一段，不是并行资源。** 没有摊薄。
2. **它慢的幅度远大于它省下的东西。** 它唯一能"省"的是接过 ROG 的 8 层，而 ROG 那 8 层
   只要 76 ms；换一台内存宽裕的机器就能免费拿到这 22.7 ms，不必牺牲算力。
3. **跨机传输是常数税。** 8 KB hidden 走 Tailscale 要 24 ms（MTU 1280、6 个包、每包一 RTT）。
   加第三段就多一次。若 Y700 在同一个 LAN（WiFi 6，<2 ms）这笔税能省掉。

---

## 3. Surface Pro 7（i5-1035G4，4C/8T）

纯 4 核 + SMT（Ice Lake-U，4×1.1–3.7 GHz，15 W，无 E 核），不是 P/E 异构。

**耗时 190–260 ms**（推测）。比 ROG 的 76 ms 慢 2.5–3.4 倍。额外的悲观因素：i5 版是
**无风扇被动散热**，持续 decode 会稳定在 ~1.8–2.0 GHz 而非 3.7 GHz 睿频，实际可能到 280 ms。

**线程数用 `-t 4`，不是 `-t 8`。** 依据：上游已把 `--threads -1` 改成优先取物理核数
（PR #31209，配套 Issue #19110 与 PR #22460）。`MEMORY-LIMITS.md` §3.4 记的 24 倍惩罚是
P/E 异构 + 绑核造成的，纯 SMT 没有这个问题。worker 有 5–6 个活动线程，8 个逻辑核：

| `-t` | 总线程 | 判断 |
| ---: | ---: | --- |
| 4 | 6 | **推荐**，留 2 个给调度 |
| 6 | 8 | 可行，会互相挤 |
| 8 | 10 | 过载 |

**16 GiB 的实际可用量 8–9.5 GiB**（Windows 11 空闲占 4.5–5.5，Edge 再吃 1.5–2）。
8 层约 4.4 GiB，占 46–55%，够且余量大——这正是 §2.2 那 22.7 ms 免费消失的原因。

风险仍在：Windows 不因超内存拒绝分配，而是分页到 swap（`MEMORY-LIMITS.md` §1 的"黑屏"）。
`chain_client.py` 已有预检：`MEM_SAFETY = 0.75`，8 层要可用 > 5.9 GiB。启动前别绕过预检。

---

## 4. Surface 4 核 vs Y700 8 核

| 维度 | Surface Pro 7 (i5-1035G4) | Y700 (8 Gen 3) | 胜 |
| --- | --- | --- | --- |
| sustained 算力 | ~130 GFLOPs（AVX2，4×2.1 GHz） | ~150 GFLOPs（NEON，1X4+5A720） | Y700 略 |
| 有效带宽 | ~30–35 GB/s | ~26–30 GB/s | Surface 略 |
| 8 层预测耗时 | 190–260 ms | 190–260 ms | 打平 |
| 散热 | **无风扇被动**，必然降频 | 平板机身余量更好（待实测） | Y700（推测） |
| 可用内存 | 8–9.5 GiB | **10–12 GiB** | **Y700** |
| 能装几层 | 8–12 层 | 8–14 层 | Y700 略 |
| Q6_K kernel | AVX2 + `vpmaddubsw` | ARM repack `q6_K_8x8` | 打平 |
| 故障模式 | swap 静默分页 → 黑屏 | LMK 杀进程 + 后台冻结 | Surface（可观测） |
| 接入手工 | 低（纯 Windows） | 中（Termux + shim 移植） | **Surface** |

性能打平，Y700 胜在内存与散热，Surface 胜在接入成本。**接入成本那点差距是一次性的，
内存和散热是持续的**——真要补一个节点，选 Y700。

---

## 5. 切点 36:40：已实测，不采纳

`32:40` 的问题是 ROG 那 76 ms 里有 22.7 ms 来自内存压力。一个不需要新设备的改法是让 ROG 只背
4 层、本机背 36 层。**已实测**（同一 prompt、同一链路、98% 接受率）：

| 切点 | 本机段 | ROG 段 | 吞吐 | 相对 |
| --- | --- | --- | ---: | ---: |
| **32:40（现行）** | 0:32 | 32:40（8 层） | **17.0 t/s** | 基线 |
| 36:40 | 0:36 | 36:40（4 层） | **17.7 t/s** | **+4%** |

预测的 +17% 没有实现（本文件的上一版按"ROG 4 层无内存压力省 22.7 ms、
本机多 4 层 +16 ms"推算出 19.6 t/s）。未做分项归因，只知道总量只省了约 13 ms。

**不采纳。** 4% 小于把 4 层负载从台式机挪到笔记本的代价：笔记本是日常使用的机器，
head 段从 32 层涨到 36 层意味着更长的持续满载、更高温度与风扇噪音，而 ROG 是专职节点。
`32:40` 保留为默认，`topology-cross-36.toml` 留作对照。

---

## 6. 证据等级

**已证实（源码/官方文档/实测）**

- `docs/android.md` 三条路都存在，Termux 与 NDK 命令是文档原文
- `examples/llama.android` 是 Kotlin/JNI binding，不是 C ABI
- `llama_set/get_embeddings_layer_inp` 在 `src/llama-context.cpp:4331/4359` 是 `extern "C"`
- shim 平台相关代码 4 处、约 20 行
- `stage_worker.py` 依赖纯 stdlib
- android CPU variant 7 个，`android_armv8.6_1` = DOTPROD+FP16+i8mm；armv9.x 需 SVE/SME
- `ggml-cpu/arch/arm/repack.cpp` 有 Q6_K 专职 GEMV/GEMM kernel（:1381/:1570/:4782）
- 上游 `-t -1` 已改为物理核优先（PR #31209 / Issue #19110 / PR #22460）
- ROG 8 层 76 ms 中 22.7 ms 来自内存压力（`README.md:173-175`）
- Surface Pro 7 i5-1035G4 = 4c/8t，LPDDR4x-3733 双通道（notebookcheck）
- 8 Gen 3 LPDDR5X 理论 76.8 GB/s；天玑 9300 平板 7B Q4 decode 8.22 tok/s
- 移动端最优线程数 = 大核数（arXiv 2410.03613）
- **切点 36:40 = 17.7 t/s，32:40 = 17.0 t/s（§5）**

**推测（未实测，标注了区间）**

- Surface / Y700 跑 8 层 4 行 = 190–260 ms
- Surface 无风扇导致持续负载再慢 20–30%
- Android 上 `/sdcard` 的 mmap 比私有目录慢（社区共识，未在 Y700 上验证）
- 跨机 24 ms 传输在网络条件变化时的分布（数据来自 Tailscale 特定路径）

---

## 7. 什么时候开始做，先做什么

1. **先不改拓扑。** §5 已证伪"换切点"这条路，`32:40` 的 22.7 ms 是这台机器上的固有代价。
2. **等模型换到 80B-A3B 级**（40–50 GiB 量级），两节点装不下时再走安卓路径。
   那时按 本机 0:20 / ROG 20:32 / 弱设备 32:40 划分，弱设备是必需容量。
3. **动手时的顺序**：Termux 内编译（预算 1 天 = 编译 40 分钟 + shim 移植 2 小时 + 联调 4 小时）
   → 模型放 Termux 私有目录 → `-t 5`、不绑核、开 wakelock →
   先只跑 4 层验证数值（`checks/` 那套逐位判据）→ 再加层。
4. **Surface 放在 Y700 之后**：同一套 shim 改动，Windows 侧没有新平台成本，但内存与散热都更差。
