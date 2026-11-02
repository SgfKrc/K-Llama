# 弱 CPU + 16 GiB 设备跑 35B-A3B 的一段

> 状态：**端到端已跑通**（2026-11-03 凌晨实测）
> 候选：Y700 安卓平板（骁龙 8 Gen 3，16 GiB）、Surface Pro 7（i5-1035G4，4C/8T，16 GiB）
> 基准：主机链（本机 head 0:32 + ROG tail 32:40）
>
> 适用范围：什么条件下该把弱设备接进链、接进来要付多少成本、安卓侧的硬约束。

---

## 0. 结论

**弱设备之间自己组一条链，实测 6.50 t/s**（Qwen3.6-35B-A3B Q6_K，21.86 GiB / 35.51 B）：

| 方案 | 每轮 | 吞吐 | 说明 |
| --- | ---: | ---: | --- |
| 主机链（本机 0:32 + ROG 32:40） | 236 ms | 16.7 t/s | 占着本机 |
| **纯弱设备链（Surface 0:24 + Y700 24:40）** | **153 ms** | **6.50 t/s** | **不占本机** |
| 单台 Surface 跑整模 | 363 ms | 2.75 t/s | 抖动区（22 GiB > 16 GiB） |

- 比主机链慢 2.6 倍，但**换出整台主机**；
- 比单台弱设备快 **2.4 倍**——单台 16 GiB 装不下整模，这条路本来跑不起来；
- 两端都是**整模 + `LLAMA_LAYER_RANGE` 选择性加载**，没有裁层工件。

**两端算力在 MoE 上是接近的**（§4）：Surface 3.76 ms/层 vs Y700 2.84 ms/层，1.3 倍。
早先"Y700 快 2.6 倍"的说法来自 **9B dense 工件**——那是带宽受限场景（Y700 45 GB/s
对 Surface 17.6 GB/s），不能外推到 MoE。

**它变成必需品的条件**：模型大到两个节点装不下。当前 35B 分两段已经贴边；换到 40–50 GiB
的 80B-A3B 级模型时两节点必然装不下，那时弱设备是容量的一环而不只是"换出主机"。

---

## 1. 安卓侧怎么跑（已验证）

### 1.1 平台选择：Termux，不是 NDK

| 形态 | 产物 | 接线 |
| --- | --- | --- |
| Termux | ELF，链 Termux 的 `libc++_shared.so` | Python `ctypes.CDLL` 直接加载，零摩擦 |
| NDK 交叉编译 | ELF，链 bionic 系统 libc++ | 要 JNI 或自己 `dlopen`；`llama.android` 是 Kotlin binding，不是 C ABI |

**Termux 版实测可跑**：cmake 配置、编译、python ctypes 加载、worker 起监听全部通过。
`GGML_NATIVE=ON` 自动检测到 `dotprod + i8mm`（`-mcpu=native+dotprod+i8mm+nosve+nosme`），
正是 Q6_K MoE GEMV 需要的两条指令。NDK 形态没有实测，保留为后备。

### 1.2 shim 的移植面：实测 3 处

`qlh_keep_head_gpu.c` 里平台相关的只有：

| 位置 | Windows | POSIX |
| --- | --- | --- |
| `qlh_resolve_layer_inp` | `GetModuleHandleA` + `GetProcAddress`（MSVC mangled 名） | `dlsym(RTLD_DEFAULT, "_Z30...")`（Itanium mangled 名） |
| `qlh_ensure_backends_loaded` | `GetModuleHandleExA` + `GetModuleFileNameA` | `dladdr()` 取本 .so 路径 |
| 三处 `_putenv_s` | `<stdlib.h>` | `setenv` / `unsetenv` |

**那两处 `GetProcAddress` 不能条件编译掉。** `llama_set/get_embeddings_layer_inp` 声明在
`src/llama-ext.h:165/169`，没有 `extern "C"`，`LLAMA_API` 也只展开成 visibility 属性。
所以两个符号在 Linux 上同样是 Itanium mangled 名，必须 `dlsym` 解析。

一分钟验证：`nm -D libllama.so | grep layer_inp`，看到 `_Z30...` 就是 mangled。

编译产物 40 KB，`clang -shared -O2 -fPIC -Wl,-rpath,'$ORIGIN'` 即可。`$ORIGIN` 不能省——
否则 `libllama.so.0` 找不到，报 `dlopen failed: library "libllama.so.0" not found`。

### 1.3 五个硬坑（全部实测确认）

1. **CPU variant 别指望 armv9.2。** 8 Gen 3 标称 Armv9.2-A 但没开 SVE/SME，实际生效
   `android_armv8.6_1`（DOTPROD + FP16 + i8mm）。手动指定是
   `-DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16`。
2. **KleidiAI 对本项目的量化无效。** 它覆盖 Q4_0/Q8_0 系；Q6_K 走
   `ggml-cpu/arch/arm/repack.cpp` 的专职 kernel（`ggml_gemv_q6_K_8x4_q8_K`:1381 等）。
3. **模型必须放 Termux 私有目录**，不能放 `/sdcard`——FUSE 上 mmap 极慢甚至失败。
   实测补充：`run-as com.termux` 启动的进程连 `/sdcard` 和 `/data/local/tmp` 都读不了
   （SELinux 拒绝），只有 Termux 自己的进程（sshd 等）有存储权限。
4. **整模 21.86 GiB 要传到设备上**。实测 `adb push` 到 `/sdcard` 后 run-as 读不了，
   最终走 `adb forward tcp:18022 tcp:8022` + `scp`，**40 MB/s**（Tailscale 直传只有 0.57 MB/s）。
5. **Android LMK**：编译时 `-j 6` 会被 SIGKILL，必须 `-j 1`；推理时 20 层（11 GiB）会 OOM，
   16 层（8.8 GiB）稳定。长时间无交互需要 wakelock。

---

## 2. 实测耗时

### 2.1 每层耗时

| 设备/平台 | 层数 | 耗时 | 每层 | 条件 |
| --- | ---: | ---: | ---: | --- |
| Y700（Android） | 16 | 45.5 ms | **2.84 ms** | tail，含 output 投影 |
| Surface（Windows） | 24 | 90.3 ms | **3.76 ms** | head，含 token embedding |
| Y700 单机 8 层 | 8 | 22.7 ms | 2.84 ms | worker 口径，热态 |
| Surface 单机 8 层 | 8 | 112 ms | 14 ms | **Vulkan 未禁用时** |

最后一行是关键对照：**Vulkan 后端只要被加载，即使 `-ngl 0`，CPU 侧权重也会进 Vulkan host
buffer**（`make_cpu_buft_list` 把 host buft 插在 CPU buft 之前）。禁掉 `ggml-vulkan.dll`
后 Surface 从 245 ms 降到 112 ms，再算上页缓存热起来，24 层只要 90 ms。

### 2.2 带宽（MoE 不等价于带宽受限）

| 设备 | 内存规格 | 理论 | dense 实测有效 | 来源 |
| --- | --- | ---: | ---: | --- |
| Surface Pro 7 | LPDDR4x-3733 双通道 | 59.7 GB/s | ~17.6 | 实测 |
| Y700 (8 Gen 3) | LPDDR5X-9600，64-bit | 76.8 GB/s | ~45 | 实测 |

dense 9B Q4_K_M 全模型（5.37 GiB）实测：Y700 7.82 t/s（45 GB/s 打满），这就是两者在被带宽
掐住时差 2.6 倍的原因。MoE 每 token 只激活一小撮专家，不落在带宽墙上，所以差距缩到 1.3 倍。

安卓侧锚点：COTS 测量（arXiv 2410.03613）里天玑 9300 平板跑 Llama-2-7B Q4 得 8.22 tok/s，
8 Gen 3 约为其 80%。

**线程数取大核数**：8 Gen 3 = 1×X4 + 3×A720@3.15 + 2×A720@2.96 + 2×A520。实测
`-t 5` 最好（37.6 t/s），`-t 8` 反而退化到 22.6（线程落到 A520），`-t 6` 抖动大。**不要绑核**。

### 2.3 长 prompt 的 prefill

同一条链路（Surface 0:24 + Y700 24:40），每轮先 RESET 再跑，prompt 用占位 token：

| prompt | head | tail | 合计 | prefill 吞吐 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 811 ms | 372 ms | 1183 ms | 13.5 tok/s |
| 64 | 2211 | 1110 | 3321 | 19.3 |
| 128 | 3941 | 1952 | 5894 | 21.7 |
| 256 | 7574 | 4087 | 11660 | **22.0** |

稳态 **22 tok/s**，约是 decode（6.5 t/s）的 3.4 倍——批量并行确实有效。另有约 **750 ms**
的首轮固定开销（图 reserve + 预热），N=16 那一行里占比过半。head 段稳定占 **65%**，
和它 24 层的份额吻合。

**体感**：一个 256 token 的 prompt 要 **11.7 秒**才出第一个 token。慢但可用；
真要优化，方向是让 Y700 多扛几层（它单机 prefill 比 Surface 快），受限于 16 层的热内存边界。

---

## 3. Surface Pro 7（i5-1035G4，4C/8T）

纯 4 核 + SMT（Ice Lake-U，4×1.1–3.7 GHz，15 W，无风扇被动散热）。

**`-t 6` 最优**（24 层 90.3 ms）。上游已把 `--threads -1` 改成优先取物理核数
（PR #31209），纯 SMT 没有 P/E 异构的绑核惩罚。

**16 GiB 的实际可用量 8–9.5 GiB**（Windows 11 空闲占 4.5–5.5）。整模 21.86 GiB 靠 mmap
选择性加载后，head 24 层约 12.6 GiB 权重——**贴边**，这是 §2.1 里页缓存必须热的根本原因。

风险：Windows 不因超内存拒绝分配，而是分页到 swap（`MEMORY-LIMITS.md` §1 的"黑屏"）。
`chain_client.py` 已有预检：`MEM_SAFETY = 0.75`。

---

## 4. Surface vs Y700（实测，不是推测）

| 维度 | Surface Pro 7 | Y700 (8 Gen 3) | 胜 |
| --- | --- | --- | --- |
| MoE 每层 | 3.76 ms | **2.84 ms** | Y700 1.3× |
| dense 有效带宽 | ~17.6 GB/s | **~45 GB/s** | Y700 2.6× |
| 可用内存 | 8–9.5 GiB | **11.95 GiB** | Y700 |
| 能装几层 | 24 层（贴边） | 16 层（安全） | Surface 略 |
| 核显 | Iris Plus，**比 CPU 慢 30%** | Adreno 750，**比 CPU 慢 5–25%** | 都不可用 |
| 散热 | 无风扇，持续降频 | 平板机身，编译时需 `-j 1` | — |
| 故障模式 | swap 静默分页 → 黑屏 | LMK 杀进程（SIGKILL） | Surface（可观测） |
| 接入手工 | 低（纯 Windows） | 中（Termux + shim 移植 + adb） | **Surface** |

**核显都不值得用**。Surface 关掉 Vulkan 后 head 从 245 ms 降到 112 ms；Y700 上
`-ngl 99` 是 29.8 t/s，纯 CPU 37.6 t/s。两者共享同一块 LPDDR，把层搬到另一组执行单元
只增加一次拷贝，不增加带宽。Adreno 750 的 Vulkan 路径要额外绕：Termux 自带的
`libvulkan.so` 找不到 driver（`vkCreateInstance` 返回 -9），必须软链到系统
`/system/lib64/libvulkan.so` 才能枚举到设备。

---

## 5. 切点 36:40：已实测，不采纳（主机链）

`32:40` 的问题是 ROG 那 76 ms 里有 22.7 ms 来自内存压力。让 ROG 只背 4 层、本机背 36 层
的实测结果：

| 切点 | 本机段 | ROG 段 | 吞吐 | 相对 |
| --- | --- | --- | ---: | ---: |
| **32:40（现行）** | 0:32 | 32:40（8 层） | **17.0 t/s** | 基线 |
| 36:40 | 0:36 | 36:40（4 层） | **17.7 t/s** | **+4%** |

预测的 +17% 没有实现。4% 小于把 4 层负载从台式机挪到笔记本的代价，**不采纳**。

---

## 6. 证据等级

**已证实（源码 / 官方文档 / 本次实测）**

- 弱设备链端到端 **153.4 ms / 6.50 t/s**（n=24，p90 165.1 ms，两次复测 6.50 / 6.52）
- 链式 prefill 稳态 **22 tok/s**（N=256，11.7 秒；N=16 时 13.5 tok/s，含约 750 ms 固定开销）
- 时间可分解：head 90.3 + Y700 计算 45.5 + 网络 17.5 = 153.3，与实测 153.4 吻合
- MoE 每层：Surface 3.76 ms / Y700 2.84 ms（1.3 倍）
- dense 9B 的 8 层工件：Surface 7.55 t/s / Y700 19.76 t/s（2.6 倍）；Y700 跑 9B 全模型 7.82 t/s（带宽打满）
- `llama_set/get_embeddings_layer_inp` 不是 `extern "C"`，符号 mangled，必须 `dlsym`
- shim 平台相关代码实测 3 处（两处 Win32 + `_putenv_s`），移植后 40 KB
- `-t 5` 在 Y700 上最优；`-t 8` 退化 40%；`GGML_NATIVE=ON` 检测到 dotprod+i8mm
- 内核补丁：qwen35moe 的 `t_layer_inp[n_layer]` 在 `n_outputs=0` 时是 `[n_embd, 0]` 空张量
  （`crop_before_nextn` 导致），需要在 relay 场景跳过一次裁剪
- 切点 36:40 = 17.7 t/s，32:40 = 17.0 t/s（§5）

**推测（未实测）**

- 80B-A3B 级模型（40–50 GiB）分两段的可行切分
- NDK 构建形态的性能

---

## 7. 已完成 / 下一步

**已完成**：平台探针、shim 移植、worker 接线、端到端 decode（6.5 t/s）与 prefill（22 tok/s）
实测、内核补丁进主树并重新生成 patch 快照。

**下一步的优先级**：

1. **Y700 的内存边界再压**：16 层是当前安全值，20 层 OOM。如果能把页缓存压力降下去
   （更小的 ctx、更紧的量化），20/20 是更均衡的切点，prefill 也会跟着受益（head 现占 65%）。
2. **prefill 的 750 ms 固定开销**：每次 RESET 后要重新 reserve 图。真实对话里 KV 是连续增长的，
   这笔开销只在首轮出现；但如果做多轮会话，值得确认它不会被反复触发。
3. **两端内核版本统一**：Surface 用"编译宏 + default 通道"，Y700 用"内核补丁 + default 通道"，
   两条路实测 hidden 逐位相同。补丁已经进 `H:\llamacpp-c`，Surface 那边可以撤掉宏了。
4. **35B 上它的意义是换出本机，不是增加总容量**；真正的容量意义要等模型换到 80B-A3B 级。
