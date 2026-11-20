# R9 复测：两项 Vulkan 调优在 rebase 后的上游新 pipeline 下是否仍有效

> 状态：**实测结论（2026-11-20）**
> 平台：Intel Iris Xe（Gen12 / XE1）+ i9-13900H，Windows 专有驱动，单卡隔离 `GGML_VK_VISIBLE_DEVICES=0`
> 触发：审计风险 **R9**——`c8f33e018`（原 `b3c134db0`）「vulkan : fix the Intel iGPU host buffer, offload and MMVQ paths」
> 的收益是在**旧 pipeline** 上测的；fork 刚 rebase 到上游 `b94959a64`（11-19，跨 170 提交，含 **12 个 vulkan 提交**，
> unary/bf16/stream_k/narrow_n_max 与 `ggml-vulkan-types.h` 大量重写），需要确认第 1、2 项是否还是正收益。
> 相关：`IGPU-MOE-MMVQ.md`（旧口径 MMVQ 归因）、`IGPU-SLOWDOWN.md` §⑥⑦（旧口径 offload 交叉点）、`MEASUREMENT-PITFALLS.md`

标注约定：**[实测]** 本机带 bracket 对照测出；**[代码]** 源码读出；**[推算]** 公式算出。

---

## 1 结论

1. **[实测] 第 1 项（XE1 MMVQ 放行）仍有效，且比旧 pipeline 略强**：35B-A3B Q2_K decode
   `8.8–9.4 → 14.3–15.3 t/s`，**+63%**（旧 pipeline 记录 `10.66 → 16.64`，+56%）。
   内核级 q2_K（`MUL_MAT_ID_VEC m=512 n=8 k=2048`）**8.0 → 33.2 GB/s**，修复后带宽与旧记录的 **33.9 GB/s 差 2%（噪声内）**。
   → **保留**。
2. **[实测] 第 2 项（op_offload iGPU 阈值 32→64）仍有效，但收益被新 pipeline 吃掉一半**：
   pp32 时 `thr32 63.9 → thr64 94.3 t/s`，**+47.6%**（旧 pipeline 同口径 `45.7 → 89.9`，**+96.9%**）。
   原因是新 pipeline 把 **iGPU op_offload 的 prefill 路径拉快了 40–100%**（同一命令 pp128：旧 109.7 → 新 218.9），
   而 CPU 侧完全没变（纯 CPU pp32 旧 112.4 / 新 110.2）。
3. **[实测] 交叉点左移，64 反而更「对症」了**：拿现行默认 64 与「纯 CPU」比——
   旧 pipeline 到 pp96 还输 29%（pp32 输 20%、pp64 输 36%）；新 pipeline 下 pp32 只输 14%、pp48 输 13%，
   **pp64 起 offload 反超 7%**（133 vs 124）、pp96 反超 17–26%。
   即阈值 64 现在正好落在「offload 开始赢」的位置：pp<64 走 CPU、pp≥64 走核显。
   → **保留 64**；回到 32 会丢 pp32 的 +48%，不建议。
4. **[实测] 附带发现（超出 R9 原问题，供决策）**：标量阈值表达不了「小 pp 完全不 offload」。
   pp32/48 时 `thr64` 仍有 nrows 型算子被搬上核显，实测比纯 CPU 低 **14% / 13%**；
   把阈值扫到 96/128/256/512 都恢复不了（pp32：91–98 vs 纯 CPU 110）。要再拿这 14%，得改判据而不是调阈值。
5. **[实测] 绝对值比旧记录低 8–12%，但不是 pipeline 的锅**：旧 build（pre-rebase）今天在**同一分钟窗**测得
   no-fix MoE decode 9.24–9.43，与新 build no-fix 的 9.25–9.46 一致；结论是机器状态（温漂/后台）而非 pipeline 变慢。

---

## 2 装置与方法

### 2.1 两份构建

| 构建 | 来源 | 用途 | 关键事实 |
|---|---|---|---|
| **NEW** | 独立 worktree `H:\llamacpp-r9` @ `c7c02d589`（rebase 后 HEAD，即 `b94959a64` + 我们的提交） | 主测量 | `GGML_VULKAN=ON`、`GGML_CUDA=OFF`、`Release`、`BUILD_SHARED_LIBS=ON`、`GGML_NATIVE=ON`、Ninja、MSVC 19.51.36247.0、VULKAN_SDK 1.4.363.0 |
| **OLD** | `H:\llamacpp-c\build-v22-vulkan`（2026-11-14 构建，pre-rebase，v2.2 分支） | 「旧 pipeline」对照 | **不含** XE1 MMVQ 放行（实测开关无差别：9.43 vs 9.24）；**含** op_offload iGPU=64 |

**[实测/方法] 为什么另建 worktree**：`H:\llamacpp-c` 当时有两个 agent 在改 `src/models/*.cpp`、`src/llama-model*`、`common/common.cpp`、
`ggml-vulkan-buffers.cpp`（后者在被改的是 iGPU host buffer 分配路径，正压在第 3 项上）。为了 A/B 里唯一变量是「被测调优」，
NEW 构建取的是 **HEAD 的干净树**（`git worktree add --detach H:/llamacpp-r9 c7c02d589`，独立 `build-r9`），
共享树的未提交改动一个字节都没进来、也没被碰。

### 2.2 开关：不靠改源码

**[代码] 两处调优都有现成 env**（`ggml-vulkan.cpp:5475` / `:17304`），所以主测量**没有改源码**：

| 调优 | 开关 | 语义 |
|---|---|---|
| 第 1 项 MMVQ | `GGML_VK_DISABLE_MMVQ=1` | `mmvq_mode=-1` → `ggml_vk_should_use_mmvq()` 直接 false |
| | `GGML_VK_FORCE_MMVQ=1` | `mmvq_mode=1` → 全类型 true（未用） |
| 第 2 项 offload | `GGML_OP_OFFLOAD_MIN_BATCH=N` | 覆盖 `ggml_vk_op_offload_min_batch_default()`；`4096` ≈ `--no-op-offload` |

**[实测] 等价性验证（第 1 项）**：env 关 MMVQ 是否等于「删掉 XE1 分支」？做了一次真正的 baseline 构建：
把 `INTEL_XE1` 分支 `#if 0` 掉 → 只重编 `ggml-vulkan.cpp`（shader 全缓存，约 1 分钟），测得

| 二进制 | 设置 | tg128 |
|---|---|---:|
| 删 XE1 分支 | 无 env | **9.38 / 9.25** |
| 删 XE1 分支 | `DISABLE_MMVQ=1` | **9.36** |
| 原二进制 | `DISABLE_MMVQ=1` | **9.46 / 9.25** |

三者一致 ⇒ 在这台机器（Intel Windows 专有驱动）上 **env 关 MMVQ ≡ 删 XE1 分支**：其它量化类型本来就走到
`driver_id == eIntelProprietaryWindows → return false`（`ggml-vulkan.cpp:7412`），没有第二种类型在走 MMVQ。
删分支的 worktree 之后 `git checkout --` 已恢复，最终 `git status` 干净。

### 2.3 口径与温漂

- **全表白名单**：只测 Iris Xe（`GGML_VK_VISIBLE_DEVICES=0`）；RTX 4060 全程不参与。
- llama-bench `-t 8`（与旧文档一致；另做 `-t 14` 敏感性，见 §4.4），`-r 3`（广扫）/ `-r 5`（交叉区聚焦），默认带 warmup。
- **[实测] 这台机器的温漂必须记**：旧 build 上同一二进制、同一配置的 MoE tg128，15:32 测 9.43、15:34 测 **8.30（−12%）**；
  新 build 的 MMVQ 开臂在 15:06–15:31 的多次测量里落在 13.04–15.37（±8%）。
  所以**所有结论只用同一时间窗内的 A/B bracket**，跨窗绝对值不采信。

---

## 3 第 1 项：XE1 MMVQ 放行

`ggml/src/ggml-vulkan/ggml-vulkan.cpp:7400` —— `INTEL_XE1` 时放行 `Q2_0 / Q2_K / Q3_K` 走 MMVQ。
测量：35B-A3B Q2_K（`Qwen3.6-35B-A3B-UD-Q2_K.gguf`，12.67 GiB，能装进 15.4 GiB 空闲的核显），`-ngl 99 -p 0 -n 128`。

### 3.1 端到端（两个独立时间窗，各自带 bracket）

| 窗 | MMVQ 开 | MMVQ 关 | 比 |
|---|---:|---:|---:|
| 15:14–15:19 | 14.41 / 14.33 | 8.85 / 8.80 | **1.63×** |
| 15:27–15:31 | 15.29 / 15.26 | 9.46 / 9.25 | **1.64×** |
| **旧 pipeline（2026-11-14 记录）** | **16.64** | **10.66** | **1.56×** |

**收益 +63%（新） vs +56%（旧）——调优没失效。**

对照组（必须成立才对）：

| 对照 | 结果 | 说明 |
|---|---|---|
| pp512，开关两臂 | 63.28 ± 1.02 vs 63.56 ± 1.52 | n>1 不走 MMVQ，差 −0.4% = 噪声 ✓ |
| 2B Q4_K_M tg128 | 26.41 / 26.35（有分支） vs 24.94（删分支构建） | Q4_K 不在放行名单，**无回归** ✓ |

### 3.2 内核级：q2_K 带宽

`GGML_VK_PERF_LOGGER=1` + `-n 8`，取 `MUL_MAT_ID_VEC q2_K m=512 n=8 k=2048 n_expert=256`（就是旧文档 §3 D 实验那个 gather 形状），
8 个 token 行取**中位数**（首行含 shader 冷启动，已排除）：

| | GFLOPS | GB/s（q2_K = 84B/256 元素 ⇒ ×0.164） |
|---|---:|---:|
| MMVQ 关 | 48.9 | **8.0** |
| MMVQ 开 | 202.4 | **33.2** |
| 旧 pipeline 记录 | — | **6.5 → 33.9** |

**[实测] 修复后的绝对值 33.2 vs 33.9 GB/s（−2%，噪声内）——新 pipeline 没有退化内核。**
倍数从旧记录的 5.2× 降到 4.1×，是因为「修复前」基线本身快了 23%（6.5 → 8.0）：其它 vulkan 提交让 dequant 路径变快了。

同口径其它类型（中位数，`-n 8`）：

| 类型 | MMVQ 关 | MMVQ 开 | 说明 |
|---|---:|---:|---|
| q2_K（放行） | 8.0 GB/s | **33.2** | 4.1× |
| q3_K（放行） | 8.5 | 11.8 | 1.4×，与旧记录 10.9 一致 |
| q4_K（未放行） | 14.4 | 13.7 | ≈不变 ✓ |
| q6_K（**故意不放行**） | 16.2 | 13.8 | 样本仅 8 行、含冷启动，仅作方向参考；旧记录是「放行反而 −17%」 |

---

## 4 第 2 项：op_offload 阈值 32 vs 64

`ggml-vulkan.cpp:5496` 的 `ggml_vk_op_offload_min_batch_default()`：iGPU 返 64、dGPU 32。
测量：2B Q4_K_M（`qwen35-2b-Q4_K_M.gguf`），`-ngl 0`（权重全在 host → `ggml-backend.cpp:1387` 的 op_offload 分支才生效），
`-p N -n 0`，`-t 8`。判据是 `ggml_vk_get_op_batch_size()`：MUL_MAT 取 `ne[1]`（= prompt 长度），其它算子取 `ggml_nrows`。

### 4.1 新 pipeline（`-r 5`，交叉区）

| pp | `MIN_BATCH=32`（offload） | `MIN_BATCH=64`（≤63 留 CPU） | 纯 CPU（=4096） | 64 相对 32 |
|---:|---:|---:|---:|---:|
| 16 | 72.6 | 73.2 | 76–86（`-nopo` 口径） | 0（两臂都不 offload）✓ |
| 32 | **63.9** | **94.3** | 110.2 | **+47.6%** |
| 48 | 91.9 | 102.4 | 118.2 | **+11.4%** |
| 64 | 133.5 | 132.2 | 124.0 | 0（两臂都 offload，`64≥64`）✓ |
| 96 | 151.0 | 161.7 | 128.9 | 0 ✓ |
| 128 | 218.9 | 223.6 | — | 0 ✓ |

（pp16/32/48 是 `-r 5` 两次重复的均值；pp64/96/128 是 `-r 3` 两次重复的均值；每格 stdev 1.8–9.2。）

### 4.2 旧 pipeline（同一台机、同一条命令、同一模型）

| pp | `32` | `64` | 纯 CPU（=4096） |
|---:|---:|---:|---:|
| 16 | 70.3 | 72.5 | — |
| 32 | **45.7** | **89.9** | 112.4 |
| 48 | 62.7 | 99.6 | 120.0 |
| 64 | 76.5–83.6 | 77.2–81.4 | 123.7 |
| 96 | 90.3–94.9 | 89.8–91.0 | 128.3 |
| 128 | 109.7–118.4 | 109.3–115.2 | — |

**新旧对照（同配置、同窗口附近）**

| 配置 | 旧 pipeline | 新 pipeline | 变化 |
|---|---:|---:|---:|
| offload 路径（thr32）pp32 | 45.7 | 63.9 | **+40%** |
| offload 路径 pp48 | 62.7 | 91.9 | **+46%** |
| offload 路径 pp64 | 79 | 133.5 | **+75%** |
| offload 路径 pp128 | 109.7–118.4 | 218.9 | **+100%** |
| CPU 侧（thr64）pp32 / pp48 | 89.9 / 99.6 | 94.3 / 102.4 | +5% / +3% |
| 纯 CPU pp32 / pp48 / pp96 | 112.4 / 120.0 / 128.3 | 110.2 / 118.2 / 128.9 | 不变 |

### 4.3 交叉点：现行默认 64 与「纯 CPU」比

决策相关的是「**默认值 64 让这个 pp 落在 CPU 还是核显，落对了没有**」，所以拿 `thr64` 与「纯 CPU（=4096）」比：

| pp | 旧 pipeline：thr64 vs 纯 CPU | 新 pipeline：thr64 vs 纯 CPU |
|---:|---|---|
| 32 | 89.9 vs 112.4 → **thr64 低 20%** | 94.3 vs 110.2 → thr64 低 **14%** |
| 48 | 99.6 vs 120.0 → **thr64 低 17%** | 102.4 vs 118.2 → thr64 低 **13%** |
| 64 | ~79 vs 123.7 → **thr64 低 36%**（offload 比不 offload 还差） | **133 vs 124 → offload 高 7%** |
| 96 | ~91 vs 128.3 → **thr64 低 29%** | **151–162 vs 128.9 → offload 高 17–26%** |
| 128 | ~112 vs — | 218.9 vs — |
| 交叉点位置 | **> pp96**（2B；9B 才是 pp≈32–48，见 `IGPU-SLOWDOWN.md` §⑥③） | **pp ≈ 56–60** |

**[实测] 结论**：新 pipeline 把 offload 路径拉快 40–100%，交叉点从「2B 上 pp>96」左移到「pp≈56–60」。
**阈值 64 从「补丁式近似」变成了「正好卡在反超点」**：pp64 时 offload 已经高 7%，pp48 时 thr64 只低 13%（而不是旧 pipeline 的 17%）。
第 2 项因此**仍然有效**（pp32 +47.6%），只是收益幅度从 +97% 降到 +48%，因为另一侧变快了。

### 4.4 敏感性：线程数

**[实测]** 阈值结论不依赖线程数（`-t 14` 全核，新 pipeline）：

| pp | 32 | 64 | 纯 CPU | 64 相对 32 |
|---:|---:|---:|---:|---:|
| 32 | 65.2 | 103.5 | 121.7 | **+59%** |
| 48 | 93.3 | 111.0 | 124.8 | **+19%** |
| 64 | 122.7 | 127.6 | 138.9 | +4% |
| 96 | 154.3 | 157.6 | 145.4 | +2% |

线程多 → CPU 更快 → 交叉点右移（`-t 14` 时 pp64 纯 CPU 仍赢 9%），但 **pp32/48 上 64 优于 32 的结论不变**。

### 4.5 附带发现：阈值扫不到「小 pp 完全不动」

**[实测]** 新 pipeline、pp32/48，阈值扫描（`-r 5`）：

| 阈值 | pp32 | pp48 |
|---:|---:|---:|
| 64（现行） | 91–94 | 102–104 |
| 96 | 94.9 | 101.1 |
| 128 | 94.4 | 104.9 |
| 256 | 95.2 | 105.4 |
| 512 | 97.8 | 103.3 |
| 4096（=不 offload） | **105.6** | **124.2** |

pp32/48 上没有任何中间阈值能恢复到纯 CPU 的水平：**64–512 全卡在 91–105，比纯 CPU（pp32 110、pp48 118）低 11–17% / 11–15%**。
机制 **[代码]**：`ggml_vk_get_op_batch_size()` 对非 MUL_MAT 返回 `ggml_nrows`，而 flash-attn / rope / 3D unary 的 `nrows = ne1·ne2·ne3`
（含 head 数与 seq），在 pp32 就已经远大于 512 → 这些算子无论阈值取多大都会被搬上核显，而小 pp 上搬它们纯亏。
**这不是 32→64 的回归，而是标量阈值这个设计的固有上限**（旧 pipeline 更差：pp32 时 thr64 也只有 89.9，比纯 CPU 低 20%）。
要拿回这 14%，得改判据（例：MUL_MAT 仍看 `ne[1]`，nrows 型算子改用「本 pp 下实测是否划算」或直接对小 pp 屏蔽），
**属于新工作、不在 R9「验证」范围**，此处只报数。

---

## 5 建议

| 调优 | 判定 | 建议 |
|---|---|---|
| **1. XE1 MMVQ 放行** | **仍有效，+63%** | **保留**。可顺手把注释里的旧数（q2_K 6.5→33.0、decode 10.7→16）改成新口径（8.0→33.2 GB/s，+63%）。 |
| **2. op_offload iGPU 64** | **仍有效，pp32 +47.6%**（旧 +97%） | **保留 64**。不要回 32（pp32 −48%）。若想再挤：pp48 处 64 vs 纯 CPU 只差 13%，阈值本身已经调到极限（§4.5）；下一步是**改判据**，不是改数字。 |
| 3. portable host malloc | 正确性项 | R9 不涉及（本次未测，也未碰） |
| 4. q4_K shader 预取 | 不在 R9 范围 | 未测 |

**[实测] 对 R9 的直接回答**：上游 12 个 vulkan 提交**没有**让这两项失效；它把 iGPU 的 offload 路径拉快了 40–100%，
所以第 2 项的**收益幅度**（不是方向）缩水一半，交叉点左移到 64 附近——**当前默认值 64 反而比 rebase 前更贴交叉点**。

---

## 6 复现

```sh
# 构建（NEW）：独立 worktree，不污染共享树
git -C H:/llamacpp-c worktree add --detach H:/llamacpp-r9 c7c02d589
# 需要 vcvars64 + VULKAN_SDK，然后
cmake -S H:/llamacpp-r9 -B H:/llamacpp-r9/build-r9 -G Ninja \
      -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build H:/llamacpp-r9/build-r9 --target llama-bench -- -j 12

# 第 1 项
export GGML_VK_VISIBLE_DEVICES=0
B=H:/llamacpp-r9/build-r9/bin
M=H:/qlh_models/qwen3-6-35b-a3b-mtp-official/Qwen3.6-35B-A3B-UD-Q2_K.gguf
$B/llama-bench.exe -m $M -ngl 99 -p 0 -n 128 -r 3 -t 8                        # MMVQ 开
GGML_VK_DISABLE_MMVQ=1 $B/llama-bench.exe -m $M -ngl 99 -p 0 -n 128 -r 3 -t 8 # MMVQ 关
# 内核级
GGML_VK_PERF_LOGGER=1 $B/llama-bench.exe -m $M -ngl 99 -p 0 -n 8 -r 1 -t 8 --no-warmup 2>&1 | grep "MUL_MAT_ID_VEC q2_K"

# 第 2 项（2B）
M2=H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf
for pp in 16 32 48 64 96 128; do
  for thr in 32 64 4096; do
    GGML_OP_OFFLOAD_MIN_BATCH=$thr $B/llama-bench.exe -m $M2 -ngl 0 -p $pp -n 0 -r 5 -t 8
  done
done
```

清理：`git -C H:/llamacpp-c worktree remove H:/llamacpp-r9`；临时删分支的构建用 `git checkout -- ggml/src/ggml-vulkan/ggml-vulkan.cpp` 恢复。

---

## 7 未做 / 风险

- **只测了 2B Q4_K_M 与 35B-A3B Q2_K**：本机没有 Q2_K/Q3_K 的 9B dense。旧文档的 offload 交叉点主表是 9B（pp32 16.5→34.5），
  本次用 2B 复测：**方向与结论一致，但 2B 的 CPU 更快、交叉点更靠右**（见 §4.3）。跨模型引用时别把 2B 的绝对数搬去 9B。
- **OLD 对照是 v2.2 分支（11-14）而非 rebase 前 HEAD**：两者都 pre-rebase、同一 Vulkan pipeline，足够判断 pipeline 效应；
  但它不含我们的 MMVQ 放行（已实测确认），所以第 1 项的「旧 pipeline + 修复」只能引用 11-14 的记录值。
- **绝对吞吐跨天不可比**：噪声来自温漂与后台负载（§2.3），本表的可靠性在「同窗 bracket 的比值」，不在绝对值。
- 未改共享树；未提交、未 rebase；临时改动只发生在独立 worktree，且已恢复。
