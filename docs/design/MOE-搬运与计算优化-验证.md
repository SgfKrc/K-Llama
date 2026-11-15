# MoE 优化方案的验证：七条前提逐条判定

> 面向 `MOE-搬运与计算优化.md`（下称「原方案」）的七条待实证前提 · 基于 2026-11-15 本机实测与代码核查 · 产出可执行的取舍结论
> 环境：i9-13900H / RTX 4060 Laptop 8 GB / 32 GB RAM，`build-cuda`（CUDA=ON, Vulkan=OFF, NATIVE=ON），HEAD `675b0e30f`
> 模型：`Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf`（23.48 GB / 21.86 GiB，`file_type=18`，专家 IQ4_XS 15.75 GiB + Q6_K 8.89 + Q5_K 7.50 + Q8_0 0.90）
> 测量纪律：每次测量独占机器，9B `-p 0 -n 64 -r 3` 作 bracket（CUDA 构建基线 6.86–6.91，波动 ±0.06）

---

## 0. 一句话

**方案 A 可以接线做（80–150 行），但它的收益前提在本机被 VRAM 瓶颈掩盖；方案 B 的核心论据不成立**——它引用的 6.5 GB/s 是 Vulkan 核显上一个**已修**的 kernel-selection bug，不是 CPU 并行度；把调度改成 claim 在 decode 形状下与非 repack 路径**逐字节等价**，收益 0–12% 而非 4 倍。原方案约六成事实可核实，但口径错误集中在「后端张冠李戴」与「基线模型不符」，直接照做会做错方向。

---

## 1. 本机实测：同口径基线（原方案 §6 第 1 条）

`llama-bench -p 941 -n 64 -r 3~6 -t 8`，CUDA 后端，模型同上。

| 配置 | 上 GPU 的专家层 | pp941 t/s | tg64 t/s | VMEM 状态 |
| --- | ---: | ---: | ---: | --- |
| `-ngl 99 -ncmoe 29` | 11 | 58.64 ± 2.69 | **15.07 ± 0.28** | **spill**（need 9.95 / free 7.93 GiB，短 2.01 GiB） |
| `-ngl 99 -ncmoe 32` | 8 | 158.95 ± 14.85 | 23.08 ± 0.86 | 未报 spill |
| `-ngl 99 -ncmoe 34` | 6 | 218.13 ± 6.86 | **24.37 ± 0.19** | 未报 spill |
| `LLAMA_LAYER_RANGE=0:24 -ngl 99 -ncmoe 24`（head 段） | 0 | 263.59 ± 24.82 | **36.75 ± 0.44** | — |

**三条结论**：

1. **原方案的 30.45 ± 0.27 在本机不可复现**。它是 `UD-Q4_K_XL`（20.82 GiB）在 `-ncmoe 29` 下的 tg256；本机是 IQ4_XS 混合（21.86 GiB），同一 `-ncmoe` 下 **VRAM 装不下、spill 2 GiB**，tg 掉到 15.07。**比本机最优配置（24.37）还慢 38%**。
2. **本机最优 `-ncmoe` 是 34 而不是 29**。29 的 tg（15.07）低于 34（24.37）——spill 的代价超过了多放 5 层专家上 GPU 的收益。原方案 §5 的「head 段有 29 层 host 专家（`-ncmoe 29`）」既把段长度搞错（三段链 head 是 `0:24`，只有 24 层），也踩在这个 spill 区间上。
3. **head 段（24 层、专家全在 CPU）单独跑是 36.75 t/s**，是链上 5.85 的 6.3 倍。这印证原方案 §1 的定性（链的瓶颈是 RTT，不是段内计算），但**具体数字全部要换**。

> **口径警告**：上表 tg 是 `llama-bench` 的合成 token，与原方案 §1 的「941 prompt / 64 gen」真实文本口径不等价（llama-bench 不做 chat template）。head 段那行的 `LLAMA_LAYER_RANGE` 是环境变量，`llama-bench` 不接受 `--split-layer-range`。

---

## 2. 七条前提逐条判定

| # | 原方案前提 | 判定 | 依据 |
| ---: | --- | --- | --- |
| 1 | head 段同口径（941+64）速度只有 `pp512/tg256` 的 30.45 | **不成立（口径与模型双错）** | 见 §1：30.45 是 Q4_K_XL + tg256；本机 IQ4_XS 在 `-ncmoe 29` 下 spill，实测 15.07；本机最优 24.37；head 段 36.75 |
| 2 | #31122 在 Q4_K_M + 4060 Laptop 上复现 | **未验证（可做，成本低）** | 上游 PR 事实全部核实无误（`S95F`、1 commit `25f222c`、Closed、三条自动检查意见、`cudaHostRegisterPortable|ReadOnly`、env gate、零 in-tree 调用者）。但**本机模型不是 Q4_K_M**，且 `-ncmoe 29` 已 spill——复现前要先定一个不 spill 的配置 |
| 3 | 打开 `moe_cache` 后的 pageable 基线 | **未测（工具受限）** | `llama-bench` 不接受 `--moe-cache-mib`；server 路径的 `/completion` 响应在本版上游**不再带 `timings` 字段**，需改用 `--timings` 或解析 server 日志。原方案「只开 cache 不 pin 是负收益（18.8 < 22.3）」是上游数据，本机未复现 |
| 4 | 6.5 GB/s 与 42.55 GB/s 是否同口径 | **不成立（不同后端、不同模型）** | 6.5 是 **Vulkan 核显 + 35B Q2_K** 的 `MUL_MAT_ID_VEC`（GPU shader op），且是**修复前**的值；`IGPU-MOE-MMVQ.md` 记录修 `ggml_vk_should_use_mmvq()` 后同 op 6.5 → **33.9 GB/s**（5.2×），归因是 **q2_K dequant 指令数**，不是并行度。41.7 是 **9B Q4_K_M 纯 CPU 的整模型推算**（5.68 GB × 7.35 t/s），与 MoE 无关 |
| 5 | 批处理改动的收益（推算 6.5 → 10/15/30） | **不成立（论据链断裂）** | 见 §3：`repack.cpp:4886-4897` 的静态切分在 decode（nth=8, ne01=1024）下**每线程 128 行、NB_COLS 整除、零 `return`**；非 repack 路径的 claim 在 decode 时 `nchunk0=nth=8` 退化为**同一个划分**。改成 claim 收益 ≈ 0 |
| 6 | WDDM cap 在 32 GB RAM 上的可用值 | **未验证（依赖第 2 条）** | 上游「无限注册到 ~15.3/31 GiB 后不稳」成立；默认 cap = RAM/4 = 8 GiB。但本机 `-ncmoe 29` 已经 spill 2 GiB，pinning 前必须先解决 VRAM 预算 |
| 7 | 分段链上 `moe_cache` 是否真的生效 | **成立（是伪问题）** | `server-context.cpp:1464` 的 `vram_in.moe_cache` 只是预检；真实链路是 `common_init_from_params` → `common.cpp:2087` → `llama-context.cpp:392` → `:622`。命令行给 `--moe-cache-mib` 且 `-ncmoe` 留下 host 专家时，链上 head 会生效 |

---

## 3. 方案 B 的归因纠正（见 `MOE-搬运与计算优化.md` §2.2 / §4）

原方案的因果链是：`6.5 GB/s`（gather 慢）→「地址惩罚≈0」→「并行度喂不饱」→ 批处理（250–500 行）→ 4 倍。

**三处断裂**：

1. **6.5 GB/s 不是 CPU 的数**。它来自 `IGPU-SLOWDOWN.md:287`，是 Vulkan 核显上 `MUL_MAT_ID_VEC q2_K` 的 GPU 侧测量。CPU 侧 MoE 的带宽**从未被单独测过**；`IGPU-MOE-MMVQ.md` 只有「整 token 平均 25.8 GB/s」（1.49 GB/token × 17.33 t/s），扣掉非专家段后 MoE 段落在 12–30 GB/s 的宽区间，误差 2–3 倍。
2. **静态切分在 decode 下不空转**。`repack.cpp:4886-4897`：`ne01=1024`、`nth=8` → 每线程 128 行；`NB_COLS=8`（`q4_K_8x8_q8_K`）整除，`return`（`:4896`，注意是 `return` 不是 `continue`）不触发。**真正的隐患是那个 `return`**——若 `nth` 增大到使 `ne01/nth < NB_COLS`，某线程会静默丢掉后续所有 expert。这是正确性隐患，不是性能瓶颈。
3. **claim 与非 claim 在 decode 下等价**。`ggml-cpu.c:1710-1724`：`nr1==1` → `chunk_size=64` → `nchunk0=16` → 因 `16 < nth*4=32` 强制 `nchunk0=nth=8`，每线程恰好 1 个 128 行 chunk，无竞争、无 steal，与 repack 的 `(ith*ne01)/nth` 逐字节等价。**把 repack 改成同文件 dense 路径（4747-4780）的 claim 模板，decode 时产出相同划分，收益 = 0，外加 atomic 开销。**

**真实瓶颈按可能性排序**（原方案都没提到）：

| 排序 | 瓶颈 | 证据 | 可测判据 |
| ---: | --- | --- | --- |
| ① | **工作块太小**：每次 gemv 只有 144 KiB/线程，8 次串行调用/层；dense 一次覆盖 3.1 MB（大 21 倍） | `repack.cpp:4900-4913` 逐 expert 逐行；同文件 dense 路径用 `ggml_threadpool_chunk_add` 做 4× oversubscription，MoE 没有 | 把 gate_up 的 8 个 expert 合成一次 `gemm(nr1=8)` 的合成基准 |
| ② | P/E 核异构拖尾（i9-13900H 6P+8E） | 静态等分下最慢线程决定总时间 | `-t 4/6/8/12` 扫 MoE 段 µs；8 vs 6 线性则无拖尾 |
| ③ | 量化 dequant 指令数（对 Q4_K 已排除，对 Q2_K/IQ4_XS 成立） | 9B Q4_K dense 41.7 GB/s 用的就是同一 repack 内核；但 **IQ4_XS 不在 repack 列表**（`repack.cpp:4979-5109`），走通用 `vec_dot` | 同模型换 Q4_K 专家后每字节耗时应下降 |

**结论**：方案 B 若只改调度，收益上界 **0–12%**（只治 ②）；要治 ① 需要「跨 expert 合并成一次多行 gemm」，那要动 gemv 内核的 `nc/nr` 语义和 dst scatter，不是搬 4747-4780 的模板——**原方案 250–500 行的估算低估了真正的改动**。

---

## 4. 方案 A 的真实工作量（原方案 §3）

**是接线，不是移植**。`ggml_backend_cuda_register_host_buffer` 的契约全在位：

- 声明 `ggml/include/ggml-cuda.h:40`，导出经 `get_proc_address`（`ggml-cuda.cu:6872-6877`）
- env gate `GGML_CUDA_REGISTER_HOST`（`:170` / `:5975`）
- `cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly)`（`:5980`），失败清错 + DEBUG + `return false`
- **零 in-tree 调用者**

要新写的只有 `MOE-搬运与计算优化.md` §3.2 第 1 步那个「按真实 OS 页对齐并合并」——`llama_moe_cache` 里有 `groups`/`layers`/`host_bytes`，但**没有任何按地址排序、合并、页对齐的结构**。落点：`src/llama-moe-cache.cpp` 的 `impl` ctor 尾部（约 655，`groups`/`layers` 已定型）注册，`impl` dtor（682）注销。

**估行数 80–150**（原方案 100–200 合理）。

**但有一个原方案没写的前提**：本机 `-ngl 99 -ncmoe 29` 已经 spill 2 GiB（§1）。pinning 只解决 H2D 拷贝效率，不解决 VRAM 不足。**复现前必须先定一个不 spill 的 `-ncmoe`**（本机是 34），否则测到的是 spill 的噪声。

---

## 5. 原方案的其他口径错误（不影响结论方向，但会误导执行）

| 位置 | 原方案 | 实际 |
| --- | --- | --- |
| §1 表·单机整模纯 CPU | 8.23，标「941 + 64」 | `DESIGN.md:25` 是 **tg256**；`INDEX.md` 现行 941 口径是 **9.09**（`-t 14`） |
| §1 表·三段链 | 5.85，标「`-ngl 0`」 | `INDEX.md:27` 无 `-ngl 0`；64 gen 是 **1522** 口径的数，941 口径没记 |
| `MOE-搬运与计算优化.md` §1 成本表·固定开销 | 3 × 14.78 ms | `14.78` 全仓库无出处；`INDEX.md` §四 的每段固定开销是 10–20 ms |
| §1 成本表·hidden 传输 | 8 KB / **2 ms** / 1% | 8 KB ÷ 100 Mbps = **0.66 ms**；`ARCHITECTURE.md:213` 明说「亚毫秒级」 |
| §1 成本表·占比 | 26+43+31+1% | 加总 **101%**；44.3+73+54+2 = 173.3 ≠ 170.9（2 ms 与 54 ms 重复计） |
| §0 | 合计 400–800 行 | 100–200 + 250–500 = **350–700** |
| `MOE-搬运与计算优化.md` §2.2 | 8 次独立的 **1.72 MB** GEMV | 1.72 MiB 是 gate_up+down **合计**；单次 `mul_mat_id` 的 gate_up 是 **1.125 MiB**，且 decode 每层有 **2 次**（gate_up + down） |
| `MOE-搬运与计算优化.md` §4.4 t/s 列 | 8.5 / 11.8 / 20.0 | 只有把**整链 40 层**都按缩放后的每层 ms 算才复现（`1000/(40×ms+17.5)`）；按「只缩 Surface 24 层」得 **8.22 / 9.79 / 12.09**，后两行低 17–40% |
| `MOE-搬运与计算优化.md` §4.4 方法 | 用 6.5 GB/s 缩放 3.76 ms/层 | 3.76 × 6.5 = **24.4 MB/层**，而 8 专家实际 ≈ 14.4 MB——只有 59% 的 3.76 ms 能归给 gather；且 6.5 测于**本机核显**，3.76 测于 **Surface i5-1035G4** |
| §6 第 4 条 | 读 Strata `Memory/LEDGER.md` L9 | **该文件在 Strata 仓库里不存在**（全树 find 无）；42.55 GB/s 只能从 `pool.hpp:3-7` 的注释得到 |
| §8 | `sycl/` 114,290 行 | `.cpp` 实测 **114,743**（差 0.4%）；含头文件共 161,384 |

---

## 6. 建议的下一步（按收益/成本排序）

| 优先级 | 动作 | 判据 | 成本 |
| ---: | --- | --- | --- |
| **1** | 用 `-ncmoe 34`（不 spill）重跑 §6 第 3 条：`--moe-cache-mib {0, 1024, 2048, 3072}` 的 tg 曲线 | 若 cache 在 pageable 下是负收益，先做方案 A；若是正收益，可以先开 cache 拿一部分 | 半天 |
| **2** | 接线方案 A（80–150 行），在 `-ncmoe 34` 下 A/B | temp 0 逐位一致；tg/pp 对照；cap 从 2 GiB 起扫 | 1–2 天 |
| **3** | 零代码实验：`-t {4,6,8,12}` 扫 MoE 段 µs（先加 5 行计时） | 若 8 线程已线性，**方案 B 作废**；若 8≈6，claim 有 ≤25% 空间 | 半天 |
| **4** | 只有第 3 步显示拖尾时，才考虑 claim；要治「工作块太小」需要另立题（跨 expert gemm） | — | — |
| **5** | 链上端到端（§6 第 7 条） | 三段链 941+64 与 5.85 对照 | 1 天 |

**不建议**：按原方案直接做方案 B 的 250–500 行——它的收益论据建立在一个 GPU 侧的已修 bug 上，且 claim 在 decode 下等价于现状。

---

## 7. 与 V3 收尾的关系

这份验证**没有推翻**原方案的方向（head 段的搬运与计算确实值得修），但**重排了优先级**：

- **方案 A 仍是唯一低风险、可量化的收益点**，且本机已经具备全部前提（API 在位、零调用者、cap 机制清楚）。
- **方案 B 需要先证伪「并行度喂不饱」**，否则 500 行换 0–12% 是亏的。
- **基线必须先钉住**：`-ncmoe 34`（tg 24.37）而不是原方案的 29（spill，15.07）。

如果按 §6 的 1→2→5 走完，head 段的 tg 从 24.37 提到 pinning 后的量级（上游 +19% 对应约 29），链上 5.85 的 31% 段内计算部分相应下降——**这是可以宣布 V3 收尾的量级**，而方案 B 不是。

---

## 8. 补充实测（2026-11-15 晚，三条全部跑完）

全部在 `-ncmoe 34`（§1 定的不 spill 最优）下，独占机器，CUDA 后端。

### 8.1 MoE cache 是负收益，且越大越慢（原方案 §6 第 3 条）

`llama-server` + `/completion`（`n_predict 128`、`cache_prompt:false`、长 prompt），每档 3 次取中位数：

| `--moe-cache-mib` | tg 中位数 | 三次 | pp |
| ---: | ---: | --- | ---: |
| **0（off）** | **23.39** | 21.49 / 23.39 / 23.92 | 61.66 |
| 1024 | 14.19 | 10.57 / 14.19 / 14.50 | 33.34 |
| 2048 | 17.88 | 14.21 / 18.16 / 17.88 | 37.84 |
| 3072 | **10.13** | 8.95 / 10.13 / 10.20 | 18.71 |

**上游的「pageable 下 cache 是负收益（18.8 < 22.3）」在本机成立，而且更严重**：本机是 **−39% 到 −57%**（上游 −16%），并且**不单调**（1024 < 2048，3072 最差）——不是「cache 越大越好」的简单关系，pageable 上传 + WDDM 的代价压过了命中收益。

**这条直接决定顺序**：**方案 A（pinning）不是「可选优化」，是打开 cache 的前置条件**。在 pinning 之前，`--moe-cache-mib` 在本机是净负收益，任何「先开 cache 拿一点」的想法都不成立。

### 8.2 线程数 4 就饱和：方案 B 作废（原方案 §6 第 4/5 条）

`llama-bench -ncmoe 34 -p 941 -n 64 -r 2`，`-t` 一次跑多值：

| threads | pp941 | tg64 |
| ---: | ---: | ---: |
| 4 | 162.57 ± 11.12 | **22.16 ± 0.40** |
| 6 | 187.51 ± 1.39 | 22.47 ± 0.02 |
| 8 | 192.33 ± 4.26 | 23.85 ± 0.26 |
| 12 | 200.82 ± 1.23 | **24.56 ± 0.22** |

**tg 从 4 线程到 12 线程只涨 10.8%（22.16 → 24.56），4→6 几乎不涨（+1.4%）**；pp 倒是接近线性（+23%）。

**解读**：decode 的 tg 已经被非 CPU 部分限制（GPU 计算 + PCIe 传输 + WDDM），**CPU 侧的线程并行度在 4 线程时就已经饱和**。原方案「每层只有 8 个专家、并行度喂不饱所有核」在这里**被实测否证**——如果真是并行度问题，4→12 线程应该接近 3 倍，实际只有 1.1 倍。

**方案 B 判定：作废**。把调度从静态切分改成 claim，能影响的只有 CPU 侧那一小段，而那段在 4 线程时已饱和；收益上界 < 10%，且 §3 已证明 claim 与非 repack 路径的现有实现在 decode 形状下逐字节等价。250–500 行换不到可测的收益。

### 8.3 `-ncmoe` 的单峰：34 是最优（原方案 §6 第 1 条）

`llama-bench -t 8 -p 941 -n 64 -r 2~3`：

| `-ncmoe` | 上 GPU 的专家层 | tg64 | 备注 |
| ---: | ---: | ---: | --- |
| 29 | 11 | 15.07 ± 0.28 | **spill**（need 9.95 / free 7.93 GiB） |
| 32 | 8 | 23.08 ± 0.86 | |
| **34** | **6** | **24.37 ± 0.19** | **峰值** |
| 36 | 4 | 23.35 ± 0.16 | |
| 38 | 2 | 22.36 ± 0.09 | |
| 40 | 0（全 CPU 专家） | 21.55 ± 0.15 | |

**单峰，峰值在 34**。往左（32/29）VRAM 开始紧张直到 spill，往右（36/38/40）GPU 利用不足。原方案假设的 29 落在**峰左的悬崖上**——这就是为什么它的 30.45 在本机不可复现。

### 8.4 三条测完之后的总判断

| 方案 | 判定 | 依据 |
| --- | --- | --- |
| **A（host 专家页 pinning）** | **做，且是前置** | cache 在 pageable 下 −39%~−57%（§8.1）；pinning 是接线不是移植，80–150 行（§4）；本机不 spill 配置已定（`-ncmoe 34`，§8.3） |
| **B（跨 expert 批处理）** | **作废** | 4 线程即饱和（§8.2），CPU 不是瓶颈；claim 在 decode 下与现状等价（§3） |
| 原方案的基线 30.45 | 不可复现 | 模型不同（Q4_K_XL vs IQ4_XS）+ `-ncmoe 29` spill（§8.3） |

**下一步只有一条路**：接线方案 A，在 `-ncmoe 34` 下做 `--moe-cache-mib {0,1024,2048,3072}` 的 A/B，看 pinning 后 cache 是否从 −39%~−57% 翻成正收益。如果是，head 段就拿到上游那 +11%~+30%；如果不是，本机的 PCIe x8 + WDDM 组合不适合 GPU 专家缓存，方案 A 也只剩「H2D 拷贝更快」这一层收益（上游 pp +27%~+44% 的那部分）。

---

## 9. 方案 A 的实现与本机结论（2026-11-15 深夜）

### 9.1 实现（`H:\llamacpp-c`，未提交）

- `src/llama-moe-cache.cpp`：`impl::pin()`（约 126 行）——env gate（`GGML_CUDA_REGISTER_HOST`）→ cap（`ggml_backend_dev_memory(cpu)/4`，`LLAMA_MOE_PIN_MIB` 覆盖，≤0 禁用）→ 遍历 `groups` 收集 host 专家 range → 排序/合并/页对齐 → 逐 range `ggml_backend_register_host_buffer`（proc address，名字不带 cuda 前缀）→ `pinned` 记录 `(ptr, unregister)`，dtor 注销。
- `src/llama-moe-cache.h`：`void pin();`
- `src/llama-context.cpp`：`sched_reserve()` 之后 `if (moe_cache) moe_cache->pin();`
- `ggml/src/ggml-cuda/ggml-cuda.cu`：`cudaHostRegisterReadOnly` 返回 `cudaErrorNotSupported` 时去掉 ReadOnly 重试。

**关键修正**：`pin()` 必须放在 `sched_reserve()` 之后。放在 ctor 里时 `ranges = 0`——`no_alloc` 模式下 model tensor 的 `data` 和 `buffer->base` 都是 null；移到 `sched_reserve()` 之后 `ranges = 102`。

### 9.2 本机结论：WDDM 不支持大范围 file-backed pin

三层错误链，逐层排除：

| 层 | 现象 | 排除方式 | 结论 |
|---|---|---|---|
| 1 | ctor 里 `ranges = 0` | 移到 `sched_reserve()` 后 | 已解决，`ranges = 102` |
| 2 | `cudaHostRegister(Portable\|ReadOnly)` → `801 operation not supported` | ctypes 直调 cudart 对照 | **ReadOnly 在本机驱动上不支持**（匿名与 file-backed 一样） |
| 3 | 去掉 ReadOnly（Portable 或 0）→ `2 out of memory` | 注册前按页 fault-in（touch 每一页） | **仍 out of memory** |

**ctypes 对照**（`cudart64_12.dll`）：

| 内存 | flags | 结果 |
|---|---|---|
| 匿名 1 MiB | `Portable\|ReadOnly` | 801 |
| 匿名 1 MiB | `Portable` / `0` | 0 ✓ |
| file-backed 1 MiB | `ReadOnly` / `P\|RO` | 801 |
| file-backed 1 MiB | `Portable` / `0` | 0 ✓ |

**1 MiB 能 pin，210 MiB 不能**，且 fault-in 无效——说明不是「页未 resident」，是 WDDM 对大范围注册的限制：Windows 的 `cudaHostRegister` 不带 ReadOnly 时把范围**复制进 pinned 池**（池大小受 WDDM 约束），而 Linux 的同一调用是**锁定原页**（零拷贝）。上游 #31122 的收益建立在后者上。

### 9.3 因此

- **方案 A 在本机（Windows + mmap）不成立**：不是实现问题，是驱动语义差异。上游的 `Portable|ReadOnly` 在 Windows 直接不支持，退到无 ReadOnly 又变成复制。
- 唯一可能的替代是 `--load-mode none`（loader 自己分配 pinned host buffer，不经过 `cudaHostRegister`），代价是 21.86 GiB 加载慢 + 锁定——这正是 `MOE-搬运与计算优化.md` §2.1 表里「`--load-mode none`：279.1 pp / 32.3 tg」那条，它在 pageable 之上确实快，但加载时间不可接受。
- **`ggml-cuda.cu` 的 ReadOnly fallback 值得保留**：它把一个真实的 Windows 平台差异变成「多一次调用」，失败时仍然 fail soft；对上游是一个独立可提交的小修复。
- **pin 实现本身保留**：它在 Linux 上是对的（上游机制），本机只是跑不到。

### 9.4 对 §8.1 的最终修正

§8.1 的 cache 曲线（23.39 → 14.19 / 17.88 / 10.13）**不能归因于 pageable 上传**——本机的 pinning 从未成功注册过任何范围。那条曲线的成因是「cache 占 VRAM → `-ncmoe 34` 从 0 spill 变成 1.36 GiB spill → decode 变慢」，与 §8.3 的 ncmoe 单峰是同一个机制的两个侧面。

**所以「先 pinning 再开 cache」这条实施顺序在本机不成立**：pinning 不可用，cache 的负收益又主要来自 VRAM 抢占，那么 cache 在本机 8 GB VRAM 上就**没有可用配置**（除了把 `-ncmoe` 再调大、让出 VRAM，但那时 cache 的命中收益也随之消失）。

**方案 A 判定：在本机不成立，需 Linux + 足够 VRAM 的机器才能验证上游那组 +11%~+30%。**

### 9.5 `--load-mode none` 对照（全 pinned，绕过 cudaHostRegister）

既然 `cudaHostRegister` 走不通，用 `-lm none`（loader 自己分配 pinned host buffer，21.86 GiB 全锁定）验证「pinned 后 cache 是否翻正」：

| cache | tg（两次） |
| ---: | --- |
| 0 | 23.29 / 24.03 |
| 2048 | 23.20 / 24.56 |

**两条结论**：

1. **cache 的负收益确实是 pageable 上传造成的**：pageable 下 cache 2048 是 17.88（−24%），pinned 下是 23.20（±0%）——上传代价消失。
2. **但 pinned 后也没有正收益**（23.29 → 23.20，噪声内）。本机 8 GB VRAM 在 `-ncmoe 34` 下已经用掉 7.3 GiB，cache 再占 2 GiB 就把 `-ncmoe 34` 推入 spill（§8.1 已测：need 9.30 / free 7.93）。命中省下的上传被 spill 抵消。

**所以方案 A 在本机不可行是双重原因**：WDDM 不支持 `cudaHostRegister` 的大范围 file-backed 注册，而即使绕开它（`-lm none`），8 GB VRAM 也没有空间让 cache 净收益为正。上游那组 +11%~+30% 需要「pinned + VRAM 有余量」两个条件同时成立。
