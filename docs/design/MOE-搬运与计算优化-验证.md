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

## 3. 方案 B 的归因纠正（原方案 §2.2 / §4）

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

要新写的只有原方案 §3.2 第 1 步那个「按真实 OS 页对齐并合并」——`llama_moe_cache` 里有 `groups`/`layers`/`host_bytes`，但**没有任何按地址排序、合并、页对齐的结构**。落点：`src/llama-moe-cache.cpp` 的 `impl` ctor 尾部（约 655，`groups`/`layers` 已定型）注册，`impl` dtor（682）注销。

**估行数 80–150**（原方案 100–200 合理）。

**但有一个原方案没写的前提**：本机 `-ngl 99 -ncmoe 29` 已经 spill 2 GiB（§1）。pinning 只解决 H2D 拷贝效率，不解决 VRAM 不足。**复现前必须先定一个不 spill 的 `-ncmoe`**（本机是 34），否则测到的是 spill 的噪声。

---

## 5. 原方案的其他口径错误（不影响结论方向，但会误导执行）

| 位置 | 原方案 | 实际 |
| --- | --- | --- |
| §1 表·单机整模纯 CPU | 8.23，标「941 + 64」 | `DESIGN.md:25` 是 **tg256**；`INDEX.md` 现行 941 口径是 **9.09**（`-t 14`） |
| §1 表·三段链 | 5.85，标「`-ngl 0`」 | `INDEX.md:27` 无 `-ngl 0`；64 gen 是 **1522** 口径的数，941 口径没记 |
| §1 成本表·固定开销 | 3 × 14.78 ms | `14.78` 全仓库无出处；`INDEX.md` §4 的每段固定开销是 10–20 ms |
| §1 成本表·hidden 传输 | 8 KB / **2 ms** / 1% | 8 KB ÷ 100 Mbps = **0.66 ms**；`ARCHITECTURE.md:213` 明说「亚毫秒级」 |
| §1 成本表·占比 | 26+43+31+1% | 加总 **101%**；44.3+73+54+2 = 173.3 ≠ 170.9（2 ms 与 54 ms 重复计） |
| §0 | 合计 400–800 行 | 100–200 + 250–500 = **350–700** |
| §2.2 | 8 次独立的 **1.72 MB** GEMV | 1.72 MiB 是 gate_up+down **合计**；单次 `mul_mat_id` 的 gate_up 是 **1.125 MiB**，且 decode 每层有 **2 次**（gate_up + down） |
| §4.4 t/s 列 | 8.5 / 11.8 / 20.0 | 只有把**整链 40 层**都按缩放后的每层 ms 算才复现（`1000/(40×ms+17.5)`）；按「只缩 Surface 24 层」得 **8.22 / 9.79 / 12.09**，后两行低 17–40% |
| §4.4 方法 | 用 6.5 GB/s 缩放 3.76 ms/层 | 3.76 × 6.5 = **24.4 MB/层**，而 8 专家实际 ≈ 14.4 MB——只有 59% 的 3.76 ms 能归给 gather；且 6.5 测于**本机核显**，3.76 测于 **Surface i5-1035G4** |
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
