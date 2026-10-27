# llama.cpp 层段推理实际生效参数审计（2026-10-27）

> 范围：master#cpu `[8,24)`、Surface `[28,32)`、y700 `[24,28)` 三台设备的 llama.cpp 层段参数。
> 方法：读代码 + 读构建缓存 + 读运行期原生日志。**每条结论给文件:行号。**
> 标注：`[代码事实]` 我读到的；`[官方文档]` 上游行为；`[计算]` 由事实推算；`[推测]` 未验证。
> 前置调研 `docs/CPU推理优化调研-2026-10-27.md` 已读，本文只写它没覆盖到的"实际生效值"。

---

## 0. 三个最硬的发现（先看这个）

**发现 1：部署中的 shim DLL 是 10-21 的旧二进制，10-26 的两项原生优化根本没进生产。**

- 生产路径指向 `build/keephead/build-cpu/bin/qlh_keep_head.dll`（`.env:53`）`[代码事实]`。
- 该文件时间戳 **2026/10/21 11:35:12**；而 `scripts/model_tools/keep_head_shim/qlh_keep_head.c` 是 **2026/10/26 01:32** `[代码事实]`。
- 用 `strings` 比对三个构建产物，事实无可争辩 `[代码事实]`：

| DLL | `ctx ready` 格式串 | `qlh_kh_set_skip_hidden_logits` |
| --- | --- | --- |
| `build-cpu/bin`（**生产**，10-21） | `n_ctx=%u n_batch=%u`（**无 n_ubatch**） | **无此符号** |
| `build-cpu-ubatchfix2/bin`（10-25） | 含 `n_ubatch=%u` | 无 |
| `build-cpu-skiplogits/bin`（10-26） | 含 `n_ubatch=%u` | **有** |

- 运行期日志直接印证：`C:\Users\...\state\native-logs\keephead-28456-25daf71cc50.err:380` 输出的正是**旧格式串**
  ```
  [qlh_keep_head] ctx ready: n_ctx=4096 n_batch=4096 (requested n_ctx=4096 n_batch=4096 n_seq_max=1)
  ```
  同一份日志 `:327` 打出 `n_ubatch = 4096` `[代码事实]`。
- ⚠️ 该日志来自一次 **8 层**工件的运行（`:kv 5: qwen35.block_count u32 = 8`、`print_info: file size = 1.53 GiB`），不是当前的 `mid8-24`（16 层）。但"生产 DLL 是旧版"的结论**不依赖层数** —— 它由 `strings` 比对格式串 + 缺符号两条独立证据支撑。下文引用的该日志数值（compute buffer 4168 MiB、52 个 repack 张量、KV 32 MiB）都是 **8 层段**的量；16 层段的 compute buffer 只会更大，这反而强化了 ubatch 封顶的必要性。
- 结论：源码里 10-25 的 `PERF-NATIVE-COST-01`（`QLH_LAYER_N_BATCH` 封顶 ubatch 到 512，`qlh_keep_head.c:129-133`）和 10-26 的 `PERF-SKIP-LOGITS-01`（`qlh_keep_head.c:610`）**都只在源码里，不在跑着的二进制里**。

**发现 2：`QLH_LAYER_SKIP_LOGITS` 在 `.env` 里根本不存在 ⇒ 中间段每 token 白读 834 MB `output.weight`。**

- `grep SKIP` 全 `.env` 无结果 `[代码事实]`；该开关缺省为关（`llama_keep_head.py:101-104`，`engine_host.py:2996` 明写"缺省**关**"）。
- 既有实测收益（`local_docs/perf-skiplogits-20261026/`，同 shim 版本、隐藏态 digest 完全一致 `c97ab4ca…`）`[代码事实]`：
  - 20 层中间段：`122.615 → 102.584 ms/步` = **1.195×**；复测 `120.249 → 106.740` = 1.127×。
  - **`hidden_digest` 前后逐位相同**、`tail_token` 相同 ⇒ 纯收益、无数值风险。
- 这是当前唯一"代码已写、实测已验证、只差一次重新构建 + 一个环境变量"的收益。

**发现 3：`n_batch` / `n_ctx` 的实际值与 `.env` 不一致，且代码里有两个**同名不同源**的构造点。**

| 项 | `.env` | 运行日志 | 代码默认 |
| --- | --- | --- | --- |
| `n_ctx` | `QLH_LAYER_N_CTX=8192`（`.env:67`） | **4096** | `engine_host.py:3118` 默认 4096 |
| `n_batch` | `QLH_LAYER_N_BATCH=4096`（`.env:66`） | 4096 | `engine_host.py:3119` 默认 2048 |

- `n_ctx=4096` 与 `.env` 的 8192 不符、却恰好等于 `engine_host.py:3118` 的默认值 ⇒ 该 worker 的 `environment` 里**没有** `.env` 的值（`engine_host.py:3106` 只 `environment.get(name,"")`）`[代码事实]`。
- 另一构造点 `llama_engine.py:2510-2517` 读的是 **`QLH_KEEP_HEAD_N_BATCH`**（另一个名字！），`.env` 里也没有 ⇒ 落到 `spec.batch_size` 或默认 2048。
- ⇒ **改 `.env` 不一定生效**，这是下次调参必须先确认的第一件事。

---

## 1. 当前实际生效参数（按设备）

### 1.1 参数传递链（三台设备共用）

```
.env (QLH_LAYER_*)                         硬编码默认
        ↓                                       ↓
engine_host.py:3118-3121  _env_int(...)  ←→  无
        ↓
llama_keep_head.py:407  lib.qlh_kh_load(path, n_ctx, n_threads, n_batch, n_seq_max, mode, cut)
        ↓
qlh_keep_head.c:109-137  llama_context_default_params() + 覆盖
```

`[代码事实]`：`llama_keep_head.py:406-410` 的 ctypes 调用只有 7 个语义参数，**其它一切（mmap/mlock/flash_attn/type_k/type_v/defrag/op_offload）都走上游默认值**，shim 侧没有任何覆盖。

### 1.2 master#cpu `[8,24)`（3.47 GB 工件，16 层）

| 参数 | 当前实际值 | 来源 | 证据 |
| --- | --- | --- | --- |
| `n_threads` | **8** | `.env:54` `QLH_LAYER_THREADS=8` | `engine_host.py:3120` |
| `n_threads_batch` | **8**（强制等于 `n_threads`） | 硬编码 | `qlh_keep_head.c:135-136` |
| `n_batch` | **4096** | `.env:66` / 默认不一致（见发现 3） | 日志 `:326` |
| `n_ubatch` | **4096**（= n_batch，**未封顶 512**） | 旧 DLL 无封顶逻辑 | 日志 `:327`；`qlh_keep_head.c:129-133` 未生效 |
| `n_ctx` | **4096**（`.env` 写 8192，未生效） | `engine_host.py:3118` 默认 | 日志 `:324` |
| `n_seq_max` | **1** | `.env` 无此键 → 默认 | `engine_host.py:3121` |
| `n_outputs_max` | **4096**（= n_batch） | 上游默认 0→n_batch | 日志 `:334` |
| `use_mmap` | **true** | 上游默认，shim 未改 | `llama-model.cpp:2313`；日志 `:266` `CPU_Mapped 1570.40 MiB` |
| `use_mlock` | **false** | 上游默认，shim 未改 | `llama-model.cpp:2315` |
| `use_extra_bufts`（repack） | **true，已生效** | 上游默认 | `llama-model.cpp:2317`；日志 `:268-320` 52 个张量 `q4_K_8x8`（8 层段；16 层段按比例 ≈100 个） |
| `flash_attn_type` | **AUTO → 实际 enabled** | 上游默认 AUTO | `llama-context.cpp:3471`；日志 `:367` `Flash Attention was auto, set to enabled` |
| `type_k` / `type_v` | **F16 / F16** | 上游默认 | `llama-context.cpp:3482-3483`；日志 `:347` |
| `defrag_thold` | **-1.0（禁用）** | 上游默认，且标 `[DEPRECATED]` | `llama-context.cpp:3479`、`llama.h:363` |
| `offload_kqv` / `op_offload` | true / true | 上游默认，CPU 单后端无对象 | `llama-context.cpp:3487,3489` |
| `no_perf` | true | 上游默认 | `llama-context.cpp:3488` |
| `n_gpu_layers` | **0**（硬编码） | shim 显式设 | `qlh_keep_head.c:101` |
| `QLH_LAYER_SKIP_LOGITS` | **未设 ⇒ 关** | 未配置 | 发现 2 |
| Compute buffer | **4168.16 MiB**（8 层段实测；16 层段更大） | 4096 ubatch 撑出 | 日志 `:389`、`:363` `max_nodes = 163840` |

### 1.3 构建档位（master 与 Surface 共用同一份 `build-cpu`）

`[代码事实]`，来自 `build/keephead/build-cpu/CMakeCache.txt`：

| 开关 | 值 | 说明 |
| --- | --- | --- |
| `GGML_NATIVE` | **OFF** | 与已知事实一致 |
| `GGML_AVX` / `GGML_AVX2` / `GGML_FMA` / `GGML_F16C` / `GGML_BMI2` / `GGML_SSE42` | **ON** | AVX2 基线 |
| `GGML_AVX_VNNI` | **OFF** | ← **Raptor Lake 支持、但没开** |
| `GGML_AVX512` / `_VNNI` / `_VBMI` / `_BF16` | OFF | i9-13900H **硬件不支持** AVX-512（实测 `Avx512F.IsSupported=False`） |
| `GGML_CPU_REPACK` | **ON** | 已开 |
| `GGML_OPENMP` | **ON** | 线程走 OpenMP + 内部线程池 |
| `GGML_LLAMAFILE` | ON | sgemm（无 AVX512 时收益有限） |
| `GGML_CPU_ALL_VARIANTS` | OFF | 无多 ISA 运行时分发 |
| `CMAKE_BUILD_TYPE` | Release，`-O3 -DNDEBUG` | 正常 |
| `GGML_CPU_KLEIDIAI` | OFF | ARM 专用，与 PC 无关 |

`build_keep_head_cpu.ps1:19` 已经预留了 `-IsaProfile avx2_vnni` 这一档（`:253` 会置 `GGML_AVX_VNNI=ON`），但**默认是 `avx2`**，从未构建过 VNNI 版本 `[代码事实]`。

### 1.4 Surface `[28,32)`（1.98 GB 工件，4 层 + lm_head）

| 参数 | 当前实际值 | 来源 |
| --- | --- | --- |
| `n_threads` / `n_threads_batch` | **8 / 8** | 代码默认 8（`llama_keep_head.py:246`、`engine_host.py:3120`）；`.env` 侧值**未找到**（仓库内只有 master 一份 `.env`） |
| `n_batch` / `n_ubatch` | 同 master（4096 / 4096，旧 DLL） | 同一份 DLL |
| `use_mmap` / `use_mlock` | true / false | 同上 |
| ISA | **AVX2 基线**，无 VNNI、无 AVX-512 | 同一份 `build-cpu` |
| `QLH_LAYER_SKIP_LOGITS` | **不可用**（末段要 argmax，`qlh_keep_head.c:64-66` 明确 argmax 入口恒开 logits） | — |

> 实测 2/4/8 线程 = 139.85/113.43/**112.86** ms/token，8 已是最优 ⇒ 线程这项**没有剩余空间**。

### 1.5 y700 `[24,28)`（Android，Kotlin worker → JNI）

`[代码事实]`，来自 `android/app/src/main/cpp/qlh_llama_jni.cpp`：

| 参数 | 当前实际值 | 证据 |
| --- | --- | --- |
| `n_threads` / `n_threads_batch` | `clamp(核数-2, 2, 8)`，**同一个值** | `:110-120`、`:288-289` |
| `n_batch` | `max(2048, min(n_ctx, 8192))` | `:284-286` |
| `n_ubatch` | **`min(512, batch_limit)`（这里已经封顶 512）** | `:287` |
| `no_perf` | true | `:290` |
| `embeddings` | = `j_extract_hidden`（中间段 true） | `:298` |
| `use_mmap` / `use_mlock` | true / false（默认） | 未覆盖 `llama_model_default_params()` `:257` |
| `n_gpu_layers` | `max(-1, j_gpu_layers)`，无 GPU 后端时强制 0 | `:258-262` |
| `flash_attn_type` | AUTO（未覆盖） | 未设 `ctx_params.flash_attn_type` |
| `type_k` / `type_v` | F16 / F16 | 未覆盖 |
| ISA | `armv8.6-a+dotprod+fp16+i8mm` | `build.gradle.kts:175` |
| `GGML_NATIVE` | **OFF**（交叉编译必需） | `CMakeLists.txt:45` |
| `GGML_OPENMP` | **OFF** | `CMakeLists.txt:46` |
| `GGML_CPU_KLEIDIAI` | **OFF** | `CMakeLists.txt:47` |
| `batch.logits[i]` | **`all_tokens!=NULL 或 out_hidden!=NULL` ⇒ 每个 token 都标 1** | `:1224`、`:1537` |

> **y700 的 `n_ubatch` 已经是对的**（512 封顶），PC 侧才是漏的那台。

---

## 2. 核心交付：当前值 vs 建议值 vs 预期收益

### 2.1 master#cpu `[8,24)`

| 参数 | 当前值 | 建议值 | 预期收益 | 依据 |
| --- | --- | --- | --- | --- |
| **shim DLL** | 10-21 旧版 | 重新构建（10-26 源码） | **前置条件**，下面两项的前提 | 发现 1 |
| **`QLH_LAYER_SKIP_LOGITS`** | 未设（关） | **`1`** | **1.13–1.20×**（实测 122.6→102.6 / 120.2→106.7），hidden 逐位不变 | `local_docs/perf-skiplogits-20261026/` |
| **`n_ubatch`** | 4096（未封顶） | **512** | compute buffer 4.17 GB → ≈0.5 GB；decode 本身 ≈1.0×，但省内存换页 | `qlh_keep_head.c:112-134` |
| `GGML_AVX_VNNI` | OFF | **ON**（`-IsaProfile avx2_vnni`） | decode **1.0–1.2×**、prefill 1.1–1.8× | `build_keep_head_cpu.ps1:253`；Raptor Lake 支持 |
| `n_threads` | 8 | **6**（先标定 6/7/8 三点） | 未知，可能 ±5% | i9-13900H = 6 P + 8 E；8 线程时 2 个线程必落 E 核 |
| CPU 亲和性 | 无（默认全核，`ggml-cpu.c:2471` 的 `SetThreadAffinityMask` 未被使用） | 进程绑到 P 核 | 未知，配合上一项一起测 | `ggml-cpu.c:2471-2500` |
| `n_ctx` | 4096（`.env` 8192 未生效） | 保持 4096；**修 `.env` 或代码，别让配置说谎** | 0（现在是对的） | 发现 3 |
| `use_mmap` / `use_mlock` | true / false | **不动** | 0 | 见 §4 |
| `type_k` / `type_v` | F16 / F16 | **不动** | 负收益（已证） | §4 |
| `flash_attn` | AUTO→enabled | **不动（已开）** | 0 | 日志 `:367` |
| repack | ON | **不动（已开）** | 0 | 日志 `:268` |
| `defrag` | 禁用 | **不动（已 DEPRECATED）** | 0 | `llama.h:363` |

**叠加估算**：skip-logits 1.19× × VNNI（decode 侧保守 1.0×）= **104.08 → ≈86–92 ms/token**。若 VNNI 在 GEMV 上有 5% 作用，则 ≈82 ms。

### 2.2 Surface `[28,32)`

| 参数 | 当前值 | 建议值 | 预期收益 | 依据 |
| --- | --- | --- | --- | --- |
| `n_threads` | 8 | **8（不动）** | 0，已最优 | 实测 2/4/8 = 139.85/113.43/112.86 |
| `n_ubatch` | 4096（旧 DLL） | **512** | 省内存（1.98 GB 工件在 15.6 GB 平板上，GB 级 compute buffer 会挤页缓存） | 同上 |
| **`output.weight` 位宽** | **Q6_K，834 MB** | **Q4_K / Q5_K（≈500/630 MB）** | **1.15–1.20×** | 见下方算术 |
| `GGML_AVX512` / `_VNNI` | OFF | **不建议**（可出但只为 prefill） | decode ≈1.0× | PR #30036；见 §4 |
| `n_ctx` | 4096 | **保持 4096，不要 8192** | 避免 1.31× 变慢 | 既有实测 |
| `use_mmap` / `mlock` | true / false | **不动** | 0 | §4 |
| 电源计划 | 未知 | 处理器最大状态 99%（抑制 turbo 抖动） | 稳定 > 提速 | 15W 平板 |

**`output.weight` 位宽这一项的算术（`[计算]`）**：
Surface 段 1.98 GB 构成 = 4 层 ≈ 540 MB + `output` Q6_K 834 MB + `token_embd` Q4_K 572 MB。末段的 decode 必须读 4 层 + `output`（出 argmax），`token_embd` 在 embd 注入路径下不参与查表。⇒ 每 token 真实流量 ≈ 1.41 GB，其中 **59% 是 `output.weight`**。把它降到 Q4_K（834→500 MB）⇒ 1.41→1.07 GB ⇒ 理论 **1.31×**，按实测有效带宽打折取 1.15–1.20×。
这是**只改工件、不改架构、不改代码**的一步，且 `lm_head` 单独降位宽对质量的影响远小于整模降位宽（输出层只做一次线性映射，误差不进 32 层残差流）。

### 2.3 y700 `[24,28)`

| 参数 | 当前值 | 建议值 | 预期收益 | 依据 |
| --- | --- | --- | --- | --- |
| `n_threads` | `clamp(核-2,2,8)` | **实机标定 4/6/8**（`-2` 是拍脑袋的） | 未知，类似 Surface 的 1.2× 空间 | `qlh_llama_jni.cpp:119` |
| `n_ubatch` | 512 | **不动（已经对）** | 0 | `:287` |
| `batch.logits[i]` | **中间段每 token 都标 1** | 按"只要 hidden"改成全 0（同 PERF-SKIP-LOGITS-01） | **1.1–1.2×**（若工件含 `output.weight`） | `:1224`、`:1537` |
| `GGML_CPU_KLEIDIAI` | OFF | ON（试） | decode 未知，prefill 有收益 | `CMakeLists.txt:47` |
| `GGML_OPENMP` | OFF | 保持 OFF（Android bionic 上 OpenMP 收益差，且打包体积） | 0 | `CMakeLists.txt:46` |
| `armv8.6-a+dotprod+fp16+i8mm` | 已开 | **不动** | 0 | `build.gradle.kts:175` |
| `use_mmap` | true | 不动 | 0 | — |

> y700 至今"未测 decode"，**任何参数建议都只能先测基线**。它的 `batch.logits` 是全表里第三处可验证的确定收益点。

---

## 3. 哪些是**真正没开的优化**（不改架构、能开就有收益）

按"确定性 × 收益"排序：

1. **重新构建 shim 并打开 `QLH_LAYER_SKIP_LOGITS=1`** — 唯一有 A/B 实测证据的一项，1.13–1.20×，hidden 逐位不变。当前生产 DLL 里连这个符号都没有。**成本：一次 `build_keep_head_cpu.ps1` + 一行 env。**
2. **`n_ubatch` 封顶 512** — 源码已写（10-25），二进制没有。省 3.6 GB compute buffer 预留，对 15.6 GB 的 Surface 是实质性的页缓存改善。
3. **`GGML_AVX_VNNI=ON` 构建** — Raptor Lake（master）确实支持 AVX-VNNI，`build_keep_head_cpu.ps1:19` 的 profile 已备好但从未构建。decode 预期小（PR #30036 的 tg 列只有 1.15–1.21×），但**不花额外成本**，且对 prefill（2644 token 那一段）有 1.1–1.8×。
4. **Surface 的 `output.weight` 降到 Q4_K/Q5_K** — 只改工件。它是该段 59% 的每 token 流量。
5. **y700 的 `batch.logits` 按"只要 hidden"关掉** — Android 已有 `out_hidden != NULL` 这个条件（`qlh_llama_jni.cpp:1224`），但方向反了：条件成立时**全标 1**而不是全标 0。
6. **修 `n_ctx` 的配置漂移** — `.env:67` 写 8192、实际跑 4096。不修的话，以后任何 `.env` 调参都可能白改；修的过程也会暴露 `QLH_LAYER_N_BATCH` vs `QLH_KEEP_HEAD_N_BATCH` 两个名字并存的问题（`engine_host.py:3119` vs `llama_engine.py:2516`）。
7. **i9-13900H 的大小核线程标定（6 vs 8）** — 8 线程 = 6 P + 2 E，两个线程必然落 E 核、且和 P 核同步在同一个 OpenMP barrier 上 ⇒ 每步等最慢线程。这一项**没有实测数据**，必须先测 6/7/8 三点再决定，不要直接改。

---

## 4. 哪些看起来能优化但其实没用（下次别再查）

以下逐条给出"为什么排掉"，理由都建立在已确认的事实上，**不需要重测**。

| 项 | 为什么没用 |
| --- | --- |
| **KV cache 量化（q8_0/q4_0）** | 8 层段 KV = 32 MiB（日志 `:346-347`），16 层段约 64 MiB，占 3.47 GB 工件的 **≈1.8%**。量化最多省一半，却被 CPU 侧的反量化开销吃掉 —— 本仓已实测为**负收益**。 |
| **`use_mmap` 关掉** | 日志 `:266` `CPU_Mapped model buffer size = 1570.40 MiB` 说明走的是页缓存复用，不是每 token 重新缺页；且 `[8,24)` 工件 3.3 GB 常驻内存后 mmap 就是纯读。关掉只会多一次 3.3 GB 拷贝 + 双份内存。 |
| **`use_mlock`** | 同机可用内存远大于工件（master 段 3.47 GB / DDR5 笔记本），没有换页可防；mlock 反而抢走页缓存给别的进程。 |
| **AVX-512 构建（master）** | **硬件不支持**：实测 `[System.Runtime.Intrinsics.X86.Avx512F]::IsSupported = False`（Raptor Lake 消费级已熔断 AVX-512）。`CMakeCache` 里 `GGML_AVX512=OFF` 是对的，不是漏。 |
| **AVX-512/VNNI 构建（Surface）** | Ice Lake 支持，但 PR #30036 的 tg 列是 **1.00×**；Surface 段 decode 是 GEMV（1 行输出），VNNI 的指令级收益进不到这条路径。**只对 prefill 有意义，而 prefill 在关键路径外。** |
| **`n_threads_batch` 分开设** | `qlh_keep_head.c:135-136` 把两者绑成同一个值。decode 时 batch=1 只走 `n_threads`；`n_threads_batch` 只在 prefill 生效。分开设**对 104 ms/token 无影响**。 |
| **加线程（>8）** | 已实测 4/8/12/16 = 109.75/118.39/128.32/146.42，单调变差。 |
| **`--no-kv-offload` / `op_offload` / `offload_kqv`** | 单 CPU 后端（日志 `:361` `backend_ptrs.size() = 1`），这些开关没有可卸载的对象。 |
| **`defrag`** | 上游 `llama.h:363` 已标 `[DEPRECATED]`，默认 -1。 |
| **NUMA / 大页 / `--cpu-affinity`** | Windows 单 NUMA 节点；`ggml-cpu.c:615-628` 的 NUMA 探测是 `#if defined(__gnu_linux__)`。 |
| **换 Q4_0 格式** | 需重新裁层 + 重过数值门；tg 列 Q4_0 10.97 vs Q4_K_S 13.24，**更慢**。 |
| **IQ2/IQ3 低比特** | CPU 上 pp 慢 3–5×，tg 不如同字节 Q2_K。 |

### 4.1 顺带修正一条既有判据（重要）

既有调研说 "master#cpu 19.7 GB/s，是 copy 上限 30.6 的 64% ⇒ 没到带宽极限"。这个比值**低估了实际带宽**，因为 `copy` 是读+写各一半，而 decode 是**纯读流**。

用本文的数据反算 `[计算]`：
- skip-logits 省下的 `output.weight` 是 **834 MB / 20 ms = 41.7 GB/s**（20 层工件，122.6→102.6）。
- master 段 decode 104.08 ms 读 16 层 2060 MB + output 834 MB = 2894 MB ⇒ **27.8 GB/s**。

41.7 GB/s 的纯读流在双通道 DDR5 上完全可达，而 27.8 GB/s > "copy 上限 30.6 × 64% = 19.7"。⇒ **CPU 段其实已经贴着纯读带宽跑，不是"算力/解量化有余量"。** 这反过来强化了唯一有效的方向：**减少每 token 字节数**（skip-logits、降 `output` 位宽、减层），而不是任何"提高算力利用率"的参数。

`[推测]`：这条与既有结论冲突，建议用一次 `q35-9b-local-mid8-24.gguf` 的 skip-logits A/B（16 层、同 shim）直接验；若实测落在 86–92 ms 区间，则 `output.weight` 确实被完整读取，27.8 GB/s 成立。

---

## 5. "13 秒固定开销"的独立线索

现象：`/api/chat` 每请求约 13 s 固定开销、`loaded=False`、128 token 稳态 598 ms/token。

**线索 A（最强）：keep-head 上游的复用键包含 `prepare_identity`，它一变就 `close()` + 全量重建。**

`engine_host.py:3068` 构造复用键：
```python
binding = (executor_id, prepare_identity, dict(identity), tuple(span), model_path, version)
```
其中 `prepare_identity` 是 `stage_identity(identity)`（`:2874`），含 **`operation_id` / `assignment_id` / `generation`**（`worker_prepare_contract.py:21`）。`:3069` 一旦不相等：
```
3090  self._layer_upstream_binding = None
...  close() → KeepHeadUpstream(...) 重新 llama_model_load_from_file
```
⇒ 只要 operation/assignment 变了（换会话、重下发 assignment、recovery 后 generation+1），**3.47 GB 工件就重新 mmap + repack**。日志显示 repack 产生 1264 MiB 匿名 buffer（`:267`），这是纯 CPU 拷贝 —— 数秒级。

**线索 B：`sched_reserve` 要做两次 4.17 GB 预留。**

日志 `:376` 与 `:389` 各一次 `CPU compute buffer size = 4104.16 / 4168.16 MiB`，`:379`+`:392` 的 `reserve took 22.74 / 30.55 ms`（偏移本身不快），但**内存压力和随之而来的页回收**在 15.6 GB 的 Surface 上是秒级。⇒ `n_ubatch` 封顶 512 直接压掉这个量级（见 §3 第 2 项）。

**线索 C：`model_host.model_loaded = False` 时的自愈路径每次都跑。**

`api_server.py:4138-4148`：
```python
if (model_host.model_loaded and model_manager.is_loaded): return
...
_auto_load_default_model()
```
`_auto_load_default_model`（`:4151`）是完整加载流程。而 `loaded=False` 是现象而非原因 —— 需查 `model_host.model_loaded` 为何在 pipeline 模式下始终为 False：`:4200`、`:4314` 只在特定分支置 True，`:2617-2620` 的置位条件是 `model_manager.is_loaded and not _pipeline_model_is_prepared()`。**pipeline 模式下 `is_pipeline_prepared=True` ⇒ `model_loaded` 永远是 False** `[代码事实]`，于是每次请求都走 `_pipeline_model_is_prepared()` 早退（`:4131-4137`），不算重复加载 —— 但**指标上永远显示 loaded=False**，容易把"pipeline 已就绪"误诊成"没加载"。

**线索 D：`_supplied_layer_caches` 按 workflow 清空。**

`engine_host.py:3232-3234`：`workflow_id` 一变就 `self._supplied_layer_caches = {}`。每请求新 workflow ⇒ 段内 KV 全清 ⇒ 本次 prefill 必须从 pos=0 重算 2644 token。4 层段 prefill 在 Surface 上就是秒级。**这与 598 ms/token 的"稳态"并不矛盾**：稳态是 decode，13 s 是 prefill + 重建。

**线索 E（可排除）：摘要校验。**

`engine_host.py:3077` `cached_sha256(artifact, ...)` 走 `model_supply_digest_cache.py:196-220`，命中缓存后只做两次 `file_stamp`（含 `_change_time` 一次 `CreateFileW`）。3.47 GB 的 sha256 若每请求重算会是 2–4 s，但**缓存命中时不是**。⇒ 先确认 `file-digests.json` 是否存在且 stamp 未变；若 stamp 每次变（硬链接/复制），这里就会退化成每请求 3 GB 哈希。`[推测]`，值得在 13 s 的火焰图上确认。

---

## 6. 下一步动作清单（按性价比）

| # | 动作 | 成本 | 预期 |
| --- | --- | --- | --- |
| 1 | `build_keep_head_cpu.ps1` 重建 shim（10-26 源码），改 `.env:53` 指向新目录 | 10 min | 解锁 skip-logits + ubatch 封顶 |
| 2 | `.env` 加 `QLH_LAYER_SKIP_LOGITS=1`，A/B 复测 master 段 | 5 min | **1.13–1.20×** |
| 3 | 确认 `n_ubatch` 真的变成 512（看新 DLL 的 `ctx ready` 是否打印 `n_ubatch=512`） | 2 min | 省 3.6 GB 预留 |
| 4 | master 段线程标定 6/7/8 三点 | 20 min | 未知，±5% |
| 5 | 出 Surface 用 `[28,32)` 工件，`output.weight` 压到 Q4_K | 数小时 | 1.15–1.20× |
| 6 | y700 测基线，再改 `batch.logits` | 数小时 | 未知 + 1.1–1.2× |
| 7 | 查 `prepare_identity.operation_id` 是否每请求变化 ⇒ 13 s 的根因 | 读日志 | 可能直接消掉 13 s |

> 一句话：**当前最大的浪费不是"参数没调好"，而是"改好的代码没进二进制"。** 先重建 DLL、再开一个环境变量，master 段预期从 104.08 掉到 86–92 ms/token，成本不到半小时。
