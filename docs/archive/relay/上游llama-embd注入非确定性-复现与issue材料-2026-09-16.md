# 上游 llama.cpp `embd` 注入路径非确定性：复现、定位与 issue 材料（2026-09-16）

> ## ⚠️ 状态更新（2026-09-16 晚）：**根因已确定并已修复** —— 本文 §5–§12 的结论**全部作废**
>
> **上游给出的根因**（[issue #28963](https://github.com/ggml-org/llama.cpp/issues/28963) 回复）：
> `llama_batch_allocr::ubatch_add` 对**调用方 `pos` 数组的堆越界读** —— M-RoPE 模型的 `embd` 批次会读
> `batch.pos[0 .. n_pos_per_embd*n_tokens)`，而 `llama_batch_init` 只分配 `n_tokens` 个条目
> （valgrind：`Invalid read of size 4 … 0 bytes after a block of size 20 alloc'd`）。
>
> **本仓库独立复现并验证**（Windows 11 / MSYS2 UCRT64 g++ 15.2.0 / Qwen3.5-2B f16 GGUF / CPU 后端；
> `qwen35.rope.dimension_sections` 为 4 段，故 `n_pos_per_embd = 4`）。按上游的**调用方修法**
> （`embd` 批次的 `pos` 按 `4*n_tokens` 分配、每段都填 `pos[j*n_tokens+i] = i`）后：
>
> | 实验 | 修复前 | 修复后 |
> | --- | --- | --- |
> | `embd` 同输入连跑两次（141 位置） | argmax `126/141`、cosine 0.85–0.99 | **`141/141`，cosine min = 1.000000** |
> | `embd` vs token 路径（同一 141-token prompt） | argmax `132/141` | **`141/141`，cosine min = 0.999999** |
> | L→L 层接力 141 步贪心生成 | 第 46–67 步分叉（逐次不同） | **141/141，可重复** |
>
> **所以**：所谓"`embd` 路径非确定性"由 **`pos` 越界读**造成，**不是** llama.cpp 的架构限制，
> 也**不是**跨框架接力的固有缺陷。本文 **§5（定位结论）、§6（已排除项）、§7（候选假设）、
> §8（绕开口径）、§9（pin 与打补丁）、§11、§12 全部建立在错误前提上，仅作历史记录保留**；
> 正确结论见 **§13**。
>
> 修复补丁（**仅位于被忽略的实验 fork，主仓运行时不依赖**）：
> `build/cross-framework-layer-poc/patches/embd-pos-overread-fix-2026-09-16.patch`。

> 状态（历史记录）：**issue 材料就绪（根因未坐实）** —— 已被上面的更新取代。
>
> 适用范围（历史）：llama.cpp **CPU 后端**在 `llama_batch.embd`（向量注入）输入路径上的**非确定性**；QLH 层间接力的可复现性口径与绕开方案。
>
> 关联：[跨框架层接力重启评估](跨框架层接力重启评估-2026-09-15.md) §7.10/§7.11 · [基线重写方案](../runtime/基线重写方案-2026-09-16.md) §1.4

## 0. 结论摘要（**已作废 —— 见文首横幅与 §13**）

> 以下五条是 2026-09-16 早期结论，其共同前提（"这是 llama.cpp 固有的非确定性"）**已被推翻**。
> 保留原文仅为追溯，**任何后续决策不得引用本节**。
>
> - ~~**现象可稳定复现**：同一进程、同一 context、同一输入、`llama_memory_clear` 之后重复 decode，`llama_batch.embd` 路径的结果**逐位不等**（cosine 0.984–0.9999）；而 **token 路径逐位完全一致**。~~
> - ~~**已排除 7 类成因**（见 §6）：未写入的 `inp->tokens`、多线程归约、KV 容量/未使用区域、残留 KV 内容、ggml 融合算子、Flash Attention、CPU repack。~~
> - ~~**已定位**：非确定性**只在 attention 层出现**，**分叉精确始于第一个 full-attention 层**，且**需要 >1 个 token**。~~
> - ~~**未找到可直接修改的代码点** → 本轮**不提供上游修复补丁**。~~
> - ~~**对 QLH 的口径**：判据用 argmax、接力实验固定单进程、跨进程"行为等价"与"逐位复现"分开声明。~~
>
> **实际结论**：以上现象的成因是调用方 `pos` 数组被越界读；**按上游修法修好后，`embd` 路径完全逐位确定**
> （cosine min = 1.000000），上述"排除项/定位/绕开口径"均不再需要。

## 1. 现象（What）

```text
同一进程 / 同一 context / 同一输入（5×2048 f32）/ 每次 decode 前 llama_memory_clear(mem, true)

#[1] token 路径（llama_batch.token）      → run0..run5 逐位完全一致      （cosine = 1.00000000）
#[2] embd  路径（llama_batch.embd），1 token → 逐位完全一致
#[3] embd  路径，2 / 3 / 5 token          → 逐位不等（cosine 0.984 – 0.9999）
     argmax 在所有情形下都稳定（= 11751），top-10 高度重合
```

- 抖动**不是**"每次完全随机"：多次运行的结果落在**少数几个值**上并有循环模式（例：`run0==run2`、`run1==run3==run4`）。
- 跨进程（同一命令独立跑 3 个进程）同样不一致（pairwise cosine 0.9946–0.9971）。
- 同进程内**两个不同 context** 各 decode 一次，也不一致（cosine 0.9885）。

## 2. 环境（Environment）

| 项 | 值 |
| --- | --- |
| commit | `6f04274cc145cf76aafc46eb8c1ea054ca221368`（2026-08-14；含 PR #27073 hidden-state 提取） |
| 版本串 | `version: 0.1.0-dev (build 10436, commit 6f04274cc)` / `built with GNU 15.2.0 for Windows AMD64`（取自同一次构建的 `llama-server --version`；`git describe` = `b10431-5-g6f04274cc`） |
| 构建 | `cmake -B build-cpu -G "MinGW Makefiles" -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF` + `mingw32-make -j4`（MSYS2 **UCRT64** g++） |
| 运行 | Windows（`windows/amd64`），**纯 CPU 后端**（`n_gpu_layers = 0`），`n_threads = n_threads_batch = 1`（除注明处） |
| 模型 | `Qwen3.5-2B`（`qwen35` 架构：混合 **SSM/Gated Delta Net** + **每 4 层一个 full attention**），f16 GGUF；另用"裁掉前 4 层"的同源模型 |
| 注入数据 | `5 × 2048` f32（来自原模型 layer 3 输出的真实 hidden） |
| 上下文参数 | `n_ctx = 512`、`n_batch = n_ubatch = 512`（另做 n_ctx 扫描，见 §6） |

## 3. 最小复现（How）

### 3.1 复现探针（自包含，直接编译）

```cpp
// embd-nondet-probe.cpp —— 同一 context 重复 decode 同一 embd，报告逐位一致性
// 编译：g++ -std=c++17 -O2 -I <llama.cpp>/include -I <llama.cpp>/ggml/include embd-nondet-probe.cpp \
//        <build>/src/libllama.a <build>/ggml/src/ggml.a <build>/ggml/src/ggml-cpu.a <build>/ggml/src/ggml-base.a \
//        -lws2_32 -lgomp -static-libgcc -static-libstdc++ -o probe.exe
// 用法：probe.exe <model.gguf> <embd.f32> [mode] [N] [threads]
//   mode: repeat（embd 重复）/ tokens（token 路径对照）
//   embd.f32 = N_tokens × n_embd 的 little-endian float32
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double cosine_of(const float * a, const float * b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) { dot += (double) a[i]*b[i]; na += (double) a[i]*a[i]; nb += (double) b[i]*b[i]; }
    return (na > 0 && nb > 0) ? dot / (std::sqrt(na)*std::sqrt(nb)) : 0.0;
}

int main(int argc, char ** argv) {
    const char * model_path = argv[1];
    const char * embd_path  = argv[2];
    const std::string mode  = argc > 3 ? argv[3] : "repeat";
    const int N = argc > 4 ? atoi(argv[4]) : 6;
    const int n_threads = argc > 5 ? atoi(argv[5]) : 1;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    const int n_embd  = llama_model_n_embd_inp(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<float> raw;
    { FILE * f = fopen(embd_path, "rb"); fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
      raw.resize((size_t) sz / 4); (void) fread(raw.data(), 4, raw.size(), f); fclose(f); }
    const int n_tokens = (int) (raw.size() / (size_t) n_embd);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = n_threads; cp.n_threads_batch = n_threads;   // 两者都要设：decode 走 n_threads_batch
    llama_context * ctx = llama_init_from_model(model, cp);

    std::vector<float> first; bool all_eq = true;
    for (int r = 0; r < N; ++r) {
        llama_memory_clear(llama_get_memory(ctx), true);         // 关键：每次都清 memory
        llama_batch b = (mode == "tokens")
            ? llama_batch_init(n_tokens, 0, 1)
            : llama_batch_init(n_tokens, n_embd, 1);
        for (int i = 0; i < n_tokens; ++i) {
            if (mode == "tokens") { int32_t t; memcpy(&t, &raw[(size_t) i * n_embd], 4); b.token[i] = t % n_vocab; }
            b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = (i == n_tokens - 1) ? 1 : 0;
        }
        b.n_tokens = n_tokens;
        if (mode != "tokens") { memcpy(b.embd, raw.data(), raw.size() * sizeof(float)); }
        llama_decode(ctx, b);
        llama_batch_free(b);
        const float * lg = llama_get_logits_ith(ctx, n_tokens - 1);
        if (r == 0) { first.assign(lg, lg + n_vocab); }
        else {
            if (memcmp(first.data(), lg, (size_t) n_vocab * 4) != 0) { all_eq = false; }
            printf("  run0 vs run%d: cosine=%.8f bitwise_equal=%s\n", r,
                   cosine_of(first.data(), lg, n_vocab), memcmp(first.data(), lg, (size_t) n_vocab * 4) == 0 ? "YES" : "NO");
        }
    }
    printf("VERDICT: %s\n", all_eq ? "BITWISE DETERMINISTIC" : "NON-DETERMINISTIC");
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
```

### 3.2 运行

```bash
# 2 个 token 就足够触发（1 个 token 不会）
python -c "import numpy as np; np.random.default_rng(0).standard_normal((5,2048), dtype=np.float32).tofile('embd5.f32')"
./probe.exe model.gguf embd5.f32 repeat 6 1     # → NON-DETERMINISTIC
head -c 8192 embd5.f32 > embd1.f32
./probe.exe model.gguf embd1.f32 repeat 6 1     # → BITWISE DETERMINISTIC
# 对照：token 路径（把同一文件前 4 字节当 int32 token 解释）
./probe.exe model.gguf embd5.f32 tokens 6 1    # → BITWISE DETERMINISTIC
```

> 说明：用**随机向量**即可复现，不依赖任何特定模型权重语义；本文档的数据则来自真实 hidden（5×2048）。

## 4. 实验矩阵与数据（Observed）

| # | 实验 | 结果 |
| --- | --- | --- |
| 1 | token 路径重复 6 次 | **逐位一致**（cosine = 1.00000000） |
| 2 | embd 注入 **1 token** 重复 6 次 | **逐位一致** |
| 3 | embd 注入 **2 token** 重复 6 次 | 逐位不等（cosine 0.9979–1.0，偶发整轮一致） |
| 4 | embd 注入 **3 token** 重复 4 次 | 逐位不等 |
| 5 | embd 注入 **5 token** 重复 6 次（连跑 3 次） | **逐位不等**（cosine 0.984–0.9999；argmax 恒为 11751） |
| 6 | 5 token + `n_threads = n_threads_batch = 1` | 仍不等 → 排除多线程 |
| 7 | 5 token + 显式清零未写入的 `inp->tokens`（改 `llm_graph_input_embd::set_input`） | 仍不等 → 排除该假设 |
| 8 | 5 token + 扫 `n_ctx ∈ {8,16,32,64,128,512}` | 全档不等 → 排除 KV 容量 |
| 9 | 5 token + `GGML_CPU_DISABLE_FUSION=1` | 仍不等 → 排除融合算子 |
| 10 | 5 token + `flash_attn_type = DISABLED` | 仍不等 → 排除 Flash Attention |
| 11 | 5 token + **全零输入** | **输出全零且逐位一致**（说明"非输入来源"只在非零输入时产生贡献） |
| 12 | 跨进程（同一命令独立 3 进程） | 不等（pairwise cosine 0.9946–0.9971） |
| 13 | 同进程、两个 context 各 decode 一次 | 不等（cosine 0.9885） |
| 14 | **逐层 × 逐位置**定位（`llama_get_hidden_state`） | **layer 0/1/2 逐位一致；layer 3 起不等** |
| 15 | 结果分布 | 落在少数几个值并循环（`run0==run2`、`run1==run3==run4` 等） |
| 16 | 重新编译同一源码后 | 现象依旧（连跑 3 次均不等；偶发某一轮"整轮一致"的巧合） |
| 17 | **增量式逐 token 注入**（每步 1 个 token、pos 递增 0..4） | **逐位不等**（cosine 0.994；两种模式各跑 2 轮均不等）→ **改变注入粒度无法恢复确定性** |
| 18 | 由此细化的定位 | **pos = 0 的单 token 确定；pos ≥ 1（有历史状态）的单 token 不确定** → 与"attention 在非空 KV 历史下的计算"一致 |

## 5. 定位结论（Where）

1. **只在 `llama_batch.embd` 路径**（`ubatch->token == NULL`）出现；token 路径完全确定。
2. **需要 >1 个 token**：1 token 确定，2 token 起不确定。
3. **分叉始于第一个 full-attention 层**：在本模型（每 4 层一个 full attention）上，**layer 0–2（SSM 层）逐位置逐位一致**，**layer 3（= 第一个 full-attention 层）起不一致**。→ 问题与 **attention 的跨位置机制**（KV / mask / 其 buffer）强相关。
4. 结果**仍由输入决定**（全零 → 全零且确定），说明不是"无中生有的随机注入"，而是某个**与输入相乘**的量在两次 decode 之间发生了变化。
5. 变化量**有限且循环**（少数状态），且**跨进程/跨 context 都复现** → 更像"**某个 buffer/状态在两次 decode 之间被复用且残留**"，而不是"每次全新的随机值"。

## 6. 已排除项（Ruled out，逐条含证据）

| 假设 | 实验 | 结果 |
| --- | --- | --- |
| 未写入的 `inp->tokens` 被读 | 在 `set_input` 中对该 input 显式清零（§4 #7） | **排除**（仍不等） |
| 多线程浮点归约顺序 | `n_threads = n_threads_batch = 1`（§4 #6） | **排除** |
| KV cache 容量 / 未使用区域 | 扫 `n_ctx` 8→512（§4 #8） | **排除** |
| 读到未初始化/残留的 KV 内容 | 全零输入 → 全零输出（§4 #11） | **排除**（若读到残留，零输入也会产生非零输出） |
| ggml 融合算子 | `GGML_CPU_DISABLE_FUSION=1`（§4 #9） | **排除** |
| CPU Flash Attention | `flash_attn_type = DISABLED`（§4 #10） | **排除** |
| CPU repack（权重重排） | 该构建日志明确 `cannot be used with preferred buffer type CPU_REPACK, using CPU instead`（未启用） | **不适用** |

## 7. 候选假设（未坐实）与建议方向

1. **attention 相关 buffer 的残留/复用**：某处 buffer 或状态在两次 decode 之间未被完全重置，其残留值只在非零输入时被"乘进"结果。建议方向：在 `llama_context` 里核对 embd 路径下 attention 的 mask/KV 相关 buffer 的初始化与复用（对比 token 路径）。
2. **embd 路径图/buffer 布局与 token 路径不同**导致的 SIMD 路径差异：异常地表现为"非确定"（更可能是某 buffer 未初始化，恰好在不同布局下被不同内容填充）。
3. 建议上游提供**最小复现脚本**并把判据定为"同进程 + 清 memory + 同输入 ⇒ 逐位一致"（本仓库可提供该探针）。

## 8. 对 QLH 的影响与绕开方案（Actionable）

**影响面**：任何用 `llama_batch.embd` 注入 hidden 的用法（跨框架层接力、多模态嵌入、**模型分片的 hidden 传递**）都**不能假设逐位可复现**。

**绕开口径（已写入项目文档）**：

1. **判据用 argmax / top-k 重合**，不用 cosine 阈值；
2. **接力实验固定在单进程内完成**（`llama-relay-gen` 即此形态）；
3. 跨进程形态（A/B 两节点）必须把"**行为等价**"与"**逐位复现**"分开声明；
4. **增量式逐 token 注入**（每步只注入 1 个新 token、pos 递增）：**已验证不成立** —— 与整段注入一样逐位不等（§4 #17，cosine 0.994）。结合"pos = 0 的单 token 确定、pos ≥ 1（有历史状态）的单 token 不确定"（§4 #18），**可复现性无法通过改变注入粒度获得**，只能依赖上面的 argmax / 单进程口径；
5. **提交给上游**：issue 正文（按 `019-bug-misc` 模板）与复现 gist 已备好，见 `docs/archive/relay/issue草稿-llama-cpp-embd非确定性-2026-09-16.md` —— 因上游有 AI 使用政策，**由人工审阅改写后用本人账号提交**。

## 9. pin 版本与打补丁流程

**当前处置（根因未坐实，无可改代码点）**：

- **pin**：以 `6f04274cc145cf76aafc46eb8c1ea054ca221368` 为实验基线版本（本机 clone：`build/cross-framework-layer-poc/llama.cpp`），并在引用了该行为的文档里标注本问题。
- **不改上游**（《基线重写方案》§5 排除项第 6 条）。

**若上游修复、或我们后续独立定位到代码点**：

```bash
# 1) 在 clone 里形成补丁
cd build/cross-framework-layer-poc/llama.cpp
git diff > ../patches/embd-nondeterminism-<日期>.patch      # patches/ 已存在且 gitignored

# 2) 应用（换机器/换版本时）
git apply --check ../patches/embd-nondeterminism-<日期>.patch && git apply ../patches/...

# 3) 重编译并复验（判据：§3.1 探针 repeat 模式 → VERDICT: BITWISE DETERMINISTIC）
cmake -B build-cpu -G "MinGW Makefiles" -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF
mingw32-make -C build-cpu -j4 llama
```

> 本轮尝试过的**未成功修复**：清零 `inp->tokens`（探针 A）—— 已回退，工作树恢复干净。

## 10. issue 正文草稿（英文，可直接粘贴）

```markdown
### Title
CPU backend: `llama_batch.embd` (embedding input) decoding is non-deterministic, while the token path is bitwise deterministic

### Summary
On the CPU backend, repeated identical decodes through the **embedding input path**
(`llama_batch.embd` set, `ubatch->token == NULL`) produce **different logits** — in the same
process, same context, same input, after `llama_memory_clear(mem, true)` between decodes.
The **token path** (`llama_batch.token`) is **bitwise identical** under exactly the same
conditions. `argmax` stays stable, so the practical impact is on *reproducibility*, not on
greedy generation.

### Environment
- commit: 6f04274cc145cf76aafc46eb8c1ea054ca221368
- build: `cmake -B build-cpu -G "MinGW Makefiles" -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF` (MSYS2 UCRT64 g++)
- runtime: Windows amd64, CPU backend, `n_gpu_layers = 0`, `n_threads = n_threads_batch = 1`
- model: Qwen3.5-2B (`qwen35`: hybrid SSM + full attention every 4 layers), f16 GGUF
- ctx: `n_ctx=512`, `n_batch=n_ubatch=512`

### Steps to reproduce
1. Prepare one `embd.f32` file = `n_tokens × n_embd` little-endian float32 (random values are fine).
2. Build and run the minimal probe in §3.1 of this report (also pasted below), mode `repeat`:
   it decodes the same batch N times in the **same context**, calling
   `llama_memory_clear(llama_get_memory(ctx), true)` before each decode, and reports
   `bitwise_equal` for each run vs run 0.
3. Also run mode `tokens` (same file interpreted as int32 tokens) as the control.

### Observed
| case | result |
| --- | --- |
| token path, 6 repeats | bitwise identical (cosine 1.00000000) |
| embd path, **1** token, 6 repeats | bitwise identical |
| embd path, **2/3/5** tokens, 6 repeats | **bitwise different** (cosine 0.984–0.9999; `argmax` constant) |
| embd path, all-zero input | all-zero output, bitwise identical |
| embd path, separate processes | different across processes (pairwise cosine 0.9946–0.9971) |

### Additional localization (this report)
- **Per-layer comparison** via `llama_get_hidden_state`: layers before the **first full-attention
  layer** are bitwise identical; divergence starts exactly at that layer (for this model:
  layers 0–2 identical, layer 3 differs). SSM layers are deterministic.
- **Requires >1 token**: 1 token is deterministic, 2 tokens already are not.
- Results fall into a **small set of states that recur** (e.g. run0==run2, run1==run3==run4),
  suggesting **buffer/state reuse** rather than fresh randomness.

### Ruled out
- unwritten `inp->tokens` (explicitly zeroing that input in `llm_graph_input_embd::set_input` does not help)
- multi-threading (`n_threads = n_threads_batch = 1`)
- KV capacity / unused KV region (swept `n_ctx` 8…512)
- reading uninitialized/stale KV content (all-zero input yields all-zero output)
- ggml fused ops (`GGML_CPU_DISABLE_FUSION=1`)
- CPU Flash Attention (`flash_attn_type = DISABLED`)

### Expected behavior
Same context + same input + cleared memory ⇒ **identical** output, as on the token path
(and as any user would expect for a pure-CPU deterministic backend).

### Extra context
This surfaced while prototyping **layer-pipeline relay** (compute layers `0..k` with one model
instance and inject the resulting hidden state into a truncated model that computes `k+1..N`
via `llama_batch.embd`). Greedy generation still matches the non-relay baseline 16/16 steps,
but bit-exact reproducibility across processes is not achievable today.
```

## 11. 补充实测（2026-09-16 晚，141-token 长批）：**argmax 判据在长序列上失效**

> 本节由主节点在本轮 XFRAME 控制组复跑中新增，**修正 §8 绕开口径第 1 条**。

**背景**：本报告 §1/§4 的非确定性观测基于 **5×2048 注入**（5 token），结论是"逐位不稳、但 **argmax 恒为 11751**"，据此给出绕开口径"判据用 argmax"。

**本轮实测（同一构建、同一 CPU 后端、同源模型 `qwen35-2b-f16.gguf`、141 tokens、`--dump-all` 逐位置 logits）**：

| # | 对比 | argmax 一致 | 说明 |
|---|---|---|---|
| 19 | token 路径 vs 既有 baseline | **141/141**（cosine min 0.999999） | token 路径**跨进程完全可复现**（复现 §4 #1） |
| 20 | **同 exe、同一 embd 输入、连续两次运行** | **126/141** | ⚠️ **argmax 本身不稳定（15 处不同）** |
| 21 | 补丁（embd 模式不构建 token 分支）后 embd vs baseline | **130/141 / 128/141**（两次） | 与 §9 已回退的"清零 `inp->tokens`"一样**未回到 141/141** |

**结论（新增）**：

1. **"argmax 稳定"只在短注入（≈1–5 token）成立**；在本模型的 **141-token 长批**上，embd 路径的 **argmax 序列也不可复现**。
2. 因此 **§8 第 1 条"判据用 argmax / top-k 重合，不用 cosine 阈值"在长序列场景下不成立** —— 逐位置 logits 的比较（argmax 或 top-k）都受同一抖动影响。
3. **可用的判据只剩"端到端的贪心生成序列一致"**（比较生成出的 token 序列本身，而非逐位置 logits）—— 这正是 **L→L 16/16** 所采用的口径：单个位置若抖动到改变 argmax，会立即改变生成序列从而被发现；而**未改变生成的抖动**本就不影响行为等价。
4. 对 XFRAME/D→L 的影响：**判据应从"逐位置 argmax（`per_token_argmax`）"改为"贪心生成序列一致"**，`RELAY_ACCEPTANCE` 的定义需同步修订（见《基线重写方案》§1.4、《跨框架层接力重启评估》）。
5. **§9 的"根因未坐实、无可改代码点"得到再次支持**：清零未写入 input（已回退）与"不构建未选中分支"（本轮编译通过）**两种补丁都未使结果回到 141/141**。

**复现命令（本轮）**：

```bash
# baseline（token 路径，完全可复现）
llama-relay-check.exe out/qwen35-2b-f16.gguf --tokens out/xframe-dl-long-prompt.txt \
  --n-ubatch 141 --dump-all --dump <A.f32>
# embd 路径：同一输入连跑两次
llama-relay-check.exe out/qwen35-2b-f16.gguf --embd <emb.f32> \
  --n-tokens 141 --n-ubatch 141 --dump-all --dump <B1.f32>
llama-relay-check.exe out/qwen35-2b-f16.gguf --embd <emb.f32> \
  --n-tokens 141 --n-ubatch 141 --dump-all --dump <B2.f32>
# 判定（compare_all.py = 逐位置 cosine + argmax 比较）
python compare_all.py <A.f32>  <B1.f32>   # → 130/141（与既有 132/141 同一噪声带）
python compare_all.py <B1.f32> <B2.f32>   # → 126/141（纯噪声，argmax 已不一致）
```

> 注：`--n-ubatch N` 会**同时**把 `n_batch` 设为 N，因此 `--n-ubatch 1` 会触发
> `GGML_ASSERT(n_tokens_all <= cparams.n_batch)`；该工具无法只改 `n_ubatch`。

## 12. 长序列补充实测（2026-09-16 晚，L→L 生成 141 步）：**问题从"不可复现"升级为"长序列生成实际出错"**

> 本节由主节点在按"改用生成序列判据"重测时新增。
> **重要：它需要修正本文 §0 与 §10 issue 草稿中的一句话** —— 原文称 "practical impact is on *reproducibility*, not on greedy generation"。

**载体**：`llama-relay-gen`（A=整模型 `qwen35-2b-f16.gguf` 算 layer 0..3 → B=裁前 4 层模型 `qwen35-2b-f16-cut.gguf` + `embd` 注入算 layer 4..N，逐 token 贪心）。prompt：`out/prompt-single.txt`。

**对照（同一工具、同一命令）**：

| 生成长度 | 运行 | baseline（token 路径） | relay（embd 注入） | 结果 |
|---|---|---|---|---|
| **16 步** | r1/r2/r3 | 3 次完全相同 | 3 次完全相同 | ✅ **16/16 一致**（复现既有结论） |
| **141 步** | L1/L2/L3/t1/plen/plong | 6 次全部生成完整 141 步且逐 token 相同 | 第 31–67 步之间分叉 | ✗ **分叉**（诊断见下表） |

**分叉诊断**（工具在分叉点自动 dump 了 A_gen 与 B 的整份 logits，逐份量化）：

| 运行 | 分叉步 | 分叉时前缀长度 | cosine(A_gen vs B) | A 的 argmax 在 B 中排名 | B 的 argmax 在 A 中排名 | top-5 交集 |
|---|---|---|---|---|---|---|
| L1 | 51 | 56 | 0.984432 | **2** | **2** | 4/5 |
| L2 | 46 | 50 | 0.998251 | **2** | **2** | 4/5 |
| L3 | 46 | 50 | 0.992647 | **2** | **2** | 4/5 |
| t1（`--threads 1`） | 46 | 50 | 0.998579 | **2** | **2** | 4/5 |
| plen | 67 | 71 | 0.987392 | **2** | **2** | 4/5 |
| plong（长 prompt 141 tok） | 31 | 171 | 0.997147 | **2** | **2** | **5/5** |

**关键事实**：

1. **6/6 次分叉的模式完全一致：`top-1 ↔ top-2` 互换** —— 双方的 argmax **互相都排在对方第 2 位**，cosine 高达 **0.984–0.999**，top-5 交集 **4–5/5**；
2. **`--threads 1` 与 `--threads 4` 的分叉步、cosine、top-5 全部一致**（threads 只改变抖动幅度：0.9926 vs 0.9986）→ 与线程数无关；
3. **baseline（token 路径）6 次全部完整生成且逐 token 相同** → 问题在 embd 注入路径，不在 token 路径；
4. **`RESULT:` 行里出现的 `223145984` / `211031952` / `0` 是工具 bug，不是模型输出** —— `relay-gen.cpp` 在分叉时 `break` 而未 `push_back`，故 `relay.size() == first_diverge`，`relay[first_diverge]` 属**越界读**（该 bug 已在实验 fork 修复）。分叉 token 以循环内 `step N: relay_argmax=...` 行为准，实测为 `804` / `198` 等**合法** id。

**结论（经上述诊断修正）**：

1. **embd 注入路径的精度与整模型非常接近**（cosine 0.99+、top-5 重合 4–5/5）；**分歧的本质是"前两名近乎并列时 argmax 翻转"**。一旦某步翻转，自回归会把差异放大成整条序列 —— 这解释了"分叉点随运行漂移（31–67 步）"。
2. **因此"端到端贪心序列逐 token 一致"这个判据对 embd 路径过于严格**：**连 L→L 自身都在 46 步内失败**。与之配套的合理判据是**容忍近并列**（见 `src/relay_contract.py` 的 `top_k_tolerant_argmax`）：relay 的 argmax 落在 baseline 的 **top-k**（实测 k=2 覆盖 6/6）即算通过。
3. **既有"L→L 16/16 行为等价"的适用边界仍是"16 步内逐 token 一致"**；更长序列应改用 top-k 容忍口径。
4. **问题性质仍是"长序列不可逐 token 复现"**（**不是"生成崩坏"**）→ issue #28963 的表述基本正确，建议补充"分歧表现为近并列 argmax 翻转、并在自回归中放大"。
5. **D→L 的结论应表述为"top-2 容忍下等价"**，而不是"逐 token 一致"；后者在 L→L 都已不成立。

**复现命令**：

```bash
export PATH="/c/msys64/ucrt64/bin:$PATH"
BIN=build/cross-framework-layer-poc/llama.cpp/build-cpu/bin/llama-relay-gen.exe
ROOT=build/cross-framework-layer-poc
"$BIN" "$ROOT/out/qwen35-2b-f16.gguf" "$ROOT/out/qwen35-2b-f16-cut.gguf" \
  --prompt "$ROOT/out/prompt-single.txt" --gen 141 --threads 4 --tag long --dump-dir "$ROOT/out"
# → RESULT: 在第 4x~5x 步分叉（两次运行分叉点不同）
```

**待验证（本节已作废，见 §13）**：~~`--threads 1` 下的长序列对照；分叉点是否随 `--gen` 与 prompt 变化；`223145984` 是否可能由工具侧越界打印造成~~ → 见 §13：分叉成因是调用方 `pos` 越界读；`223145984` 确为工具侧越界打印（已修）。

## 13. 根因确定与修复验证（2026-09-16 晚 —— **本文的有效结论**）

> 本节是本文的**有效结论**；§5–§12 全部作废（原因见文首横幅）。

### 13.1 根因（上游给出，本仓库独立复现确认）

`llama_batch_allocr::ubatch_add`（`llama-batch.cpp`）：

```cpp
for (size_t j = 0; j < (size_t)n_pos_per_embd; ++j) {
    // ... otherwise, the input batch is image embeddings, we copy the positions as-is
    size_t src_off = batch.token ? 0 : j*batch.n_tokens;
    udata->pos[j*n_tokens + i] = batch.pos[src_off + idxs[i]];
}
```

对 **M-RoPE 模型的 `embd` 批次**，该循环读 `batch.pos[0 .. n_pos_per_embd*n_tokens)`；而 `llama.h`
声明"数组必须为 `n_tokens` 大小"、`llama_batch_init` 也**只分配 `n_tokens` 个 `pos`**
→ **sections 1..n-1 属堆越界读**。上游 valgrind 实证：

```
Invalid read of size 4
   at llama_batch_allocr::ubatch_add(...)
 Address 0xb4e4244 is 0 bytes after a block of size 20 alloc'd   (20 bytes = 5 positions)
```

**这解释了此前所有"看着像变量"的现象**：N 的大小、graph reuse 与 fresh context、权重类型
（F16/Q8_0 的"确定"只是碰巧堆内容）、`X_PERTURB_`、以及"全零输入稳定"（垃圾位置旋转零仍是零）。
它与 `pos == NULL` 的 #28902 / #28910 是同一循环的孪生情形。

### 13.2 我们的独立验证（Windows / MSYS2 UCRT64 / Qwen3.5-2B f16 / CPU 后端）

`qwen35.rope.dimension_sections` 为 4 段 → `n_pos_per_embd = 4`。采用上游建议的**调用方修法**：
`embd` 批次的 `pos` 按 `4*n_tokens` 分配，且每段都填 `pos[j*n_tokens + i] = i`。

| 实验（同一 exe、同一输入） | 修复前 | 修复后 |
| --- | --- | --- |
| `embd` 连跑两次（141 位置，`relay-check --dump-all`） | argmax `126/141`、cosine 0.85–0.99 | **`141/141`、cosine min = `1.000000`** |
| `embd` vs token 路径（同一 141-token prompt） | argmax `132/141` | **`141/141`、cosine min = `0.999999`** |
| token 路径对照（连跑两次） | bitwise 一致 | bitwise 一致（不变） |
| `llama-relay-gen` L→L 141 步贪心生成（连跑 2 次） | 第 46 / 51 步分叉 | **`全部 141 步一致`（2/2）** |
| `llama-relay-gen-dl` **D→L 逐步生成**（A=PyTorch 每步 layer 0..3 → B=裁层 llama.cpp，141 步） | 无法进行（`embd` 每步结果不可复现） | **`全部 141 步一致`**（驱动 `drive_dl_relay.py`；报告 `local_docs/evidence/relay-xframe/CORE-RELAY-XFRAME-01-dl-stepwise-141-2026-09-16.json`） |

修法实现见 `build/cross-framework-layer-poc/patches/embd-pos-overread-fix-2026-09-16.patch`
（`relay-check.cpp` / `relay-gen.cpp` / `embd-inject-test.cpp` 的 `pos4` 分配）。
**主仓运行时不依赖该补丁**；主仓只记录契约事实 —— **在 M-RoPE 模型上使用 `embd` 批次时，
调用方的 `pos` 必须按 `n_pos_per_embd * n_tokens` 提供**。

### 13.3 对项目结论的影响（改写）

1. **"`embd` 路径进程内固有非确定性"** ✗ 不成立 —— 是 `pos` 越界读，且**已有可行修复**；
2. **"D→L 长序列拒绝（132/141、128/141）"** ✗ 不成立 —— 修复后 **141/141**；
3. **"L→L 16/16 的边界 ≈16 步、长序列会分叉"** ✗ 不成立 —— 修复后 **141/141 全一致**；
4. **"分歧是 top-1↔top-2 互换、严格判据不可达"** ✗ —— 那些"近并列翻转"同源于越界读；
5. **"分段短序列接力 / `top_k_tolerant_argmax` 是必要绕开"** ✗ 不再必要 —— 严格判据在修复后可达。
   `top_k_tolerant_argmax`（`src/relay_contract.py`）**保留为诊断档**，但**不再作为准入依据**；
6. **未改变的**：Relay 用 argmax 判定、接力实验固定单进程、"行为等价"与"逐位复现"分开声明 —— 这些仍是好实践。

### 13.5 触发条件与模型影响面（2026-09-16，本仓库实测）

**触发需要同时满足两条**（源码即判据）：

1. **走 `embd` 批次**（`ubatch->token == NULL`）—— **普通 token 路径永不触发**；
2. **`n_pos_per_embd > 1`**，即模型声明了多段 `rope.dimension_sections`（**M-RoPE**）。
   其中 `n_pos_per_embd = len(<arch>.rope.dimension_sections)`；**无该键的模型即为 1 段**。

**实测**（同一 exe，用 `QLH_NO_POS_FIX=1` 控制是否做调用方修复；随机 embd 向量 × 5 token；各连跑两次）：

| 组 | 模型 | `rope.dimension_sections` | `n_pos_per_embd` | 是否修复 | 两次结果 |
| --- | --- | --- | --- | --- | --- |
| A | `qwen35-2b-f16.gguf`（arch `qwen35`） | `[11, 11, 10, 0]` | **4** | ✗ | **`identical=False`（maxabsdiff 0.959）→ 触发** |
| B | 同上 | 同上 | 4 | ✓ | `identical=True` |
| C | `Qwen-1_8B-Chat.Q4_K_M.gguf`（arch `qwen`） | **无该键** | **1** | ✗ | **`identical=True` → 不触发** |
| D | 同上 | 同上 | 1 | ✓ | `identical=True` |

**结论**：

- **不是"多模态模型"的问题，而是「M-RoPE（`n_pos_per_embd > 1`）+ `embd` 批次」的组合**；
- **1D RoPE 模型即使走 `embd` 批次也不受影响**（C 组：不做修复也逐位一致）—— 因为 `j` 循环只跑 `j = 0`，只读 `pos[0..n)`；
- **任何模型走 token 路径都不受影响**。

**模型族对照**：

| 模型族 | `rope.dimension_sections` | `embd` 批次下是否受影响 |
| --- | --- | --- |
| Qwen3.5 / Qwen3-VL / Qwen2-VL / Qwen2.5-VL（M-RoPE） | 多段 | ⚠️ **受影响**（须按契约填 `pos`） |
| Qwen2.5 文本 / Qwen1.5 / Qwen2 / Llama / Mistral / DeepSeek | 无（=1 段） | ✅ 不受影响 |
| 任意模型 + token 路径 | — | ✅ 不受影响 |

**复现命令**（随机向量即可，不依赖真模型权重）：

```bash
export PATH="/c/msys64/ucrt64/bin:$PATH"
BIN=build/cross-framework-layer-poc/llama.cpp/build-cpu/bin
D=build/cross-framework-layer-poc/out/npe-check
# 造随机 embd（5 × 2048）
python -c "import numpy as np; np.random.default_rng(0).standard_normal((5,2048),dtype=np.float32).tofile('$D/rand5x2048.f32')"
# A) M-RoPE 模型 + 跳过修复 → 两次不一致
QLH_NO_POS_FIX=1 $BIN/llama-relay-check.exe out/qwen35-2b-f16.gguf --embd $D/rand5x2048.f32 --n-tokens 5 --dump-all --dump /tmp/a1.f32
QLH_NO_POS_FIX=1 $BIN/llama-relay-check.exe out/qwen35-2b-f16.gguf --embd $D/rand5x2048.f32 --n-tokens 5 --dump-all --dump /tmp/a2.f32
# C) 1D RoPE 模型 + 跳过修复 → 仍逐位一致
QLH_NO_POS_FIX=1 $BIN/llama-relay-check.exe H:/qlh_models/Qwen-1_8B-Chat.Q4_K_M.gguf --embd $D/rand5x2048.f32 --n-tokens 5 --dump-all --dump /tmp/c1.f32
QLH_NO_POS_FIX=1 $BIN/llama-relay-check.exe H:/qlh_models/Qwen-1_8B-Chat.Q4_K_M.gguf --embd $D/rand5x2048.f32 --n-tokens 5 --dump-all --dump /tmp/c2.f32
```

**对 QLH 的含义**：

1. **主线（L 档单机推理 / TUI 对话 / RPC 分片）都是 token 路径 → 任何模型都不触发**；
2. **仅"跨框架层接力"（`embd` 注入）轨道受影响**：选 M-RoPE 模型须按契约填 `pos`；**选 1D RoPE 模型则天然安全**；
3. 本项目接力实验恰好选用 **Qwen3.5-2B（M-RoPE，最坏情况）**；此前"现象随权重类型变化"的假象，实质是该 20 字节 malloc 之后恰好是什么堆内容；
4. **MTMD（图像 embedding）同样走 `embd` 批次** → 在 M-RoPE 模型上命中同一处（这正是上游 #28910 修 `pos == NULL` 的动因）。

### 13.6 采样矩阵（2026-09-16，本仓库实测）

在已 dump 的两份 141 位置 logits（A=整模型 token 路径、B=裁层模型 + torch hidden 注入）上做自回归采样，
两侧各用独立的 `np.random.default_rng(seed)`；判据 = 逐 token 相同：

| case | 一致步数 | 判读 |
| --- | --- | --- |
| `T=0` 贪心（各自 argmax） | **141/141** | 与 §13.2 一致 |
| `T=0.8, top_p=1` 两侧同 `seed=0` | **30/141**（第 31 步分叉） | ⚠️ **同 seed 也不够** |
| `T=0.8, top_p=1` 两侧不同 seed（0 vs 1） | **1/141**（第 2 步分叉） | RNG 状态决定性 |
| `T=0.8, top_p=0.9` 两侧同 `seed=0` | **141/141** | 截尾后一致 |
| `T=0.8, top_p=0.9` 两侧不同 seed（0 vs 2） | **1/141** | — |
| `T=1.5, top_p=1` 两侧同 `seed=0` | **0/141**（第 1 步分叉） | 高温放大微小差异 |

**结论**：

1. **`T=0`（贪心）确定** → XFRAME 准入契约要求 `temperature=0, top_p=1` **保守但正确**；
2. **不同 RNG 必然第 2 步分叉** —— 采样正确性与 logits 等价性**正交**；
3. **"共享 RNG 状态"不是充分条件**：两侧概率向量并非逐位相同（cosine 0.9999995），`top_p=1` 下采样落在 CDF 边界时会翻转（30/141）；**只有截尾（`top_p<1`）让保留集上的分布差异变小后，同 seed 才一致**（141/141）；
4. **可放宽路径**：真实的 D→L 形态是 **A 侧只提供 hidden、只由 B 侧采样**（RNG 只有一个）→ 不存在"两侧 RNG 不一致"问题，因此**采样本身并非不可行**；契约约束的实质应表述为 **"禁止两侧各自采样"**，而不是"禁止采样"。

脚本：`build/cross-framework-layer-poc/sampling_matrix_check.py`（读两份 dump，纯 numpy，跑完释放约 1.2 GB）；
报告：`local_docs/evidence/relay-xframe/CORE-RELAY-XFRAME-01-dl-sampling-matrix-2026-09-16.json`。

### 13.7 f32 权重对照（2026-09-16，本仓库实测）

目的：确认 `pos` 契约与 D→L 等价性**不依赖于权重格式**。

| 步骤 | 命令要点 | 结果 |
| --- | --- | --- |
| baseline | `qwen35-2b-f32.gguf`（整模型 f32）+ `--tokens`（141 tokens） | 写出 141×248320 logits |
| D→L | `qwen35-2b-f32-cut.gguf`（裁层 f32）+ `--embd torch-hs-l3.f32` | 写出 141×248320 logits |
| 判定 | `compare_all.py` | **argmax 一致 `141/141`，cosine min `0.999999`** |

**内存**：整模型 f32 为 7.8 GB（mmap 按需读），实测 **Free 9416 MB → 9458 MB**，**未出现内存压力**；
因此 f32 对照**不需要**额外驻留 torch fp32 模型（直接复用既有的 `torch-hs-l3.f32` 导出）。

**结论**：**f16 与 f32 两种格式下 D→L 141-token prefill 均为 141/141** —— 与「#28963 是 `pos` 越界读、
与权重类型无关」的定性一致；此前"换成 F16/Q8_0 就变确定"的观感，正来自越界读到的堆内容。

### 13.4 待办

- 若上游在**库侧**修复（对单段位置做广播，或扩展 header 语义），本地补丁即可移除；
- 重跑此前被"非确定性"阻断的 D→L 正式实验，重新给出准入证据；
- 回复上游：**仅做独立复现确认**，草稿见 `local_docs/plans/给上游的回复草稿-embd-pos越界-2026-09-16.md`。

### 13.8 上游社区状态追踪（2026-09-19 核查）

本节追踪与本仓库 #28963 同源的上游讨论。**四个线程共用同一条根因链**：`llama_batch_allocr::ubatch_add`
在 M-RoPE 模型 + `embd` 批次时按 `n_pos_per_embd * n_tokens`（典型 = 4×）读 `pos`，而
`include/llama.h:248` 只承诺 `n_tokens` 项。

| 线程 | 类型 | 状态 | 作者 | 摘要 |
| --- | --- | --- | --- | --- |
| **#28441** | issue | **closed** | `ardan-bkennedy` | 现场案例：Kronk(Go) + Qwen2.5-Omni Q8 + Metal，750-token 音频 prefill 间歇性静默损坏。实测 `llama_batch_init(750, 2048, 1)` → `pos` 750 项(3000B)，llama.cpp 读 3000 项(12000B) → **越界 9000B**；改分配 `n_tokens*4` 后 4 次全新进程均相干并到 EOS |
| **#28902** | issue | **open** | `devYRPauli` | 根因定位：`src/llama-batch.cpp:786-787` 的读越界；**两条路径** —— ① 调用方照文档分配；② `pos == NULL` 时库内 fallback(`llama-batch.cpp:91` `pos.resize(n_tokens)`)同样越界，**调用方无法避免** |
| **#28910** | **PR** | **open**（`mergeable=True`，未合并） | `shashb27` | **修复路径②**：fallback 尺寸改为 `n_tokens * n_pos_per_embd`，各 section 填同一自增序列；并在 `llama.h` 文档化 embedding 批次的 `pos` 布局。`+39 -3`，3 文件。分支 `fix/28902-mrope-auto-pos-overread` |
| **#28963** | issue | **open** | **`SgfKrc`（本仓库）** | 我们的原始报告：CPU 后端 `llama_batch.embd` 解码非确定性、token 路径逐位确定。`mitoyosh` 独立复现并定位同一根因；我们随后在 Windows/MSYS2 + Qwen3.5-2B 上独立确认修复有效（141/141，cosine min = 1.000000） |

**交叉独立验证证据**：

| 来源 | 平台 | 方法 | 结果 |
| --- | --- | --- | --- |
| `devYRPauli`（#28910 评论） | macOS arm64，`-DLLAMA_SANITIZE_ADDRESS=ON` | 三棵树的对照构建 | master 未改 rc=0(30 tests)；**只加测试 rc=134（ASan heap-buffer-overflow @ `llama-batch.cpp:787`）**；加完整 PR rc=0(31 tests) |
| `mitoyosh`（#28902 评论） | Apple clang 21 / arm64，ASan，mock-vocab 单元测试 | 无模型参与 | 同样在 `ubatch_add` 报 heap-buffer-overflow（8-byte 区域后越界读） |
| `mitoyosh`（#28963 评论） | Debian 13 x86-64，CPU build `930e2fa`，`n_threads=1`，flash attn off，Qwen3.5-0.8B BF16 | 同形状探针 | 复现我们的现象（embd 5 次运行 `bitwise_equal=NO`，`max_abs_diff` 1.85–2.4）并给出根因 |
| **本仓库**（#28963 评论） | Windows 11 x86-64 / MSYS2 UCRT64 g++ 15.2.0 / CPU / Qwen3.5-2B f16（`dimension_sections` 4 项 → `n_pos_per_embd=4`） | 按对方所述调用方修复 | 修复前 argmax **126/141**、cosine 0.85–0.99 → 修复后 **141/141**、**cosine min = 1.000000** |

**⚠️ 修复的覆盖面（关键）**：#28910 **只修路径②（`pos == NULL` 的库内 fallback）**。
「路径①」（调用方按 `include/llama.h:248` 的文档分配 `n_tokens`）被作者明确留作
**「maintainer decision」**（API 语义问题）—— 即 **header 承诺的尺寸与实际读取的尺寸不一致**这一点，
截至 2026-09-19 仍未被上游以代码形式解决。

**对 QLH 的含义**：

1. **本仓库的调用方补丁仍然必要**（§13.2 / §8）：在 M-RoPE 模型上使用 `embd` 批次时，`pos` 必须按
   `n_pos_per_embd * n_tokens` 提供。§13.4 待办中「若上游在库侧修复，本地补丁即可移除」——
   **目前只对 `pos == NULL` 的调用方成立；我们走的是显式 `pos`，仍须保留补丁**。
2. **「同一循环的孪生情形」判断已被上游独立坐实**（§13.1 中我们提出的表述），且同一根因还波及
   **MTMD 图像 embedding** 与 **Metal 多模态** —— 说明该缺陷的影响面比我们最初报告的 CPU 文本场景更广。
3. 追踪状态（截至 2026-09-19）：**#28910 尚未有维护者表态**，5 条评论均为 bot 检查 + 贡献者互验。

**数据来源**：GitHub REST API（`api.github.com/repos/ggml-org/llama.cpp/{issues,pulls}/{28441,28902,28910,28963}`
及其 `/comments`），核查时间 2026-09-19。

## 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-09-19 | **新增 §13.8 上游社区状态追踪**（核查 `api.github.com`）：确认 **#28963 与 #28441 / #28902 / #28910 共用同一条根因链**（`ubatch_add` 在 M-RoPE + `embd` 批次按 `4×n_tokens` 读 `pos`，而 `llama.h:248` 只承诺 `n_tokens`）。记录 #28441 现场数据（Kronk + Qwen2.5-Omni + Metal，越界 9000B）、#28902 的两条路径、**#28910 PR 当前 `open`/`mergeable`/未合并**，以及四方（`devYRPauli` / `mitoyosh` / `shashb27` / 本仓库）的独立验证证据。**结论修正**：§13.4「若上游库侧修复即可移除本地补丁」**只对 `pos == NULL` 成立** —— #28910 仅修路径②，我们走显式 `pos`，**补丁必须保留**；「路径①」被作者留作 maintainer decision，截至核查日未被代码解决。 |
| 2026-09-16 | 新建：把 §7.10/§7.11 发现的 `embd` 非确定性整理成可提交的 issue 材料。含环境、最小复现探针（自包含源码）、16 项实验数据、逐层定位（分叉始于第一个 full-attention 层、需要 >1 token）、7 条已排除成因、候选假设、QLH 侧绕开口径、pin 版本与打补丁流程、英文 issue 正文草稿。**根因未坐实，本轮不提供上游修复补丁**；曾尝试的"清零 `inp->tokens`"已回退，工作树干净。 |
| 2026-09-16 | **补实验 #17/#18（增量注入）**：新增 `relay-incr-probe`（整段注入 vs 逐 token 增量注入对比）→ **增量注入同样逐位不等**（cosine 0.994），"改用更细注入粒度即可复现"的期望**被证伪**；并据此细化定位为"**pos = 0 单 token 确定、pos ≥ 1（有历史状态）单 token 不确定**"。同步更新 §8 绕开口径（可复现性只能靠 argmax / 单进程口径）与提交指引。 |
| 2026-09-16 | **issue 已提交**：[ggml-org/llama.cpp#28963](https://github.com/ggml-org/llama.cpp/issues/28963)（作者 `SgfKrc`，`open`，0 评论）—— `Misc. bug: CPU backend: llama_batch.embd (embedding input) decoding is non-deterministic`。 |
| 2026-09-16 | **长序列实测（§12，主节点）**：按"改用生成序列判据"重测 —— `llama-relay-gen --gen 16` 的 L→L 三次全一致（复现 16/16）；**`--gen 141` 两次分别在 51 / 46 步分叉**（relay 侧出现非法 token `223145984` ≫ vocab 248320），baseline（token 路径）的 141 步两次完全一致 → **问题性质从"不可逐位复现"升级为"长序列生成实际出错"，且 L→L 自身即复现，无需跨框架**；"16/16" 的适用边界 ≈16 步。同时给出 141-token 长批下 **argmax 判据失效**（同命令两次仅 126/141）的证据。**§0/§10 中 "impact is reproducibility, not greedy generation" 需更正（建议人工补 issue 评论）。** |
| 2026-09-16 | **⚠️ 更正上一行与 §12（主节点，同日）**：上一行的"**非法 token `223145984`**"**是工具 bug，不是模型输出** —— `relay-gen.cpp` 在分叉时 `break` 而未 `push_back`，`relay.size() == first_diverge`，故 `relay[first_diverge]` 属**越界读**（已在实验 fork 修复）。经分叉点自动 dump 的整份 logits 逐份量化（**6/6 次**：L1/L2/L3/t1/plen/plong）：**cosine 0.984–0.999、top-5 交集 4–5/5，且双方的 argmax 互相都排在对方第 2 位** → **分歧 100% 是 `top-1 ↔ top-2` 互换**，由 embd 路径非确定性触发、被自回归放大（分叉步随运行漂移 31–67）。**故问题性质是"长序列不可逐 token 复现"，而不是"生成崩坏"**；`--threads 1` 与 `--threads 4` 的分叉步/cosine/top-5 完全一致。据此在 `src/relay_contract.py` 新增 **`top_k_tolerant_argmax`**（`RELAY_TOLERANT_TOP_K = 2`）与 `judge_relay_generation_tolerant`，Relay 定向回归 **49 passed**。 |
