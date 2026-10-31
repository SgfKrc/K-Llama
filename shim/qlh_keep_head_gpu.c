/*
 * qlh_keep_head.c — keep-head 上游的 C shim（P2「C 路线」的接入点）
 * ==================================================================
 *
 * 为什么需要 shim，而不是直接在 Python 里 ctypes 调 libllama：
 * `llama_context_params` / `llama_model_params` 是**按值传参的大结构体**，其字段布局
 * 随 llama.cpp 版本变动。用 PyPI `llama-cpp-python` 的 ctypes 声明去调**自建**的
 * libllama，实测报 `llama_init_from_model: failed to initialize the context:
 * Unsupported ctx type` —— 即写入的字段错位（把 n_ctx 写到了 DLL 眼中的 `type`）。
 * 本 shim 用**与被调 DLL 同一份 `include/llama.h`** 编译，从根上消除该风险；
 * Python 侧只剩 4 个平凡签名的函数。
 *
 * 语义（与 `src/llama_keep_head.py` 的文档一致）：
 *   mode 0 = nextn   → 模型**自身的最后一层输出**（`output_norm` 之前）。配 head 裁层工件
 *                      （保留 blk.0..K-1）⇒ 只跑 K 层，正是 relay 上游该交的 hidden。
 *   mode 1 = layer_inp → **第 cut_layer 层的输入**（= 前 cut_layer 层的输出）。配整模工件
 *                      可验证语义，但会跑满全部层。
 *
 * ⚠️ 2026-09-23：`mode 0` 不再走 `llama_set_embeddings_nextn` —— 各架构把 `t_h_nextn` 挂在不同
 *   位置（qwen2 在 `output_norm` 之前、qwen35 在之后），拿它当上游会随架构而变（Qwen3.5 上多一次
 *   RMSNorm ⇒ 接力数值分叉）。现统一改用 `layer_inp` 的 `lid == n_layer` 槽位（"第 n_layer 层的
 *   输入" = 末层输出）；该槽位由 `llama-context.cpp` 多分配一个、由各架构 graph 在 `output_norm`
 *   之前登记。**目前登记该槽位的架构：`qwen2`、`qwen35`**（其他架构会因槽位为空而在 decode 时报
 *   `layer input tensor not null` 断言）。
 *
 * 编译（Windows / MinGW，见 scripts/model_tools/build_keep_head_shim.ps1）：
 *   gcc -shared -O2 -o qlh_keep_head.dll qlh_keep_head.c \
 *       -I<llama.cpp>/include -L<llama.cpp>/build/src -lllama
 *
 * ⚠️ 依赖上游实验性形态的 API（`llama_set_embeddings_layer_inp` /
 * `llama_get_embeddings_nextn_ith` 等），由
 * `scripts/model_tools/patches/llama-cpp-layer-forward-api.patch` 导出；升级 llama.cpp
 * 时必须重新核对（见 `scripts/model_tools/llama_quantize.lock.json` 的 `marker`）。
 */

#include "llama.h"
#include "ggml-backend.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

/* The two layer_inp entry points are a QLH-local extension, not upstream llama.cpp,
 * so no published llama.h declares them. The DLL exports them (verified with
 * dumpbin), and the prototype builds against a stock llama.h, so declare them here.
 * Signatures are read off the call sites below: the setter takes a layer index and a
 * flag, the getter returns that layer's input embedding for the last decode. */
extern void         llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool enable);
extern const float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

/* [C-scheme] "relay hidden" = the output of layer K-1, before the model's final
 * output_norm. Two ways to get it depending on which llama.cpp the shim links:
 *
 *   QLH DLL  : llama_set_embeddings_layer_inp(ctx, K, true) -> a dedicated channel.
 *   upstream : cparams.embeddings + cparams.embeddings_pre_norm, then the ordinary
 *              llama_get_embeddings() returns the pre-norm hidden of the last layer.
 *
 * With upstream the slot number is implicit: the context only ever holds the layers it was
 * told to load (LLAMA_LAYER_RANGE), so "layer K" is just the last layer. The lid argument is
 * therefore unused on that path -- it is kept so both builds share the call sites.
 *
 * It must be llama_get_embeddings() and NOT llama_get_embeddings_ith(-1): the latter returns
 * one row, while qlh_extract_hidden copies n_tokens * n_embd floats out of it. That mismatch
 * read past the buffer and every token after the first was garbage -- which looked like the
 * model being broken, not like a bug in the plumbing.
 */
/* plot twist（2026-10-31，DFLASH-01）：这两个入口在 llama.dll 里导出的是 **C++ mangled
 * 名**，不是 C 名：
 *   ?llama_set_embeddings_layer_inp@@YAXPEAUllama_context@@I_N@Z
 *   ?llama_get_embeddings_layer_inp@@YAPEAMPEAUllama_context@@I@Z
 * （dumpbin /exports 实测）。本文件是 C，`extern` 声明得到的是 C 链接名，链接期直接
 * LNK1120。两个选择：把整个 shim 改成 C++，或者运行时按 mangled 名解析。选后者 ——
 * 代价是一次 GetProcAddress，改动面只有一个函数。
 *
 * 为什么之前没人踩到：原代码在 QLH_ABI_UPSTREAM_HEADER 分支里把 SET 做成空操作、GET
 * 退化成 llama_get_embeddings()（那是 C 导出的），所以从来没有引用过这两个符号。 */
typedef void          (*qlh_set_layer_inp_fn)(struct llama_context *, uint32_t, bool);
typedef const float * (*qlh_get_layer_inp_fn)(struct llama_context *, uint32_t);

static qlh_set_layer_inp_fn qlh_p_set_layer_inp;
static qlh_get_layer_inp_fn qlh_p_get_layer_inp;

static void qlh_resolve_layer_inp(void) {
    static int done = 0;
    if (done) {
        return;
    }
    done = 1;
    HMODULE m = GetModuleHandleA("llama.dll");
    if (m == NULL) {
        return;
    }
    qlh_p_set_layer_inp = (qlh_set_layer_inp_fn) (void *) GetProcAddress(
        m, "?llama_set_embeddings_layer_inp@@YAXPEAUllama_context@@I_N@Z");
    qlh_p_get_layer_inp = (qlh_get_layer_inp_fn) (void *) GetProcAddress(
        m, "?llama_get_embeddings_layer_inp@@YAPEAMPEAUllama_context@@I@Z");
    if (qlh_p_set_layer_inp == NULL || qlh_p_get_layer_inp == NULL) {
        fprintf(stderr, "[qlh_keep_head] WARNING: layer_inp symbols not resolved "
                        "(set=%p get=%p) -- multi-slot hidden output unavailable\n",
                (void *) qlh_p_set_layer_inp, (void *) qlh_p_get_layer_inp);
    }
}

/* ★ 2026-10-31（DFLASH-01）修正：**默认槽和多槽必须走两条不同的路径**。
 *
 * 前一版把它们统一成 layer_inp，结果接力直接崩在
 *   llama-graph.cpp:1476  GGML_ASSERT(t_layer_inp[il] != nullptr)
 * 原因是 `qwen35moe.cpp:178` 只登记 `t_layer_inp[0..n_layer-1]`，**没有 t_layer_inp[n_layer]**，
 * 而 mode 0 的默认槽恰好是 lid=n_layer（"末层输出"）——它在 layer_inp 通道里不存在。
 *
 * 接力一直能用的真实原因是：`cparams.embeddings = !needs_logits`（本文件 L470），
 * hidden 出口把 embeddings 打开，于是 `llama_get_embeddings()` 返回**末层 output_norm 之前**
 * 的 hidden。所以默认槽继续走它，多槽走真正的 layer_inp。 */
#define QLH_DEFAULT_SET(ctx, lid) ((void) (lid))
#define QLH_DEFAULT_GET(ctx, lid) (llama_get_embeddings((ctx)))

/* 多槽（add_layer_inp）解析 C++ 符号后走 layer_inp */
#define QLH_SLOT_SET(ctx, lid) do {                                \
    qlh_resolve_layer_inp();                                       \
    if (qlh_p_set_layer_inp != NULL) {                             \
        qlh_p_set_layer_inp((ctx), (lid), true);                   \
    }                                                              \
} while (0)

#define QLH_SLOT_GET(ctx, lid)                                     \
    (qlh_resolve_layer_inp(),                                      \
     qlh_p_get_layer_inp != NULL ? qlh_p_get_layer_inp((ctx), (lid)) : NULL)


/* PROTO: load whatever ggml backends sit next to this DLL (ggml-cuda.dll in the
 * prototype build). Upstream QLH never needed this because the shim only ever ran
 * on CPU; llama.cpp's own tools call it once at startup. With GGML_BACKEND_DL=ON the
 * backends are separate DLLs and the loader searches the executable directory and
 * the CWD, neither of which is guaranteed to be our directory when the shim runs
 * inside a detached worker process. Pass our own directory explicitly, falling back
 * to the default search when the path cannot be resolved. */
static void qlh_ensure_backends_loaded(void) {
    static int loaded = 0;
    if (loaded) return;
    /* PROTO: prefer an explicit directory (the worker knows where the shim lives),
     * fall back to this DLL's own directory, then to the default search paths. */
    const char * dir = getenv("SA_BACKEND_DIR");
    fprintf(stderr, "[qlh_keep_head] SA_BACKEND_DIR=%s\n", (dir != NULL && dir[0] != '\0') ? dir : "(unset)");
    if (dir != NULL && dir[0] != '\0') {
        ggml_backend_load_all_from_path(dir);
    }
    /* Resolve this DLL's own path. Passing a function pointer as an HMODULE does not
     * work: GetModuleFileNameA then fails and the fallback silently finds nothing.
     * GetModuleHandleExA with FROM_ADDRESS is the documented way to map an address
     * inside the module back to its handle. */
    char self[MAX_PATH];
    HMODULE self_module = NULL;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR) &qlh_ensure_backends_loaded, &self_module) && self_module != NULL) {
        DWORD n = GetModuleFileNameA(self_module, self, (DWORD) sizeof(self));
        if (n > 0 && n < (DWORD) sizeof(self)) {
            char * slash = strrchr(self, '\\');
            if (slash != NULL) {
                *slash = '\0';
                fprintf(stderr, "[qlh_keep_head] loading backends from self dir=%s\n", self);
                ggml_backend_load_all_from_path(self);
            }
        }
    }
    ggml_backend_load_all();
    loaded = 1;
    fprintf(stderr, "[qlh_keep_head] backends loaded: count=%zu\n", ggml_backend_reg_count());
}

/* Declared in llama.cpp's src/llama-ext.h rather than the public header: both are exported
 * from the DLL, but a pip wheel's llama.h has no trace of them. */
extern void    llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);
extern float * llama_get_embeddings_nextn(struct llama_context * ctx);

typedef struct qlh_keep_head {
    struct llama_model   *model;
    struct llama_context *ctx;
    /* Second context on the same model, built with ctx_type = LLAMA_CONTEXT_TYPE_MTP.
     * A hybrid model's draft head is a distinct graph (LLM_GRAPH_TYPE_DECODER_MTP), not a flag
     * on the main graph, so it needs its own context. NULL when the model carries no NextN head.
     * Its batch takes the target's last hidden via llama_batch_ext_set_embd_state(). */
    struct llama_context *mtp_ctx;
    int n_embd;
    int n_layer;
    int mode;        /* 0 = nextn, 1 = layer_inp */
    int cut_layer;
    /* ★ 2026-10-31（DFLASH-01）：**多槽** layer_inp。
     *
     * 原来只有一个槽（mode 0 用 `n_layer`、mode 1 用 `cut_layer`），够接力用；但 DFlash 的
     * draft 一次要 8 路中间层的输入 hidden（GGUF 的 `target_layers`，每个 id 一个槽位）。
     * 上游 `llama_set_embeddings_layer_inp(ctx, lid, true)` 本来就是按 lid 独立开的
     * （`llama-cparams.h::embeddings_layer_inp` 是 per-layer 的 vector），所以这里**只需记住
     * 开了哪些**，一行 llama.cpp 都不用改。
     *
     * `lids[]` 只记录通过 `qlh_kh_add_layer_inp()` 显式开的槽；mode 0/1 那一个仍走
     * `cut_layer`/`n_layer`，因此既有行为完全不变。
     * `last_n_tokens` 是 getter 需要的行数 —— getter 不接收 n_tokens，因为它取的是"最近一次
     * forward"的 buffered 结果，行数由那个 forward 决定。 */
    uint32_t lids[16];
    int32_t  n_lids;
    int32_t  last_n_tokens;
    /* ★ 2026-10-30：context 的并行序列上限，从 qlh_kh_load 的入参归一化后存下。
     * 批次分配的 `n_seq_max` 必须与它一致 —— 详见 qlh_fill_explicit 上方的说明。 */
    int32_t n_seq_max;
    /* 是否启用统一 KV。= 0 时 llama.cpp 要求每个 token 恰好属于一个序列
     * （`llama-kv-cache.cpp` 在 `n_stream > 1` 时断言 `n_seq_id == 1`，而
     * `llama-context.cpp` 用 `n_stream = kv_unified ? 1 : n_seq_max`）。
     * 这里保持 llama.cpp 的默认（分开 KV），因此"一个 token 属于多条序列"会被
     * `qlh_fill_explicit` 明确拒绝，而不是走到 decode 深处才报 KV 槽位找不到。 */
    int kv_unified;
    int n_pos_per_embd;  /* ★ M-RoPE 模型 = 4（每个 token 的位置分量数），否则 1 */
    /* ★ 2026-10-26（PERF-SKIP-LOGITS-01）：**输出 hidden 的入口**是否连 logits 一起要。
     *
     * 0（缺省，= 既有行为）：每行都标 `batch.logits[i] = 1`。
     * 1：输出 hidden 的入口（`qlh_kh_forward` / `qlh_kh_forward_embd` /
     *    `qlh_kh_forward_embd_seq`）**一行都不标** ⇒ llama.cpp 的 `n_outputs = 0`，
     *    `lm_head` 矩阵乘退化为 0 行 ⇒ 不再读 `output.weight`。
     *
     * 为什么这是纯收益：这些入口的语义就是"交出 hidden"，调用方（`KeepHeadUpstream.
     * forward_tokens_to_hidden` / `forward_hidden_to_hidden`）从不取 logits；而
     * `output.weight` 在 9B 中间段工件上是**每步几百 MB** 的内存流量，本项目瓶颈正是
     * 内存带宽（见 docs/本机跨框架层段成本与干扰实测-2026-10-25.md §2）。
     *
     * ⚠️ **argmax 入口不受本开关影响**（`qlh_kh_forward_embd_token` /
     *    `qlh_kh_forward_embd_tokens` 恒为每行标 1）：末段要靠 logits 出 token，
     *    若把它们也关掉，末段会直接失败 —— 这是本改动的**数值门**。
     *
     * 由 `qlh_kh_set_skip_hidden_logits()` 设置（不读环境变量：开关属于调用方的配置，
     * 不该藏在 native 层里，否则同进程多实例无法分别控制）。 */
    int32_t skip_hidden_logits;
    /* ★ 2026-09-24：最近一次 `llama_decode` 的**原始**返回码（0 = 无错误）。
     * 此前 decode 失败一律折成 -2，把 llama.cpp 的具体错误码丢了 ⇒ 长时 decode 的 ctx 边界
     * 问题无法定位。每次 forward 入口清零（读到的就是本次错误），`qlh_kh_reset` 也清零。 */
    int32_t last_error;
    /* ★ 2026-10-29（SAMPLING-01）：**采样链**。NULL = argmax（默认，行为与旧版完全一致）。
     *
     * 非 NULL 时 `qlh_kh_forward_embd_token` 改用上游 `llama_sampler_sample` 选 token，
     * 而不是自己遍历 logits 取最大值。链由 `qlh_kh_set_sampling` 建立，顺序遵循
     * llama.cpp 惯例：penalties → top_k → top_p → temp → dist（dist 必须最后）。
     *
     * 为什么把采样放在 shim 里而不是 Python 侧：段间只传 hidden（8 KB/token），
     * 传完整 logits 是 944 KB/token（241,664 × f32），会毁掉链路的带宽特性；
     * 采样在末段本地做完，跨机协议一个字都不用改。
     *
     * 生命周期：`qlh_kh_reset` 调 `llama_sampler_reset`（否则 penalties 的历史会跨序列污染），
     * `qlh_kh_close` 调 `llama_sampler_free`，重复调 setter 先 free 旧的再建（不泄漏）。 */
    struct llama_sampler * sampler;
    /* 实际生效的采样参数（回显用；`sampler == NULL` 时全为零/缺省）。 */
    float    sampling_temperature;
    float    sampling_top_p;
    int32_t  sampling_top_k;
    float    sampling_repeat_penalty;
    int32_t  sampling_repeat_last_n;
    float    sampling_penalty_freq;
    float    sampling_penalty_present;
    uint32_t sampling_seed;
} qlh_keep_head;

static void qlh_set_err(char *err, size_t errlen, const char *msg) {
    if (err != NULL && errlen > 0) {
        snprintf(err, errlen, "%s", msg);
    }
}

/* Vocabulary width of a loaded handle, 0 when unavailable. Hoisted to a static
 * helper because the sampling setter needs it as a *size* (upper bound for
 * repeat_last_n and for detecting "top_k disables everything"), while the exported
 * `qlh_kh_n_vocab` lives further down the file. Relying on an implicit declaration
 * there would have compiled and linked while silently returning the wrong value. */
static int32_t qlh_vocab_size(const qlh_keep_head * handle) {
    if (handle == NULL || handle->model == NULL) {
        return 0;
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(handle->model);
    return vocab == NULL ? 0 : llama_vocab_n_tokens(vocab);
}

/* 返回值：句柄；失败返回 NULL 并把原因写进 err。
 * `n_seq_max` ★ P3：context 的并行序列上限（多序列数据流所必需）；<=0 视为 1（旧行为）。 */
void * qlh_kh_load(const char * model_path,
                   int32_t n_ctx, int32_t n_threads, int32_t n_batch,
                   int32_t n_seq_max,
                   int32_t mode, int32_t cut_layer,
                   int32_t * out_n_embd, int32_t * out_n_layer,
                   char * err, size_t errlen) {
    if (model_path == NULL) {
        qlh_set_err(err, errlen, "model_path is NULL");
        return NULL;
    }
    if (mode != 0 && mode != 1) {
        qlh_set_err(err, errlen, "mode must be 0 (nextn) or 1 (layer_inp)");
        return NULL;
    }

    qlh_ensure_backends_loaded();

    struct llama_model_params mparams = llama_model_default_params();
    /* PROTO: n_gpu_layers is configurable here. Upstream QLH hardcodes 0 because its
     * GPU stage runs on PyTorch and the shim only ever serves CPU stages. This
     * prototype puts the first-stage expert weights on the GPU, so the shim must be
     * able to offload. SA_N_GPU_LAYERS=0 keeps the original behaviour. */
    {
        const char * gpu_layers_raw = getenv("SA_N_GPU_LAYERS");
        if (gpu_layers_raw != NULL && gpu_layers_raw[0] != '\0') {
            char * gpu_layers_end = NULL;
            const long parsed = strtol(gpu_layers_raw, &gpu_layers_end, 10);
            mparams.n_gpu_layers = (gpu_layers_end != gpu_layers_raw && *gpu_layers_end == '\0')
                ? (int32_t) parsed : 0;
        } else {
            mparams.n_gpu_layers = 0;
        }
    }

    /* PROTO: keep a pure-CPU stage off the GPU entirely.
     *
     * n_gpu_layers = 0 does not mean "do not touch the GPU". Measured on the three-stage
     * chain: the middle stage runs 16 layers on the CPU, yet llama.cpp still picked
     * CUDA0 as its device and reserved a 1012 MiB compute buffer there. Two processes
     * holding CUDA contexts on one card made the GPU stage's decoder step go from
     * 5.9 ms standalone to 61.9 ms inside the chain -- a 10x regression caused purely
     * by a stage that was supposed to be CPU-only.
     *
     * `devices` is a NULL-terminated list of the devices a model may use. Passing only
     * the CPU device keeps that stage out of VRAM and out of the CUDA context, which is
     * what n_gpu_layers = 0 was always meant to imply here.
     */
    static ggml_backend_dev_t s_cpu_only[2];
    if (mparams.n_gpu_layers == 0) {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev != NULL) {
            s_cpu_only[0] = cpu_dev;
            s_cpu_only[1] = NULL;
            mparams.devices = s_cpu_only;
            fprintf(stderr, "[qlh_keep_head] cpu-only: pinned devices to CPU (%s)\n",
                    ggml_backend_dev_name(cpu_dev));
        } else {
            /* Should not happen: the CPU backend is always registered. Say so rather
             * than silently keeping every device, which is the bug this exists to fix. */
            fprintf(stderr, "[qlh_keep_head] cpu-only: no CPU device found, "
                            "falling back to all devices\n");
        }
    }

    /* NOTE 2026-10-29: an SA_NO_EXTRA_BUFTYPES switch lived here and was removed.
     *
     * It set mparams.use_extra_bufts = false to skip ggml's 4.6 GB repack buffer, on
     * the theory that the buffer was starving page cache on a 15.2 GB host and making
     * the mmap'd 9.17 GB of weights re-read from disk every step. Two measurements
     * killed that:
     *
     *   1. The switch changed nothing here. The log printed "repack disabled" and the
     *      loader still reported `CPU_REPACK model buffer size = 4608.00 MiB`; this
     *      DLL build does not honour use_extra_bufts, so the knob was decorative.
     *   2. With the flag on, resident cost went UP, not down: 10.51 GB peak versus
     *      9.18 GB with repack left alone. Much of that is confounded by the GPU
     *      engine loaded in the same process during the comparison, so the honest
     *      reading is "no measurable benefit", not "actively harmful".
     *
     * Repack is a pure-CPU decode optimisation; its cost only shows up when the GPU
     * and CPU stages share a machine. It is not the lever for a memory-tight tail
     * host, so nothing here tries to disable it until a real switch exists upstream
     * (a `--no-repack` equivalent in llama_model_params).
     */

    /* PROTO: mirror llama-bench's --n-cpu-moe. Expert tensors are the bulk of the
     * weights (88% for Qwen3.6-35B-A3B) but only 8 of 256 are touched per token, so
     * leaving them on the host while everything else sits in VRAM is what makes the
     * split worth doing. Implemented the same way llama-bench does it: pin the
     * first N layers' expert tensors to the CPU buffer type via tensor overrides.
     * SA_N_CPU_MOE unset or 0 leaves every expert on the GPU. */
    static char * s_moe_patterns[512];
    static struct llama_model_tensor_buft_override s_moe_overrides[513];
    {
        const char * cpu_moe_raw = getenv("SA_N_CPU_MOE");
        long cpu_moe = 0;
        if (cpu_moe_raw != NULL && cpu_moe_raw[0] != '\0') {
            char * end = NULL;
            const long parsed = strtol(cpu_moe_raw, &end, 10);
            if (end != cpu_moe_raw && *end == '\0' && parsed > 0) cpu_moe = parsed;
        }
        if (cpu_moe > 0 && cpu_moe <= 512) {
            for (long i = 0; i < cpu_moe; ++i) {
                const size_t need = strlen("blk\\.\\.ffn_(up|down|gate|gate_up)_(ch|)exps") + 16;
                s_moe_patterns[i] = (char *) malloc(need);
                if (s_moe_patterns[i] == NULL) break;
                snprintf(s_moe_patterns[i], need, "blk\\.%ld\\.ffn_(up|down|gate|gate_up)_(ch|)exps", i);
                s_moe_overrides[i].pattern = s_moe_patterns[i];
                s_moe_overrides[i].buft = ggml_backend_cpu_buffer_type();
            }
            s_moe_overrides[cpu_moe].pattern = NULL;
            s_moe_overrides[cpu_moe].buft = NULL;
            mparams.tensor_buft_overrides = s_moe_overrides;
        }
    }

    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == NULL) {
        qlh_set_err(err, errlen, "llama_model_load_from_file failed");
        return NULL;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) (n_ctx > 0 ? n_ctx : 4096);
    cparams.n_batch         = (uint32_t) (n_batch > 0 ? n_batch : 512);
    /* ★ 2026-10-25（PERF-NATIVE-COST-01）：**n_ubatch 不再跟着 n_batch**。
     *
     * llama.cpp 按 `min(n_ctx, n_ubatch)` 做 worst-case 图预留，而 QWEN35 的
     * `graph_max_nodes = n_tokens * 40` ⇒ n_ubatch=4096 时每次 memory update 都要
     * 预留约 16.4 万节点。ROG 的 native 日志实测 `sched_reserve: CPU compute
     * buffer size = 4168.16 MiB`，就是这个值撑出来的（而它只有 15.15 GB 可见内存、
     * 热数据是 1646 MiB 的匿名 repack buffer，4 GB 预留会直接挤压它）。
     *
     * `n_batch` 仍必须够装下整段序列 —— M-RoPE 会把 embd 批次按 4× 放大
     * （见 llama_engine.py 关于上游 #28963 的记录），所以保留调用方传入值不动。
     * 但 **`n_ubatch` 只需覆盖一次内部微批量**。
     *
     * Android 侧早已如此（android/.../qlh_llama_jni.cpp：`n_ubatch = min(512,
     * batch_limit)`），且真机实测过不放大的代价：`n_batch = n_ubatch = n_ctx = 32768`
     * ⇒ 1062 tokens × 4 层 **>43 秒**（超 stage 超时）。PC shim 至今未移植，
     * 这里对齐同一取值。 */
    {
        uint32_t ubatch = (uint32_t) (n_batch > 0 ? n_batch : 512);
        if (ubatch > 512) {
            ubatch = 512;
        }
        cparams.n_ubatch = ubatch;
    }
    cparams.n_threads       = (n_threads > 0 ? n_threads : 4);
    cparams.n_threads_batch = (n_threads > 0 ? n_threads : 4);
    cparams.n_seq_max       = (uint32_t) (n_seq_max > 0 ? n_seq_max : 1);
    /* Zero the DLL's own n_outputs_max slot. The newer llama.h declares it at a
     * different offset than this build uses, so the guard in llama.h removes the
     * field and we write it by raw offset instead.
     *
     * Measured on the real DLL (build-headers/probe_slots.py): n_ctx 0, n_batch 4,
     * n_ubatch 8, n_seq_max 12, slot16, slot20, n_threads 24, n_threads_batch 28.
     * n_threads at 24 means exactly two extra slots, and the second (offset 20) is
     * n_outputs_max. The DLL asserts `n_outputs_max <= cparams.n_outputs_max` on
     * decode; leaving whatever default_params() put there makes the bound garbage and
     * the assert fires at the first forward. 0 means "derive it". */
#if defined(QLH_ABI_UPSTREAM_HEADER)
    /* Compiled against a llama.cpp checkout rather than the pip wheel: the struct has the
     * fields by name, three of them before n_threads (n_threads sits at 28, not 24). Writing
     * through the names is the whole point of switching headers -- the raw-offset version
     * below is what has to be kept in sync with a layout measured from a binary. */
    /* Recurrent-state snapshots. A hybrid model (Qwen3.5's LLL F) keeps linear-attention
     * state that has no position index, so a speculative decode cannot trim it by position the
     * way plain KV can: it needs a snapshot per sequence to fall back to. 0 disables that, which
     * is what every non-speculative chain wants (it is pure overhead otherwise). SA_N_RS_SEQ sets
     * it; the value must be >= the number of tokens one verification batch covers. */
    {
        const char * rs_env = getenv("SA_N_RS_SEQ");
        const long rs = (rs_env != NULL && rs_env[0] != '\0') ? strtol(rs_env, NULL, 10) : 0;
        cparams.n_rs_seq = (rs > 0 && rs <= 64) ? (uint32_t) rs : 0;
    }
    cparams.n_outputs_max          = 0;   /* derive from n_batch */
    cparams.n_outputs_max_per_seq  = 0;   /* derive from n_outputs_max */
#else
    {
        unsigned char * base = (unsigned char *) &cparams;
        uint32_t zero = 0;
        memcpy(base + 16, &zero, sizeof(zero));   /* n_rs_seq slot on this build */
        memcpy(base + 20, &zero, sizeof(zero));   /* n_outputs_max */
    }
#endif

#if defined(QLH_ABI_UPSTREAM_HEADER)
    /* The relay channel must be requested at context creation on upstream llama.cpp:
     * `embeddings` makes the graph expose a hidden state at all, and
     * `embeddings_pre_norm` moves that tap before the final output_norm (see llama.h).
     * The QLH DLL instead has llama_set_embeddings_layer_inp(), called after creation. */
    /* A relay segment returns the hidden state and skips the LM head; the tail returns
     * logits and must NOT skip it. The role arrives as SA_NEEDS_LOGITS (set by
     * spawn_worker) because context creation happens at load time, before any forward
     * call could pass want_logits. Absent means "needs logits", i.e. the old behaviour. */
    const char * needs_logits_env = getenv("SA_NEEDS_LOGITS");
    const bool needs_logits = (needs_logits_env == NULL) || (atoi(needs_logits_env) != 0);
    cparams.embeddings          = !needs_logits;
    cparams.embeddings_pre_norm = !needs_logits;
#endif

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        llama_model_free(model);
        qlh_set_err(err, errlen, "llama_init_from_model failed");
        return NULL;
    }
    /* ★ 2026-09-24：把**实际生效**的 ctx 参数打出来。长时 decode 在 ~481 帧报
     * `llama_decode rc=1`（llama.cpp 的 "failed to find a memory slot for batch"）时，必须能
     * 一眼看出 llama.cpp 是否真按我们请求的 `n_ctx` / `n_batch` 分配了 KV。 */
    fprintf(stderr, "[qlh_keep_head] ctx ready: n_ctx=%u n_batch=%u n_ubatch=%u "
                    "(requested n_ctx=%d n_batch=%d n_seq_max=%d)\n",
            llama_n_ctx(ctx), llama_n_batch(ctx), cparams.n_ubatch,
            n_ctx, n_batch, n_seq_max);

    const int n_layer = llama_model_n_layer(model);
    if (mode == 1 && (cut_layer < 0 || cut_layer >= n_layer)) {
        llama_free(ctx);
        llama_model_free(model);
        qlh_set_err(err, errlen, "cut_layer out of range for layer_inp mode");
        return NULL;
    }

    qlh_keep_head * handle = (qlh_keep_head *) calloc(1, sizeof(qlh_keep_head));
    if (handle == NULL) {
        llama_free(ctx);
        llama_model_free(model);
        qlh_set_err(err, errlen, "out of memory");
        return NULL;
    }
    handle->model     = model;
    handle->ctx       = ctx;
    handle->n_embd    = llama_model_n_embd(model);
    handle->n_layer   = n_layer;
    handle->mode      = mode;
    handle->cut_layer = cut_layer;

    /* Build the draft-head context when the file has one and the caller asks for it.
     *
     * The MTP head is a separate graph (LLM_GRAPH_TYPE_DECODER_MTP) selected by the context
     * type, so it cannot be reached from the main context. Same model, second context, tiny
     * KV: it only ever holds one verified position at a time.
     * SA_MTP=1 opts in; absent keeps a single context, which is the non-speculative path. */
    {
        const char * mtp_env = getenv("SA_MTP");
        const bool want_mtp = (mtp_env != NULL) && (atoi(mtp_env) != 0);
        if (want_mtp) {
            const int n_nextn = llama_model_n_layer_nextn(model);
            if (n_nextn <= 0) {
                fprintf(stderr, "[qlh_keep_head] SA_MTP=1 but the model has no NextN head "
                                "(n_layer_nextn=%d); draft disabled\n", n_nextn);
            } else {
                struct llama_context_params mp = llama_context_default_params();
                mp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
                mp.n_ctx         = 64;
                mp.n_batch       = 8;
                mp.n_ubatch      = 8;
                mp.n_seq_max     = 1;
                mp.n_threads     = (uint32_t) (n_threads > 0 ? n_threads : 4);
                mp.n_threads_batch = mp.n_threads;
                mp.n_rs_seq      = 0;
                mp.embeddings   = false;
                mp.pooling_type = LLAMA_POOLING_TYPE_NONE;
                /* Device placement is decided when the model is loaded (mparams), so the
                 * draft context inherits it; llama_context_params has no device list. */
                handle->mtp_ctx = llama_init_from_model(model, mp);
                if (handle->mtp_ctx == NULL) {
                    fprintf(stderr, "[qlh_keep_head] MTP context creation failed; draft disabled\n");
                } else {
                    /* The draft needs the target's last hidden, which the trunk graph only
                     * exports when this flag is set (t_h_nextn). masked=false keeps a row per
                     * token rather than only for the rows flagged via batch.logits. */
                    llama_set_embeddings_nextn(handle->ctx, true, false);
                    fprintf(stderr, "[qlh_keep_head] MTP context ready: n_layer_nextn=%d\n", n_nextn);
                }
            }
        }
    }
    /* ★ P3 (2026-10-30)：把 `n_seq_max` 记进 handle。
     *
     * 之前它只写进了 `cparams`（context 层），批次分配处却硬编码 1：
     *
     *     llama_batch_init(n_tokens, embd, 1)
     *
     * 于是 `seq_id[i]` 只拿到 1 个槽位。而 `qlh_fill_explicit` 允许调用方传
     * `n_seq_id[i] > 1`（一个 token 同时属于多条序列 —— 多序列交错推进的正常用法），
     * 它会写 `batch->n_seq_id[i] = 2`，却只填 `seq_id[i][0]`：llama.cpp 随后读
     * `seq_id[i][1]` 是**未初始化内存**，`llama_batch_free` 遍历时也按这个数字走。
     * 崩溃点在 free 而不是 decode，所以现场看起来与模型无关。
     *
     * 现在 batch 按 handle 记录的上限分配，填充处越界时直接拒绝，见 qlh_fill_explicit。 */
    handle->n_seq_max = (int32_t) cparams.n_seq_max;
    handle->kv_unified = 0;   /* llama.cpp 默认 false；见结构体注释 */
    /* ★ 2026-10-29（SAMPLING-01）：**中性**默认值，不是 calloc 的零。
     * `calloc` 会把 top_p / repeat_penalty 留成 0.0，而 0.0 在这套接口里的含义是
     * "极端截断"和"非法惩罚"，与"未启用"正好相反。回显函数会把这些字段原样交给
     * 客户端，所以必须在加载时就写成各自的关闭值，否则 argmax 模式会对外声称
     * top_p=0.0（听起来像最贪婪的采样）而不是"不适用"。 */
    handle->sampler                 = NULL;   /* argmax，见 last_error 上方的说明 */
    handle->sampling_temperature    = 0.0f;   /* <= 0 = greedy */
    handle->sampling_top_p          = 1.0f;   /* >= 1 = 关闭 */
    handle->sampling_top_k          = 0;      /* <= 0 = 关闭 */
    handle->sampling_repeat_penalty = 1.0f;   /* 1.0 = 关闭 */
    handle->sampling_repeat_last_n  = 0;      /* 0 = 关闭 */
    handle->sampling_penalty_freq   = 0.0f;
    handle->sampling_penalty_present= 0.0f;
    handle->sampling_seed           = LLAMA_DEFAULT_SEED;
    /* ★ M-RoPE（Qwen3.5 等）：每个 token 有 4 个位置分量，**embd 通道**下必须按 planar 提供
     * （见 `qlh_fill_pos_planar`）。按模型实际 rope 类型判定，不对架构硬编码。 */
    {
        const enum llama_rope_type rope = llama_model_rope_type(model);
        handle->n_pos_per_embd = (rope == LLAMA_ROPE_TYPE_MROPE ||
                                  rope == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;
    }

    /* ★ 2026-09-23：两个模式现在走**同一条** `layer_inp` 通道（只有槽位不同，见
     * `qlh_extract_hidden`）：
     *   mode 0 ⇒ lid = n_layer = 「第 n_layer 层的输入」= **末层输出（`output_norm` 之前）**
     *   mode 1 ⇒ lid = cut_layer = 「第 cut_layer 层的输入」
     * 为什么不再用 `llama_set_embeddings_nextn`：各架构把 `t_h_nextn` 挂在不同位置 ——
     * qwen2 在 `output_norm` **之前**（QLH 2026-09-20 补丁），而 qwen35 在**之后**
     * （那里的消费方是 MTP head）⇒ 同一个 `mode 0` 在 Qwen3.5 上拿到的不是层输出，接力
     * 数值首步即分叉（实测 9B）。`layer_inp` 的 `n_layer` 槽位在语义上**只会**是"末层输出"，
     * 与架构无关（槽位由 llama-context.cpp 多分配一个，由各架构 graph 在 `output_norm`
     * 之前登记）。 */
    if (mode == 0) {
        QLH_DEFAULT_SET(ctx, (uint32_t) n_layer);
    } else {
        QLH_DEFAULT_SET(ctx, (uint32_t) cut_layer);
    }

    if (out_n_embd  != NULL) { *out_n_embd  = handle->n_embd;  }
    if (out_n_layer != NULL) { *out_n_layer = handle->n_layer; }
    return handle;
}

/* 从当前 batch 结果里取出本次前向的 hidden（按模式），写进 out。返回 0 或负错误码。
 *
 * ★ 2026-09-23：两个模式统一走 `layer_inp` 通道，差别只是槽位号 ——
 *   mode 0（nextn，语义 = **末层输出 / output_norm 之前**）⇒ lid = n_layer
 *   mode 1（layer_inp，语义 = 第 cut_layer 层的输入）⇒ lid = cut_layer
 * 这样既不依赖各架构 `t_h_nextn` 的挂点差异，返回值也天然是**稠密的 `[n_tokens, n_embd]`**
 * （旧实现按 token 逐个取 `llama_get_embeddings_nextn_ith`）。 */
static int32_t qlh_extract_hidden(qlh_keep_head * handle, int32_t n_tokens, float * out) {
    const uint32_t lid = (uint32_t) ((handle->mode == 0) ? handle->n_layer : handle->cut_layer);
    const float * ptr = QLH_DEFAULT_GET(handle->ctx, lid);
    if (ptr == NULL) {
        return -3;
    }
    memcpy(out, ptr, (size_t) n_tokens * (size_t) handle->n_embd * sizeof(float));
    return 0;
}

/* ★ 2026-10-31（DFLASH-01）：额外开一个 layer_inp 槽位。
 *
 * 为什么需要：DFlash 的 draft 一次吃 target 的 8 路中间层输入 hidden（GGUF 的
 * `target_layers`），而 shim 原来只有一个槽（mode 0 用 n_layer、mode 1 用 cut_layer）。
 * 上游 `llama_set_embeddings_layer_inp()` 本来就是 per-lid 独立开关的，所以叠加即可。
 *
 * 返回 0；-1 参数无效、-2 槽位已满（上限 16）、-3 llama.cpp 拒绝该 lid
 * （常见原因：lid 超出本地层数，或该架构没有为它登记 t_layer_inp）。
 * 幂等：重复开同一个 lid 直接返回 0 —— 调用方每轮重建 driver 时会重复调用。 */
int32_t qlh_kh_add_layer_inp(void * handle_void, uint32_t lid) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || handle->ctx == NULL) {
        return -1;
    }
    const int32_t cap = (int32_t) (sizeof(handle->lids) / sizeof(handle->lids[0]));
    for (int32_t i = 0; i < handle->n_lids; ++i) {
        if (handle->lids[i] == lid) {
            return 0;
        }
    }
    if (handle->n_lids >= cap) {
        return -2;
    }
    QLH_SLOT_SET(handle->ctx, lid);
    /* 这里**不能**用 getter 验证：`embd_layer_inp` 要到 `extract_layer_inputs()`（即
     * decode 内）才被填，forward 之前 getter 恒为 NULL —— 早先在这里检查导致每次
     * add 都返回 -3（实测）。lid 是否真有效，由 get_layer_inp 在 forward 后报 -3。 */
    handle->lids[handle->n_lids++] = lid;
    return 0;
}

/* 取某个已开槽位、**最近一次 forward** 的 hidden（`last_n_tokens` 行）。
 *
 * 为什么行数从 handle 读而不让调用方传：它必须与产生这次数据的那次 forward 严格一致，
 * 传错就是整段拷到陈旧数据且不会报错。放在 handle 里就传不错。
 * 返回 0；-1 参数无效或 capacity 不足、-3 该槽本次没有数据、-4 槽位没开。 */
int32_t qlh_kh_get_layer_inp(void * handle_void, uint32_t lid, float * out, int32_t capacity) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || handle->ctx == NULL || out == NULL || handle->n_embd <= 0) {
        return -1;
    }
    int32_t open = 0;
    for (int32_t i = 0; i < handle->n_lids; ++i) {
        if (handle->lids[i] == lid) {
            open = 1;
            break;
        }
    }
    /* mode 0/1 的默认槽不是经 add_layer_inp 开的，但它同样有效（接力一直在用它）——
     * 漏掉这一条会让 lid=n_layer 报 -4「槽位没开」，而它其实是最老的那个槽。 */
    if (!open && handle->mode == 0 && lid == (uint32_t) handle->n_layer) {
        open = 1;
    }
    if (!open && handle->mode == 1 && lid == (uint32_t) handle->cut_layer) {
        open = 1;
    }
    if (!open) {
        return -4;
    }
    if (handle->last_n_tokens <= 0 ||
        capacity < handle->last_n_tokens * handle->n_embd) {
        return -1;
    }
    const float * ptr = QLH_SLOT_GET(handle->ctx, lid);
    if (ptr == NULL) {
        return -3;
    }
    memcpy(out, ptr, (size_t) handle->last_n_tokens * (size_t) handle->n_embd * sizeof(float));
    return 0;
}

/* 最近一次 forward 的行数（getter 用它决定拷贝多少行）。0 = 还没跑过。 */
int32_t qlh_kh_last_n_tokens(void * handle_void) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    return handle == NULL ? 0 : handle->last_n_tokens;
}

/* 统一标注：默认**每个 token 都要标输出**。
 * 历史原因：旧的 nextn 通道只对「有输出的行」计算末层 hidden，若只标最后一行，取 n_tokens 行
 * 就会踩到 GGML_ASSERT "tensor read out of bounds"（实测踩过）。现在两个模式都走 layer_inp
 * 通道，这条约束不再是硬性的，但保持"全行标输出"可以不必依赖具体的 sched 行为。
 *
 * ★ 2026-10-26（PERF-SKIP-LOGITS-01）：`want_logits = 0` 时**一行都不标**（`n_outputs = 0`）。
 *   这条路正是为"只要 hidden、不要 logits"的段准备的 —— 它换来的是 llama.cpp 不再执行
 *   `output.weight` 的矩阵乘。是否走这条路由 `qlh_kh_set_skip_hidden_logits()` 控制，
 *   缺省仍是 `want_logits = 1`（既有行为不变）。
 *   ⚠️ `layer_inp` 的提取与 `batch.logits` **无关**（`llama-context.cpp` 的
 *   `extract_layer_inputs()` 不看 `n_outputs`，`llama-graph.cpp::set_outputs()` 也按
 *   `cparams.embeddings_layer_inp` 独立 `ggml_set_output`），因此 hidden 仍然完整。 */
static void qlh_fill_common(struct llama_batch * batch, int32_t n_tokens, int32_t n_past,
                            int32_t want_logits) {
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch->pos[i]       = n_past + i;
        batch->n_seq_id[i]  = 1;
        batch->seq_id[i][0] = 0;
    #if defined(QLH_ABI_UPSTREAM_HEADER)
    /* Upstream derives the embedding rows from batch.logits, so marking nothing makes
     * llama_get_embeddings_ith() index past the end of an empty tensor (access violation,
     * measured). The LM head is skipped inside the graph instead, by the C-scheme patch,
     * which costs nothing -- so marking every row is cheap here. */
    batch->logits[i]    = 1;
#else
    batch->logits[i]    = want_logits ? 1 : 0;
#endif
    }
    batch->n_tokens = n_tokens;
}

/* ★ P3：多序列显式绑定 —— 每个 token 自带 `n_seq_id / seq_id / pos`，**不依赖隐式位置递增**。
 * 与主仓 `llama_engine.forward_layers_from_hidden(seq_ids=..., positions=...)` 同一契约；
 * 多序列交错推进时，位置由调用方决定（每序列可各自递增）。
 * ⚠️ `positions == NULL` 时回落到 `n_past + i`（单序列增量语义）—— 少了这一项，
 *    单序列 decode 的第二步会从位置 0 重放，直接触发
 *    "tokens ... have inconsistent sequence positions" 而 decode 失败（实测踩过）。
 *
 * ★ 2026-10-30：`seq_ids` 是**扁平**数组，按 `sum(n_seq_id[0..i-1])` 偏移取第 i 个 token
 *   的各个序列号 —— 这正是 llama_batch 的语义（`LLAMA_API` 头文件里
 *   `seq_id[i][j]` 的二维视角在内存里就是连续的）。旧实现只写 `seq_id[i][0]`，
 *   于是 `n_seq_id[i] > 1` 时后面的槽位保持未初始化，llama.cpp 读它、`llama_batch_free`
 *   也按它遍历。`cap_seq` 是 batch 实际分配到的槽位数，超出即拒绝，不再越界。 */
static int32_t qlh_fill_explicit(struct llama_batch * batch, int32_t n_tokens, int32_t n_past,
                                 const int32_t * n_seq_id, const int32_t * seq_ids,
                                 const int32_t * positions, int32_t want_logits,
                                 int32_t cap_seq, int kv_unified) {
    int32_t off = 0;
    for (int32_t i = 0; i < n_tokens; ++i) {
        const int32_t per_token = (n_seq_id == NULL) ? 1 : n_seq_id[i];
        if (per_token <= 0 || per_token > cap_seq) {
            return -5;   /* 调用方要的序列数超过批次能容纳的槽位 */
        }
        if (per_token > 1 && !kv_unified) {
            /* 不是越界，是不支持：kv_unified = false 时 llama.cpp 的 KV 为每个序列
             * 保留独立池，一个 token 只能归属一个序列。放行会让它走到 decode 深处才
             * 报 "failed to find a memory slot"，那句话指不到真正的原因。 */
            return -5;
        }
        batch->n_seq_id[i] = per_token;
        for (int32_t j = 0; j < per_token; ++j) {
            batch->seq_id[i][j] = (seq_ids == NULL) ? 0 : seq_ids[off + j];
        }
        off += per_token;
        batch->pos[i]    = (positions == NULL) ? (n_past + i) : positions[i];
    #if defined(QLH_ABI_UPSTREAM_HEADER)
    batch->logits[i] = 1;   /* see the note above: embeddings follow batch.logits */
#else
    batch->logits[i] = want_logits ? 1 : 0;
#endif
    }
    batch->n_tokens = n_tokens;
    return 0;
}

/* ★ M-RoPE 的 **embd 通道**位置布局（真 bug 修复，2026-09-23）：
 *
 * `llama-batch.cpp::llama_batch_allocr::ubatch_add` 里取位置的写法是
 *
 *     size_t src_off = batch.token ? 0 : j*batch.n_tokens;      // j = 0 .. n_pos_per_embd-1
 *     udata->pos[j*n_tokens + i] = batch.pos[src_off + idxs[i]];
 *
 * ⇒ **token 通道**（`batch.token != NULL`）把同一个位置**广播**到全部 RoPE 分量（文本语义）；
 *    **embd 通道**（`batch.token == NULL`）按 **planar**（分量分块）读，要求调用方给出
 *    `n_tokens * n_pos_per_embd` 个位置。
 *
 * 而 `llama_batch_init(n_tokens, ...)` **只分配 n_tokens 个** `pos`。于是对 M-RoPE 模型
 * （`n_pos_per_embd() == 4`：Qwen3.5 的 `rope type = 40 = IMROPE`）走 embd 通道时，
 * j=1..3 会读到缓冲区之外 ⇒ 位置语义错 ⇒ 层段接力数值错。
 * 实测症状：0.5B（`LLAMA_ROPE_TYPE_NORM`，n_pos_per_embd=1）历史记录 32/32 PASS；
 * Qwen3.5-9B（IMROPE）两段接力首步即分叉（`l2l-net-local-9b-2seg*.json`）。
 *
 * 本函数把单分量 `positions`（`NULL` ⇒ `n_past + i`）广播成 planar 布局，与 token 通道的
 * 语义对齐。返回 0 成功，-5 表示分配失败（调用方按参数错处理）。 */
static int32_t qlh_fill_pos_planar(struct llama_batch * batch, int32_t n_tokens, int32_t n_past,
                                   const int32_t * positions, int32_t n_pos_per_embd) {
    if (batch == NULL || n_tokens <= 0) {
        return -5;
    }
    if (n_pos_per_embd <= 1) {
        return 0;   /* 单分量：qlh_fill_explicit 填的就是正确布局 */
    }
    llama_pos * pos_full = (llama_pos *) malloc(
        sizeof(llama_pos) * (size_t) n_tokens * (size_t) n_pos_per_embd);
    if (pos_full == NULL) {
        return -5;
    }
    for (int32_t j = 0; j < n_pos_per_embd; ++j) {
        for (int32_t i = 0; i < n_tokens; ++i) {
            pos_full[(size_t) j * (size_t) n_tokens + (size_t) i] =
                (llama_pos) ((positions == NULL) ? (n_past + i) : positions[i]);
        }
    }
    free(batch->pos);       /* 旧的 n_tokens 缓冲；llama_batch_free 释放的是我们替换后的指针 */
    batch->pos = pos_full;
    return 0;
}

/* 前向：跑 tokens（从 n_past 起），把结果写进 out（容量 n_tokens * n_embd 个 float）。
 * 返回 0 成功；否则为负的错误码（-1 参数错，-2 decode 失败，-3/-4 取 hidden 失败）。 */
int32_t qlh_kh_forward(void * handle_void,
                       const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                       float * out) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || tokens == NULL || out == NULL || n_tokens <= 0) {
        return -1;
    }
    handle->last_error = 0;   /* ★ 本次 forward 的错误从这里重新计 */
    handle->last_n_tokens = n_tokens; /* ★ DFLASH-01: getter 的行数来源 */

    struct llama_batch batch = llama_batch_init(n_tokens, 0, handle->n_seq_max);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
    }
    qlh_fill_common(&batch, n_tokens, n_past, handle->skip_hidden_logits ? 0 : 1);

    const int32_t rc = llama_decode(handle->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        /* ★ 2026-09-24：把原始 rc 透出给调用方（**返回码仍是 -2，不破坏既有契约**）。 */
        handle->last_error = rc;
        fprintf(stderr,
                "[qlh_keep_head] llama_decode failed rc=%d (tokens: n_tokens=%d, n_past=%d)\n",
                rc, n_tokens, n_past);
        return -2;
    }
    return qlh_extract_hidden(handle, n_tokens, out);
}

/* ★ 中间段能力：吃 hidden（`embd` 注入）→ 吐 hidden（本模型末层 / 第 cut_layer 层输入）。
 * 这是「1 个 torch 上游 + n 个 llama 下游」链式拼接的关键 —— 中间的 llama 段必须能
 * 既接受上游 hidden 又交出 hidden。返回码同 qlh_kh_forward（另有 -5 = embd 参数错）。 */
int32_t qlh_kh_forward_embd_seq(void * handle_void,
                                const float * embd, int32_t n_tokens, int32_t n_past,
                                const int32_t * n_seq_id, const int32_t * seq_ids,
                                const int32_t * positions, float * out);

/* ★ P4.5 退化路径（无 PC 集群）：吃 hidden（embd 注入）→ 吐 **token**（末位 argmax）。
 * 末段能力：集群里可能没有任何能跑 torch 的节点，此时层接力必须全部由 llama.cpp 承载，
 * 末段就要能"吃 hidden 出 token"。与 `qlh_kh_forward_embd_seq` 共用同一份 decode 路径，
 * 只把输出从 hidden 换成末位 argmax ⇒ 语义天然对齐（同一份 C 源、同一份 llama.cpp，
 * 不引入第二个 llama.cpp 版本）。
 * 返回码同 `qlh_kh_forward_embd_seq`（另有 -6 = 取 logits 失败）。 */
int32_t qlh_kh_forward_embd_token(void * handle_void,
                                  const float * embd, int32_t n_tokens, int32_t n_past,
                                  const int32_t * n_seq_id, const int32_t * seq_ids,
                                  const int32_t * positions, int32_t * out_token);

int32_t qlh_kh_forward_embd(void * handle_void,
                            const float * embd, int32_t n_tokens, int32_t n_past,
                            float * out) {
    return qlh_kh_forward_embd_seq(handle_void, embd, n_tokens, n_past, NULL, NULL, NULL, out);
}

/* ★ P3：多序列版 `embd` 前向 —— 显式 `n_seq_id / seq_ids / positions`（长度均为 n_tokens）。
 * 三个数组都可为 NULL（等价于旧的单序列、位置自 n_past 起递增）⇒ 向后兼容。
 * ⚠️ 需要 `qlh_kh_load(..., n_seq_max >= 序列数)`，否则 llama.cpp 拒绝 >0 的 seq_id。 */
int32_t qlh_kh_forward_embd_seq(void * handle_void,
                                const float * embd, int32_t n_tokens, int32_t n_past,
                                const int32_t * n_seq_id, const int32_t * seq_ids,
                                const int32_t * positions, float * out) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || embd == NULL || out == NULL || n_tokens <= 0) {
        return -5;
    }
    handle->last_error = 0;
    handle->last_n_tokens = n_tokens; /* ★ DFLASH-01: getter 的行数来源 */

    struct llama_batch batch = llama_batch_init(n_tokens, handle->n_embd, handle->n_seq_max);
    if (qlh_fill_explicit(&batch, n_tokens, n_past, n_seq_id, seq_ids, positions,
                          handle->skip_hidden_logits ? 0 : 1,
                          handle->n_seq_max, handle->kv_unified) != 0) {
        llama_batch_free(batch);
        return -5;
    }
    if (qlh_fill_pos_planar(&batch, n_tokens, n_past, positions,
                            handle->n_pos_per_embd) != 0) {
        llama_batch_free(batch);
        return -5;
    }
    memcpy(batch.embd, embd, (size_t) n_tokens * (size_t) handle->n_embd * sizeof(float));

    const int32_t rc = llama_decode(handle->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        handle->last_error = rc;
        fprintf(stderr,
                "[qlh_keep_head] llama_decode failed rc=%d (embd: n_tokens=%d, n_past=%d)\n",
                rc, n_tokens, n_past);
        return -2;
    }
    return qlh_extract_hidden(handle, n_tokens, out);
}

/* ★ P4.5：末段能力实现 —— 与 `qlh_kh_forward_embd_seq` 共用同一条 decode 路径，
 * 只把输出从 hidden 换成末位 argmax（因此语义天然对齐）。 */
int32_t qlh_kh_forward_embd_token(void * handle_void,
                                  const float * embd, int32_t n_tokens, int32_t n_past,
                                  const int32_t * n_seq_id, const int32_t * seq_ids,
                                  const int32_t * positions, int32_t * out_token) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (out_token != NULL) {
        *out_token = -1;
    }
    if (handle == NULL || embd == NULL || out_token == NULL || n_tokens <= 0) {
        return -5;
    }
    handle->last_error = 0;
    handle->last_n_tokens = n_tokens; /* ★ DFLASH-01: getter 的行数来源 */

    struct llama_batch batch = llama_batch_init(n_tokens, handle->n_embd, handle->n_seq_max);
    /* ★ PERF-SKIP-LOGITS-01：本入口要**末位 argmax** ⇒ logits 恒开，**不跟随**
     *   `skip_hidden_logits`（末段是数值门的终点，缺了 logits 就直接失败）。 */
    if (qlh_fill_explicit(&batch, n_tokens, n_past, n_seq_id, seq_ids, positions, 1,
                          handle->n_seq_max, handle->kv_unified) != 0) {
        llama_batch_free(batch);
        return -5;
    }
    if (qlh_fill_pos_planar(&batch, n_tokens, n_past, positions,
                            handle->n_pos_per_embd) != 0) {
        llama_batch_free(batch);
        return -5;
    }
    memcpy(batch.embd, embd, (size_t) n_tokens * (size_t) handle->n_embd * sizeof(float));

    const int32_t rc = llama_decode(handle->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        handle->last_error = rc;
        fprintf(stderr,
                "[qlh_keep_head] llama_decode failed rc=%d (embd_token: n_tokens=%d, n_past=%d)\n",
                rc, n_tokens, n_past);
        return -2;
    }

    const struct llama_model * model = llama_get_model(handle->ctx);
    const struct llama_vocab * vocab = model == NULL ? NULL : llama_model_get_vocab(model);
    const int32_t n_vocab = vocab == NULL ? 0 : llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        return -6;
    }
    int32_t best = 0;
    float best_logit = 0.0f;
    /* All rows are marked as outputs, including prefill. Reject any NaN/Inf
     * before publishing the last row's argmax. Keep last_error as decode rc. */
    for (int32_t row = 0; row < n_tokens; ++row) {
        const float * logits = llama_get_logits_ith(handle->ctx, row);
        if (logits == NULL) {
            return -6;
        }
        for (int32_t i = 0; i < n_vocab; ++i) {
            const float value = logits[i];
            if (!isfinite(value)) {
                fprintf(stderr, "[qlh_keep_head] non-finite logits (row=%d, vocab_index=%d)\n",
                        row, i);
                return -7;
            }
            if (handle->sampler == NULL && row == n_tokens - 1 && (i == 0 || value > best_logit)) {
                best_logit = value;
                best = i;
            }
        }
    }
    /* ★ 2026-10-29（SAMPLING-01）：末段分叉 —— 有采样链就交给上游，否则走上面的 argmax。
     *
     * `idx = n_tokens - 1`：本入口给**每一行**都标了 `batch.logits = 1`（见上面的
     * `qlh_fill_explicit(..., 1)`），因此 output 序号与行号一一对应，最后一行的 output
     * index 就是 n_tokens - 1。这与 llama.h 的 "Equivalent to
     * llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab" 一致 —— 也意味着上面那个
     * 逐行循环读到的 `llama_get_logits_ith(ctx, n_tokens-1)` 正是采样器将要读的那一块，
     * 所以非有限门检查的正是采样输入。（实测见 docs/SAMPLING.md §6.1 的记录。）
     *
     * ⚠️ **这里刻意不调 `llama_sampler_accept()`**。`llama_sampler_sample` 自己就
     * 先 `iface->apply` 再 `iface->accept`（llama-sampling.cpp 的实现，且已用
     * dumpbin 反汇编在本机 llama.dll 上确认：函数体内两次间接调用，先 apply 后 accept），
     * 再 accept 一次会让 penalties 把同一个 token 记两遍。实测（shim/_test/probe_sampler.c）：
     * 同一个 token accept 两次，该 token 的 logit 从 -2.4999 变成 -4.4999 —— 不是无害的空操作，
     * 而是把重复惩罚翻倍。SAMPLING.md §3 的那一行是错的，以本处实现为准。 */
    if (handle->sampler != NULL) {
        const llama_token sampled = llama_sampler_sample(handle->sampler, handle->ctx,
                                                         n_tokens - 1);
        if (sampled == LLAMA_TOKEN_NULL || sampled < 0 || sampled >= n_vocab) {
            fprintf(stderr, "[qlh_keep_head] sampler returned an invalid token id: %d\n",
                    (int) sampled);
            return -6;
        }
        *out_token = (int32_t) sampled;
        return 0;
    }
    *out_token = best;
    return 0;
}

/* COMM-MULTIPOS-01: one argmax per input row, for one contiguous sequence.
 * out_capacity counts int32_t elements (not bytes); it must be >= n_tokens.
 * NULL metadata means n_seq_id=1, seq_id=0, positions=n_past+i. Explicit
 * positions take precedence over n_past and must be contiguous nonnegative
 * int32 positions starting at positions[0]. A constant nonzero seq_id is valid.
 * Returns -5 for invalid input/allocation failure, -2 for decode failure,
 * -6 for missing logits/vocab, -7 for non-finite logits. On ANY failure the
 * caller's output remains untouched. This does not roll back decoded KV.
 * The scalar entry point and its ABI/metadata semantics remain unchanged. */
int32_t qlh_kh_forward_embd_tokens(void * handle_void,
                                   const float * embd, int32_t n_tokens, int32_t n_past,
                                   const int32_t * n_seq_id, const int32_t * seq_ids,
                                   const int32_t * positions,
                                   int32_t * out_tokens, int32_t out_capacity) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle != NULL) {
        handle->last_error = 0;
        handle->last_n_tokens = n_tokens; /* ★ DFLASH-01: getter 的行数来源 */
    }
    if (handle == NULL || handle->ctx == NULL || handle->n_embd <= 0 ||
        embd == NULL || out_tokens == NULL || n_tokens <= 0 ||
        out_capacity < n_tokens ||
        (size_t) n_tokens > SIZE_MAX / sizeof(int32_t) ||
        (size_t) n_tokens > SIZE_MAX / sizeof(float) / (size_t) handle->n_embd) {
        return -5;
    }
    const int32_t pos_start = positions == NULL ? n_past : positions[0];
    if (pos_start < 0 || (int64_t) pos_start + n_tokens - 1 > INT32_MAX) {
        return -5;
    }
    const int32_t sequence = seq_ids == NULL ? 0 : seq_ids[0];
    if (sequence < 0) {
        return -5;
    }
    for (int32_t i = 0; i < n_tokens; ++i) {
        if ((n_seq_id != NULL && n_seq_id[i] != 1) ||
            (seq_ids != NULL && seq_ids[i] != sequence) ||
            (positions != NULL && positions[i] != pos_start + i)) {
            return -5;
        }
    }

    int32_t * result = (int32_t *) malloc((size_t) n_tokens * sizeof(int32_t));
    if (result == NULL) {
        return -5;
    }
    struct llama_batch batch = llama_batch_init(n_tokens, handle->n_embd, handle->n_seq_max);
    /* ★ PERF-SKIP-LOGITS-01：多位置 argmax 同样**恒开** logits（原因同上）。 */
    if (qlh_fill_explicit(&batch, n_tokens, pos_start, n_seq_id, seq_ids, positions, 1,
                          handle->n_seq_max, handle->kv_unified) != 0) {
        llama_batch_free(batch);
        return -5;
    }
    if (qlh_fill_pos_planar(&batch, n_tokens, pos_start, positions,
                            handle->n_pos_per_embd) != 0) {
        llama_batch_free(batch);
        free(result);
        return -5;
    }
    memcpy(batch.embd, embd, (size_t) n_tokens * (size_t) handle->n_embd * sizeof(float));
    const int32_t decode_rc = llama_decode(handle->ctx, batch);
    llama_batch_free(batch);
    if (decode_rc != 0) {
        handle->last_error = decode_rc;
        free(result);
        return -2;
    }

    const struct llama_model * model = llama_get_model(handle->ctx);
    const struct llama_vocab * vocab = model == NULL ? NULL : llama_model_get_vocab(model);
    const int32_t n_vocab = vocab == NULL ? 0 : llama_vocab_n_tokens(vocab);
    int32_t rc = n_vocab > 0 ? 0 : -6;
    for (int32_t row = 0; rc == 0 && row < n_tokens; ++row) {
        const float * logits = llama_get_logits_ith(handle->ctx, row);
        if (logits == NULL) {
            rc = -6;
            break;
        }
        int32_t best = 0;
        float best_logit = logits[0];
        for (int32_t i = 0; i < n_vocab; ++i) {
            const float value = logits[i];
            if (!isfinite(value)) {
                fprintf(stderr, "[qlh_keep_head] non-finite logits (row=%d, vocab_index=%d)\n",
                        row, i);
                rc = -7;
                break;
            }
            if (value > best_logit) {
                best_logit = value;
                best = i;
            }
        }
        result[row] = best;
    }
    if (rc == 0) {
        memcpy(out_tokens, result, (size_t) n_tokens * sizeof(int32_t));
    }
    free(result);
    return rc;
}

/* Optional capabilities; older shims do not export these symbols. */
int32_t qlh_kh_finite_logits_guard(void) {
    return 1;
}



/* Copy the trunk's last hidden state from the most recent decode.
 *
 * Only populated when the handle was loaded with SA_MTP=1 on a model that has a NextN head;
 * otherwise returns -5. `capacity` is in floats and must be >= n_embd.
 */
int32_t qlh_kh_last_hidden(void * handle_void, float * out, int32_t capacity) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || handle->ctx == NULL || out == NULL ||
        handle->n_embd <= 0 || capacity < handle->n_embd) {
        return -5;
    }
    const float * src = llama_get_embeddings_nextn(handle->ctx);
    if (src == NULL) {
        return -6;
    }
    memcpy(out, src, (size_t) handle->n_embd * sizeof(float));
    return 0;
}

/* Draft one token from the MTP head.
 *
 * The draft graph takes two inputs per position: the token id (its embedding is looked up
 * inside the graph) and the target model's last hidden state, handed over through the
 * extended batch. Output is that graph's logits, argmaxed here because the port only ever
 * exposes argmax (see the sampling note in qlh_kh_forward_embd_token).
 *
 * hidden must be the target's final hidden for the position that FOLLOWS `token` in the
 * sequence -- the head predicts token n+1 from (h_n, x_{n+1}).
 *
 * Returns 0 on success, -5 for bad input, -2 when llama_process fails, -6 when no logits,
 * -7 when the logits are not finite.
 */
int32_t qlh_kh_mtp_draft(void * handle_void,
                         const float * hidden,
                         int32_t n_embd,
                         int32_t token,
                         int32_t n_past,
                         int32_t * out_token) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle != NULL) {
        handle->last_error = 0;
    }
    if (handle == NULL || handle->mtp_ctx == NULL || hidden == NULL ||
        out_token == NULL || n_embd != handle->n_embd || token < 0 || n_past < 0) {
        if (handle != NULL) handle->last_error = -5;
        return -5;
    }

    struct llama_batch_ext * batch = llama_batch_ext_init(handle->mtp_ctx);
    if (batch == NULL) {
        handle->last_error = -5;
        return -5;
    }

    int32_t rc = 0;
    do {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, (llama_token) token);
        if (idx < 0) { rc = -5; break; }

        struct llama_embd state;
        state.data   = hidden;
        state.n_rows = 1;
        state.n_embd = (size_t) n_embd;
        if (!llama_batch_ext_set_embd_state(batch, idx, state)) { rc = -5; break; }

        const llama_pos pos = (llama_pos) n_past;
        if (!llama_batch_ext_set_pos(batch, idx, &pos)) { rc = -5; break; }
        if (!llama_batch_ext_set_output_logits(batch, idx, true)) { rc = -5; break; }

        const int32_t drc = llama_process(handle->mtp_ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        if (drc != 0) { rc = -2; break; }

        const float * logits = llama_get_logits_ith(handle->mtp_ctx, 0);
        if (logits == NULL) { rc = -6; break; }

        const int n_vocab = qlh_vocab_size(handle);
        int best = 0;
        float best_v = logits[0];
        for (int i = 1; i < n_vocab; ++i) {
            const float v = logits[i];
            if (!isfinite(v)) { rc = -7; break; }
            if (v > best_v) { best_v = v; best = i; }
        }
        if (rc == -7) break;
        *out_token = best;
    } while (0);

    llama_batch_ext_free(batch);
    handle->last_error = rc;
    return rc;
}

/* Drop KV/state at and after `p0` for one sequence, keeping [0, p0).
 *
 * Speculative decode verifies k tokens, accepts j <= k, and has to retract the rest. Plain
 * KV can be trimmed by position, but a hybrid model's linear layers keep recurrent state
 * that has no position index; llama.cpp covers that with n_rs_seq snapshots (SA_N_RS_SEQ),
 * and refuses a rollback that would land behind them. */
int32_t qlh_kh_seq_rm(void * handle_void, int32_t seq_id, int32_t p0) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || handle->ctx == NULL || seq_id < 0 || p0 < 0) {
        return -5;
    }
    llama_memory_t mem = llama_get_memory(handle->ctx);
    if (mem == NULL) {
        return -5;
    }
    return llama_memory_seq_rm(mem, (llama_seq_id) seq_id, (llama_pos) p0, -1) ? 0 : -6;
}

int32_t qlh_kh_n_vocab(void * handle_void) {
    return qlh_vocab_size((const qlh_keep_head *) handle_void);
}

int32_t qlh_kh_n_embd(void * handle_void) {    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    return handle == NULL ? 0 : handle->n_embd;
}

int32_t qlh_kh_n_layer(void * handle_void) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    return handle == NULL ? 0 : handle->n_layer;
}

/* ★ 2026-09-24：把**最近一次** `llama_decode` 的原始返回码透出（0 = 本轮无错误）。
 *
 * 动机：`qlh_kh_forward*` 的返回码是稳定的负值（-2 = decode 失败），但 llama.cpp 的具体
 * 错误码（例如显存/ctx 相关）以前被丢掉 ⇒ 长时 decode 在特定 ctx 下失败时无法定位。
 * 约定：每次 forward 入口清零，因此调用方在收到 -2 之后立刻读到的就是**本次**的 rc。
 * 旧版 shim 没有这个符号 ⇒ 调用方**必须**按 optional 处理（`hasattr`），缺了就走旧路径。 */
int32_t qlh_kh_last_error(void * handle_void) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL) {
        return 0;
    }
    return handle->last_error;
}

/* ★ 2026-10-26（PERF-SKIP-LOGITS-01）：**只影响"输出 hidden"的入口**的 logits 策略。
 *
 * `skip = 1` ⇒ `qlh_kh_forward` / `qlh_kh_forward_embd` / `qlh_kh_forward_embd_seq`
 * 一行都不标 `batch.logits`，llama.cpp 的 `n_outputs` 变 0，`lm_head`（`output.weight`）
 * 不再产生有效计算 ⇒ 省掉每步几百 MB 的权重读取。hidden 输出**不变**（`layer_inp`
 * 通道与 logits 掩码无关）。
 *
 * `skip = 0`（`qlh_kh_load` 后的初值）⇒ 保持既有行为（每行都标）。
 *
 * 返回 0 成功，-1 参数非法（NULL 句柄或 skip 不是 0/1）。
 *
 * ⚠️ 这是**可选符号**：旧版 shim 没有它 ⇒ Python 侧必须按 optional 处理
 *    （`hasattr`），缺了又要求跳过时必须 fail-loud，不能静默退回"仍算 logits"。 */
int32_t qlh_kh_set_skip_hidden_logits(void * handle_void, int32_t skip) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || (skip != 0 && skip != 1)) {
        return -1;
    }
    handle->skip_hidden_logits = skip;
    return 0;
}

/* ★ 2026-10-29（SAMPLING-01）：建立/替换本句柄的采样链。
 *
 * 语义（"关闭"取值与调用方的直觉一致，越界值一律按关闭处理，而不是报错，
 * 因为 OpenAI 客户端会传 0 / 1.0 这类"关掉它"的值）：
 *   temperature    <= 0     ⇒ greedy（确定性；与 argmax 逐位等价，见下）
 *   top_p          >= 1.0   ⇒ 关闭
 *   top_k          <= 0     ⇒ 关闭
 *   repeat_penalty == 1.0   ⇒ 关闭，除非 freq/present 非零（那两个也需要历史）
 *   repeat_last_n  <= 0     ⇒ 用 QLH_PENALTY_LAST_N_DEFAULT（llama.cpp CLI 的默认 64）
 *   seed                    ⇒ 直接交给 dist 采样器
 *   temperature    == QLH_SAMPLING_DISABLE (-1.0f)
 *                           ⇒ **拆掉采样链，回到 argmax 缺省**（见下方"为什么需要它"）
 *
 * 为什么 temperature <= 0 建 greedy 链、而不是把 `handle->sampler` 置回 NULL：
 *   1. 二者输出**逐位相同** —— 实测 2000 轮随机 logits（含 500 轮强制并列）零分歧
 *      （shim/_test/probe_sampler.c Q2）。所以"确定性"不受影响。
 *   2. `sampler == NULL` 在本文件里的含义是"**从未开过采样**"，那是缺省路径，
 *      也是既有吞吐数据的基线。让显式要求 temperature=0 的调用方也真正走一遍
 *      采样路径，才能验证 llama_sampler_sample 的 idx、accept 状态机与 greedy 链
 *      本身是对的 —— 否则 §6.1 的回归测试根本碰不到新代码。
 *   代价是 greedy 要为一次 sample 分配 n_vocab 大小的候选数组；实测约 0.15 ms
 *   （n_vocab=248320），相对约 50 ms 的 step 是 0.3%。
 *
 * ⚠️ 为什么还需要那个显式的 -1 关闭值：长驻服务在同一句柄上服务多个请求，而
 *    "temperature <= 0 ⇒ greedy" 会把"客户端没传任何采样参数"变成"建了一条 greedy 链"。
 *    greedy 与 argmax 逐位等价，所以数值上无害 —— 但 sampler != NULL 会让回显报告
 *    mode=greedy 而不是 argmax，客户端无法区分"我没要求采样"和"我要求了 temperature=0"。
 *    更实际的问题是状态残留：会改采样配置的请求之后，下一个什么都不传的请求若只靠
 *    "不调用 setter"，就会**静默沿用**上一个请求的参数（penalties 尤其明显）。
 *    所以"回到缺省"必须是能主动表达的动作，而不是"省略"。
 *
 * 链的添加顺序按 llama.cpp 惯例：penalties → top_k → top_p → temp → dist。
 * `dist` 必须最后（它是唯一"选出一个 token"的采样器，前面几个只改分布）。
 *
 * 返回 0 成功；-1 参数非法（NULL 句柄 / NaN / Inf）；-2 采样器分配失败
 * （此时 handle 回到 argmax，不会半残）。重复调用先 free 旧链，不泄漏。 */
#define QLH_PENALTY_LAST_N_DEFAULT 64
#define QLH_SAMPLING_DISABLE (-1.0f)

int32_t qlh_kh_set_sampling(void * handle_void,
                            float    temperature,
                            float    top_p,
                            int32_t  top_k,
                            float    repeat_penalty,
                            int32_t  repeat_last_n,
                            float    penalty_freq,
                            float    penalty_present,
                            uint32_t seed) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL) {
        return -1;
    }
    /* NaN 会让 temp / top_p 的比较全部为假，静默退化成"看起来设了、其实没设"。
     * Inf 在 top_p 上等于"全关"，但遇到就报错更能暴露调用方的 bug。 */
    if (!isfinite(temperature) || !isfinite(top_p) ||
        !isfinite(repeat_penalty) || !isfinite(penalty_freq) || !isfinite(penalty_present)) {
        return -1;
    }
    if (top_k < 0) {
        top_k = 0;
    }

    /* 显式"回到 argmax"：拆掉旧链（无论是否真的有一条），并把回显字段恢复成加载时的
     * 中性值。幂等 —— 本来就是缺省时再关一次也算成功。 */
    if (temperature == QLH_SAMPLING_DISABLE) {
        if (handle->sampler != NULL) {
            llama_sampler_free(handle->sampler);
            handle->sampler = NULL;
        }
        handle->sampling_temperature    = 0.0f;
        handle->sampling_top_p          = 1.0f;
        handle->sampling_top_k          = 0;
        handle->sampling_repeat_penalty = 1.0f;
        handle->sampling_repeat_last_n  = 0;
        handle->sampling_penalty_freq   = 0.0f;
        handle->sampling_penalty_present= 0.0f;
        handle->sampling_seed           = LLAMA_DEFAULT_SEED;
        return 0;
    }

    /* 先拆旧链：无论下面成不成功，句柄都不能同时指向两条链。 */
    if (handle->sampler != NULL) {
        llama_sampler_free(handle->sampler);
        handle->sampler = NULL;
    }

    /* n_vocab 有三个用途：penalties 的窗口上限、判断 top_k 是否"等于全开"
     * （top_k >= n_vocab 就是不截断）、以及如实回显。取不到就失败，不猜值。 */
    const int32_t n_vocab = qlh_vocab_size(handle);
    if (n_vocab <= 0) {
        return -1;
    }

    const int greedy = temperature <= 0.0f;
    const int want_penalties = (repeat_penalty != 1.0f && repeat_last_n > 0) ||
                               penalty_freq != 0.0f || penalty_present != 0.0f;
    int32_t last_n = repeat_last_n;
    if (want_penalties && last_n <= 0) {
        last_n = QLH_PENALTY_LAST_N_DEFAULT;
    }
    if (want_penalties && last_n > n_vocab) {
        last_n = n_vocab;      /* llama.cpp 自己也会 clamp，这里同步以便如实回显 */
    }
    const int want_top_k = !greedy && top_k > 0 && top_k < n_vocab;
    const int want_top_p = !greedy && top_p > 0.0f && top_p < 1.0f;

    struct llama_sampler * chain = NULL;
    if (!want_penalties && greedy) {
        /* 最常见也最轻的一条：纯 greedy，不需要链，也不需要候选数组排序。 */
        chain = llama_sampler_init_greedy();
    } else {
        struct llama_sampler * built = llama_sampler_chain_init(
            llama_sampler_chain_default_params());
        if (built == NULL) {
            return -2;
        }
        /* chain_add 接管所有权，失败时只需 free 链本身。 */
        if (want_penalties) {
            struct llama_sampler * pen = llama_sampler_init_penalties(
                n_vocab, last_n,
                repeat_penalty,
                penalty_freq,
                penalty_present);
            if (pen == NULL) {
                llama_sampler_free(built);
                return -2;
            }
            llama_sampler_chain_add(built, pen);
        }
        if (want_top_k) {
            struct llama_sampler * s = llama_sampler_init_top_k(top_k);
            if (s == NULL) { llama_sampler_free(built); return -2; }
            llama_sampler_chain_add(built, s);
        }
        if (want_top_p) {
            /* min_keep = 1：与 llama.cpp 的 common 默认一致，保证 top_p 不会把所有候选砍光。 */
            struct llama_sampler * s = llama_sampler_init_top_p(top_p, 1);
            if (s == NULL) { llama_sampler_free(built); return -2; }
            llama_sampler_chain_add(built, s);
        }
        if (greedy) {
            struct llama_sampler * s = llama_sampler_init_greedy();
            if (s == NULL) { llama_sampler_free(built); return -2; }
            llama_sampler_chain_add(built, s);
        } else {
            struct llama_sampler * s = llama_sampler_init_temp(temperature);
            if (s == NULL) { llama_sampler_free(built); return -2; }
            llama_sampler_chain_add(built, s);
            s = llama_sampler_init_dist(seed);      /* 必须最后 */
            if (s == NULL) { llama_sampler_free(built); return -2; }
            llama_sampler_chain_add(built, s);
        }
        chain = built;
    }
    if (chain == NULL) {
        return -2;
    }

    handle->sampler = chain;
    /* 记录**实际生效**的值（而不是调用方传来的原始值），供 Python 侧如实回显。 */
    handle->sampling_temperature    = greedy ? 0.0f : temperature;
    handle->sampling_top_p          = want_top_p ? top_p : 1.0f;
    handle->sampling_top_k          = want_top_k ? top_k : 0;
    handle->sampling_repeat_penalty = want_penalties ? repeat_penalty : 1.0f;
    handle->sampling_repeat_last_n  = want_penalties ? last_n : 0;
    handle->sampling_penalty_freq   = want_penalties ? penalty_freq : 0.0f;
    handle->sampling_penalty_present= want_penalties ? penalty_present : 0.0f;
    /* ⚠️ 实测（shim/_test/probe_sampler.c Q4，2026-10-29）：把 LLAMA_DEFAULT_SEED
     * （0xFFFFFFFF）交给 `llama_sampler_init_dist` **不是**"用 4294967295 当种子"，
     * 而是"随机抽一个"：两条链分别拿到 0xBEEE9BDC / 0x2ECD8F8F，序列也不同。
     * 所以这里必须回读 `llama_sampler_get_seed()` 的**实际**种子 —— 原样回显
     * 0xFFFFFFFF 等于对客户端撒谎，说了一个它并没有用的种子。greedy 链没有 dist，
     * get_seed 会返回 LLAMA_DEFAULT_SEED，那种情况下种子无意义，原值即可。 */
    handle->sampling_seed           = greedy
        ? seed
        : llama_sampler_get_seed(chain);
    return 0;
}

/* 回显**实际生效**的采样配置。返回 1 = 采样链在（含 greedy），0 = argmax 缺省。
 * 为什么要有它：server.py 必须告诉客户端"你的参数真的被采纳了多少"，而 clamp 与
 * "关闭"判定都发生在本函数里（top_k >= n_vocab 会被关掉、repeat_last_n 会被补成
 * 默认值……）。让 Python 自己复算一遍这些规则，等于把同一份语义抄成两份，
 * 一旦漂移，回显就是对客户端撒谎。任何 out 指针都可以是 NULL。
 *
 * 这是**可选符号**：旧 shim 没有它 ⇒ Python 侧按 optional 处理。 */
int32_t qlh_kh_get_sampling(void * handle_void,
                            float * temperature, float * top_p, int32_t * top_k,
                            float * repeat_penalty, int32_t * repeat_last_n,
                            float * penalty_freq, float * penalty_present,
                            uint32_t * seed) {
    const qlh_keep_head * handle = (const qlh_keep_head *) handle_void;
    if (handle == NULL) {
        return 0;
    }
    const int on = handle->sampler != NULL ? 1 : 0;
    if (temperature     != NULL) { *temperature     = handle->sampling_temperature; }
    if (top_p           != NULL) { *top_p           = handle->sampling_top_p; }
    if (top_k           != NULL) { *top_k           = handle->sampling_top_k; }
    if (repeat_penalty  != NULL) { *repeat_penalty  = handle->sampling_repeat_penalty; }
    if (repeat_last_n   != NULL) { *repeat_last_n   = handle->sampling_repeat_last_n; }
    if (penalty_freq    != NULL) { *penalty_freq    = handle->sampling_penalty_freq; }
    if (penalty_present != NULL) { *penalty_present = handle->sampling_penalty_present; }
    if (seed            != NULL) { *seed            = handle->sampling_seed; }
    return on;
}

/* ★ P3：清空 KV / recurrent 记忆。跨机服务在**同一进程**里服务多条连接时必须调用 ——
 * 否则新连接从位置 0 开始会与上一条连接留下的位置冲突（llama.cpp 报
 * "tokens ... have inconsistent sequence positions"，实测表现为远端 runner_failed）。
 *
 * ★ 2026-10-29（SAMPLING-01）：同时 reset 采样链。penalties 采样器保留最近
 * `repeat_last_n` 个 token，不清就会让**上一条序列的结尾**去惩罚**下一条序列的开头**；
 * dist 的 RNG 状态同理。注意不能改成"重建链"：重建会把 seed 重置回初始状态，
 * 那样每条请求从同一种子起步是合理的，但也会丢掉我们刻意要保留的链配置。
 * `llama_sampler_reset` 正是上游为此提供的接口。 */
/* Clear the MTP context's memory. Independent of the trunk so a caller can recycle the draft
 * head alone (recursive drafting reuses one mtp_ctx for many candidate positions). */
static void qlh_mtp_clear(qlh_keep_head * handle) {
    if (handle == NULL || handle->mtp_ctx == NULL) {
        return;
    }
    llama_memory_t mem = llama_get_memory(handle->mtp_ctx);
    if (mem != NULL) {
        llama_memory_clear(mem, true);
    }
}

int32_t qlh_kh_mtp_reset(void * handle_void) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || handle->mtp_ctx == NULL) {
        return -5;
    }
    qlh_mtp_clear(handle);
    return 0;
}

void qlh_kh_reset(void * handle_void) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || handle->ctx == NULL) {
        return;
    }
    handle->last_error = 0;
    if (handle->sampler != NULL) {
        llama_sampler_reset(handle->sampler);
    }
    llama_memory_t mem = llama_get_memory(handle->ctx);
    if (mem != NULL) {
        llama_memory_clear(mem, true);
    }
    /* The draft head is stateless per call, but its context is not: positions accumulate in
     * mtp_ctx's KV, so a second generate in the same process would draft from a context still
     * sitting at the old position and every draft would be wrong. Clear it here so one reset
     * means "same state as a fresh process". */
    qlh_mtp_clear(handle);
}

void qlh_kh_close(void * handle_void) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL) {
        return;
    }
    /* ★ 2026-10-29（SAMPLING-01）：先free采样链。它不持有 ctx/model 的引用
     * （greedy 与 penalties 只自带计数器），所以放在 free(ctx) 之前或之后都安全，
     * 但放在之前更明确：句柄里所有堆对象都按"后建先拆"的顺序释放。 */
    if (handle->sampler != NULL) {
        llama_sampler_free(handle->sampler);
        handle->sampler = NULL;
    }
    if (handle->mtp_ctx != NULL) {
        llama_free(handle->mtp_ctx);
        handle->mtp_ctx = NULL;
    }
    if (handle->ctx != NULL) {
        llama_free(handle->ctx);
        handle->ctx = NULL;
    }
    if (handle->model != NULL) {
        llama_model_free(handle->model);
        handle->model = NULL;
    }
    free(handle);
}

/* Read-only R3 load receipt. Values come from the loaded native handle. */
uint64_t qlh_kh_model_bytes(void * ptr) {
    qlh_keep_head * h = (qlh_keep_head *) ptr;
    return h && h->model ? llama_model_size(h->model) : 0;
}
uint32_t qlh_kh_context_size(void * ptr) {
    qlh_keep_head * h = (qlh_keep_head *) ptr;
    return h && h->ctx ? llama_n_ctx(h->ctx) : 0;
}
uint32_t qlh_kh_batch_size(void * ptr) {
    qlh_keep_head * h = (qlh_keep_head *) ptr;
    return h && h->ctx ? llama_n_batch(h->ctx) : 0;
}

/* ---------------------------------------------------------------------------
 * Text I/O for the OpenAI-compatible endpoint and the TUI.
 *
 * Everything above deals in token ids, which is enough to measure throughput but
 * not to show a human what the model produced. These three entry points close that
 * gap without touching the forward path.
 *
 * `token_to_piece` writes into a caller-owned buffer instead of returning a heap
 * pointer, so the binding does not have to agree with this DLL on an allocator.
 * It returns the number of bytes written, or the required size when `buf` is NULL
 * or too small -- the usual two-call pattern.
 * ------------------------------------------------------------------------- */

int32_t qlh_kh_token_to_piece(void * handle_void, int32_t token, char * buf, int32_t buf_len) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || handle->model == NULL) return -1;
    const struct llama_vocab * vocab = llama_model_get_vocab(handle->model);
    if (vocab == NULL) return -1;

    /* A single token is at most a few bytes, but a lenient upper bound avoids
     * growing the buffer in a loop; llama_token_to_piece reports the real size. */
    char local[256];
    const int32_t written = llama_token_to_piece(vocab, token, local, (int32_t) sizeof(local), 0, false);
    if (written < 0) return written;
    if (buf == NULL || buf_len < written) return written;
    memcpy(buf, local, (size_t) written);
    return written;
}

/* Returns the token count, or a negative llama error code. Same two-call pattern:
 * pass tokens == NULL to learn the required length. */
int32_t qlh_kh_tokenize(void * handle_void, const char * text, int32_t text_len,
                        int32_t * tokens, int32_t max_tokens) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || handle->model == NULL || text == NULL) return -1;
    const struct llama_vocab * vocab = llama_model_get_vocab(handle->model);
    if (vocab == NULL) return -1;

    /* add_special = true so the caller can pass raw chat text.
     *
     * parse_special used to be false, reasoning that template markers are literal text.
     * That was wrong for chat prompts: with parse_special=false the tokenizer splits
     * "<|im_start|>" into ordinary fragments, so the model never sees a chat structure and
     * answers everything with degenerate repetition ("是 是 是 ..."). It is an environment
     * switch now so plain-text callers keep the old behaviour. */
    const char * parse_special_env = getenv("SA_PARSE_SPECIAL");
    const bool parse_special = (parse_special_env != NULL) && (atoi(parse_special_env) != 0);
    const int32_t n = llama_tokenize(vocab, text, text_len, tokens, max_tokens, true, parse_special);
    return n;
}

/* True when this token ends the generation (EOS or an equivalent stop token). */
int32_t qlh_kh_is_eog(void * handle_void, int32_t token) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || handle->model == NULL) return 0;
    const struct llama_vocab * vocab = llama_model_get_vocab(handle->model);
    return vocab != NULL && llama_vocab_is_eog(vocab, token) ? 1 : 0;
}
