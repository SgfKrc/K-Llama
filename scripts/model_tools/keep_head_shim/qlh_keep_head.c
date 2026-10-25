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

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct qlh_keep_head {
    struct llama_model   *model;
    struct llama_context *ctx;
    int n_embd;
    int n_layer;
    int mode;        /* 0 = nextn, 1 = layer_inp */
    int cut_layer;
    int n_pos_per_embd;  /* ★ M-RoPE 模型 = 4（每个 token 的位置分量数），否则 1 */
    /* ★ 2026-09-24：最近一次 `llama_decode` 的**原始**返回码（0 = 无错误）。
     * 此前 decode 失败一律折成 -2，把 llama.cpp 的具体错误码丢了 ⇒ 长时 decode 的 ctx 边界
     * 问题无法定位。每次 forward 入口清零（读到的就是本次错误），`qlh_kh_reset` 也清零。 */
    int32_t last_error;
} qlh_keep_head;

static void qlh_set_err(char *err, size_t errlen, const char *msg) {
    if (err != NULL && errlen > 0) {
        snprintf(err, errlen, "%s", msg);
    }
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

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

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
        llama_set_embeddings_layer_inp(ctx, (uint32_t) n_layer, true);
    } else {
        llama_set_embeddings_layer_inp(ctx, (uint32_t) cut_layer, true);
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
    const float * ptr = llama_get_embeddings_layer_inp(handle->ctx, lid);
    if (ptr == NULL) {
        return -3;
    }
    memcpy(out, ptr, (size_t) n_tokens * (size_t) handle->n_embd * sizeof(float));
    return 0;
}

/* 统一标注：**每个 token 都要标输出**。
 * 历史原因：旧的 nextn 通道只对「有输出的行」计算末层 hidden，若只标最后一行，取 n_tokens 行
 * 就会踩到 GGML_ASSERT "tensor read out of bounds"（实测踩过）。现在两个模式都走 layer_inp
 * 通道，这条约束不再是硬性的，但保持"全行标输出"可以不必依赖具体的 sched 行为 ——
 * 上游不需要 logits，只需要**行数完整**。 */
static void qlh_fill_common(struct llama_batch * batch, int32_t n_tokens, int32_t n_past) {
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch->pos[i]       = n_past + i;
        batch->n_seq_id[i]  = 1;
        batch->seq_id[i][0] = 0;
        batch->logits[i]    = 1;
    }
    batch->n_tokens = n_tokens;
}

/* ★ P3：多序列显式绑定 —— 每个 token 自带 `n_seq_id / seq_id / pos`，**不依赖隐式位置递增**。
 * 与主仓 `llama_engine.forward_layers_from_hidden(seq_ids=..., positions=...)` 同一契约；
 * 多序列交错推进时，位置由调用方决定（每序列可各自递增）。
 * ⚠️ `positions == NULL` 时回落到 `n_past + i`（单序列增量语义）—— 少了这一项，
 *    单序列 decode 的第二步会从位置 0 重放，直接触发
 *    "tokens ... have inconsistent sequence positions" 而 decode 失败（实测踩过）。 */
static void qlh_fill_explicit(struct llama_batch * batch, int32_t n_tokens, int32_t n_past,
                              const int32_t * n_seq_id, const int32_t * seq_ids,
                              const int32_t * positions) {
    for (int32_t i = 0; i < n_tokens; ++i) {
        const int32_t per_token = (n_seq_id == NULL) ? 1 : n_seq_id[i];
        batch->n_seq_id[i]  = per_token > 0 ? per_token : 1;
        batch->seq_id[i][0] = (seq_ids == NULL) ? 0 : seq_ids[i];
        batch->pos[i]       = (positions == NULL) ? (n_past + i) : positions[i];
        batch->logits[i]    = 1;
    }
    batch->n_tokens = n_tokens;
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

    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
    }
    qlh_fill_common(&batch, n_tokens, n_past);

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

    struct llama_batch batch = llama_batch_init(n_tokens, handle->n_embd, 1);
    qlh_fill_explicit(&batch, n_tokens, n_past, n_seq_id, seq_ids, positions);
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

    struct llama_batch batch = llama_batch_init(n_tokens, handle->n_embd, 1);
    qlh_fill_explicit(&batch, n_tokens, n_past, n_seq_id, seq_ids, positions);
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
            if (row == n_tokens - 1 && (i == 0 || value > best_logit)) {
                best_logit = value;
                best = i;
            }
        }
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
    struct llama_batch batch = llama_batch_init(n_tokens, handle->n_embd, 1);
    qlh_fill_explicit(&batch, n_tokens, pos_start, n_seq_id, seq_ids, positions);
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

int32_t qlh_kh_n_vocab(void * handle_void) {
    const qlh_keep_head * handle = (const qlh_keep_head *) handle_void;
    if (handle == NULL || handle->model == NULL) {
        return 0;
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(handle->model);
    return vocab == NULL ? 0 : llama_vocab_n_tokens(vocab);
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

/* ★ P3：清空 KV / recurrent 记忆。跨机服务在**同一进程**里服务多条连接时必须调用 ——
 * 否则新连接从位置 0 开始会与上一条连接留下的位置冲突（llama.cpp 报
 * "tokens ... have inconsistent sequence positions"，实测表现为远端 runner_failed）。 */
void qlh_kh_reset(void * handle_void) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL || handle->ctx == NULL) {
        return;
    }
    handle->last_error = 0;
    llama_memory_t mem = llama_get_memory(handle->ctx);
    if (mem != NULL) {
        llama_memory_clear(mem, true);
    }
}

void qlh_kh_close(void * handle_void) {
    qlh_keep_head * handle = (qlh_keep_head *) handle_void;
    if (handle == NULL) {
        return;
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
