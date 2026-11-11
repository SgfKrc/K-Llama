// [QLH-SPLIT] 回归探针：seq_rm(p, -1) 之后重算第 p 行，结果必须与第一次算第 p 行逐位相同。
//
// 为什么这条必须成立：删掉 p 之后的 KV，不改变"第 p 行"的输入 —— 它的注意力只覆盖
// 0..p（含自己），第 p 个 token 也没变。所以重算得到的 logits 必须逐位相同。
// 如果不同，任何"回滚后重算"的路径都不可信：投机回退、prefix cache 复用、ckpt 恢复。
//
// 帧形状可以选 k：k=1 是"删一行再算一行"，k=3 是"删一帧（p..p+2）再重放这一帧"。
// 后者才是投机回退真正做的事，也是观察到偏差的地方。
//
// 两种形态（都不需要 split、不需要 stage、不需要服务端）：
//   n_rs_seq = 0  -> 绕开 recurrent snapshot，seq_rm 走"只能全删"一侧（不投机 server 的默认）
//   n_rs_seq = 3  -> 走 recurrent per-token snapshot 的 partial rollback（rs_idx 延后消费）
//
// 用法：seq_rm_exact.exe <model.gguf> <n_rs_seq> [p=29] [k=1]
// 退出码：0 = PASS（seq_rm 成功、pos_max 降到 p-1、重放第 p 行 logits 逐位相同），1 = FAIL，2 = 错误。
// 输出是固定格式的行，供 build-split/verify_seq_rm_exact.py 解析。
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <vector>

static const char * kPrompt = "The user asked for the weather. I will call the tool now:";

static int argmax_of(const float * v, size_t n) {
    int best = 0;
    for (size_t i = 1; i < n; ++i) {
        if (v[i] > v[best]) {
            best = (int) i;
        }
    }
    return best;
}

// 解码 [pos0, pos0+k) 这几行，只为首行要 logits（比较的对象就是它）。
static int decode_rows(llama_context * ctx, const std::vector<llama_token> & toks, int pos0, int k) {
    llama_batch b = llama_batch_init(k, 0, 1);
    b.n_tokens = k;
    for (int i = 0; i < k; ++i) {
        b.token[i]    = toks[(size_t) (pos0 + i)];
        b.pos[i]      = pos0 + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]   = i == 0 ? 1 : 0;
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <n_rs_seq> [p=29] [k=1] [k_replay=k]\n", argv[0]);
        return 2;
    }
    const char * path   = argv[1];
    const int    rs_seq = atoi(argv[2]);
    const int    p      = argc > 3 ? atoi(argv[3]) : 29;
    const int    k      = argc > 4 ? atoi(argv[4]) : 1;
    const int    k_rep  = argc > 5 ? atoi(argv[5]) : k;

    if (k < 1 || k > 8) {
        fprintf(stderr, "[seq_rm_exact] k must be 1..8\n");
        return 2;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        fprintf(stderr, "[seq_rm_exact] model load failed: %s\n", path);
        return 2;
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const size_t        n_vocab = (size_t) llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 1024;
    cp.n_batch         = 256;
    cp.n_ubatch        = 256;
    cp.n_rs_seq        = (uint32_t) rs_seq;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        fprintf(stderr, "[seq_rm_exact] context init failed\n");
        return 2;
    }

    const int prompt_len = -llama_tokenize(vocab, kPrompt, (int32_t) strlen(kPrompt), nullptr, 0, true, true);
    std::vector<llama_token> toks((size_t) prompt_len + (size_t) p + (size_t) k + 8, 0);
    llama_tokenize(vocab, kPrompt, (int32_t) strlen(kPrompt), toks.data(), prompt_len, true, true);
    toks.resize((size_t) prompt_len);

    if (llama_decode(ctx, llama_batch_get_one(toks.data(), prompt_len)) != 0) {
        fprintf(stderr, "[seq_rm_exact] prompt decode failed\n");
        return 2;
    }

    // 贪心追加到 pos = p-1（即 toks.size() == p），这样下一行就是第 p 行
    while ((int) toks.size() < p) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        llama_token t = (llama_token) argmax_of(lg, n_vocab);
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) {
            fprintf(stderr, "[seq_rm_exact] decode at pos %zu failed\n", toks.size() - 1);
            return 2;
        }
        toks.push_back(t);
    }

    // 第 p 行的 token：由前一行贪心得到；其余 k-1 行随便填（因果掩码下不影响第 p 行）
    {
        const float * lg = llama_get_logits_ith(ctx, -1);
        toks.push_back((llama_token) argmax_of(lg, n_vocab));
    }
    while ((int) toks.size() < p + k) {
        toks.push_back(toks[(size_t) p]);
    }

    // 第一次：帧 [p, p+k)，取第 p 行 logits
    if (decode_rows(ctx, toks, p, k) != 0) {
        fprintf(stderr, "[seq_rm_exact] first frame decode failed\n");
        return 2;
    }
    std::vector<float> first(n_vocab);
    std::memcpy(first.data(), llama_get_logits_ith(ctx, 0), n_vocab * sizeof(float));
    const int argmax_first = argmax_of(first.data(), n_vocab);

    auto * mem = llama_get_memory(ctx);
    const int pos_before = (int) llama_memory_seq_pos_max(mem, 0);

    const bool rm_ok = llama_memory_seq_rm(mem, 0, p, -1);
    const int  pos_after = (int) llama_memory_seq_pos_max(mem, 0);

    // 重放同一帧
    const int rc = decode_rows(ctx, toks, p, k_rep);
    const int pos_replay = rc == 0 ? (int) llama_memory_seq_pos_max(mem, 0) : -1;

    size_t diff_bytes = 0;
    size_t first_diff = (size_t) -1;
    double max_abs    = 0.0;
    int    argmax_second = -1;
    if (rc == 0) {
        const float * second = llama_get_logits_ith(ctx, 0);
        argmax_second = argmax_of(second, n_vocab);
        for (size_t i = 0; i < n_vocab; ++i) {
            if (first[i] != second[i]) {
                ++diff_bytes;
                if (first_diff == (size_t) -1) {
                    first_diff = i;
                }
            }
            const double d = std::fabs((double) first[i] - (double) second[i]);
            if (d > max_abs) {
                max_abs = d;
            }
        }
    }

    printf("[seq_rm_exact] n_rs_seq=%d p=%d k=%d k_replay=%d prompt_tokens=%d n_vocab=%d\n",
           rs_seq, p, k, k_rep, prompt_len, (int) n_vocab);
    printf("[seq_rm_exact] seq_rm=%s pos_max %d -> %d\n", rm_ok ? "true" : "false", pos_before, pos_after);
    printf("[seq_rm_exact] replay rc=%d pos_max=%d\n", rc, pos_replay);
    printf("[seq_rm_exact] argmax first=%d second=%d\n", argmax_first, argmax_second);
    printf("[seq_rm_exact] logits diff=%zu/%d first_diff=%lld max_abs=%.9g\n",
           diff_bytes, (int) n_vocab,
           (long long) (first_diff == (size_t) -1 ? -1 : (long long) first_diff), max_abs);

    const bool pass = rm_ok && pos_after == p - 1 && rc == 0 && pos_replay == p + k_rep - 1 &&
                      diff_bytes == 0 && argmax_first == argmax_second;
    printf("[seq_rm_exact] RESULT %s\n", pass ? "PASS" : "FAIL");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return pass ? 0 : 1;
}
