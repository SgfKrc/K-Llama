// [QLH-SPLIT] v2.2 段级 offload 的测量装置：一个进程只加载一个 range，只跑它自己的计算量。
//
// 为什么不用链路测：链路里 head 和 tail 的收益混在一起，还叠着分段协议与回环网络；而问题
// 只有一个——"这一段放到 GPU 上，它的那部分计算是快了还是慢了"。一个 range + 一个 context
// + 一个 decode 循环就是这个问题的完整实验，而且退出时 llama_moe_cache 的析构会打印命中率
// （走 llama-server/split-stage 就拿不到：那两个进程是被 taskkill /F 杀的，不跑析构）。
//
// head：token 输入，embeddings=true（输出 pre-norm hidden 的同一套图），测 32 层的 pp/tg。
// tail：hidden 行输入（合成随机行，尺度按 n_embd 归一），测 8 层 + 输出层的 pp/tg。
//
// 用法：
//   seg_probe -m MODEL --layer-range lo:hi --role head|tail [-ngl N|all]
//             [--moe-cache-mib N] [--moe-cache-cpu-share F] [--no-repack] [--no-host]
//             [-t N] [-c N] [-b N] [-n N] [-pp N] [--tag STR]
//
// 输出一行机器可读的汇总：PROBE tag=... pp=<t/s> tg=<t/s> ...
#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    std::vector<llama_token> toks(text.size() + 8);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), add_special, true);
    if (n < 0) {
        toks.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), add_special, true);
    }
    toks.resize(n > 0 ? n : 0);
    return toks;
}

static void die(const char * msg) {
    fprintf(stderr, "seg_probe: %s\n", msg);
    exit(1);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string role;
    std::string tag = "run";
    int32_t lo = -1, hi = -1;
    int32_t ngl = 0;
    int32_t n_threads = 8;
    int32_t n_ctx = 2048;
    int32_t n_batch = 512;
    int32_t n_predict = 32;
    int32_t n_pp = 256;
    int32_t moe_cache_mib = 0;
    float   moe_cpu_share = 0.0f;
    bool    no_repack = false;
    bool    no_host   = false;
    bool    no_op_offload = false;
    int32_t n_cpu_moe = 0;   // -ncmoe: keep the expert weights of the first N local blocks in the CPU

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * name) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(1); }
            return argv[++i];
        };
        if      (a == "-m" || a == "--model")          { model_path = next(a.c_str()); }
        else if (a == "--role")                        { role = next("--role"); }
        else if (a == "--layer-range") {
            const std::string v = next("--layer-range");
            if (sscanf(v.c_str(), "%d:%d", &lo, &hi) != 2) { fprintf(stderr, "bad --layer-range %s\n", v.c_str()); return 1; }
        }
        else if (a == "-ngl" || a == "--gpu-layers")   {
            const std::string v = next("-ngl");
            ngl = (v == "all") ? -1 : (int32_t) std::strtol(v.c_str(), nullptr, 10);
        }
        else if (a == "--moe-cache-mib")               { moe_cache_mib = (int32_t) std::strtol(next(a.c_str()).c_str(), nullptr, 10); }
        else if (a == "--moe-cache-cpu-share")         { moe_cpu_share = std::stof(next(a.c_str())); }
        else if (a == "--no-repack" || a == "-nr")     { no_repack = true; }
        else if (a == "--no-host")                     { no_host = true; }
        else if (a == "--no-op-offload")               { no_op_offload = true; }
        else if (a == "-ncmoe" || a == "--n-cpu-moe")  { n_cpu_moe = (int32_t) std::strtol(next(a.c_str()).c_str(), nullptr, 10); }
        else if (a == "-t" || a == "--threads")        { n_threads = (int32_t) std::strtol(next(a.c_str()).c_str(), nullptr, 10); }
        else if (a == "-c" || a == "--ctx-size")       { n_ctx = (int32_t) std::strtol(next(a.c_str()).c_str(), nullptr, 10); }
        else if (a == "-b" || a == "--batch-size")     { n_batch = (int32_t) std::strtol(next(a.c_str()).c_str(), nullptr, 10); }
        else if (a == "-n" || a == "--n-predict")      { n_predict = (int32_t) std::strtol(next(a.c_str()).c_str(), nullptr, 10); }
        else if (a == "--pp")                          { n_pp = (int32_t) std::strtol(next(a.c_str()).c_str(), nullptr, 10); }
        else if (a == "--tag")                         { tag = next("--tag"); }
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || (role != "head" && role != "tail") || lo < 0 || hi <= lo) {
        fprintf(stderr, "usage: seg_probe -m MODEL --layer-range lo:hi --role head|tail [-ngl N|all]\n"
                        "                 [--moe-cache-mib N] [--moe-cache-cpu-share F] [--no-repack] [--no-host]\n"
                        "                 [-t N] [-c N] [-b N] [-n N] [-pp N] [--tag STR]\n");
        return 1;
    }
    const bool is_head = (role == "head");

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.split_layer_lo = lo;
    mparams.split_layer_hi = hi;
    mparams.n_gpu_layers   = ngl;
    mparams.no_host        = no_host;
    mparams.use_extra_bufts = !no_repack;

    // -ncmoe：把前 N 个本地块的专家权重钉在 CPU（common/arg.cpp 的同一正则），层本身照常
    // 按 -ngl 决定在不在 GPU。这是 MoE 专家缓存唯一有意义的形态：层在 GPU、专家在 host。
    std::vector<std::string>                    ov_strs;
    std::vector<llama_model_tensor_buft_override> ovs;
    {
        static const char * const exps = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
        for (int32_t i = 0; i < n_cpu_moe; ++i) {
            ov_strs.push_back("blk\\." + std::to_string(i) + exps);
            static ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            ovs.push_back({ ov_strs.back().c_str(), ggml_backend_dev_buffer_type(cpu_dev) });
        }
        if (!ovs.empty()) {
            ovs.push_back({ nullptr, nullptr });
            mparams.tensor_buft_overrides = ovs.data();
        }
    }

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) die("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_embd = llama_model_n_embd(model);

    auto cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) n_ctx;
    cparams.n_batch         = (uint32_t) n_batch;
    cparams.n_ubatch        = (uint32_t) n_batch;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    // 与拆段工具同形：链上 head 只回一行 hidden、tail 只回一行 logits（n_outputs_max=1），
    // 这里保持同一套输出缓冲，测出来的计算量才是那一段真实的计算量。
    //
    // 但 head 例外：cparams.embeddings=true 时 llama-context 会把 prefill 的每一行都算成
    // output（"embeddings required but some input tokens were not marked as outputs ->
    // overriding"），n_outputs 从 1 变成 n_batch，于是 n_outputs_max=1 会在 output_reserve
    // 的 GGML_ASSERT 上炸掉。head 的 output buffer 是 embd（logits = no，这段没有 output
    // 层），n_batch*2048*4 = 1 MiB，给满不心疼；tail 才需要卡在 1 行。
    cparams.n_outputs_max   = is_head ? (uint32_t) n_batch : 1;
    cparams.op_offload          = !no_op_offload;
    cparams.moe_cache_size      = (size_t) moe_cache_mib * 1024 * 1024;
    cparams.moe_cache_cpu_share = moe_cpu_share;
    if (is_head) {
        cparams.embeddings          = true;
        cparams.embeddings_pre_norm = true;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) die("context init failed");

    fprintf(stderr, "[seg_probe] tag=%s role=%s range=%d:%d ngl=%d n_embd=%d n_ctx=%d n_batch=%d n_threads=%d "
                    "moe_cache=%d MiB cpu_share=%.2f repack=%s no_host=%d\n",
            tag.c_str(), role.c_str(), lo, hi, ngl, n_embd, n_ctx, n_batch, n_threads,
            moe_cache_mib, moe_cpu_share, no_repack ? "off" : "on", (int) no_host);

    // 输入：head 是真 token；tail 用合成 hidden 行。tail 的隐藏行尺度取 0.05（RMSNorm 之后
    // 的 hidden 量级），MoE gate 的选择分布因此接近真实，不至于全选同一个专家。
    const std::string prompt =
        "The capital of France is Paris. Explain in detail why the city became the political, "
        "cultural and economic centre of the country, and describe the role of the Seine in its "
        "development. Then name three other French cities and one thing each is known for. "
        "Answer in one paragraph, then list the cities again with one sentence each.";
    std::vector<llama_token> toks;
    if (is_head) {
        toks = tokenize(vocab, prompt, true);
        while ((int32_t) toks.size() < n_pp) {
            const auto more = tokenize(vocab, prompt, false);
            toks.insert(toks.end(), more.begin(), more.end());
        }
        toks.resize(n_pp);
    }

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 0.05f);

    auto fill_embd = [&](llama_batch & b, int32_t n) {
        for (int32_t i = 0; i < n; ++i) {
            for (int32_t j = 0; j < n_embd; ++j) {
                b.embd[(size_t) i * n_embd + j] = nd(rng);
            }
        }
    };

    auto make_batch = [&](int32_t n) {
        llama_batch b = llama_batch_init(n, is_head ? 0 : n_embd, 1);
        b.n_tokens = n;
        for (int32_t i = 0; i < n; ++i) {
            b.pos[i]       = 0; // 调用方填
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = 0;
        }
        return b;
    };

    // ---- prefill ----
    llama_batch bp = make_batch(n_pp);
    for (int32_t i = 0; i < n_pp; ++i) {
        bp.pos[i] = i;
        if (is_head) bp.token[i] = toks[i];
    }
    if (!is_head) fill_embd(bp, n_pp);
    bp.logits[n_pp - 1] = 1;

    const double t0 = now_ms();
    if (llama_decode(ctx, bp) != 0) die("prefill decode failed");
    const double t1 = now_ms();
    llama_batch_free(bp);
    const double pp_tps = (double) n_pp / ((t1 - t0) / 1000.0);

    // ---- decode ----
    int32_t  next_tok = is_head ? toks[n_pp - 1] : 0;
    llama_pos pos = n_pp;
    double dec_ms = 0.0;
    int32_t dec_steps = 0;
    for (int32_t step = 0; step < n_predict; ++step) {
        llama_batch b = make_batch(1);
        b.pos[0] = pos++;
        if (is_head) {
            b.token[0] = next_tok;
        } else {
            for (int32_t j = 0; j < n_embd; ++j) b.embd[j] = nd(rng);
        }
        b.logits[0] = 1;

        const double d0 = now_ms();
        const int rc = llama_decode(ctx, b);
        const double d1 = now_ms();
        if (rc != 0) { llama_batch_free(b); die("decode failed"); }

        if (step > 0) { // 第一步吃 cold kernel / 图捕获，不计入
            dec_ms += d1 - d0;
            dec_steps++;
        }
        if (!is_head) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (logits != nullptr) {
                int best = 0;
                const int nv = llama_vocab_n_tokens(vocab);
                for (int i = 1; i < nv; ++i) if (logits[i] > logits[best]) best = i;
                next_tok = best;
            }
        } else {
            const float * h = llama_get_embeddings_ith(ctx, -1);
            if (h == nullptr) die("head produced no hidden row");
        }
        llama_batch_free(b);
    }
    const double tg_tps = dec_steps > 0 ? (double) dec_steps / (dec_ms / 1000.0) : 0.0;

    printf("PROBE tag=%s role=%s range=%d:%d ngl=%d moe_cache_mib=%d cpu_share=%.2f repack=%s no_host=%d "
           "pp=%.2f tg=%.3f pp_rows=%d tg_steps=%d\n",
           tag.c_str(), role.c_str(), lo, hi, ngl, moe_cache_mib, moe_cpu_share,
           no_repack ? "off" : "on", (int) no_host, pp_tps, tg_tps, n_pp, dec_steps);
    fflush(stdout);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
