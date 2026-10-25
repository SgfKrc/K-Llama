// Standalone CPU repack A/B probe. Build against the public llama.h belonging
// to the linked libllama, including its layer_inp exports. No shim is used.
// CLI: MODEL REPACK THREADS CTX PROMPT_CSV FIXED_CSV GREEDY_STEPS REPEATS HIDDEN_OUT
// Fixed outputs predict the next token after each prompt/fixed input row.
// Timings cover llama_decode calls only; greedy output is a parity check,
// not a measurement of complete product generation throughput.

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <locale>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr const char * schema = "qlh.llama_kernel_probe.v1";
constexpr std::uint64_t max_int32 = std::numeric_limits<std::int32_t>::max();
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "Hidden output requires IEEE-754 binary32 floats");

struct ProbeError : std::exception {
    const char * code;
    explicit ProbeError(const char * value) : code(value) {}
    const char * what() const noexcept override { return code; }
};

void require(bool condition, const char * code) {
    if (!condition) {
        throw ProbeError(code);
    }
}

std::uint64_t decimal(std::string_view text, std::uint64_t minimum,
                      std::uint64_t maximum, const char * code) {
    require(!text.empty(), code);
    std::uint64_t value = 0;
    for (char digit : text) {
        require(digit >= '0' && digit <= '9', code);
        const auto next = static_cast<std::uint64_t>(digit - '0');
        require(value <= maximum / 10 &&
                (value != maximum / 10 || next <= maximum % 10), code);
        value = value * 10 + next;
    }
    require(value >= minimum, code);
    return value;
}

std::vector<llama_token> tokens(std::string_view csv, const char * code) {
    require(!csv.empty(), code);
    std::vector<llama_token> result;
    std::size_t start = 0;
    for (;;) {
        const auto comma = csv.find(',', start);
        const auto end = comma == std::string_view::npos ? csv.size() : comma;
        result.push_back(static_cast<llama_token>(
            decimal(csv.substr(start, end - start), 0, max_int32, code)));
        require(result.size() <= 4096, code);
        if (comma == std::string_view::npos) {
            return result;
        }
        start = comma + 1;
    }
}

struct Options {
    const char * model;
    const char * hidden_out;
    bool repack;
    std::int32_t threads;
    std::int32_t ctx;
    std::int32_t greedy_steps;
    std::int32_t repeats;
    std::vector<llama_token> prompt;
    std::vector<llama_token> fixed;
};

Options arguments(int argc, char ** argv) {
    if (argc != 10) {
        std::fputs("Usage: llama_kernel_probe MODEL REPACK(0/1) THREADS CTX "
                   "PROMPT_CSV FIXED_CSV GREEDY_STEPS REPEATS HIDDEN_OUT\n", stderr);
        throw ProbeError("invalid_argument_count");
    }
    require(argv[1][0] != '\0', "empty_model_path");
    require(argv[9][0] != '\0', "empty_hidden_path");
    const std::string_view repack(argv[2]);
    require(repack == "0" || repack == "1", "invalid_repack");
    Options opt;
    opt.model = argv[1];
    opt.hidden_out = argv[9];
    opt.repack = repack == "1";
    opt.threads = static_cast<std::int32_t>(decimal(argv[3], 1, 128, "invalid_threads"));
    opt.ctx = static_cast<std::int32_t>(decimal(argv[4], 1, 32768, "invalid_ctx"));
    opt.prompt = tokens(argv[5], "invalid_prompt_csv");
    // A fixed token is required for the prompt + first-fixed-token warmup.
    opt.fixed = tokens(argv[6], "invalid_fixed_csv");
    opt.greedy_steps = static_cast<std::int32_t>(decimal(argv[7], 0, 4096, "invalid_greedy_steps"));
    opt.repeats = static_cast<std::int32_t>(decimal(argv[8], 1, 20, "invalid_repeats"));
    const auto needed = static_cast<std::uint64_t>(opt.prompt.size()) +
        std::max<std::uint64_t>(opt.fixed.size(), opt.greedy_steps);
    require(needed <= static_cast<std::uint64_t>(opt.ctx), "ctx_too_small");
    // Do not truncate the model, including through a symlink or hard link.
    std::error_code error;
    require(std::string_view(opt.model) != opt.hidden_out &&
            !std::filesystem::equivalent(opt.model, opt.hidden_out, error),
            "hidden_output_is_model");
    return opt;
}

void log_to_stderr(enum ggml_log_level, const char * text, void *) {
    if (text) {
        std::fputs(text, stderr);
    }
}

struct Backend {
    Backend() { llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
    Backend(const Backend &) = delete;
    Backend & operator=(const Backend &) = delete;
};

using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;

class Batch {
public:
    explicit Batch(std::int32_t capacity)
        : data_(llama_batch_init(capacity, 0, 1)), capacity_(capacity) {}
    ~Batch() { llama_batch_free(data_); }
    Batch(const Batch &) = delete;
    Batch & operator=(const Batch &) = delete;

    llama_batch fill(const llama_token * input, std::int32_t count, llama_pos start) {
        require(count > 0 && count <= capacity_ && start >= 0 &&
                static_cast<std::uint64_t>(start) + count <= max_int32,
                "invalid_batch_range");
        require(data_.token && data_.pos && data_.n_seq_id && data_.seq_id && data_.logits,
                "batch_allocation_failed");
        data_.n_tokens = count;
        for (std::int32_t row = 0; row < count; ++row) {
            require(data_.seq_id[row] != nullptr, "batch_allocation_failed");
            data_.token[row] = input[row];
            data_.pos[row] = start + row;
            data_.n_seq_id[row] = 1;
            data_.seq_id[row][0] = 0;
            // Identical output work for warmup, every repeat, and greedy parity.
            data_.logits[row] = 1;
        }
        return data_;
    }

private:
    llama_batch data_;
    std::int32_t capacity_;
};

double elapsed(Clock::time_point start, Clock::time_point end) {
    const double seconds = std::chrono::duration<double>(end - start).count();
    require(std::isfinite(seconds) && seconds >= 0, "invalid_timing");
    return seconds;
}

double decode(llama_context * ctx, llama_batch batch) {
    const auto start = Clock::now();
    const auto rc = llama_decode(ctx, batch);
    if (rc != 0) {
        std::fprintf(stderr, "llama_kernel_probe: llama_decode returned %d\n", static_cast<int>(rc));
        throw ProbeError("decode_failed");
    }
    // Include queued backend work in every measured decode, including repeats
    // without output scans. Output access and comparisons remain outside timers.
    llama_synchronize(ctx);
    const auto end = Clock::now();
    return elapsed(start, end);
}

void reset(llama_context * ctx) {
    const auto memory = llama_get_memory(ctx);
    require(memory != nullptr, "missing_model_memory");
    llama_memory_clear(memory, true);
}

struct Prediction {
    llama_token token;
    double margin;
};

Prediction prediction(llama_context * ctx, std::int32_t row, std::int32_t n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, row);
    require(logits != nullptr, "missing_logits");
    double first = -std::numeric_limits<double>::infinity();
    double second = first;
    llama_token best = 0;
    for (std::int32_t token = 0; token < n_vocab; ++token) {
        const double value = logits[token];
        require(std::isfinite(value), "nonfinite_logits");
        if (value > first) {
            second = first;
            first = value;
            best = token;
        } else if (value > second) {
            second = value;
        }
    }
    // Strict '>' retains the first maximum; tied maxima yield a zero margin.
    const double margin = first - second;
    require(std::isfinite(margin) && margin >= 0, "invalid_margin");
    return {best, margin};
}

struct Result {
    bool requested_repack;
    bool default_repack;
    std::int32_t n_embd;
    std::int32_t n_layer;
    std::int32_t n_vocab;
    std::uint32_t n_ctx;
    std::uint32_t n_batch;
    std::uint32_t n_ubatch;
    std::uint32_t n_seq_max;
    std::int32_t n_threads;
    std::int32_t n_threads_batch;
    std::size_t hidden_rows;
    double load_seconds;
    double context_seconds;
    std::vector<double> prefill_seconds;
    std::vector<double> decode_seconds;
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> fixed_tokens;
    std::vector<llama_token> fixed_argmax;
    std::vector<double> fixed_margins;
    std::vector<llama_token> greedy_tokens;
};

void capture(llama_context * ctx, std::int32_t count, Result & result, std::ofstream & output) {
    for (std::int32_t row = 0; row < count; ++row) {
        const auto next = prediction(ctx, row, result.n_vocab);
        result.fixed_argmax.push_back(next.token);
        result.fixed_margins.push_back(next.margin);
    }
    const float * hidden = llama_get_embeddings_layer_inp(ctx, static_cast<std::uint32_t>(result.n_layer));
    require(hidden != nullptr, "missing_hidden");
    const auto values = static_cast<std::size_t>(count) * result.n_embd;
    for (std::size_t i = 0; i < values; ++i) {
        require(std::isfinite(hidden[i]), "nonfinite_hidden");
    }
    // All rows of this decode, row-major binary32, before the next decode.
    output.write(reinterpret_cast<const char *>(hidden),
                 static_cast<std::streamsize>(values * sizeof(float)));
    require(output.good(), "hidden_write_failed");
}

Result run(const Options & opt) {
    llama_log_set(log_to_stderr, nullptr);
    Backend backend;
    auto mparams = llama_model_default_params();
    const bool default_repack = mparams.use_extra_bufts;
    mparams.n_gpu_layers = 0;
    mparams.use_extra_bufts = opt.repack;
    const auto load_start = Clock::now();
    Model model(llama_model_load_from_file(opt.model, mparams), llama_model_free);
    const auto load_end = Clock::now();
    require(model != nullptr, "model_load_failed");

    Result result{};
    result.requested_repack = opt.repack;
    result.default_repack = default_repack;
    result.load_seconds = elapsed(load_start, load_end);
    result.n_embd = llama_model_n_embd(model.get());
    result.n_layer = llama_model_n_layer(model.get());
    const auto * vocab = llama_model_get_vocab(model.get());
    require(vocab != nullptr, "missing_vocab");
    result.n_vocab = llama_vocab_n_tokens(vocab);
    require(result.n_embd > 0 && result.n_layer > 0 && result.n_vocab >= 2,
            "invalid_model_dimensions");
    for (const auto * sequence : {&opt.prompt, &opt.fixed}) {
        for (llama_token token : *sequence) {
            require(token >= 0 && token < result.n_vocab, "token_out_of_vocab");
        }
    }
    result.prompt_tokens = opt.prompt;
    result.fixed_tokens = opt.fixed;
    result.hidden_rows = opt.prompt.size() + opt.fixed.size();
    const auto max_values = std::min<std::uint64_t>(
        std::numeric_limits<std::size_t>::max() / sizeof(float),
        static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float));
    require(result.hidden_rows <= max_values / static_cast<std::uint64_t>(result.n_embd),
            "hidden_size_overflow");
    const auto prompt_count = static_cast<std::int32_t>(opt.prompt.size());
    const auto batch_size = std::max<std::int32_t>(prompt_count, 1);
    auto cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<std::uint32_t>(opt.ctx);
    cparams.n_batch = static_cast<std::uint32_t>(batch_size);
    cparams.n_ubatch = static_cast<std::uint32_t>(batch_size);
    cparams.n_threads = opt.threads;
    cparams.n_threads_batch = opt.threads;
    cparams.n_seq_max = 1;
    const auto context_start = Clock::now();
    Context ctx(llama_init_from_model(model.get(), cparams), llama_free);
    const auto context_end = Clock::now();
    require(ctx != nullptr, "context_creation_failed");
    result.context_seconds = elapsed(context_start, context_end);
    result.n_ctx = llama_n_ctx(ctx.get());
    result.n_batch = llama_n_batch(ctx.get());
    result.n_ubatch = llama_n_ubatch(ctx.get());
    result.n_seq_max = llama_n_seq_max(ctx.get());
    result.n_threads = llama_n_threads(ctx.get());
    result.n_threads_batch = llama_n_threads_batch(ctx.get());
    require(result.n_ctx >= static_cast<std::uint32_t>(opt.ctx), "effective_ctx_too_small");
    require(result.n_batch >= static_cast<std::uint32_t>(batch_size) &&
            result.n_ubatch >= static_cast<std::uint32_t>(batch_size),
            "effective_batch_too_small");
    require(result.n_seq_max == 1 && result.n_threads > 0 && result.n_threads_batch > 0,
            "invalid_effective_configuration");
    llama_set_embeddings_layer_inp(ctx.get(), static_cast<std::uint32_t>(result.n_layer), true);
    Batch batch(batch_size);

    // One unreported warmup with the same batch shapes as the measured work.
    reset(ctx.get());
    decode(ctx.get(), batch.fill(opt.prompt.data(), prompt_count, 0));
    decode(ctx.get(), batch.fill(opt.fixed.data(), 1, prompt_count));

    std::ofstream hidden(opt.hidden_out, std::ios::binary | std::ios::out | std::ios::trunc);
    require(hidden.is_open(), "hidden_open_failed");
    result.prefill_seconds.reserve(opt.repeats);
    result.decode_seconds.reserve(opt.repeats);
    result.fixed_argmax.reserve(result.hidden_rows);
    result.fixed_margins.reserve(result.hidden_rows);
    result.greedy_tokens.reserve(opt.greedy_steps);
    for (std::int32_t repeat = 0; repeat < opt.repeats; ++repeat) {
        reset(ctx.get());
        const double prefill = decode(ctx.get(), batch.fill(opt.prompt.data(), prompt_count, 0));
        if (repeat == 0) {
            capture(ctx.get(), prompt_count, result, hidden);
        }
        double fixed_seconds = 0;
        for (std::size_t i = 0; i < opt.fixed.size(); ++i) {
            const auto pos = static_cast<llama_pos>(opt.prompt.size() + i);
            fixed_seconds += decode(ctx.get(), batch.fill(&opt.fixed[i], 1, pos));
            if (repeat == 0) {
                capture(ctx.get(), 1, result, hidden);
            }
        }
        require(std::isfinite(fixed_seconds), "invalid_timing");
        result.prefill_seconds.push_back(prefill);
        result.decode_seconds.push_back(fixed_seconds);
    }
    hidden.close();
    require(!hidden.fail(), "hidden_close_failed");
    require(result.fixed_argmax.size() == result.hidden_rows &&
            result.fixed_margins.size() == result.hidden_rows, "invalid_output_rows");

    reset(ctx.get());
    decode(ctx.get(), batch.fill(opt.prompt.data(), prompt_count, 0));
    for (std::int32_t step = 0; step < opt.greedy_steps; ++step) {
        const auto next = prediction(ctx.get(), step == 0 ? prompt_count - 1 : 0, result.n_vocab);
        result.greedy_tokens.push_back(next.token);
        // The first prediction already counts; only feed back when another is needed.
        if (step + 1 < opt.greedy_steps) {
            const auto pos = static_cast<llama_pos>(opt.prompt.size() + static_cast<std::size_t>(step));
            decode(ctx.get(), batch.fill(&next.token, 1, pos));
        }
    }
    return result;
}

template <typename T>
void array(std::ostream & out, const std::vector<T> & values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << values[i];
    }
    out << ']';
}

std::string success_json(const Result & result) {
    std::ostringstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\"schema\":\"" << schema << "\",\"ok\":true,\"requested_repack\":"
        << (result.requested_repack ? "true" : "false")
        << ",\"default_repack\":" << (result.default_repack ? "true" : "false")
        << ",\"n_embd\":" << result.n_embd
        << ",\"n_layer\":" << result.n_layer
        << ",\"n_vocab\":" << result.n_vocab
        << ",\"n_ctx\":" << result.n_ctx
        << ",\"n_batch\":" << result.n_batch
        << ",\"n_ubatch\":" << result.n_ubatch
        << ",\"n_seq_max\":" << result.n_seq_max
        << ",\"n_threads\":" << result.n_threads
        << ",\"n_threads_batch\":" << result.n_threads_batch
        << ",\"load_seconds\":" << result.load_seconds
        << ",\"context_seconds\":" << result.context_seconds
        << ",\"prefill_seconds\":";
    array(out, result.prefill_seconds);
    out << ",\"decode_seconds\":";
    array(out, result.decode_seconds);
    out << ",\"prompt_tokens\":";
    array(out, result.prompt_tokens);
    out << ",\"fixed_tokens\":";
    array(out, result.fixed_tokens);
    out << ",\"fixed_argmax\":";
    array(out, result.fixed_argmax);
    out << ",\"fixed_margins\":";
    array(out, result.fixed_margins);
    out << ",\"greedy_tokens\":";
    array(out, result.greedy_tokens);
    out << ",\"hidden_rows\":" << result.hidden_rows
        << ",\"hidden_columns\":" << result.n_embd << "}\n";
    return out.str();
}

int failure(const char * code) {
    // Codes are fixed source literals; no user input needs JSON escaping.
    std::fprintf(stderr, "llama_kernel_probe: %s\n", code);
    std::fprintf(stdout, "{\"schema\":\"%s\",\"ok\":false,\"error\":\"%s\"}\n", schema, code);
    return 1;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const auto opt = arguments(argc, argv);
        const auto json = success_json(run(opt));
        const auto written = std::fwrite(json.data(), 1, json.size(), stdout);
        return written == json.size() && std::fflush(stdout) == 0 ? 0 : 1;
    } catch (const ProbeError & error) {
        return failure(error.code);
    } catch (const std::bad_alloc &) {
        return failure("out_of_memory");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "llama_kernel_probe: %s\n", error.what());
        return failure("unexpected_exception");
    } catch (...) {
        return failure("unknown_exception");
    }
}
