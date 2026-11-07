// 本文件不是上游文件 —— 它是 shim/chat/ 里唯一的本地新增单元（见 UPSTREAM.txt）。
//
// 为什么需要它：vendored 的 chat.cpp 除了模板/Jinja/JSON 这些自带的东西之外，还从
// common.cpp（字符串工具、tty 判定）、sampling.cpp（采样参数补齐）、llen.cpp
// （ggml_abort）和 llama.cpp（几个 vocab 查询）引用了一小批**非内联**符号。我们不把那
// 三个 20k+ 行的巨型 .cpp 编进来（那样就不是零 llama 依赖了），所以在这里补齐链接面。
//
// 做法纪律（与 AGENTS.md §7"不重复造轮子"一致）：
//   1. 能从上游逐字搬过来的实现，一律**逐字复制**，只标来源行号，绝不凭记忆重写；
//   2. 只有真正依赖 vocab / ggml 运行时的少数符号给最小桩，并在每个桩上注明
//      "为什么在 nullptr model 路径上不可能被调用"。
//
// 桩为什么安全：本 DLL 永远用 common_chat_templates_init(nullptr, override, ...)
// 构造模板（chat.cpp:884 起）。override 非空时走 else 分支，model 相关的整块
// `if (model) { ... }`（chat.cpp:957-984）整体跳过，因此 llama_model_get_vocab /
// llama_vocab_bos / common_token_to_piece 这些都不会被执行。common_tokenize 只在
// common_chat_msg_delimiters::tokenize(vocab) 里用，而那个调用点是
// `if (vocab) { ... delimiters.tokenize(vocab); }`（chat.cpp:1885 附近），vocab 为
// nullptr 时同样不进。纯文本解析路径（common_chat_input(std::string)）不消费 token
// （peg-parser.cpp / chat-peg-parser.cpp 里 grep `\.tokens` 零命中）。

#include "common.h"
#include "ggml.h"
#include "llama.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#    include <io.h>
#    include <windows.h>
#else
#    include <unistd.h>
#endif

//
// ggml_abort — 上游在 ggml/src/ggml.c:253。原实现会调 ggml_print_backtrace()，
// 那属于 ggml 库内部（不在本 DLL 里），所以这里只保留"格式化 + stderr + abort"，
// 这是 GGML_ASSERT 失败时唯一有观感差别的部分（少了调用栈）。
//
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);

    char message[2048];
    int  offset = snprintf(message, sizeof(message), "%s:%d: ", file, line);

    va_list args;
    va_start(args, fmt);
    vsnprintf(message + offset, sizeof(message) - offset, fmt, args);
    va_end(args);

    fprintf(stderr, "%s\n", message);
    abort();
}

//
// 字符串工具 —— 逐字复制自 common/common.cpp:528 / 550 / 561 / 577
//

void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

//
// TTY 判定 —— 逐字复制自 common/common.cpp:1067 / 1077 / 1097
// （log.cpp 在初始化 logger 时无条件调用 tty_can_use_colors / tty_enable_ansi）
//

bool common_is_tty(FILE * file) {
#if defined(_WIN32)
    return _isatty(_fileno(file));
#else
    return isatty(fileno(file));
#endif
}

bool tty_can_use_colors() {
    // Check NO_COLOR environment variable (https://no-color.org/)
    if (const char * no_color = std::getenv("NO_COLOR")) {
        if (no_color[0] != '\0') {
            return false;
        }
    }

    // Check TERM environment variable
    if (const char * term = std::getenv("TERM")) {
        if (std::strcmp(term, "dumb") == 0) {
            return false;
        }
    }

    // Check if stdout and stderr are connected to a terminal
    // We check both because log messages can go to either
    return common_is_tty(stdout) || common_is_tty(stderr);
}

bool tty_enable_ansi() {
#if defined(_WIN32)
    // a Windows console renders ANSI sequences only in virtual terminal mode, pipes and files take them as is
    for (DWORD id : { STD_OUTPUT_HANDLE, STD_ERROR_HANDLE }) {
        HANDLE h    = GetStdHandle(id);
        DWORD  mode = 0;
        if (GetConsoleMode(h, &mode) && !SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
            return false;
        }
    }
#endif
    return true;
}

//
// 采样参数补齐 —— 逐字复制自 common/sampling.cpp:1055 / 1065
// 两者第一行都是 GGML_ASSERT(vocab != nullptr)，调用点全在 `if (vocab)` 分支里。
//

void common_sampling_add_preserved_tokens(common_params_sampling & sampling, const llama_vocab * vocab, const std::vector<std::string> & tokens) {
    GGML_ASSERT(vocab != nullptr);
    for (const auto & t : tokens) {
        auto ids = common_tokenize(vocab, t, false, true);
        if (ids.size() == 1) {
            sampling.preserved_tokens.insert(ids[0]);
        }
    }
}

void common_sampling_add_grammar_triggers(common_params_sampling & sampling, const llama_vocab * vocab, std::vector<common_grammar_trigger> triggers) {
    GGML_ASSERT(vocab != nullptr);
    for (auto & trigger : triggers) {
        if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
            const auto & word = trigger.value;
            auto ids = common_tokenize(vocab, word, false, true);
            if (ids.size() == 1) {
                auto token = ids[0];
                if (std::find(sampling.preserved_tokens.begin(), sampling.preserved_tokens.end(), (llama_token) token) == sampling.preserved_tokens.end()) {
                    throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                }
            }
        }
        sampling.grammar_triggers.push_back(std::move(trigger));
    }
}

//
// 以下是真正的"桩"：DLL 里没有 tokenizer，也没有 llama 运行时。
// 每个都标注了它在本 DLL 的调用路径上为什么不会被执行。
//

// 只在 meta 非空时被调用（chat.cpp:118 的 delimiters.tokenize / chat.cpp:955，
// 两者都在 `if (vocab)` 内）。返回空表示"没有 token"，调用方按 0 token 处理。
std::vector<llama_token> common_tokenize(const struct llama_vocab * vocab,
                                         const std::string &        text,
                                         bool                       add_special,
                                         bool                       parse_special) {
    (void) vocab;
    (void) text;
    (void) add_special;
    (void) parse_special;
    return {};
}

// 只在 model 非空时被调用（chat.cpp:955 的 get_token lambda）。空串即为"没有该 token"。
std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    (void) vocab;
    (void) token;
    (void) special;
    return {};
}

// llama ABI 桩：只为满足链接。本 DLL 永远以 model == nullptr 调用 chat 初始化，
// 所以这些函数一次都不会被真正执行；真被执行说明有人把 model 传了进来，那是契约违背。
extern "C" {

const llama_vocab * llama_model_get_vocab(const struct llama_model * model) {
    (void) model;
    return nullptr;
}

const char * llama_model_chat_template(const struct llama_model * model, const char * name) {
    (void) model;
    (void) name;
    return nullptr;
}

llama_token llama_vocab_bos(const struct llama_vocab * vocab) {
    (void) vocab;
    return LLAMA_TOKEN_NULL;
}

llama_token llama_vocab_eos(const struct llama_vocab * vocab) {
    (void) vocab;
    return LLAMA_TOKEN_NULL;
}

bool llama_vocab_get_add_bos(const struct llama_vocab * vocab) {
    (void) vocab;
    return false;
}

bool llama_vocab_get_add_eos(const struct llama_vocab * vocab) {
    (void) vocab;
    return false;
}

int32_t llama_chat_apply_template(const char *               tmpl,
                                  const struct llama_chat_message * chat,
                                  size_t                     n_msg,
                                  bool                       add_ass,
                                  char *                     buf,
                                  int32_t                    length) {
    (void) tmpl;
    (void) chat;
    (void) n_msg;
    (void) add_ass;
    (void) buf;
    (void) length;
    return -1;
}

} // extern "C"
