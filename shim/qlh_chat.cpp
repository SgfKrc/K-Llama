// qlh_chat.dll —— llama.cpp `common/chat` 的 C ABI 外壳。
//
// 设计约束（来自架构调研，已实测）：
//   * 独立 DLL，零 llama/ggml 运行时依赖。模板用 common_chat_templates_init(nullptr,
//     模板串) 构造：chat.cpp 里整块 `if (model)` 被跳过，所以不需要 tokenizer、不需要
//     llama.dll。
//   * 只支持千问系。这条约束的落点是 qlh_chat_new() 里的契约检查（模板必须同时含
//     <tool_call> / <function= / <parameter= 三个特征），不是靠删上游代码。
//   * 边界上只走 JSON 字符串，绝不传 C++ 对象；handle 里可以随便放 C++ 类型。
//   * 跨 DLL 边界的堆内存由分配方释放：本 DLL 用 malloc 分配，调用方用
//     qlh_chat_str_free() 释放（它在本 DLL 内 free）。
//
// 编译：shim/build_chat_msvc.ps1 / shim/build_chat_posix.py
// 测试：checks/verify_chat.py

#include "chat.h"

#include "common.h"
#include "json.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#if defined(_WIN32)
#    define QLH_CHAT_API extern "C" __declspec(dllexport)
#else
#    define QLH_CHAT_API extern "C" __attribute__((visibility("default")))
#endif

// ---------------------------------------------------------------------------
// handle：所有 C++ 状态都关在这里面（包括累计的 prev 消息和 PEG arena）
// ---------------------------------------------------------------------------
struct qlh_chat {
    common_chat_templates_ptr tmpls;

    // apply() 装配好的解析上下文：format / generation_prompt / PEG arena
    common_chat_parser_params parser_params;

    // 上一次 parse() 的累计结果，diff 就是 next - prev
    common_chat_msg prev;

    // 上一次 parse() 的输入文本，用来识别"调用方重开了一条新消息"
    std::string last_text;

    bool applied = false;
};

namespace {

void set_err(const char * err_out, size_t err_cap, const std::string & msg) {
    if (err_out == nullptr || err_cap == 0) {
        return;
    }
    const size_t n = msg.size() < err_cap - 1 ? msg.size() : err_cap - 1;
    std::memcpy(const_cast<char *>(err_out), msg.data(), n);
    const_cast<char *>(err_out)[n] = '\0';
}

char * dup_to_c(const std::string & s) {
    char * p = static_cast<char *>(std::malloc(s.size() + 1));
    if (p == nullptr) {
        return nullptr;
    }
    std::memcpy(p, s.data(), s.size());
    p[s.size()] = '\0';
    return p;
}

// 失败统一走这里：out_json 拿到 {"error": "..."}，返回 -1。
// 这样调用方既能读错误原因，也不会误把半截数据当成功。
int fail(char ** out_json, const std::string & msg) {
    if (out_json != nullptr) {
        common_json out = common_json::object();
        out["error"] = msg;
        *out_json = dup_to_c(out.dump());
    }
    return -1;
}

// 「只支持千问系」的契约检查。千问系的工具调用模板特征：XML 版 <tool_call> +
// <function= + <parameter=。Qwen2.5/3/3-Coder/3.5 全系列都命中这三条。
bool check_qwen_contract(const std::string & src, std::string & why) {
    if (src.empty()) {
        why = "chat template is empty";
        return false;
    }
    struct {
        const char * needle;
        const char * what;
    } const need[] = {
        { "<tool_call>",  "<tool_call> 开标签" },
        { "<function=",   "<function=...> 函数名标签" },
        { "<parameter=",  "<parameter=...> 参数标签" },
    };
    for (const auto & n : need) {
        if (src.find(n.needle) == std::string::npos) {
            why = std::string("unsupported chat template: only Qwen-family templates with XML tool calls are supported, but the template has no ") +
                  n.what + " (searched for \"" + n.needle + "\")";
            return false;
        }
    }
    return true;
}

// 把 apply 的结果装进 parser_params。上游 common_chat_parser_params 的拷贝构造只带
// format 与 generation_prompt（parser 要单独搬），这里照着 common_chat_session 构造
// 函数的做法手动装配（chat.cpp:1886 附近）。
void install_parser(qlh_chat * c, const common_chat_params & params) {
    c->parser_params                 = common_chat_parser_params();
    c->parser_params.format          = params.format;
    c->parser_params.generation_prompt = common_chat_input(params.generation_prompt);
    c->parser_params.parser          = params.parser;
}

} // namespace

// ---------------------------------------------------------------------------
// 1) 构造 / 析构
// ---------------------------------------------------------------------------

// tmpl_src : 模板 jinja 源码（必填）
// err_out  : 可选错误缓冲；失败时会写入 NUL 结尾的原因
// err_cap  : err_out 的字节数
// 返回     : handle（调用方负责 qlh_chat_del）；失败返回 nullptr
QLH_CHAT_API qlh_chat * qlh_chat_new(const char * tmpl_src, const char * err_out, size_t err_cap) {
    set_err(err_out, err_cap, "");

    if (tmpl_src == nullptr) {
        set_err(err_out, err_cap, "chat template pointer is null");
        return nullptr;
    }

    const std::string src(tmpl_src);
    std::string       why;
    if (!check_qwen_contract(src, why)) {
        set_err(err_out, err_cap, why);
        return nullptr;
    }

    try {
        // model == nullptr + 非空 override：chat.cpp 的 `if (model)` 分支整体跳过，
        // 不碰 vocab，也不需要 bos/eos 覆盖。
        auto tmpls = common_chat_templates_init(/* model = */ nullptr, src);

        auto * c   = new qlh_chat();
        c->tmpls   = std::move(tmpls);
        return c;
    } catch (const std::exception & e) {
        set_err(err_out, err_cap, std::string("failed to init chat template: ") + e.what());
        return nullptr;
    } catch (...) {
        set_err(err_out, err_cap, "failed to init chat template: unknown error");
        return nullptr;
    }
}

QLH_CHAT_API void qlh_chat_del(qlh_chat * c) {
    delete c;
}

// ---------------------------------------------------------------------------
// 2) apply：messages/tools -> prompt + 采样约束
// ---------------------------------------------------------------------------

// req_json 里认这些键（其余忽略）：
//   messages[].role/content/tool_calls/tool_call_id/name  必填，OpenAI 格式
//   tools[]                                               可选
//   tool_choice ("auto"/"none"/"required")                可选，默认 auto
//   parallel_tool_calls (bool)                             可选，默认 false
//   enable_thinking (bool)                                 可选，默认 true
//   reasoning_format ("none"/"auto"/"deepseek"/"deepseek-legacy")
//                                                          可选，默认 deepseek
//      —— 这一项不控制 prompt 长什么样，但它决定 apply 产出的 PEG arena 把
//         `...` 归到 reasoning_content 还是留在 content 里（上游 test-chat.cpp
//         对同一模板同一文本有两条相反期望，就是靠它选的）。
//         llama-server 的默认也是 COMMON_REASONING_FORMAT_DEEPSEEK（common.h:666）。
//   add_generation_prompt (bool)                           可选，默认 true
//   json_schema (string)                                   可选
//   chat_template_kwargs (object)                          可选，值按 JSON dump 成字符串
// 成功：*out_json = {prompt, generation_prompt, stops[], thinking_start_tag,
//                    thinking_end_tags[], grammar, grammar_lazy, format,
//                    supports_thinking}，返回 0
// 失败：*out_json = {"error": "..."}，返回 -1
QLH_CHAT_API int qlh_chat_apply(qlh_chat * c, const char * req_json, char ** out_json) {
    if (out_json != nullptr) {
        *out_json = nullptr;
    }
    if (c == nullptr) {
        return fail(out_json, "null handle");
    }
    if (req_json == nullptr) {
        return fail(out_json, "null request json");
    }

    common_chat_templates_inputs inputs;
    try {
        const common_json j = common_json::parse(req_json);

        if (!j.contains("messages")) {
            return fail(out_json, "request json has no \"messages\" array");
        }
        inputs.messages = common_chat_msgs_parse_oaicompat(j.at("messages"));

        if (j.contains("tools") && !j.at("tools").empty()) {
            inputs.tools = common_chat_tools_parse_oaicompat(j.at("tools"));
        }
        if (j.contains("tool_choice")) {
            inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(j.at("tool_choice").get<std::string>());
        }
        if (j.contains("json_schema")) {
            inputs.json_schema = j.at("json_schema").get<std::string>();
        }
        if (j.contains("chat_template_kwargs")) {
            for (const auto & item : j.at("chat_template_kwargs").items()) {
                // 与 tools/server/server-common.cpp:1484 一致：值按 JSON 字面量存成字符串
                inputs.chat_template_kwargs[item.key()] = item.value().dump();
            }
        }
        // 与 llama-server 的默认对齐（common.h:666 的 common_params::reasoning_format）
        inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        if (j.contains("reasoning_format")) {
            inputs.reasoning_format = common_reasoning_format_from_name(j.at("reasoning_format").get<std::string>());
        }
        inputs.parallel_tool_calls  = j.value("parallel_tool_calls", false);
        inputs.enable_thinking      = j.value("enable_thinking", true);
        inputs.add_generation_prompt = j.value("add_generation_prompt", true);
        inputs.use_jinja            = true;
    } catch (const std::exception & e) {
        return fail(out_json, std::string("bad request json: ") + e.what());
    }

    try {
        const common_chat_params params = common_chat_templates_apply(c->tmpls.get(), inputs);

        install_parser(c, params);

        // apply 是一次新的对话/新的一条待生成消息：解析状态从头开始
        c->prev       = common_chat_msg();
        c->prev.role  = "assistant";
        c->last_text.clear();
        c->applied    = true;

        common_json out = common_json::object();
        out["prompt"]               = params.prompt.str();
        out["generation_prompt"]    = params.generation_prompt;
        out["stops"]                = params.additional_stops;
        out["thinking_start_tag"]   = params.thinking_start_tag;
        out["thinking_end_tags"]    = params.thinking_end_tags;
        out["grammar"]              = params.grammar;
        out["grammar_lazy"]         = params.grammar_lazy;
        out["format"]               = std::string(common_chat_format_name(params.format));
        out["supports_thinking"]    = params.supports_thinking;

        if (out_json != nullptr) {
            *out_json = dup_to_c(out.dump());
        }
        return 0;
    } catch (const std::exception & e) {
        return fail(out_json, std::string("chat template apply failed: ") + e.what());
    }
}

// ---------------------------------------------------------------------------
// 3) parse：模型原始输出 -> msg + 增量 diff
// ---------------------------------------------------------------------------

// text       : 到目前为止模型输出的**全量**文本（不是这一轮的增量）
// is_partial : 非 0 = 流式中间态（宽松解析）；0 = 最终态（要求完整）
// 成功：*out_json = {msg: {...OpenAI assistant 消息...},
//                    diff: [{reasoning_content_delta, content_delta,
//                            tool_call_index (整数或 null), tool_call_delta{name,arguments,id}}]}
//       返回 0
// 失败：*out_json = {"error": "..."}，返回 -1
//
// 全量语义是刻意的：Python 侧每拿到一段新文本就整段重发，handle 内部只记住上一次的
// 结果，diff 就是"这次新增了什么"。如果新文本不是上一次文本的延长（调用方换了一条
// 消息却没重新 apply），prev 会自动清空，diff 从头算。
QLH_CHAT_API int qlh_chat_parse(qlh_chat * c, const char * text, int is_partial, char ** out_json) {
    if (out_json != nullptr) {
        *out_json = nullptr;
    }
    if (c == nullptr) {
        return fail(out_json, "null handle");
    }
    if (!c->applied) {
        return fail(out_json, "qlh_chat_apply() must be called before qlh_chat_parse()");
    }

    try {
        const std::string cur = text != nullptr ? std::string(text) : std::string();

        // 新文本不是上次文本的延长 -> 这是一条新消息，解析状态重置
        const bool continued = cur.size() >= c->last_text.size() && cur.compare(0, c->last_text.size(), c->last_text) == 0;
        if (!continued) {
            c->prev = common_chat_msg();
            c->prev.role = "assistant";
        }

        common_chat_input input(cur);
        const common_chat_msg msg = common_chat_parse(input, is_partial != 0, c->parser_params);

        const std::vector<common_chat_msg_diff> diffs = common_chat_msg_diff::compute_diffs(c->prev, msg);

        common_json arr = common_json::array();
        for (const auto & d : diffs) {
            common_json o = common_json::object();
            o["reasoning_content_delta"] = d.reasoning_content_delta;
            o["content_delta"]           = d.content_delta;
            if (d.tool_call_index == std::string::npos) {
                o["tool_call_index"] = nullptr;
            } else {
                o["tool_call_index"] = static_cast<unsigned long long>(d.tool_call_index);
            }
            common_json tc = common_json::object();
            tc["name"]      = d.tool_call_delta.name;
            tc["arguments"] = d.tool_call_delta.arguments;
            tc["id"]        = d.tool_call_delta.id;
            o["tool_call_delta"] = tc;
            arr.push_back(o);
        }

        c->prev      = msg;
        c->last_text = cur;

        common_json out = common_json::object();
        out["msg"]  = msg.to_json_oaicompat();
        out["diff"] = arr;

        if (out_json != nullptr) {
            *out_json = dup_to_c(out.dump());
        }
        return 0;
    } catch (const std::exception & e) {
        return fail(out_json, std::string("chat parse failed: ") + e.what());
    }
}

// ---------------------------------------------------------------------------
// 4) 释放本 DLL 分配出来的字符串
// ---------------------------------------------------------------------------

QLH_CHAT_API void qlh_chat_str_free(char * s) {
    std::free(s);
}
