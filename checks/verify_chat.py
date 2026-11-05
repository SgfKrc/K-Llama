"""验证 qlh_chat.dll：C ABI 契约、apply 的 prompt 渲染、parse 的工具调用解析。

这个脚本测的是「vendored 的 llama.cpp common/chat 在 C ABI 后面是否还按上游行为工作」，
不是测 qlh 的业务逻辑。三组判据：

  A. 契约 —— 只支持千问系。非千问模板必须在 qlh_chat_new() 就被拒，且错误可读。
  B. apply —— 4 个固定用例（单轮/多轮/含 tools/关思考）的 prompt 与 GOLDEN 逐字节相等。
     GOLDEN 的来源见下面「预期值从哪来」，不是从本 DLL 回读的。
  C. parse —— 工具调用与思考段。用两类输入：
     C1 上游 tests/test-chat.cpp 里对 Qwen3.5-4B.jinja 的**权威期望**（直接证据）；
     C2 .qlh_tmp/tools-p0/results.md 里那份真实模型输出（含 <tool_call> XML），
        并额外验证流式 partial 的增量拼接等于终态。

预期值从哪来（重要）
--------------------
* C1 的期望直接抄自 H:/llamacpp-c/tests/test-chat.cpp 的 Qwen3.5-4B 段
  （message_assist_thoughts / message_assist_call：见 test-chat.cpp:1065/1069 与
  2182 起的 peg 用例）。那是上游自己的黄金值，跟本仓库的实现无关 —— 直接证据。
* B 的 prompt 黄金值是**间接交叉验证**得来的：同一份 jinja 模板用 Python 官方
  jinja2 独立渲染一遍，与 DLL 输出逐字节比较，双方一致后才固化进 GOLDEN。
  再加一条直接证据：无 tools 用例的 generation_prompt
  `"<|im_start|>assistant\\n<think>\\n"` 与 test-chat.cpp:7571 的硬编码期望相同。
  跑 `--cross-check` 会现场再用 jinja2 渲染一遍（需要 pip 里的 jinja2；没有就跳过并注明）。
  llama-cli 那条路走不通：本机没有可加载的完整模型，只有 models/ggml-vocab-*.gguf
  （vocab-only 的 gguf 在 llama_model_load 就失败），所以拿不到 llama-cli 的真·上游渲染。

用法：
    python checks/verify_chat.py                        # 用 .qlh_tmp/chatout 下的产物
    python checks/verify_chat.py --dll path/to/qlh_chat.dll --llama-root H:\\llamacpp-c
    python checks/verify_chat.py --cross-check          # 顺带用 jinja2 独立渲染交叉验证
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYS = "You are a helpful assistant."

# 与上游 test-chat.cpp:442 的 special_function_tool 保持一致（参数是 integer，文档现象：
# 参数值走 schema 分支解析，arguments 用 JSON 数字而不是字符串）
SPECIAL_FUNCTION_TOOL = {
    "type": "function",
    "function": {
        "name": "special_function",
        "description": "I'm special",
        "parameters": {
            "type": "object",
            "properties": {"arg1": {"type": "integer", "description": "The arg."}},
            "required": ["arg1"],
        },
    },
}

WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def cases() -> dict[str, dict]:
    """B 组的 4 个 apply 用例。改这里必须同步改 GOLDEN。"""
    return {
        "no_tools_think": {
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": "你好，介绍一下你自己。"},
            ],
            "enable_thinking": True,
        },
        "no_tools_nothink": {
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": "你好，介绍一下你自己。"},
            ],
            "enable_thinking": False,
        },
        "multiturn": {
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": "1+1 等于几？"},
                {"role": "assistant", "content": "1+1 等于 2。"},
                {"role": "user", "content": "那 2+2 呢？"},
            ],
            "enable_thinking": True,
        },
        "tools": {
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": "北京现在天气怎么样？"},
            ],
            "tools": WEATHER_TOOLS,
            "enable_thinking": True,
        },
    }


# prompt 黄金值：jinja2 独立渲染 + DLL 输出逐字节一致后固化（见文件头说明）。
GOLDEN = {
    'no_tools_think': {
        'prompt': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好，介绍一下你自己。<|im_end|>\n<|im_start|>assistant\n<think>\n',
        'generation_prompt': '<|im_start|>assistant\n<think>\n',
        'format': 'peg-native',
        'thinking_start_tag': '<think>',
        'thinking_end_tags': ['\n</think>', '</think>', '<tool_call>'],
        'stops': [],
        'grammar_empty': True,
    },
    'no_tools_nothink': {
        'prompt': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好，介绍一下你自己。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
        'generation_prompt': '<|im_start|>assistant\n<think>\n\n</think>\n\n',
        'format': 'peg-native',
        'thinking_start_tag': '<think>',
        'thinking_end_tags': ['\n</think>', '</think>', '<tool_call>'],
        'stops': [],
        'grammar_empty': True,
    },
    'multiturn': {
        'prompt': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n1+1 等于几？<|im_end|>\n<|im_start|>assistant\n1+1 等于 2。<|im_end|>\n<|im_start|>user\n那 2+2 呢？<|im_end|>\n<|im_start|>assistant\n<think>\n',
        'generation_prompt': '<|im_start|>assistant\n<think>\n',
        'format': 'peg-native',
        'thinking_start_tag': '<think>',
        'thinking_end_tags': ['\n</think>', '</think>', '<tool_call>'],
        'stops': [],
        'grammar_empty': True,
    },
    'tools': {
        'prompt': '<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}\n{"type": "function", "function": {"name": "get_time", "description": "Get current time", "parameters": {"type": "object", "properties": {}}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n北京现在天气怎么样？<|im_end|>\n<|im_start|>assistant\n<think>\n',
        'generation_prompt': '<|im_start|>assistant\n<think>\n',
        'format': 'peg-native',
        'thinking_start_tag': '<think>',
        'thinking_end_tags': ['\n</think>', '</think>', '<tool_call>'],
        'stops': [],
        'grammar_empty': False,
    },
}

# C1：上游 test-chat.cpp 对 Qwen3.5-4B.jinja 的权威用例
UPSTREAM_TOOL_CALL_INPUT = (
    '<tool_call>\n'
    '<function=special_function>\n'
    '<parameter=arg1>\n1\n</parameter>\n'
    '</function>\n'
    '</tool_call>'
)
UPSTREAM_THOUGHTS_INPUT = "I'm\nthinking\n</think>\n\nHello, world!\nWhat's up?"

# 同一段文本在 reasoning_format=none 时的 content 期望（test-chat.cpp:2193）
UPSTREAM_NONE_CONTENT = "<think>\nI'm\nthinking\n</think>\n\nHello, world!\nWhat's up?"

# 同一段 thinking 文本在 reasoning_format=auto 时的 content（test-chat.cpp:1065 的 message_assist_thoughts）
UPSTREAM_THOUGHTS_CONTENT = "Hello, world!\nWhat's up?"

# test-chat.cpp:7571 对 Qwen3.5-4B.jinja 的 basic generation_prompt 硬编码期望（直接证据）
UPSTREAM_BASIC_GENERATION_PROMPT = '<|im_start|>assistant\n<think>\n'

# C2：.qlh_tmp/tools-p0/results.md 里 case `tools_thinking` 的逐字原文
P0_OUTPUT_FALLBACK = (
    "用户询问北京的天气情况，我需要使用get_weather工具查询北京的天气信息。\n"
    "</think>\n\n"
    "<tool_call>\n"
    "<function=get_weather>\n"
    "<parameter=city>\n"
    "北京\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        ok = False


def first_diff(a: str, b: str) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return (f"首个差异在第 {i} 个字符：\n"
                    f"    golden={a[max(0, i - 40):i + 40]!r}\n"
                    f"    actual={b[max(0, i - 40):i + 40]!r}")
    return f"前缀相同，长度 {len(a)} vs {len(b)}；尾部 golden={a[len(b):]!r} actual={b[len(a):]!r}"


# ---------------------------------------------------------------------------
# DLL 绑定
# ---------------------------------------------------------------------------


class ChatDll:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.qlh_chat_new.restype = ctypes.c_void_p
        self.lib.qlh_chat_new.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.qlh_chat_del.argtypes = [ctypes.c_void_p]
        self.lib.qlh_chat_apply.restype = ctypes.c_int
        self.lib.qlh_chat_apply.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                            ctypes.POINTER(ctypes.c_char_p)]
        self.lib.qlh_chat_parse.restype = ctypes.c_int
        self.lib.qlh_chat_parse.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                            ctypes.POINTER(ctypes.c_char_p)]
        self.lib.qlh_chat_str_free.argtypes = [ctypes.c_char_p]
        self.lib.qlh_chat_str_free.restype = None

    def new(self, tmpl: str) -> tuple[int | None, str]:
        err = ctypes.create_string_buffer(4096)
        h = self.lib.qlh_chat_new(tmpl.encode("utf-8"), err, len(err))
        return h, err.value.decode("utf-8", "replace")

    def _read_release(self, out) -> dict | None:
        """读 out_json 再交给分配方（本 DLL 的 str_free）释放。"""
        raw = out.value
        self.lib.qlh_chat_str_free(out)  # out 仍持有那个指针
        return json.loads(raw.decode("utf-8")) if raw else None

    def apply(self, h, req: dict) -> tuple[int, dict | None]:
        out = ctypes.c_char_p()
        rc = self.lib.qlh_chat_apply(h, json.dumps(req, ensure_ascii=False).encode("utf-8"),
                                     ctypes.byref(out))
        return rc, self._read_release(out)

    def parse(self, h, text: str, partial: bool) -> tuple[int, dict | None]:
        out = ctypes.c_char_p()
        rc = self.lib.qlh_chat_parse(h, text.encode("utf-8"), 1 if partial else 0,
                                     ctypes.byref(out))
        return rc, self._read_release(out)


# ---------------------------------------------------------------------------
# A. 契约
# ---------------------------------------------------------------------------


def test_contract(dll: ChatDll) -> None:
    print("\n=== A. 只支持千问系的契约检查 ===")

    chatml = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + "
        "message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    h, err = dll.new(chatml)
    check("ChatML 被拒（无 <tool_call>/<function=/<parameter=）", h is None)
    check("拒绝原因写到 err_out 且可读", "Qwen" in err and "<tool_call>" in err, f"err={err!r}")

    h, err = dll.new("")
    check("空模板被拒", h is None and bool(err), f"err={err!r}")

    # 只有 <tool_call> 的残缺模板也必须拒（三特征缺一不可）
    h, err = dll.new("{% if tools %}<tool_call>{% endif %}")
    check("只有 <tool_call> 的模板被拒", h is None, f"err={err[:60]!r}")


# ---------------------------------------------------------------------------
# B. apply
# ---------------------------------------------------------------------------


def test_apply(dll: ChatDll, tmpl_src: str, cross_check: bool) -> None:
    print("\n=== B. apply：prompt 与 GOLDEN 逐字节比较 ===")

    jinja2_render = None
    if cross_check:
        try:
            jinja2_render = _make_jinja2_renderer(tmpl_src)
            print("  （jinja2 可用，现场再做一次独立渲染交叉验证）")
        except ImportError:
            print("  （没有 jinja2，跳过交叉验证；GOLDEN 仍来自此前那次 jinja2 比对）")

    for name, req in cases().items():
        h, err = dll.new(tmpl_src)
        if h is None:
            check(f"{name}: 建 handle", False, err)
            continue
        try:
            rc, got = dll.apply(h, req)
            check(f"{name}: apply rc", rc == 0, f"rc={rc} err={(got or {}).get('error')}")
            if rc != 0:
                continue
            g = GOLDEN[name]
            check(f"{name}: prompt 逐字节一致", got["prompt"] == g["prompt"],
                  "" if got["prompt"] == g["prompt"] else first_diff(g["prompt"], got["prompt"]))
            check(f"{name}: generation_prompt", got["generation_prompt"] == g["generation_prompt"],
                  repr(got["generation_prompt"]))
            check(f"{name}: format", got["format"] == g["format"], got["format"])
            check(f"{name}: thinking_start_tag", got["thinking_start_tag"] == g["thinking_start_tag"],
                  repr(got["thinking_start_tag"]))
            check(f"{name}: thinking_end_tags", got["thinking_end_tags"] == g["thinking_end_tags"],
                  repr(got["thinking_end_tags"]))
            check(f"{name}: stops", got["stops"] == g["stops"], repr(got["stops"]))
            if g["grammar_empty"]:
                check(f"{name}: 无 grammar", got["grammar"] == "", f"len={len(got['grammar'])}")
            else:
                check(f"{name}: 有 grammar", len(got["grammar"]) > 0 and "<tool_call>" in got["grammar"],
                      f"len={len(got['grammar'])}")

            if jinja2_render is not None:
                expect = jinja2_render(req)
                check(f"{name}: jinja2 独立渲染 == DLL", expect == got["prompt"], first_diff(expect, got["prompt"]))
        finally:
            dll.lib.qlh_chat_del(h)

    # 上游 test-chat.cpp:7571 的硬编码期望，作为 generation_prompt 的直接证据
    check("上游 test-chat 的 generation_prompt 期望相符",
          GOLDEN["no_tools_think"]["generation_prompt"] == UPSTREAM_BASIC_GENERATION_PROMPT)


def _make_jinja2_renderer(tmpl_src: str):
    from jinja2 import Environment  # noqa: F401  (ImportError 由调用方处理)

    env = Environment()
    # 上游 minja 的 tojson 用 nlohmann dump() 的默认分隔符（", " 和 ": "），
    # 与 jinja2 内建 tojson 的默认参数不同但一致于 json.dumps 的默认值。
    env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
    tpl = env.from_string(tmpl_src)

    def render(req: dict) -> str:
        kw = {
            "messages": req["messages"],
            "enable_thinking": req.get("enable_thinking", True),
            "add_generation_prompt": req.get("add_generation_prompt", True),
        }
        if req.get("tools"):
            kw["tools"] = req["tools"]
        return tpl.render(**kw)

    return render


# ---------------------------------------------------------------------------
# C. parse
# ---------------------------------------------------------------------------


def _tool_calls_semantic(msg: dict) -> list[dict]:
    """把 msg 里的 tool_calls 归一成 [{name, args(解析后的 dict)}]。

    test-chat.cpp:437 比较时也是 renormalize_json —— arguments 是 JSON 字符串，
    空白不同不算差异（上游自己都不按字节比）。
    """
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        raw = fn.get("arguments", "")
        try:
            args = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            args = raw
        out.append({"name": fn.get("name", ""), "args": args})
    return out


def test_parse_upstream(dll: ChatDll, tmpl_src: str) -> None:
    print("\n=== C1. parse：上游 test-chat.cpp 的权威期望 ===")

    # 1) 纯工具调用（enable_thinking=false，对应 test-chat Qwen3.5 段的第 3 个用例）
    h, err = dll.new(tmpl_src)
    if h is None:
        check("C1.1: 建 handle", False, err)
        return
    try:
        rc, _ = dll.apply(h, {
            "messages": [{"role": "user", "content": "Hey there!"}],
            "tools": [SPECIAL_FUNCTION_TOOL],
            "enable_thinking": False,
            "reasoning_format": "auto",
        })
        check("C1.1: apply rc", rc == 0)
        rc, got = dll.parse(h, UPSTREAM_TOOL_CALL_INPUT, partial=False)
        check("C1.1: parse rc", rc == 0, str((got or {}).get("error")))
        if rc == 0:
            tcs = _tool_calls_semantic(got["msg"])
            # test-chat 期望 message_assist_call = simple_assist_msg("", "", "special_function", "{\"arg1\": 1}")
            check("C1.1: 工具调用 == message_assist_call", tcs == [{"name": "special_function", "args": {"arg1": 1}}],
                  str(tcs))
            check("C1.1: content 为空", (got["msg"].get("content") or "") == "",
                  repr(got["msg"].get("content")))
            check("C1.1: 无 reasoning", (got["msg"].get("reasoning_content") or "") == "",
                  repr(got["msg"].get("reasoning_content")))
    finally:
        dll.lib.qlh_chat_del(h)

    # 2) 思考段（对应 test-chat 第 1 个用例，reasoning_format=auto）
    h, _ = dll.new(tmpl_src)
    if h is None:
        return
    try:
        rc, _ = dll.apply(h, {
            "messages": [{"role": "user", "content": "Hey there!"}],
            "enable_thinking": True,
            "reasoning_format": "auto",
        })
        check("C1.2: apply rc", rc == 0)
        rc, got = dll.parse(h, UPSTREAM_THOUGHTS_INPUT, partial=False)
        check("C1.2: parse rc", rc == 0, str((got or {}).get("error")))
        if rc == 0:
            # test-chat 期望 message_assist_thoughts = simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking")
            check("C1.2: reasoning_content == I'm\\nthinking",
                  got["msg"].get("reasoning_content") == "I'm\nthinking",
                  repr(got["msg"].get("reasoning_content")))
            check("C1.2: content == Hello, world!\\nWhat's up?",
                  got["msg"].get("content") == UPSTREAM_THOUGHTS_CONTENT,
                  repr(got["msg"].get("content")))
    finally:
        dll.lib.qlh_chat_del(h)

    # 3) reasoning_format=none 时同一段文本必须原样留在 content（test-chat 第 2 个用例）
    h, _ = dll.new(tmpl_src)
    if h is None:
        return
    try:
        rc, _ = dll.apply(h, {
            "messages": [{"role": "user", "content": "Hey there!"}],
            "enable_thinking": True,
            "reasoning_format": "none",
        })
        check("C1.3: apply rc", rc == 0)
        rc, got = dll.parse(h, UPSTREAM_THOUGHTS_INPUT, partial=False)
        if rc == 0:
            check("C1.3: reasoning_format=none 时 content 保留 think 标签",
                  got["msg"].get("content") == UPSTREAM_NONE_CONTENT and not got["msg"].get("reasoning_content"),
                  repr(got["msg"].get("content")))
        else:
            check("C1.3: parse rc", False, str((got or {}).get("error")))
    finally:
        dll.lib.qlh_chat_del(h)


def load_p0_output() -> tuple[str, str]:
    """从 results.md 里抠出 case tools_thinking 的逐字原文；文件不在就用内嵌副本。"""
    path = ROOT / ".qlh_tmp" / "tools-p0" / "results.md"
    if not path.exists():
        return P0_OUTPUT_FALLBACK, "内嵌副本（results.md 不在）"
    text = path.read_text(encoding="utf-8")
    marker = "# case `tools_thinking`"
    idx = text.find(marker)
    if idx < 0:
        return P0_OUTPUT_FALLBACK, "内嵌副本（results.md 里找不到 tools_thinking）"
    seg = text[idx:]
    start = seg.find("```text")
    if start < 0:
        return P0_OUTPUT_FALLBACK, "内嵌副本（tool_thinking 段没有 ```text 块）"
    start += len("```text")
    # 跳过紧随其后的一个换行
    if seg[start:start + 2] == "\r\n":
        start += 2
    elif seg[start:start + 1] == "\n":
        start += 1
    end = seg.find("```", start)
    body = seg[start:end] if end >= 0 else seg[start:]
    return body, "results.md"


def test_parse_p0(dll: ChatDll, tmpl_src: str, verbose: bool) -> None:
    print("\n=== C2. parse：P0 真实模型输出（含 <tool_call> XML）===")

    p0, source = load_p0_output()
    print(f"  输入来源：{source}，{len(p0)} 字符")
    if source == "results.md":
        # results.md 里的块末尾可能被 markdown 的换行带出来，这里只要求以 </tool_call> 结尾
        check("C2: results.md 的原文与内嵌副本同前缀",
              p0.startswith("用户询问北京的天气情况"), repr(p0[:40]))
    if verbose:
        print("  p0 原文 =", repr(p0))

    h, err = dll.new(tmpl_src)
    if h is None:
        check("C2: 建 handle", False, err)
        return
    try:
        rc, _ = dll.apply(h, {
            "messages": [{"role": "user", "content": "北京现在天气怎么样？"}],
            "tools": WEATHER_TOOLS,
            "enable_thinking": True,
            "reasoning_format": "auto",
        })
        check("C2: apply rc", rc == 0)

        rc, got = dll.parse(h, p0, partial=False)
        check("C2: parse rc", rc == 0, str((got or {}).get("error")))
        if rc == 0:
            msg = got["msg"]
            tcs = _tool_calls_semantic(msg)
            check("C2: 解析出 1 个工具调用", len(tcs) == 1, str(tcs))
            if tcs:
                check("C2: 函数名 == get_weather", tcs[0]["name"] == "get_weather", tcs[0]["name"])
                check("C2: 参数 == {\"city\": \"北京\"}", tcs[0]["args"] == {"city": "北京"}, str(tcs[0]["args"]))
            check("C2: content 为空（全是思考+工具调用）", (msg.get("content") or "") == "",
                  repr(msg.get("content")))
            check("C2: reasoning_content 含'北京'",
                  "北京" in (msg.get("reasoning_content") or ""), repr((msg.get("reasoning_content") or "")[:40]))
            if verbose:
                print("  msg  =", json.dumps(msg, ensure_ascii=False))

        # 流式：按字符位置切分（不切 UTF-8 中间），验证增量拼接 == 终态
        cuts = [20, 45, 80, len(p0)]
        acc_reason, acc_content, acc_args = "", "", {}
        final = None
        for i, cut in enumerate(cuts):
            partial = 1 if i < len(cuts) - 1 else 0
            rc, got = dll.parse(h, p0[:cut], partial=bool(partial))
            if rc != 0:
                check(f"C2 流式: cut={cut} parse rc", False, str((got or {}).get("error")))
                break
            for d in got["diff"]:
                acc_reason += d["reasoning_content_delta"]
                acc_content += d["content_delta"]
                if d["tool_call_index"] is not None:
                    k = d["tool_call_index"]
                    acc_args.setdefault(k, "")
                    acc_args[k] += d["tool_call_delta"]["arguments"]
            final = got["msg"]
        if final is not None:
            check("C2 流式: reasoning 增量拼接 == 终态",
                  acc_reason == (final.get("reasoning_content") or ""),
                  f"acc={acc_reason[-30:]!r} final={(final.get('reasoning_content') or '')[-30:]!r}")
            check("C2 流式: content 增量拼接 == 终态", acc_content == (final.get("content") or ""),
                  f"acc={acc_content!r} final={(final.get('content') or '')!r}")
            check("C2 流式: tool args 增量拼接 == 终态",
                  list(acc_args.values()) == [tc["function"]["arguments"] for tc in final.get("tool_calls", [])],
                  str(list(acc_args.values())))
    finally:
        dll.lib.qlh_chat_del(h)


# ---------------------------------------------------------------------------
# D. 错误路径
# ---------------------------------------------------------------------------


def test_errors(dll: ChatDll, tmpl_src: str) -> None:
    print("\n=== D. 错误路径（都要返回 -1 且带 error 字段，不能崩）===")

    h, _ = dll.new(tmpl_src)
    if h is None:
        check("D: 建 handle", False)
        return
    try:
        rc, got = dll.parse(h, "hello", partial=False)
        check("D1: 未 apply 就 parse -> -1", rc == -1 and "apply" in (got or {}).get("error", ""),
              str((got or {}).get("error")))

        rc, got = dll.apply(h, {"messages": "not-an-array"})
        check("D2: messages 类型错 -> -1", rc == -1 and bool((got or {}).get("error")),
              str((got or {}).get("error"))[:80])

        rc, got = dll.apply(h, {"nope": 1})
        check("D3: 缺 messages -> -1", rc == -1 and bool((got or {}).get("error")),
              str((got or {}).get("error"))[:80])

        # 坏 JSON 走 apply 的字节入口
        out = ctypes.c_char_p()
        rc = dll.lib.qlh_chat_apply(h, b"{not json", ctypes.byref(out))
        detail = out.value.decode("utf-8", "replace") if out.value else ""
        dll.lib.qlh_chat_str_free(out)
        check("D4: 坏 JSON -> -1", rc == -1 and "error" in detail, detail[:80])
    finally:
        dll.lib.qlh_chat_del(h)


# ---------------------------------------------------------------------------


def find_dll(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for name in ("qlh_chat.dll", "libqlh_chat.so"):
        p = ROOT / ".qlh_tmp" / "chatout" / name
        if p.exists():
            return p
    raise SystemExit("找不到产物：先跑 shim/build_chat_msvc.ps1（或 build_chat_posix.py），"
                     "或用 --dll 指定")


def find_template(explicit: str | None, llama_root: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"模板不存在：{p}")
        return p
    roots = []
    if llama_root:
        roots.append(Path(llama_root))
    if os.environ.get("QLH_LLAMA_ROOT"):
        roots.append(Path(os.environ["QLH_LLAMA_ROOT"]))
    roots += [Path(r"H:\llamacpp-c"), ROOT.parent / "llamacpp-c", Path.home() / "llamacpp-c"]
    for r in roots:
        p = r / "models" / "templates" / "Qwen3.5-4B.jinja"
        if p.exists():
            return p
    # 退而求其次：任何 Qwen 模板
    for r in roots:
        d = r / "models" / "templates"
        if d.is_dir():
            cands = sorted(d.glob("Qwen*.jinja"))
            if cands:
                print(f"  （没有 Qwen3.5-4B.jinja，退用 {cands[0].name}；GOLDEN 可能不匹配）")
                return cands[0]
    raise SystemExit("找不到 Qwen3.5-4B.jinja：用 --template 指定")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dll", default=None, help="qlh_chat.dll / libqlh_chat.so")
    ap.add_argument("--template", default=None, help="千问模板 jinja 路径")
    ap.add_argument("--llama-root", default=None, help="llama.cpp 源码根（用来找模板）")
    ap.add_argument("--cross-check", action="store_true",
                    help="再用 jinja2 独立渲染一遍，做交叉验证（需要 jinja2）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    dll_path = find_dll(args.dll)
    tmpl_path = find_template(args.template, args.llama_root)
    tmpl_src = tmpl_path.read_text(encoding="utf-8")

    print(f"dll      : {dll_path}")
    print(f"template : {tmpl_path}（{len(tmpl_src)} 字符）")

    dll = ChatDll(dll_path)

    test_contract(dll)
    test_apply(dll, tmpl_src, args.cross_check)
    test_parse_upstream(dll, tmpl_src)
    test_parse_p0(dll, tmpl_src, args.verbose)
    test_errors(dll, tmpl_src)

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
