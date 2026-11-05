"""验证 server.py 的 OpenAI 工具调用（非流式）：真实 DLL + 真实模板，不 mock。

测的是三件事：

  A. 渲染 —— `tools` 进不进提示词。判据用 `verify_chat.py` 的 GOLDEN（那套值先由
     jinja2 独立渲染交叉验证过），另加负向断言：不带 tools 时提示词里不得出现工具段。
  B. 解析 —— 同一 handle 上 render 之后 parse 一次真实模型输出（P0 阶段的逐字文本），
     再交给 `server.openai_assistant_message` 组装成 OpenAI 响应，检查形状与
     `finish_reason`。
  C. 回传 —— 把 parse 出来的 assistant 消息（`arguments` 是 JSON 字符串，且没有 id，
     正是客户端回传时的形状）连同 `role: "tool"` 的结果一起再渲染一次。

不覆盖 HTTP 层与本机模型：前者由 `python -m unittest` 与 9105 端到端跑，后者需要真
模型，见本期报告。

用法：
    python checks/verify_openai_tools.py
    python checks/verify_openai_tools.py --dll path/to/qlh_chat.dll --template path/to/qwen.jinja
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "checks"))

import server  # noqa: E402
import verify_chat  # noqa: E402  - 真实的 find_dll / find_template / WEATHER_TOOLS / GOLDEN
from chat import Chat  # noqa: E402

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        ok = False


def _first_diff(a: str, b: str) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return (f"首个差异在第 {i} 个字符：golden={a[max(0, i - 30):i + 30]!r} "
                    f"actual={b[max(0, i - 30):i + 30]!r}")
    return f"前缀相同，长度 {len(a)} vs {len(b)}"


def assistant_history(arguments) -> dict:
    """一条历史 assistant 工具调用，按客户端回传时的 OpenAI 形状（arguments 是字符串）。"""
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_hist", "type": "function",
         "function": {"name": "get_weather", "arguments": arguments}}]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dll", default=None)
    ap.add_argument("--template", default=None)
    ap.add_argument("--llama-root", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    dll_path = verify_chat.find_dll(args.dll)
    tmpl_path = verify_chat.find_template(args.template, args.llama_root)
    tmpl_src = tmpl_path.read_text(encoding="utf-8")
    case = verify_chat.cases()["tools"]
    golden = verify_chat.GOLDEN["tools"]

    print(f"dll      : {dll_path}")
    print(f"template : {tmpl_path}（{len(tmpl_src)} 字符）")

    # -- A. 渲染 ---------------------------------------------------------------
    print("\n=== A. 渲染：tools 进提示词，没有 tools 就不进 ===")
    no_tools_msgs = verify_chat.cases()["no_tools_think"]["messages"]
    with Chat(tmpl_src) as renderer:
        plain = server.render_messages(renderer, no_tools_msgs).prompt
    check("无工具：prompt == GOLDEN[no_tools_think]", plain == verify_chat.GOLDEN["no_tools_think"]["prompt"],
          "" if plain == verify_chat.GOLDEN["no_tools_think"]["prompt"] else _first_diff(plain, verify_chat.GOLDEN["no_tools_think"]["prompt"]))
    # 负向断言：工具段的每个结构标记都不该出现。只查一个词不够——模板可能只漏掉其中一段。
    for marker in ("# Tools", "<tools>", "</tools>", "<tool_call>", "<function=", "<parameter="):
        check(f"无工具：提示词不含 {marker!r}", marker not in plain)

    with Chat(tmpl_src) as renderer:
        with_tools = server.render_messages(renderer, case["messages"], tools=case["tools"]).prompt
    check("带工具：prompt == GOLDEN[tools]", with_tools == golden["prompt"],
          "" if with_tools == golden["prompt"] else _first_diff(with_tools, golden["prompt"]))
    check("带工具：含 <tools> 段与函数名", "<tools>" in with_tools and "get_weather" in with_tools)
    check("带工具：含格式说明 <function=example_function_name>",
          "<function=example_function_name>" in with_tools)

    # -- B. 解析 ---------------------------------------------------------------
    print("\n=== B. 解析：真实模型输出 -> OpenAI tool_calls + finish_reason ===")
    p0, source = verify_chat.load_p0_output()
    print(f"  输入来源：{source}，{len(p0)} 字符")
    if args.verbose:
        print("  p0 =", repr(p0))

    with Chat(tmpl_src) as renderer:
        server.render_messages(renderer, case["messages"], tools=case["tools"])
        parsed = renderer.parse(p0, partial=False)["msg"]
    if args.verbose:
        print("  parse.msg =", json.dumps(parsed, ensure_ascii=False))
    check("同一 handle：apply 之后 parse 成功", isinstance(parsed, dict) and bool(parsed))
    check("parse 出的 tool_calls 自带 arguments 字符串（无需再序列化）",
          isinstance((parsed.get("tool_calls") or [{}])[0].get("function", {}).get("arguments"), str),
          repr((parsed.get("tool_calls") or [{}])[0].get("function", {}).get("arguments"))[:60])
    check("parse 出的 tool_calls 没有 id（由 server 生成）",
          "id" not in (parsed.get("tool_calls") or [{}])[0])

    message, finish = server.openai_assistant_message(parsed, tokens=30, max_tokens=256)
    if args.verbose:
        print("  response.message =", json.dumps(message, ensure_ascii=False), "finish =", finish)
    check("finish_reason == tool_calls", finish == "tool_calls", finish)
    check("message.content 为 null（有调用无正文）", message.get("content") is None,
          repr(message.get("content")))
    check("tool_calls 数量 == 1", len(message.get("tool_calls") or []) == 1,
          str(len(message.get("tool_calls") or [])))
    call = (message.get("tool_calls") or [{}])[0]
    check("tool_calls[0].id 形如 call_<24 hex>",
          isinstance(call.get("id"), str) and call["id"].startswith("call_") and len(call["id"]) == 29,
          repr(call.get("id")))
    check("tool_calls[0].type == function", call.get("type") == "function")
    check("函数名 == get_weather", call.get("function", {}).get("name") == "get_weather",
          repr(call.get("function", {}).get("name")))
    args_text = call.get("function", {}).get("arguments")
    check("arguments 是字符串（OpenAI 规范）", isinstance(args_text, str), repr(args_text)[:60])
    check("arguments 解析后 == {\"city\": \"北京\"}",
          isinstance(args_text, str) and json.loads(args_text) == {"city": "北京"},
          repr(args_text))
    check("reasoning_content 带上了思考段", "北京" in (message.get("reasoning_content") or ""),
          repr((message.get("reasoning_content") or "")[:40]))

    # tool_choice 真的到了解析器：none 时同一段文本不得产出 tool_calls。
    with Chat(tmpl_src) as renderer:
        server.render_messages(renderer, case["messages"], tools=case["tools"],
                               tool_choice="none")
        none_parsed = renderer.parse(p0, partial=False)["msg"]
    check("tool_choice=none 时 parse 不出 tool_calls",
          not (none_parsed.get("tool_calls") or []), str(none_parsed.get("tool_calls")))
    check("tool_choice=none 时 XML 留在 content",
          "<tool_call>" in (none_parsed.get("content") or ""),
          repr((none_parsed.get("content") or "")[:60]))

    # -- C. role:tool 回传 ------------------------------------------------------
    print("\n=== C. 回传：历史 assistant.tool_calls + role:tool 一起渲染 ===")
    history = [*case["messages"], parsed, {"role": "tool", "tool_call_id": call["id"],
                                           "content": "晴 25 度"}]
    with Chat(tmpl_src) as renderer:
        try:
            replied = server.render_messages(renderer, history, tools=case["tools"]).prompt
            render_error = ""
        except Exception as exc:  # noqa: BLE001 - 失败的细节要打出来，而不是让脚本崩
            replied, render_error = "", f"{type(exc).__name__}: {exc}"
    check("带 tool 结果的历史渲染成功", not render_error, render_error)
    if not render_error:
        check("提示词里出现工具返回值", "晴 25 度" in replied)
        check("提示词里出现那次调用", "<function=get_weather>" in replied
              and "<parameter=city>" in replied)
        # 已知行为：Qwen 模板把 role:tool 合并进 user 块，不渲染 tool_call_id。
        # 断言「内容出现了」而不是断言 id 出现，否则这条会稳定误报。
        print(f"  （观察）提示词里是否出现 tool_call_id：{'call_' in replied}——"
              f"Qwen 模板不渲染它，属于已知行为")

        # 历史 arguments 的两种形状（JSON 字符串 / 已经是 mapping）必须渲染成同一份提示词。
        # 这是 (b) 的机制判据：归一之后，客户端回传的字符串不会再比 dict 少东西。
        tool_msg = {"role": "tool", "tool_call_id": "call_hist", "content": "晴 25 度"}
        with Chat(tmpl_src) as r_str:
            from_string = server.render_messages(
                r_str, [*case["messages"], assistant_history('{"city": "北京"}'), tool_msg],
                tools=case["tools"]).prompt
        with Chat(tmpl_src) as r_map:
            from_mapping = server.render_messages(
                r_map, [*case["messages"], assistant_history({"city": "北京"}), tool_msg],
                tools=case["tools"]).prompt
        check("历史 arguments：字符串与 mapping 渲染逐字节相同", from_string == from_mapping,
              "" if from_string == from_mapping else _first_diff(from_string, from_mapping))

    # -- D. 流式 ---------------------------------------------------------------
    print("\n=== D. 流式：逐片 partial=True，拼回的调用必须与一次性解析一致 ===")
    with Chat(tmpl_src) as renderer:
        server.render_messages(renderer, case["messages"], tools=case["tools"])
        seen: dict[int, str] = {}
        deltas: list[dict] = []
        step = max(1, len(p0) // 40)        # ~40 片，必然跨过每个标签边界
        for end in range(step, len(p0) + step, step):
            for diff in renderer.parse(p0[:end], partial=True)["diff"]:
                if delta := server.stream_delta(diff, seen):
                    deltas.append(delta)
        # 尾声用最终态补一次，与 server._stream_response 的收尾一致
        for diff in renderer.parse(p0, partial=False)["diff"]:
            if delta := server.stream_delta(diff, seen):
                deltas.append(delta)

    calls: dict[int, dict] = {}
    announced = 0
    text = ""
    for delta in deltas:
        text += delta.get("content") or ""
        for piece in delta.get("tool_calls") or []:
            slot = calls.setdefault(piece["index"],
                                    {"id": "", "name": "", "arguments": ""})
            if piece.get("id"):
                announced += 1
                slot["id"] = piece["id"]
            function = piece.get("function") or {}
            if function.get("name"):
                slot["name"] = function["name"]
            slot["arguments"] += function.get("arguments") or ""

    print(f"  {len(deltas)} 个 delta，{len(calls)} 个调用，id 出现 {announced} 次")
    if args.verbose:
        print("  deltas =", json.dumps(deltas, ensure_ascii=False)[:1200])
    check("流式重建出 1 个调用", len(calls) == 1, str(len(calls)))
    only = calls.get(0) or {}
    check("流式拼出的函数名 == get_weather", only.get("name") == "get_weather",
          repr(only.get("name")))
    check("流式拼出的 arguments 解出 {\"city\": \"北京\"}",
          json.loads(only.get("arguments") or "{}") == {"city": "北京"},
          repr(only.get("arguments"))[:80])
    check("id 只宣告一次（否则客户端会建出多个调用）", announced == 1, str(announced))
    check("流式与一次性的 arguments 逐字节相同",
          only.get("arguments") == args_text,
          _first_diff(only.get("arguments") or "", args_text or ""))
    check("流式 delta 里没有裸 XML（调用部分已被解析掉）",
          "<parameter=" not in text and "</function>" not in text, repr(text[:80]))

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
