"""Unit tests for the OpenAI tool-call plumbing in `server.py`.

The rendering DLL is not loaded here on purpose: everything under test is the Python
glue around it -- normalising historical `arguments`, forwarding `tools`/`tool_choice`,
and turning a parsed message into an OpenAI response. The real DLL and the real template
are exercised by `checks/verify_openai_tools.py`.

The renderer is a stub because the point is what `render_messages` sends *into* the
template engine, not what the engine returns; that comparison is the check script's job.
"""
from __future__ import annotations

import json
import unittest

import server


class StubRenderer:
    """Records the arguments of the last `render` call and returns a fake prompt."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def render(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return type("Rendered", (), {"prompt": "PROMPT"})()

    def thinking_start(self) -> str:
        return "<think>"


def assistant_with(arguments) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_x", "type": "function",
         "function": {"name": "get_weather", "arguments": arguments}}]}


class NormalizeToolCallArguments(unittest.TestCase):
    def test_json_string_becomes_mapping(self):
        out = server.normalize_tool_call_arguments([assistant_with('{"city": "北京"}')])
        self.assertEqual({"city": "北京"}, out[0]["tool_calls"][0]["function"]["arguments"])

    def test_mapping_is_kept_as_is(self):
        mapping = {"city": "北京"}
        out = server.normalize_tool_call_arguments([assistant_with(mapping)])
        self.assertIs(mapping, out[0]["tool_calls"][0]["function"]["arguments"])

    def test_empty_string_becomes_empty_mapping(self):
        for raw in ("", "   ", "\n"):
            out = server.normalize_tool_call_arguments([assistant_with(raw)])
            self.assertEqual({}, out[0]["tool_calls"][0]["function"]["arguments"], repr(raw))

    def test_malformed_json_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            server.normalize_tool_call_arguments([assistant_with("{not json")])
        self.assertIn("arguments", str(caught.exception))

    def test_non_object_json_is_refused(self):
        # The template iterates arguments with `| items`; a scalar or array would fail
        # there with a message that names neither the field nor the message.
        for raw in ("3", "[1, 2]", "true", "null"):
            with self.assertRaises(ValueError, msg=raw):
                server.normalize_tool_call_arguments([assistant_with(raw)])

    def test_input_is_not_mutated(self):
        messages = [{"role": "user", "content": "hi"}, assistant_with('{"city": "北京"}'),
                    {"role": "tool", "tool_call_id": "call_x", "content": "晴"}]
        before = json.loads(json.dumps(messages, ensure_ascii=False))
        server.normalize_tool_call_arguments(messages)
        self.assertEqual(before, messages)

    def test_untouched_messages_are_passed_through(self):
        plain = {"role": "user", "content": "hi"}
        system = {"role": "system", "content": "sys"}
        tool = {"role": "tool", "tool_call_id": "call_1", "content": "r"}
        no_calls = {"role": "assistant", "content": "hello"}
        not_a_list = {"role": "assistant", "content": "",
                      "tool_calls": {"function": {"name": "x", "arguments": "{}"}}}
        messages = [plain, system, tool, no_calls, not_a_list]
        out = server.normalize_tool_call_arguments(messages)
        for got, want in zip(out, messages):
            self.assertIs(want, got)

    def test_multiple_calls_are_each_normalised(self):
        msg = {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "a", "arguments": '{"x": 1}'}},
            {"type": "function", "function": {"name": "b", "arguments": {"y": 2}}},
            {"type": "function", "function": {"name": "c"}}]}
        out = server.normalize_tool_call_arguments([msg])[0]["tool_calls"]
        self.assertEqual({"x": 1}, out[0]["function"]["arguments"])
        self.assertEqual({"y": 2}, out[1]["function"]["arguments"])
        self.assertNotIn("arguments", out[2]["function"])


class RenderMessages(unittest.TestCase):
    def test_tools_and_choice_are_forwarded(self):
        renderer = StubRenderer()
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        server.render_messages(renderer, [{"role": "user", "content": "hi"}],
                               tools=tools, tool_choice="required",
                               parallel_tool_calls=True)
        call = renderer.calls[-1]
        self.assertEqual(tools, call["tools"])
        self.assertEqual("required", call["tool_choice"])
        self.assertIs(True, call["parallel_tool_calls"])

    def test_no_tools_means_none_not_empty_list(self):
        renderer = StubRenderer()
        server.render_messages(renderer, [{"role": "user", "content": "hi"}])
        self.assertIsNone(renderer.calls[-1]["tools"])
        self.assertIsNone(renderer.calls[-1]["tool_choice"])
        self.assertIsNone(renderer.calls[-1]["parallel_tool_calls"])

    def test_system_is_merged_in_front_and_arguments_are_fixed(self):
        renderer = StubRenderer()
        messages = [{"role": "system", "content": "be brief"},
                    {"role": "user", "content": "北京天气？"},
                    assistant_with('{"city": "北京"}'),
                    {"role": "tool", "tool_call_id": "call_x", "content": "晴 25 度"}]
        server.render_messages(renderer, messages)
        turns = renderer.calls[-1]["messages"]
        self.assertEqual("system", turns[0]["role"])
        self.assertEqual("be brief", turns[0]["content"])
        self.assertEqual({"city": "北京"}, turns[2]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual("call_x", turns[3]["tool_call_id"])


class RequestMaxTokens(unittest.TestCase):
    """`max_completion_tokens` is the name DSH actually sends; `max_tokens` is the old one."""

    def test_old_name_alone_is_honoured(self):
        self.assertEqual(99, server.request_max_tokens({"max_tokens": 99}, 512))

    def test_new_name_alone_is_honoured(self):
        self.assertEqual(99, server.request_max_tokens({"max_completion_tokens": 99}, 512))

    def test_new_name_wins_when_both_are_sent(self):
        body = {"max_tokens": 11, "max_completion_tokens": 99}
        self.assertEqual(99, server.request_max_tokens(body, 512))

    def test_neither_name_falls_back_to_the_server_default(self):
        self.assertEqual(512, server.request_max_tokens({}, 512))

    def test_null_and_zero_count_as_absent(self):
        self.assertEqual(512, server.request_max_tokens({"max_tokens": None}, 512))
        self.assertEqual(512, server.request_max_tokens({"max_tokens": 0}, 512))


class OpenAIAssistantMessage(unittest.TestCase):
    def test_tool_call_shape(self):
        parsed = {"role": "assistant", "content": "", "reasoning_content": "想想",
                  "tool_calls": [{"type": "function",
                                  "function": {"name": "get_weather",
                                               "arguments": "{\"city\": \"北京\"}"}}]}
        message, finish = server.openai_assistant_message(
            parsed, tokens=10, max_tokens=100, new_id=lambda: "call_1")
        self.assertEqual("tool_calls", finish)
        self.assertIsNone(message["content"])
        self.assertEqual("想想", message["reasoning_content"])
        call = message["tool_calls"][0]
        self.assertEqual("call_1", call["id"])
        self.assertEqual("function", call["type"])
        self.assertEqual("get_weather", call["function"]["name"])
        self.assertIsInstance(call["function"]["arguments"], str)
        self.assertEqual({"city": "北京"}, json.loads(call["function"]["arguments"]))

    def test_content_before_a_tool_call_is_kept(self):
        parsed = {"role": "assistant", "content": "我查一下。",
                  "tool_calls": [{"type": "function",
                                  "function": {"name": "get_weather", "arguments": "{}"}}]}
        message, _ = server.openai_assistant_message(
            parsed, tokens=1, max_tokens=100, new_id=lambda: "call_1")
        self.assertEqual("我查一下。", message["content"])

    def test_each_call_gets_its_own_id(self):
        ids = iter(["call_1", "call_2"])
        parsed = {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"type": "function", "function": {"name": "b", "arguments": "{}"}}]}
        message, _ = server.openai_assistant_message(
            parsed, tokens=1, max_tokens=100, new_id=lambda: next(ids))
        self.assertEqual(["call_1", "call_2"], [c["id"] for c in message["tool_calls"]])

    def test_plain_answer_keeps_the_old_shape(self):
        parsed = {"role": "assistant", "content": "北京晴，25 度。"}
        message, finish = server.openai_assistant_message(
            parsed, tokens=5, max_tokens=100)
        self.assertEqual({"role": "assistant", "content": "北京晴，25 度。"}, message)
        self.assertEqual("stop", finish)

    def test_length_wins_only_without_a_tool_call(self):
        parsed = {"role": "assistant", "content": "半句"}
        message, finish = server.openai_assistant_message(
            parsed, tokens=100, max_tokens=100)
        self.assertEqual("length", finish)
        self.assertEqual("半句", message["content"])

    def test_missing_reasoning_is_not_emitted(self):
        parsed = {"role": "assistant", "content": "答"}
        message, _ = server.openai_assistant_message(parsed, tokens=1, max_tokens=10)
        self.assertNotIn("reasoning_content", message)

    def test_missing_content_becomes_empty_string(self):
        message, _ = server.openai_assistant_message(
            {"role": "assistant"}, tokens=1, max_tokens=10)
        self.assertEqual("", message["content"])

    def test_default_ids_look_like_openai_ids(self):
        parsed = {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "a", "arguments": "{}"}}]}
        message, _ = server.openai_assistant_message(parsed, tokens=1, max_tokens=10)
        self.assertTrue(message["tool_calls"][0]["id"].startswith("call_"))
        self.assertEqual(29, len(message["tool_calls"][0]["id"]))   # call_ + 24 hex


class StreamDelta(unittest.TestCase):
    """`qlh_chat_parse`'s per-chunk diff, turned into OpenAI streaming deltas.

    The observed partial-mode behaviour (real DLL, real template) is the contract these
    encode: the first slice names the function and leaves `id` empty, every later slice
    carries only an `arguments` fragment, and `index` is constant for one call.
    """

    @staticmethod
    def diff(arguments="", name="", index=0, reasoning="", content=""):
        return {"reasoning_content_delta": reasoning, "content_delta": content,
                "tool_call_index": index,
                "tool_call_delta": {"name": name, "arguments": arguments, "id": ""}}

    def test_first_slice_announces_the_call(self):
        seen: dict[int, str] = {}
        delta = server.stream_delta(self.diff(name="get_weather", arguments="{"), seen,
                                    new_id=lambda: "call_1")
        self.assertEqual([{"index": 0, "id": "call_1", "type": "function",
                           "function": {"name": "get_weather", "arguments": "{"}}],
                         delta["tool_calls"])

    def test_later_slices_only_append_arguments(self):
        seen: dict[int, str] = {}
        server.stream_delta(self.diff(name="get_weather", arguments="{"), seen,
                            new_id=lambda: "call_1")
        delta = server.stream_delta(self.diff(arguments='"city":'), seen,
                                    new_id=lambda: "call_2")
        self.assertEqual([{"index": 0, "function": {"arguments": '"city":'}}],
                         delta["tool_calls"])
        self.assertEqual({"0": "call_1"}, {str(k): v for k, v in seen.items()})

    def test_empty_argument_slice_is_dropped(self):
        seen: dict[int, str] = {}
        server.stream_delta(self.diff(name="f", arguments="{"), seen, new_id=lambda: "c")
        self.assertEqual({}, server.stream_delta(self.diff(arguments=""), seen,
                                                 new_id=lambda: "c"))

    def test_reasoning_comes_before_content_in_one_delta(self):
        delta = server.stream_delta(self.diff(index=None, reasoning="想", content="答"),
                                    {}, new_id=lambda: "c")
        self.assertEqual({"reasoning_content": "想", "content": "答"}, delta)

    def test_no_tool_call_index_means_no_tool_calls_key(self):
        delta = server.stream_delta(self.diff(index=None, content="答"), {},
                                    new_id=lambda: "c")
        self.assertNotIn("tool_calls", delta)

    def test_second_call_gets_its_own_id(self):
        seen: dict[int, str] = {}
        ids = iter(["call_1", "call_2"])
        first = server.stream_delta(self.diff(name="a", arguments="{}", index=0), seen,
                                    new_id=lambda: next(ids))
        second = server.stream_delta(self.diff(name="b", arguments="{}", index=1), seen,
                                     new_id=lambda: next(ids))
        self.assertEqual("call_1", first["tool_calls"][0]["id"])
        self.assertEqual("call_2", second["tool_calls"][0]["id"])


class StreamResponseFailures(unittest.TestCase):
    """A stream that cannot start must not claim 200 first.

    DSH reads the status line as the answer: "200, then one error frame, then EOF" is
    indistinguishable from a connection that broke mid-answer, and that is exactly how a
    request that was merely too large for the KV window got reported as WinError 10053.
    """

    def handler(self, chain, write_error=None):
        handler = type("H", (), {})()
        handler.chain = chain
        handler.heads = 0
        handler.frames = []
        handler.jsons = []

        def head():
            handler.heads += 1

        def sse(payload):
            if write_error:
                raise write_error
            handler.frames.append(payload)

        handler._sse_head = head
        handler._sse = sse
        handler._json = lambda code, payload: handler.jsons.append((code, payload))
        return handler

    def test_overflow_before_the_first_byte_is_a_plain_400(self):
        handler = self.handler(FakeChain([], server.ContextOverflow("too big")))
        server.Handler._stream_response(handler, "cid", 0, "m", "prompt", 8, [])
        self.assertEqual(0, handler.heads)          # nothing was written yet
        self.assertEqual(400, handler.jsons[0][0])
        self.assertIn("too big", handler.jsons[0][1]["error"]["message"])

    def test_overflow_mid_stream_ends_with_an_error_frame(self):
        handler = self.handler(FakeChain(["一", "二"], server.ContextOverflow("too big")))
        server.Handler._stream_response(handler, "cid", 0, "m", "prompt", 8, [])
        self.assertEqual(1, handler.heads)          # the stream had already started
        self.assertEqual([], handler.jsons)         # and cannot be re-typed as a 400
        self.assertIn("error", handler.frames[-2])
        self.assertEqual("[DONE]", handler.frames[-1])

    def test_a_hung_up_client_is_not_reported_as_a_failure(self):
        handler = self.handler(FakeChain(["一"]), write_error=ConnectionAbortedError(10053))
        server.Handler._stream_response(handler, "cid", 0, "m", "prompt", 8, [])
        # The status line did go out (that is when the write failed), but nothing is
        # turned into a 500 and no second write is attempted on the dead socket.
        self.assertEqual([], handler.jsons)
        self.assertEqual([], handler.frames)

    def test_connection_aborted_counts_as_a_hung_up_client(self):
        self.assertIn(ConnectionAbortedError, server.CLIENT_GONE)


class FakeChain:
    """Feeds pre-baked text through the sink, then raises the configured error."""

    def __init__(self, texts, error=None):
        self.texts = list(texts)
        self.error = error

    def drain(self, prompt, max_tokens, sampling, sink=None):
        for text in self.texts:
            sink(text)
        if self.error is not None:
            raise self.error
        return "".join(self.texts), len(self.texts), 3, 0.1, {}

    def sampling_state(self):
        return {}


class AssistantTurnClosing(unittest.TestCase):
    """The last assistant turn has to be rendered in the shape the template wrote it.

    Qwen-style templates open the generation prompt with `<think>` but render a historical
    assistant message without it, so a conversation re-rendered from history diverges from
    the KV at the assistant header. Measured before this existed: reuse was 0% on every
    turn because the common prefix stopped at that header.
    """

    def render(self, messages):
        renderer = StubRenderer()
        server.render_messages(renderer, messages)
        return renderer.calls[-1]["messages"]

    def test_the_last_assistant_turn_gets_the_marker(self):
        turns = self.render([{"role": "user", "content": "问"},
                             {"role": "assistant", "content": "答"}])
        self.assertEqual(" <think>\n答", turns[-1]["content"])

    def test_a_marker_already_present_is_not_doubled(self):
        turns = self.render([{"role": "user", "content": "问"},
                             {"role": "assistant", "content": "<think>\n答"}])
        self.assertEqual("<think>\n答", turns[-1]["content"])

    def test_a_history_turn_is_marked_even_when_the_request_ends_in_user(self):
        # The interesting case: every real follow-up ends in a user turn, so the assistant
        # message needing the marker is never the last one.
        turns = self.render([{"role": "user", "content": "问"},
                             {"role": "assistant", "content": "答"},
                             {"role": "user", "content": "再问"}])
        self.assertEqual(" <think>\n答", turns[1]["content"])

    def test_an_assistant_turn_carrying_tool_calls_is_left_alone(self):
        turns = self.render([{"role": "user", "content": "问"},
                             {"role": "assistant", "content": None, "tool_calls": [
                                 {"id": "c", "type": "function",
                                  "function": {"name": "f", "arguments": "{}"}}]}])
        self.assertIsNone(turns[-1]["content"])

    def test_the_input_messages_are_not_mutated(self):
        messages = [{"role": "user", "content": "问"},
                    {"role": "assistant", "content": "答"}]
        server.render_messages(StubRenderer(), messages)
        self.assertEqual("答", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
