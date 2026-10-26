# -*- coding: utf-8 -*-
"""`#126`：思考内容被抑制时，流式通道必须发心跳而不是干等。

背景（`docs/已知问题记录.md` 的 #126）：

- 三机线性路径在 `enable_thinking=True` 且未要求展示思考时，会进入
  `NativePorts.suppress_thinking`（`native_thinking_suppression_required`：
  `not show_thinking` 且 prompt 留着没闭合的 `<think` 块）。
- 这个抑制的**本意是"不要把思考文本发给客户端"**，但此前的实现是裸 `return`
  —— 整个思考期**一个字节都不发**。实测真机 21:34 请求 `duration_ms=10555`，
  期间无任何 SSE 帧；用户观感是"不开 thinking 还凑合、开了就超时"，
  TUI 则表现为"停刷"（Textual 只能靠增量事件驱动重绘）。
- 修法：**发心跳不发内容**。`suppress_thinking` 的承诺（不泄露思考文本）仍然
  成立，但连接能看出"还在推进"。

这几个测试锁住的就是这条边界。
"""

import pytest

from qlh_core.bootstrap.execution import NativePorts


def _ports(*, suppress: bool, events: list) -> NativePorts:
    """只构造 `emit` 需要的那几个属性（不去碰 host/scheduler 那些重依赖）。"""
    ports = NativePorts.__new__(NativePorts)
    ports.options = {"_stream_callback": events.append}
    ports.suppress_thinking = suppress
    ports.thinking_buffer = ""
    return ports


def test_suppressed_thinking_emits_heartbeat_and_hides_content():
    events: list = []
    ports = _ports(suppress=True, events=events)

    ports.emit({"token": "思"})
    ports.emit({"token": "考"})

    assert events, "抑制期间不能完全没有帧（否则客户端会以为连接死了）"
    for event in events:
        assert event.get("heartbeat") is True
        assert event.get("phase") == "thinking"
        # 承诺的核心：心跳里**不能**夹带任何思考文本
        assert "token" not in event
        assert "thinking" not in event
    # 计数递增，客户端据此区分"卡住"与"仍在推进"
    assert [event["suppressed_tokens"] for event in events] == [1, 2]


def test_content_resumes_after_the_thinking_block_closes():
    events: list = []
    ports = _ports(suppress=True, events=events)

    ports.emit({"token": "思考中"})
    # `</think>` 一出现就结束抑制；其后同一 token 里的正文照常发
    ports.emit({"token": "</think>正文"})

    heartbeats = [e for e in events if e.get("heartbeat")]
    bodies = [e for e in events if e.get("token")]
    assert len(heartbeats) == 1
    assert bodies == [{"token": "正文"}]
    # 抑制已解除，后续 token 不再走心跳
    ports.emit({"token": "继续"})
    assert events[-1] == {"token": "继续"}


def test_no_suppression_forwards_tokens_directly():
    events: list = []
    ports = _ports(suppress=False, events=events)

    ports.emit({"token": "答"})
    ports.emit({"token": "案"})

    assert events == [{"token": "答"}, {"token": "案"}]
    assert not any(event.get("heartbeat") for event in events)


def test_missing_stream_callback_is_a_noop():
    """没有流式消费者时什么都不该发生（也不能抛）。"""
    events: list = []
    ports = _ports(suppress=True, events=events)
    ports.options = {}

    ports.emit({"token": "思"})

    assert events == []


@pytest.mark.parametrize("suppress", [True, False])
def test_heartbeat_and_token_events_are_mutually_exclusive(suppress):
    """同一帧不能既算心跳又算正文 —— 客户端按字段分派，混在一起会重复渲染。"""
    events: list = []
    ports = _ports(suppress=suppress, events=events)

    ports.emit({"token": "x"})

    for event in events:
        assert ("heartbeat" in event) != ("token" in event)
