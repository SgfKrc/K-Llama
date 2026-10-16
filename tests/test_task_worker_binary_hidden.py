"""★ 票 21a（COMM-WIRE-01）：二进制 hidden 帧的端到端往返。

覆盖两层各自的测试之外**跨层**的那一段：`split_inline_hidden`（协议层）产出的
二进制体经 `tcp_comm.build_message` / `recv_data` 的帧层原样到达对端，再由
`restore_inline_hidden` 还原成 base64 内联 hidden —— 逐位无损。

不依赖真实 socket：用 `pack_data` 拼帧、按 `recv_data` 的读取顺序解析，
与 `TCPClient.recv_data` / 服务端收帧循环的布局完全一致。
"""
import base64
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tcp_comm import (  # noqa: E402
    HEADER_LEN,
    MessageType,
    build_message,
    pack_data,
    unpack_header,
)
from task_worker_protocol import (  # noqa: E402
    restore_inline_hidden,
    split_inline_hidden,
)

#: 一段 32 字节的 hidden（8 个 f32 的字节数），逐位可核。
HIDDEN = bytes(range(32))


def _recv_frame(packet):
    """按 `recv_data` 的顺序解析一帧：返回 (meta dict, binary|None)。

    与 `TCPClient.recv_data` 及服务端收帧循环同构：先读元数据段，
    若 `_needs_binary` 再读一个长度前缀的数据段。
    """
    meta_len = unpack_header(packet[:HEADER_LEN])
    meta = packet[HEADER_LEN:HEADER_LEN + meta_len]
    meta = json.loads(meta.decode("utf-8"))
    rest = packet[HEADER_LEN + meta_len:]
    body = None
    if meta.get("format") == "binary":
        body_len = unpack_header(rest[:HEADER_LEN])
        body = rest[HEADER_LEN:HEADER_LEN + body_len]
    return meta, body


def _wire(snapshot, binary_body=None):
    """按 `send_to_client` / `send_data` 的方式封帧（`data` 进 meta）。"""
    return build_message(
        MessageType.TASK_WORKER, snapshot, binary_body=binary_body,
    )


def _make_offer():
    return {
        "message_type": "stage_offer",
        "payload": {
            "root_input": {
                "hidden_f32": base64.b64encode(HIDDEN).decode("ascii"),
                "other": "keep",
            },
            "input_sha256": "abc",
        },
    }


def test_binary_hidden_offer_roundtrip_is_bit_identical():
    """入程：抽体 → 帧 → 还原 ⇒ 与原 stage_offer 逐字段相同。"""
    offer = _make_offer()
    snapshot, body = split_inline_hidden(offer, enabled=True)
    assert body == HIDDEN

    # 发送：JSON 里已无 hidden_f32，字节走独立数据段
    packet = _wire(snapshot, body)
    meta, received = _recv_frame(packet)

    assert meta["format"] == "binary"
    assert meta["binary_bytes"] == len(HIDDEN)
    assert received == HIDDEN
    assert "hidden_f32" not in meta["data"]["payload"]["root_input"]

    # 接收：decode 之前把体放回 ⇒ 与原始消息逐字段相同
    assert restore_inline_hidden(meta["data"], received) == offer


def test_binary_hidden_result_roundtrip_is_bit_identical():
    """回程：`stage_result.output.hidden_out_f32` 同理（位置按消息类型推导）。"""
    result = {
        "message_type": "stage_result",
        "payload": {
            "output": {"hidden_out_f32": base64.b64encode(HIDDEN).decode("ascii")},
            "attempt_id": "a1",
        },
    }
    snapshot, body = split_inline_hidden(result, enabled=True)
    assert body == HIDDEN
    meta, received = _recv_frame(_wire(snapshot, body))

    assert received == HIDDEN
    assert restore_inline_hidden(meta["data"], received) == result


def test_unadvertised_peer_keeps_the_old_base64_frame():
    """未声明能力 ⇒ 不抽体 ⇒ 帧里没有数据段，JSON 与旧路径逐字节一致。"""
    offer = _make_offer()
    snapshot, body = split_inline_hidden(offer, enabled=False)
    assert body is None
    assert snapshot == offer

    packet = _wire(snapshot, body)
    meta, received = _recv_frame(packet)

    # 与不传 binary_body 的既有帧逐字节相同
    assert packet == build_message(MessageType.TASK_WORKER, offer)
    assert meta["format"] == "json"
    assert meta.get("binary_bytes") is None
    assert received is None
    assert restore_inline_hidden(meta["data"], received) == offer


def test_binary_hidden_chunk_roundtrip_is_bit_identical():
    """分片：`stage_chunk` / `stage_result_chunk` 的 `payload_b64` 也走二进制。

    载荷就在 `payload` 顶层，所以路径表用容器键 `None` 表示；入程与回程分片共用
    同一形状（`build_stage_chunk` 的 `message_type` 参数），因此一条定义覆盖两个方向。
    """
    chunk = bytes(range(64))
    for message_type in ("stage_chunk", "stage_result_chunk"):
        message = {
            "message_type": message_type,
            "payload": {
                "payload_b64": base64.b64encode(chunk).decode("ascii"),
                "chunk_index": 0,
                "chunk_count": 1,
            },
        }

        # 抽体：字段从 JSON 删除，raw 走独立数据段
        snapshot, body = split_inline_hidden(message, enabled=True)
        assert body == chunk
        assert "payload_b64" not in snapshot["payload"]
        assert snapshot["payload"]["chunk_index"] == 0

        meta, received = _recv_frame(_wire(snapshot, body))
        assert received == chunk
        # 还原 ⇒ 与原分片消息逐字段相同
        assert restore_inline_hidden(meta["data"], received) == message

        # 未声明 ⇒ 原样返回，帧与旧路径逐字节一致
        same, none = split_inline_hidden(message, enabled=False)
        assert none is None
        assert same == message


def test_binary_body_is_not_inlined_into_json_metadata():
    """体不进 JSON 元数据：元数据段远小于体，也不含 base64。"""
    offer = _make_offer()
    snapshot, body = split_inline_hidden(offer, enabled=True)
    packet = _wire(snapshot, body)

    meta_len = unpack_header(packet[:HEADER_LEN])
    assert meta_len < 1024
    assert base64.b64encode(HIDDEN) not in packet[:HEADER_LEN + meta_len]
