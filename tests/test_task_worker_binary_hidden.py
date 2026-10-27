"""Binary hidden regressions through real framing, validation and scheduler paths.

Small round trips preserve wire bytes. Larger layer-stage fixtures exercise
negotiation, dispatch, chunk assembly and return without loading any model.
"""
import base64
import hashlib
import io
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

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

from task_worker_protocol import (
    MAX_MESSAGE_BYTES, MAX_BINARY_FRAME_BYTES,
    WorkerProtocolError, build_message as build_worker_message,
    canonical_sha256, decode_message, hidden_fits_stage_frame,
    max_hidden_tokens,
)
from tcp_comm import TCPClient, TCPServer
from scheduler_task_worker import SchedulerTaskWorkerMixin
from task_worker_adapter import TaskWorkerControlPlane, RemoteFullWorkerProvider
from execution_provider import ModelIdentity, StageRequest, StageAttempt
from tcp_comm import ClientConn, parse_message

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




class _MemorySocket:
    def __init__(self, packet=b""):
        self.stream = io.BytesIO(packet)
        self.reads = []
        self.packets = []
        self.closed = False

    def recv(self, size):
        self.reads.append(size)
        return self.stream.read(size)

    def sendall(self, packet):
        self.packets.append(packet)

    def settimeout(self, _timeout):
        pass

    def close(self):
        self.closed = True


def _identity():
    return dict(workflow_id="wf_binaryregression", stage_id="segment_1",
                attempt_id="att_binaryregression", lease_id="lease_binaryregression",
                lease_epoch=1, provider_id="remote_worker_01")


def _result_payload(raw):
    output = {"hidden_out_f32": base64.b64encode(raw).decode("ascii"),
              "hidden_out_sha256": hashlib.sha256(raw).hexdigest()}
    return {**_identity(), "output": output,
            "output_sha256": canonical_sha256(output), "metadata": {}}


def _hello_payload(value):
    caps = {"stage_types": ["layer_forward"], "engines": ["llama_cpp"],
            "models": [], "max_concurrency": 1}
    if value is not None:
        caps["binary_hidden"] = value
    return {"node_id": "worker_01", "worker_kind": "pc_full_worker",
            "min_version": 3, "max_version": 3, "capabilities": caps}


@pytest.mark.parametrize("value", [0, 1, "false", "true", [], {}])
def test_binary_capability_rejects_non_boolean_values(value):
    with pytest.raises(WorkerProtocolError) as invalid:
        build_worker_message("hello", _hello_payload(value),
                             message_id="msg_binaryhello01", sent_at_ms=1)
    assert invalid.value.code == "invalid_boolean"


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("embd", [1, 8, 2048, 4096])
def test_max_binary_tokens_matches_the_actual_frame_boundary(dtype, embd):
    limit = max_hidden_tokens(embd, dtype, binary=True)
    assert hidden_fits_stage_frame(limit, embd, dtype, binary=True)
    assert not hidden_fits_stage_frame(limit + 1, embd, dtype, binary=True)


def test_binary_only_sized_result_passes_real_validation_in_both_directions():
    raw = b"x" * (7 * 1024 * 1024)
    payload = _result_payload(raw)
    with pytest.raises(WorkerProtocolError, match="maximum size"):
        build_worker_message("stage_result", payload,
                             message_id="msg_binaryresult01", sent_at_ms=1)
    message = build_worker_message("stage_result", payload,
                                  message_id="msg_binaryresult01", sent_at_ms=1,
                                  binary_hidden=True)
    snapshot, body = split_inline_hidden(message.snapshot(), enabled=True)
    packet = _wire(snapshot, body)
    assert len(packet) < MAX_MESSAGE_BYTES
    received = TCPClient(client_id="binary-test").recv_data(_MemorySocket(packet))
    restored = restore_inline_hidden(received["data"], received["binary"])
    assert decode_message(restored, binary_hidden=True).snapshot() == message.snapshot()
    # The same enlarged inline message is still invalid without an actual
    # binary context, even when its content happens to be valid hidden.
    with pytest.raises(WorkerProtocolError, match="maximum size"):
        decode_message(restored)


@pytest.mark.parametrize("declared", [1, 5, 4.0, True, "4", None])
@pytest.mark.parametrize("side", ["client", "server"])
def test_binary_length_mismatch_rejected_before_body_read(declared, side):
    meta = {"type": "task_worker", "format": "binary", "binary_bytes": declared,
            "data": {}}
    meta_bytes = json.dumps(meta).encode("utf-8")
    packet = pack_data(meta_bytes) + pack_data(b"abcd")
    sock = _MemorySocket(packet)
    if side == "client":
        with pytest.raises(ConnectionError):
            TCPClient(client_id="binary-test").recv_data(sock)
    else:
        server = TCPServer(host="127.0.0.1", port=0)
        server._running = True
        delivered = []
        server.on_message = lambda *args: delivered.append(args)
        server._handle_client(sock, ("127.0.0.1", 1234), "pending_1234")
        assert delivered == []
        assert sock.closed
    assert sock.stream.tell() <= 4 + len(meta_bytes) + 4


@pytest.mark.parametrize("side", ["client", "server"])
def test_task_worker_binary_budget_rejected_before_body_read(side):
    meta = {"type": "task_worker", "format": "binary",
            "binary_bytes": MAX_BINARY_FRAME_BYTES, "data": {}}
    meta_bytes = json.dumps(meta).encode("utf-8")
    # Do not allocate or upload the oversized body: its length alone must fail.
    import struct
    sock = _MemorySocket(pack_data(meta_bytes) + struct.pack(">I", MAX_BINARY_FRAME_BYTES))
    if side == "client":
        with pytest.raises(ConnectionError, match="maximum size"):
            TCPClient(client_id="binary-test").recv_data(sock)
    else:
        server = TCPServer(host="127.0.0.1", port=0)
        server._running = True
        server._handle_client(sock, ("127.0.0.1", 1234), "pending_1234")
        assert sock.closed
    assert sock.reads == [4, len(meta_bytes), 4]


@pytest.mark.parametrize("message", [
    {"message_type": "hello", "payload": {}},
    {"message_type": "stage_offer", "payload": {}},
    {"message_type": "stage_offer", "payload": {"root_input": {"hidden_f32": "AAAA"}}},
    {"message_type": "stage_offer", "payload": {"root_input": {"hidden_ref": {}}}},
    {"message_type": "stage_result", "payload": {"output": {"hidden_out_ref": {}}}},
])
def test_binary_body_has_one_unambiguous_destination(message):
    with pytest.raises(WorkerProtocolError) as invalid:
        restore_inline_hidden(message, b"abcd")
    assert invalid.value.code == "invalid_binary_hidden"




def _negotiate(worker, binary=True, ack_value=True):
    owner = worker._tcp_client.capture_registration_owner()
    hello = worker._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_hello_payload(binary)["capabilities"],
        registration_owner=owner,
    )
    coordinator = TaskWorkerControlPlane()
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    raw_ack = ack.snapshot()
    if ack_value is None:
        raw_ack["payload"].pop("binary_hidden", None)
    else:
        raw_ack["payload"]["binary_hidden"] = ack_value
    worker._task_worker_control.receive_on_worker(raw_ack)
    return coordinator, ack


class _Worker(SchedulerTaskWorkerMixin):
    def __init__(self):
        self._task_worker_control = TaskWorkerControlPlane()
        self._tcp_client = TCPClient(client_id="worker_01")
        self._tcp_client.sock = _MemorySocket()
        self._tcp_client._registered = True
        self.offers = []

    def _effective_role(self):
        return "client"

    def _prepare_task_worker_request(self, message):
        # Observe a fully decoded production offer without invoking any model.
        return True, [message]

    def _replay_task_worker_responses(self, responses):
        self.offers.extend(responses)


@pytest.mark.parametrize("advertised", [None, False, True])
def test_ack_extension_is_only_sent_to_explicitly_opted_in_workers(advertised):
    coordinator = TaskWorkerControlPlane()
    hello = build_worker_message("hello", _hello_payload(advertised),
                                 message_id="msg_binaryhello02", sent_at_ms=1)
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    assert ack.payload["accepted"]
    expected = {"coordinator_node_id", "accepted", "selected_version", "reason_code"}
    if advertised is True:
        expected.add("binary_hidden")
        assert ack.payload["binary_hidden"] is True
    assert set(ack.payload) == expected


@pytest.mark.parametrize("value", [0, 1, 1.0, "false", None, [], {}])
def test_ack_binary_hidden_is_strictly_boolean(value):
    with pytest.raises(WorkerProtocolError) as invalid:
        build_worker_message("hello_ack", {
            "coordinator_node_id": "master", "accepted": True,
            "selected_version": 3, "reason_code": "", "binary_hidden": value,
        }, message_id="msg_binaryack001", sent_at_ms=1)
    assert invalid.value.code == "invalid_boolean"


@pytest.mark.parametrize("version,accepted,selected,reason", [
    (3, False, 0, "unsupported_protocol_version"), (2, True, 2, ""),
])
def test_binary_ack_cannot_enable_a_rejected_or_old_protocol(version, accepted, selected, reason):
    with pytest.raises(WorkerProtocolError) as invalid:
        build_worker_message("hello_ack", {
            "coordinator_node_id": "master", "accepted": accepted,
            "selected_version": selected, "reason_code": reason, "binary_hidden": True,
        }, message_id="msg_binaryack002", sent_at_ms=1, version=version)
    assert invalid.value.code == "invalid_negotiation_result"


@pytest.mark.parametrize("advertised,ack", [(True, True), (True, False), (True, None),
                                           (None, True), (False, True)])
def test_binary_return_requires_both_hello_and_explicit_ack(advertised, ack):
    worker = _Worker()
    _negotiate(worker, advertised, ack)
    enabled = advertised is True and ack is True
    assert worker._task_worker_binary_hidden_enabled() is enabled
    payload = _result_payload(b"abcd")
    worker._send_task_worker_to_master(build_worker_message(
        "stage_result", payload, message_id="msg_binaryresult02", sent_at_ms=1))
    frame = TCPClient(client_id="reader").recv_data(
        _MemorySocket(worker._tcp_client.sock.packets[-1]))
    assert frame["format"] == ("binary" if enabled else "json")


def test_binary_capability_is_retired_on_disconnect_and_reconnect():
    worker = _Worker()
    _negotiate(worker)
    assert worker._task_worker_binary_hidden_enabled()
    worker._task_worker_control.disconnect_coordinator()
    assert not worker._task_worker_binary_hidden_enabled()
    _negotiate(worker)
    worker._tcp_client.sock = _MemorySocket()
    worker._tcp_client._registration_epoch += 1
    assert not worker._task_worker_binary_hidden_enabled()


@pytest.mark.parametrize("declared", [1.0, True, "1", None, 0, -1])
def test_parse_binary_declaration_rejects_invalid_integer_types(declared):
    with pytest.raises(ConnectionError):
        parse_message(json.dumps({"format": "binary", "binary_bytes": declared}).encode())


def test_binary_and_json_frames_share_one_stream_without_losing_boundaries():
    snapshot, raw = split_inline_hidden(_make_offer(), enabled=True)
    packet = _wire(snapshot, raw) + build_message(MessageType.HEARTBEAT, {"ok": True})
    sock = _MemorySocket(packet)
    client = TCPClient(client_id="reader")
    first, second = client.recv_data(sock), client.recv_data(sock)
    assert restore_inline_hidden(first["data"], first["binary"]) == _make_offer()
    assert second == {"type": "heartbeat", "format": "json", "data": {"ok": True}}
    assert client.recv_data(sock) is None


def test_binary_sender_enforces_the_complete_outer_frame_budget():
    with pytest.raises(ConnectionError, match="maximum size"):
        build_message(MessageType.TASK_WORKER, {}, binary_body=b"x" * MAX_BINARY_FRAME_BYTES)


@pytest.mark.parametrize("master_enabled", [True, False])
@pytest.mark.parametrize("binary,raw_bytes", [(True, 7 * 1024 * 1024),
                                             (True, 9 * 1024 * 1024),
                                             (None, 7 * 1024 * 1024)])
def test_layer_stage_offer_chunks_and_return_use_production_scheduler_paths(
    binary, raw_bytes, master_enabled, monkeypatch,
):
    # All frames pass through TCPClient/TCPServer framing, both scheduler receive
    # handlers and the real provider state machine. No model loader is present.
    raw = b"x" * raw_bytes
    monkeypatch.setenv("QLH_TASK_WORKER_BINARY_HIDDEN", "1" if master_enabled else "0")
    worker = _Worker()
    negotiated = binary is True and master_enabled
    coordinator, ack = _negotiate(worker, binary, negotiated if binary is True else None)
    assert ack.payload.get("binary_hidden", False) is negotiated
    assert worker._task_worker_binary_hidden_enabled() is negotiated
    model = ModelIdentity(model_id="segment_4_16", engine="llama_cpp", format="gguf",
                          revision="local-v1", sha256="b" * 64)
    caps = {**_hello_payload(binary)["capabilities"], "models": [model.snapshot()],
            "stage_chunked_input": True, "layer_ranges": [[4, 16]]}
    # The helper negotiated the same binary bit. Model/range inventory is a
    # prevalidated fixture, independent of real registry state and model files.
    coordinator._workers["worker_01"]["capabilities"] = caps
    master = SimpleNamespace(
        _task_worker_control=coordinator, _nodes_lock=threading.RLock(),
        nodes={"worker_01": SimpleNamespace(node_type="pc", role="client")},
        _effective_role=lambda: "master", _remote_task_worker_providers={},
        _tcp_server=TCPServer(host="127.0.0.1", port=0),
    )
    outbound = _MemorySocket()
    master._tcp_server.clients["worker_01"] = ClientConn(
        client_id="worker_01", sock=outbound, addr=("127.0.0.1", 1234),
        registration_confirmed=True, registration_epoch=1,
    )
    snapshot = lambda: {**coordinator.worker_snapshot("worker_01"),
                        "layer_stage_dispatch_enabled": True}
    sent_types = []

    def send(message):
        sent_types.append(message.message_type)
        SchedulerTaskWorkerMixin._send_task_worker_to_node(master, "worker_01", message)
        frame = TCPClient(client_id="reader").recv_data(_MemorySocket(outbound.packets[-1]))
        worker._handle_task_worker_message("master", frame)
        if message.message_type != "stage_offer":
            return
        assert len(worker.offers) == 1
        offer = worker.offers[0].payload
        assembled = worker._assemble_stage_root_input(offer)
        assert base64.b64decode(assembled["hidden_f32"]) == raw
        identity = {key: offer[key] for key in _identity()}
        accept = build_worker_message("stage_accept", {
            **identity, "accepted": True, "reason_code": "", "retryable": False,
        }, message_id="msg_binaryaccept01", sent_at_ms=1)
        worker._send_task_worker_to_master(accept)
        result_payload = {**_result_payload(raw), **identity}
        for response in worker._task_worker_stage_result_messages(
            result_payload, offer=offer, attempt_id=offer["attempt_id"],
            provider_id=offer["provider_id"],
        ):
            worker._send_task_worker_to_master(response)
        for packet in worker._tcp_client.sock.packets:
            response_frame = TCPClient(client_id="reader").recv_data(_MemorySocket(packet))
            SchedulerTaskWorkerMixin._handle_task_worker_message(master, "worker_01", response_frame)

    provider = RemoteFullWorkerProvider(node_id="worker_01", peer_snapshot=snapshot,
                                        send_message=send)
    master._remote_task_worker_providers["worker_01"] = provider
    request = StageRequest(
        workflow_id="wf_binaryregression", request_id="request-binary-regression",
        stage_id="segment_1", stage_type="layer_forward", provider_id=provider.provider_id,
        dependencies={}, root_input={"hidden_f32": base64.b64encode(raw).decode("ascii"),
                                     "want_hidden": True}, model_identity=model,
        stage_fields={"layer_range": [4, 16], "handoff_at": 4,
                      "hidden_sha256": hashlib.sha256(raw).hexdigest(),
                      "hidden_spec": {"n_tokens": raw_bytes // (2048 * 4),
                                      "n_embd": 2048, "dtype": "float32"}},
    )
    reservation = provider.reserve(request)
    attempt = StageAttempt(attempt_id="att_binaryregression", request=request,
                           provider_id=provider.provider_id, lease_id="lease_binaryregression",
                           lease_epoch=1, lease_expires_at=time.time() + 30)
    try:
        result = provider.execute(attempt, reservation, threading.Event())
        assert base64.b64decode(result.output["hidden_out_f32"]) == raw
        assert sent_types[-1] == "stage_offer"
        chunked = raw_bytes > 8 * 1024 * 1024 or not negotiated
        assert ("stage_chunk" in sent_types) is chunked
        result_frames = [TCPClient(client_id="reader").recv_data(_MemorySocket(packet))
                         for packet in worker._tcp_client.sock.packets]
        result_types = [frame["data"]["message_type"] for frame in result_frames]
        assert ("stage_result_chunk" in result_types) is chunked
        assert all(frame["format"] == ("binary" if negotiated else "json")
                   for frame in result_frames if frame["data"]["message_type"] == "stage_result_chunk")
    finally:
        provider.release(reservation.reservation_id)
        provider.close()


def test_binary_frame_at_the_exact_outer_limit_is_accepted():
    meta = {"type": "task_worker", "format": "binary",
            "binary_bytes": MAX_BINARY_FRAME_BYTES, "data": {}}
    body_size = MAX_BINARY_FRAME_BYTES - 4 - len(json.dumps(meta).encode())
    packet = build_message(MessageType.TASK_WORKER, {}, binary_body=b"x" * body_size)
    assert len(packet) == 4 + MAX_BINARY_FRAME_BYTES
    frame = TCPClient(client_id="reader").recv_data(_MemorySocket(packet))
    assert len(frame["binary"]) == body_size
    with pytest.raises(ConnectionError, match="maximum size"):
        build_message(MessageType.TASK_WORKER, {}, binary_body=b"x" * (body_size + 1))


def test_binary_return_cannot_write_to_a_replacement_socket(monkeypatch):
    import scheduler_task_worker as scheduler_module

    worker = _Worker()
    _negotiate(worker)
    old_sock = worker._tcp_client.sock
    new_sock = _MemorySocket()
    split = scheduler_module.split_task_worker_hidden

    def reconnect(*args, **kwargs):
        result = split(*args, **kwargs)
        worker._tcp_client.sock = new_sock
        worker._tcp_client._registration_epoch += 1
        return result

    monkeypatch.setattr(scheduler_module, "split_task_worker_hidden", reconnect)
    message = build_worker_message("stage_result", _result_payload(b"abcd"),
                                   message_id="msg_binaryresult03", sent_at_ms=1)
    with pytest.raises(ConnectionError, match="新代际替换"):
        worker._send_task_worker_to_master(message)
    assert old_sock.packets == new_sock.packets == []


def test_binary_return_classifies_a_socket_write_failure(monkeypatch):
    worker = _Worker()
    _negotiate(worker)

    def fail(_packet):
        raise OSError(10038, "socket retired")

    monkeypatch.setattr(worker._tcp_client.sock, "sendall", fail)
    message = build_worker_message("stage_result", _result_payload(b"abcd"),
                                   message_id="msg_binaryresult04", sent_at_ms=1)
    with pytest.raises(ConnectionError) as failed:
        worker._send_task_worker_to_master(message)
    assert isinstance(failed.value.__cause__, OSError)


@pytest.mark.parametrize("ack_value,expected", [(None, False), (False, False), (True, True)])
def test_worker_ack_info_reports_the_negotiated_binary_bit(caplog, ack_value, expected):
    import logging

    worker = _Worker()
    worker._task_worker_refresh_lock = threading.Lock()
    worker._task_worker_refresh_requested = False
    hello = worker._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_hello_payload(True)["capabilities"],
        registration_owner=worker._tcp_client.capture_registration_owner(),
    )
    ack = TaskWorkerControlPlane().receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master").snapshot()
    if ack_value is None:
        ack["payload"].pop("binary_hidden")
    else:
        ack["payload"]["binary_hidden"] = ack_value
    with caplog.at_level(logging.INFO, logger="scheduler"):
        worker._handle_task_worker_message("master", {"data": ack})
    records = [record for record in caplog.records
               if "event=task_worker_hello_ack_received" in record.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert f"binary_hidden_negotiated={expected}" in records[0].getMessage()
    assert worker._task_worker_binary_hidden_enabled() is expected



def test_large_binary_result_replay_keeps_the_negotiated_wire_budget():
    worker = _Worker()
    _negotiate(worker)
    raw = b"x" * (7 * 1024 * 1024)
    response = build_worker_message(
        "stage_result", _result_payload(raw), message_id="msg_binaryreplay01",
        sent_at_ms=1, binary_hidden=True,
    )
    SchedulerTaskWorkerMixin._replay_task_worker_responses(worker, [response.snapshot()])
    frame = TCPClient(client_id="reader").recv_data(
        _MemorySocket(worker._tcp_client.sock.packets[-1]))
    assert frame["format"] == "binary"
    assert frame["binary"] == raw



def test_master_switch_disables_ack_provider_and_pipeline_budget_together(monkeypatch):
    from scheduler_pipeline import _node_capabilities, _assert_layer_stage_offer_fits_frame, LayerStageFrameTooLarge

    monkeypatch.setenv("QLH_TASK_WORKER_BINARY_HIDDEN", "0")
    worker = _Worker()
    coordinator, ack = _negotiate(worker, True, False)
    assert ack.payload["binary_hidden"] is False
    snapshot = coordinator.worker_snapshot("worker_01")
    # Keep the worker's actual declaration for diagnosis; decisions use ACK state.
    assert snapshot["capabilities"]["binary_hidden"] is True
    assert snapshot["binary_hidden"] is False
    control = SimpleNamespace(status=lambda role: {"workers": [snapshot]})
    caps = _node_capabilities(control, "worker_01")
    assert caps["binary_hidden"] is False
    with pytest.raises(LayerStageFrameTooLarge):
        _assert_layer_stage_offer_fits_frame(node_id="worker_01", n_tokens=1,
                                            n_embd=2_000_000, binary_hidden=caps["binary_hidden"])
    provider = RemoteFullWorkerProvider(node_id="worker_01", peer_snapshot=lambda: snapshot,
                                        send_message=lambda _: None)
    try:
        assert not provider._supports_binary_hidden()
    finally:
        provider.close()


def test_pipeline_admission_requires_ack_even_with_true_worker_capability():
    from scheduler_pipeline import _node_declares_binary_hidden

    for negotiated, expected in [(None, False), (False, False), (True, True)]:
        worker = {"node_id": "worker_01", "capabilities": {"binary_hidden": True}}
        if negotiated is not None:
            worker["binary_hidden"] = negotiated
        control = SimpleNamespace(status=lambda role: {"workers": [worker]})
        assert _node_declares_binary_hidden(control, "worker_01") is expected



@pytest.mark.parametrize("encoded", ["AB==", "AAB="])
def test_noncanonical_base64_is_not_extracted_and_rewritten(encoded):
    # A different spelling of the same bytes still changes the envelope digest.
    # Keep the old JSON path rather than canonicalizing during binary restore.
    message = {"message_type": "stage_offer", "payload": {"root_input": {"hidden_f32": encoded}}}
    snapshot, body = split_inline_hidden(message, enabled=True)
    assert snapshot == message
    assert body is None


def test_mid_binary_body_timeout_is_a_connection_error():
    import socket

    snapshot, raw = split_inline_hidden(_make_offer(), enabled=True)
    packet = _wire(snapshot, raw)
    meta_size = unpack_header(packet[:4])

    class TimeoutSocket(_MemorySocket):
        def recv(self, size):
            # Consume metadata and body header, then fail inside the body.
            if self.stream.tell() >= 4 + meta_size + 4:
                raise socket.timeout("body timed out")
            return super().recv(size)

    with pytest.raises(ConnectionError, match="帧不完整"):
        TCPClient(client_id="reader").recv_data(TimeoutSocket(packet))



def test_full_frame_overflow_chunks_result_before_sending_any_response():
    from task_worker_chunks import StageChunkAssembler

    worker = _Worker()
    _negotiate(worker)
    raw = b"x" * 8_389_080
    payload = _result_payload(raw)
    # Match the small envelope that exposed the gap: the raw and the canonical
    # inner metadata fit, but the actual spaced/escaped outer TCP metadata does not.
    payload["output"].pop("hidden_out_sha256")
    payload["output_sha256"] = canonical_sha256(payload["output"])
    messages = worker._task_worker_stage_result_messages(
        payload, offer=_identity(), attempt_id=payload["attempt_id"],
        provider_id=payload["provider_id"],
    )
    assert [message.message_type for message in messages] == ["stage_result_chunk"] * 9 + ["stage_result"]
    assert worker._tcp_client.sock.packets == []
    assembler = StageChunkAssembler()
    for message in messages:
        worker._send_task_worker_to_master(message)
        packet = worker._tcp_client.sock.packets[-1]
        assert len(packet) - HEADER_LEN <= MAX_BINARY_FRAME_BYTES
        frame = TCPClient(client_id="reader").recv_data(_MemorySocket(packet))
        if message.message_type == "stage_result_chunk":
            restored = decode_message(
                restore_inline_hidden(frame["data"], frame["binary"]), binary_hidden=True,
            ).payload
            chunk = base64.b64decode(restored["payload_b64"])
            assembler.add(attempt_id=restored["attempt_id"], chunk_index=restored["chunk_index"],
                          chunk_count=restored["chunk_count"], payload=chunk,
                          payload_sha256=restored["payload_sha256"])
        else:
            final = decode_message(frame["data"]).payload
            assert "hidden_out_f32" not in final["output"]
            assert final["output"]["hidden_out_ref"]["total_bytes"] == len(raw)
            assert final["output_sha256"] == canonical_sha256(final["output"])
    assert len(worker._tcp_client.sock.packets) == 10
    assert assembler.assemble(payload["attempt_id"]) == raw


@pytest.mark.parametrize("model", ["plain", '层段\n"路径\\' * 20])
def test_binary_result_validation_uses_exact_complete_frame_boundary(model):
    payload = _result_payload(b"x")
    payload["metadata"] = {"model": model}
    small = build_worker_message("stage_result", payload, message_id="msg_fullframeedge01",
                                 sent_at_ms=1, binary_hidden=True)
    stripped, _ = split_inline_hidden(small.snapshot(), enabled=True)
    meta = {"type": "task_worker", "format": "binary",
            "binary_bytes": MAX_BINARY_FRAME_BYTES, "data": stripped}
    raw_size = MAX_BINARY_FRAME_BYTES - HEADER_LEN - len(json.dumps(meta).encode("utf-8"))
    for delta in (-1, 0, 1):
        raw = b"x" * (raw_size + delta)
        payload = _result_payload(raw)
        payload["metadata"] = {"model": model}
        if delta > 0:
            with pytest.raises(WorkerProtocolError) as oversized:
                build_worker_message("stage_result", payload, message_id="msg_fullframeedge01",
                                     sent_at_ms=1, binary_hidden=True)
            assert oversized.value.code == "message_too_large"
            continue
        message = build_worker_message("stage_result", payload, message_id="msg_fullframeedge01",
                                       sent_at_ms=1, binary_hidden=True)
        stripped, body = split_inline_hidden(message.snapshot(), enabled=True)
        packet = _wire(stripped, body)
        assert len(packet) - HEADER_LEN == MAX_BINARY_FRAME_BYTES + delta
        restored = TCPClient(client_id="reader").recv_data(_MemorySocket(packet))
        assert decode_message(restore_inline_hidden(restored["data"], restored["binary"]),
                              binary_hidden=True).snapshot() == message.snapshot()



