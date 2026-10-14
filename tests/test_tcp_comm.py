"""
单元测试 — TCP 通信协议
=======================
测试消息打包/解包、MessageType 枚举、消息构建。
纯逻辑测试，无需网络连接。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import logging
import socket
import struct
import threading
import time
from unittest.mock import MagicMock
from types import SimpleNamespace
import pytest
import torch

import tcp_comm as tcp_comm_mod
from tcp_comm import (
    MessageType, pack_data, unpack_header,
    build_message, parse_message, recv_exact,
    HEADER_LEN, MAX_PACKET_SIZE,
    serialize_tensor, deserialize_tensor,
    serialize_tensor_fast, deserialize_tensor_fast,
    TCPServer, TCPClient, ClientConn,
    parse_host_port, format_host_port,
)


def test_detect_lan_ip_rejects_tailscale_apipa_and_wsl_virtual_adapter(monkeypatch):
    """未登录 Tailscale 的 169.254/16 不能作为主节点广告地址。"""
    monkeypatch.setattr(tcp_comm_mod, "_HAS_PSUTIL", True)
    monkeypatch.setattr(
        tcp_comm_mod.psutil,
        "net_if_addrs",
        lambda: {
            "Tailscale": [SimpleNamespace(
                family=socket.AF_INET, address="169.254.83.107",
            )],
            "vEthernet (WSL)": [SimpleNamespace(
                family=socket.AF_INET, address="172.28.176.1",
            )],
            "WLAN": [SimpleNamespace(
                family=socket.AF_INET, address="10.61.50.55",
            )],
        },
    )
    monkeypatch.setattr(
        tcp_comm_mod.psutil,
        "net_if_stats",
        lambda: {
            "Tailscale": SimpleNamespace(isup=True),
            "vEthernet (WSL)": SimpleNamespace(isup=True),
            "WLAN": SimpleNamespace(isup=True),
        },
    )

    assert tcp_comm_mod.detect_lan_ip() == "10.61.50.55"


def test_detect_lan_ip_prefers_assigned_tailscale_address(monkeypatch):
    monkeypatch.setattr(tcp_comm_mod, "_HAS_PSUTIL", True)
    monkeypatch.setattr(
        tcp_comm_mod.psutil,
        "net_if_addrs",
        lambda: {
            "Tailscale": [SimpleNamespace(
                family=socket.AF_INET, address="100.88.9.10",
            )],
            "WLAN": [SimpleNamespace(
                family=socket.AF_INET, address="10.61.50.55",
            )],
        },
    )
    monkeypatch.setattr(
        tcp_comm_mod.psutil,
        "net_if_stats",
        lambda: {
            "Tailscale": SimpleNamespace(isup=True),
            "WLAN": SimpleNamespace(isup=True),
        },
    )

    assert tcp_comm_mod.detect_lan_ip() == "100.88.9.10"


class TestRegistrationAuthentication:
    def test_verify_auth_signature_rejects_missing_fields(self, monkeypatch):
        monkeypatch.setattr(tcp_comm_mod, "_get_cluster_secret", lambda: "s" * 32)
        monkeypatch.setattr(tcp_comm_mod.time, "time", lambda: 1000.0)

        for auth_data in (None, {}, {"auth_timestamp": 1000.0}, {"auth_signature": "x"}):
            ok, _ = tcp_comm_mod.verify_auth_signature("client-auth", auth_data)
            assert ok is False

    def test_verify_auth_signature_rejects_missing_secret_stale_and_bad_hmac(self, monkeypatch):
        monkeypatch.setattr(tcp_comm_mod.time, "time", lambda: 1000.0)
        monkeypatch.setattr(tcp_comm_mod, "_get_cluster_secret", lambda: "")
        ok, _ = tcp_comm_mod.verify_auth_signature(
            "client-auth", {"auth_timestamp": 1000.0, "auth_signature": "x"},
        )
        assert ok is False

        monkeypatch.setattr(tcp_comm_mod, "_get_cluster_secret", lambda: "s" * 32)
        stale = tcp_comm_mod.build_auth_signature("client-auth", timestamp=100.0)
        stale_ok, _ = tcp_comm_mod.verify_auth_signature("client-auth", stale)
        assert stale_ok is False

        invalid = tcp_comm_mod.build_auth_signature("client-auth", timestamp=1000.0)
        invalid["auth_signature"] = "0" * 64
        invalid_ok, _ = tcp_comm_mod.verify_auth_signature("client-auth", invalid)
        assert invalid_ok is False

    def test_verify_auth_signature_accepts_matching_hmac_inside_time_window(self, monkeypatch):
        monkeypatch.setattr(tcp_comm_mod, "_get_cluster_secret", lambda: "s" * 32)
        monkeypatch.setattr(tcp_comm_mod.time, "time", lambda: 1000.0)

        auth = tcp_comm_mod.build_auth_signature("client-auth", timestamp=999.0)
        assert tcp_comm_mod.verify_auth_signature("client-auth", auth) == (True, "ok")

    def test_registration_nonce_is_consumed_once_across_connections(self, monkeypatch):
        monkeypatch.setattr(tcp_comm_mod, "_get_cluster_secret", lambda: "s" * 32)
        monkeypatch.setattr(tcp_comm_mod, "_get_cluster_secret_epoch", lambda: 7)
        monkeypatch.setattr(tcp_comm_mod.time, "time", lambda: 1000.0)
        guard = tcp_comm_mod.RegistrationReplayGuard()
        auth = tcp_comm_mod.build_auth_signature(
            "client-auth", timestamp=999.0, secret_epoch=7,
        )

        assert tcp_comm_mod.verify_auth_signature(
            "client-auth", auth, replay_guard=guard,
        ) == (True, "ok")
        replay_ok, replay_reason = tcp_comm_mod.verify_auth_signature(
            "client-auth", auth, replay_guard=guard,
        )
        assert replay_ok is False
        assert "replay" in replay_reason

    @pytest.mark.parametrize("timestamp", [True, "1000", float("nan"), float("inf")])
    def test_registration_rejects_non_finite_or_non_numeric_timestamp(
        self, monkeypatch, timestamp,
    ):
        monkeypatch.setattr(tcp_comm_mod, "_get_cluster_secret", lambda: "s" * 32)
        auth = tcp_comm_mod.build_auth_signature("client-auth", timestamp=1000.0)
        auth["auth_timestamp"] = timestamp
        ok, reason = tcp_comm_mod.verify_auth_signature("client-auth", auth)
        assert ok is False
        assert "时间戳" in reason

    def test_registration_rejects_old_secret_epoch(self, monkeypatch):
        monkeypatch.setattr(tcp_comm_mod, "_get_cluster_secret", lambda: "s" * 32)
        monkeypatch.setattr(tcp_comm_mod, "_get_cluster_secret_epoch", lambda: 3)
        auth = tcp_comm_mod.build_auth_signature(
            "client-auth", timestamp=1000.0, secret_epoch=2,
        )
        monkeypatch.setattr(tcp_comm_mod.time, "time", lambda: 1000.0)
        ok, reason = tcp_comm_mod.verify_auth_signature("client-auth", auth)
        assert ok is False
        assert "epoch" in reason


# ================================================================
# pack_data / unpack_header 测试
# ================================================================

class TestPackUnpack:
    """测试封包/解包"""

    def test_pack_data_format(self):
        """封包格式: 4字节长度头 + 数据体"""
        payload = b"hello world"
        packed = pack_data(payload)

        # 长度头 = 4 字节
        header = packed[:HEADER_LEN]
        body = packed[HEADER_LEN:]

        assert len(header) == HEADER_LEN
        assert body == payload
        assert unpack_header(header) == len(payload)

    def test_pack_empty_payload(self):
        """空 payload"""
        packed = pack_data(b"")
        assert len(packed) == HEADER_LEN
        assert unpack_header(packed[:HEADER_LEN]) == 0

    def test_pack_large_payload(self):
        """较大 payload（模拟中间特征张量）"""
        payload = b"x" * 100000  # 100KB
        packed = pack_data(payload)
        assert len(packed) == HEADER_LEN + 100000
        assert unpack_header(packed[:HEADER_LEN]) == 100000

    def test_unpack_header_value(self):
        """解包长度头应返回正确的整数值"""
        for size in [0, 1, 255, 256, 65535, 1048576]:
            header = struct.pack(">I", size)
            assert unpack_header(header) == size

    def test_header_big_endian(self):
        """长度头应为大端序"""
        header = struct.pack(">I", 0x12345678)
        # 大端序: 12 34 56 78
        assert header[0] == 0x12
        assert header[1] == 0x34
        assert header[2] == 0x56
        assert header[3] == 0x78
        assert unpack_header(header) == 0x12345678


# ================================================================
# MessageType 枚举测试
# ================================================================

class TestMessageType:
    """测试消息类型枚举"""

    def test_all_types_defined(self):
        """所有消息类型应定义完整"""
        expected = {
            "register", "heartbeat", "heartbeat_ack",
            "task_start", "task_stop", "task_done", "error",
            "tensor", "result",
            "status_req", "status_res",
            "infer_forward", "infer_result", "infer_cancel",
            "layer_config", "layer_config_ack",
            "layer_worker_opt_out", "layer_worker_opt_in",
            "task_worker",
            "role_transfer", "role_transfer_ack",
            "spare_master_designate", "spare_master_designate_ack",
            "spare_master_activate", "spare_master_activate_ack",
            "spare_master_deactivate",
            "node_list_sync", "node_update",
            "layer_forward", "layer_result", "chain_forward",
            "pipeline_done", "pipeline_abort",
            "pipeline_pause", "pipeline_resume",
            # L5: 多节点日志聚合
            "log_request", "log_response",
        }
        actual = set(MessageType.__members__.keys())
        for t in expected:
            assert t.upper() in actual, f"缺少消息类型: {t}"

    def test_string_enum(self):
        """MessageType 应为字符串枚举"""
        assert MessageType.REGISTER.value == "register"
        assert MessageType.HEARTBEAT.value == "heartbeat"
        assert isinstance(MessageType.REGISTER, str)

    def test_infer_forward_type(self):
        """分布式推理转发类型"""
        assert MessageType.INFER_FORWARD.value == "infer_forward"
        assert MessageType.INFER_RESULT.value == "infer_result"
        assert MessageType.INFER_CANCEL.value == "infer_cancel"

    def test_layer_config_type(self):
        """分层配置推送类型"""
        assert MessageType.LAYER_CONFIG.value == "layer_config"
        assert MessageType.LAYER_CONFIG_ACK.value == "layer_config_ack"

    def test_local_model_hash_prefers_pytorch_weights(self, tmp_path, monkeypatch):
        """流水线注册摘要必须优先取 PyTorch 权重，而不是同目录的 GGUF。"""
        import config as cfg
        import hashlib
        from tcp_comm import TCPClient

        model_dir = tmp_path / "qwen"
        model_dir.mkdir()
        weight = model_dir / "model.safetensors"
        weight.write_bytes(b"pytorch-weights")
        gguf = tmp_path / "qwen.gguf"
        gguf.write_bytes(b"gguf-weights")

        monkeypatch.setattr(cfg, "MODEL_PATH", str(model_dir))
        monkeypatch.setattr(cfg, "GGUF_MODEL_PATH", str(gguf))

        actual = TCPClient._compute_local_model_sha256()
        file_sha = hashlib.sha256(b"pytorch-weights").hexdigest()
        expected = hashlib.sha256(
            f"model.safetensors\0{weight.stat().st_size}\0{file_sha}\n".encode()
        ).hexdigest()
        assert actual == expected

    def test_chain_forward_type(self):
        """master 向下一逻辑段转发的消息类型保持 wire 兼容。"""
        assert MessageType.CHAIN_FORWARD.value == "chain_forward"

    def test_chain_forward_ack_type(self):
        """链式转发 ACK 类型"""
        assert MessageType.CHAIN_FORWARD_ACK.value == "chain_forward_ack"


class TestWindowsProbeSubprocesses:
    def test_netsh_probe_is_hidden(self, monkeypatch):
        import platform

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return type("Result", (), {"stdout": "Ethernet Connected", "returncode": 0})()

        monkeypatch.setattr(tcp_comm_mod, "_HAS_PSUTIL", False)
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(tcp_comm_mod.subprocess, "run", fake_run)

        assert tcp_comm_mod.detect_network_type() == "ethernet"
        assert calls[0][1]["creationflags"] == tcp_comm_mod._WINDOWS_NO_WINDOW
        assert calls[0][1]["encoding"] == "utf-8"
        assert calls[0][1]["errors"] == "replace"

    def test_getmac_probe_is_hidden(self, monkeypatch):
        import platform
        import uuid

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return type(
                "Result",
                (),
                {
                    "stdout": "Header\nEthernet,AA-BB-CC-DD-EE-FF,Connected\n",
                    "returncode": 0,
                },
            )()

        monkeypatch.setattr(tcp_comm_mod, "_HAS_PSUTIL", False)
        monkeypatch.setattr(uuid, "getnode", lambda: 0)
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(tcp_comm_mod.subprocess, "run", fake_run)

        assert tcp_comm_mod.get_mac_addresses() == ["aa-bb-cc-dd-ee-ff"]
        assert calls[0][1]["creationflags"] == tcp_comm_mod._WINDOWS_NO_WINDOW
        assert calls[0][1]["encoding"] == "utf-8"
        assert calls[0][1]["errors"] == "replace"


class TestLayerConfigSocketRoundTrip:
    """Use loopback TCP to verify the layer-config and ACK wire contract."""

    def test_direct_assignment_round_trips_with_ack(self, monkeypatch):
        import config as cfg

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        monkeypatch.setattr(cfg, "CLUSTER_SECRET", "test-layer-config-secret")
        monkeypatch.setattr(
            TCPClient, "_compute_local_model_sha256", staticmethod(lambda: "sha-qwen")
        )
        monkeypatch.setattr(
            TCPClient,
            "_heartbeat_loop",
            lambda self, connection_generation=None, connection_sock=None: None,
        )
        monkeypatch.setattr(tcp_comm_mod, "detect_network_type", lambda: "ethernet")
        monkeypatch.setattr(tcp_comm_mod, "detect_lan_ip", lambda: "127.0.0.1")

        ack_event = threading.Event()
        received_configs = []
        received_acks = []
        server = TCPServer(host="127.0.0.1", port=port)

        def on_server_message(client_id, msg):
            if msg.get("type") == MessageType.REGISTER.value:
                server.confirm_registration(client_id)
            elif msg.get("type") == MessageType.LAYER_CONFIG_ACK.value:
                received_acks.append((client_id, msg["data"]))
                ack_event.set()

        server.start(on_message=on_server_message)
        client = TCPClient(
            server_host="127.0.0.1",
            server_port=port,
            client_id="worker1",
            role="client",
            advertise_host="127.0.0.1",
            advertise_port=port,
        )

        def on_client_message(msg):
            if msg.get("type") != MessageType.LAYER_CONFIG.value:
                return
            assignment = msg["data"]
            received_configs.append(assignment)
            client.send_data(
                {
                    "node_id": "worker1",
                    "config_id": assignment["config_id"],
                    "status": "ready",
                    "start_layer": assignment["start_layer"],
                    "end_layer": assignment["end_layer"],
                    "model_sha256": assignment["model_sha256"],
                    "engine": "pytorch",
                },
                MessageType.LAYER_CONFIG_ACK,
            )

        try:
            assert client.connect(on_message=on_client_message) is True
            assignment = {
                "node_id": "worker1",
                "config_id": "cfg-loopback",
                "start_layer": 4,
                "end_layer": 8,
                "has_embedding": False,
                "has_lm_head": True,
                "model_id": "deepseek-r1-distill-qwen-1.5b",
                "model_sha256": "sha-qwen",
                "total_layers": 28,
                "master_api_port": 8000,
            }
            server.send_layer_config("worker1", assignment)

            assert ack_event.wait(3), "layer_config_ack timed out"
            assert received_configs == [assignment]
            assert received_acks == [
                (
                    "worker1",
                    {
                        "node_id": "worker1",
                        "config_id": "cfg-loopback",
                        "status": "ready",
                        "start_layer": 4,
                        "end_layer": 8,
                        "model_sha256": "sha-qwen",
                        "engine": "pytorch",
                    },
                )
            ]
        finally:
            client.disconnect()
            server.stop()


class TestNetworkPathLoopbackBoundary:
    """One local TCP flow proving path metrics do not alter wire semantics."""

    def test_registration_heartbeat_cancel_and_result_keep_wire_contract(
            self, monkeypatch):
        import config as cfg

        monkeypatch.setattr(cfg, "CLUSTER_SECRET", "nw1-4-loopback-secret")
        monkeypatch.setattr(
            TCPClient,
            "_compute_local_model_sha256",
            staticmethod(lambda: "sha-nw1-4"),
        )
        monkeypatch.setattr(
            TCPClient,
            "_heartbeat_loop",
            lambda self, connection_generation=None, connection_sock=None: None,
        )
        monkeypatch.setattr(tcp_comm_mod, "detect_network_type", lambda: "ethernet")
        monkeypatch.setattr(tcp_comm_mod, "detect_lan_ip", lambda: "127.0.0.1")

        received_by_server = []
        received_by_client = []
        cancel_received = threading.Event()
        result_received = threading.Event()
        heartbeat_ack = threading.Event()
        server = TCPServer(host="127.0.0.1", port=1)
        server.port = 0

        def on_server_message(client_id, msg):
            if msg.get("type") == MessageType.REGISTER.value:
                assert server.confirm_registration(client_id) is True
                return
            received_by_server.append((client_id, msg["type"], msg["data"]))
            if msg.get("type") == MessageType.INFER_CANCEL.value:
                cancel_received.set()

        server.start(on_message=on_server_message)
        port = server.sock.getsockname()[1]
        client = TCPClient(
            server_host="127.0.0.1",
            server_port=port,
            client_id="worker-nw1-4",
            role="client",
            advertise_host="127.0.0.1",
            advertise_port=port,
        )
        original_handle_ack = client._handle_heartbeat_ack

        def on_heartbeat_ack(msg, connection_generation=None):
            original_handle_ack(msg, connection_generation)
            heartbeat_ack.set()

        client._handle_heartbeat_ack = on_heartbeat_ack

        def on_client_message(msg):
            received_by_client.append((msg["type"], msg["data"]))
            if msg.get("type") == MessageType.INFER_RESULT.value:
                result_received.set()

        try:
            assert client.connect(on_message=on_client_message) is True
            assert client.is_registered is True
            assert server.get_client_ids() == ["worker-nw1-4"]

            sent_at = time.time()
            client._last_heartbeat_send = sent_at
            client._heartbeat_quality.record_send(
                client._connection_generation,
                sent_at,
            )
            client.send_data({"t_send": sent_at}, MessageType.HEARTBEAT)
            assert heartbeat_ack.wait(3), "loopback heartbeat ACK timed out"
            assert client.get_network_quality_snapshot()["sample_count"] == 1

            cancel = {"request_id": "req-nw1-4", "reason": "user_cancelled"}
            client.send_data(cancel, MessageType.INFER_CANCEL)
            assert cancel_received.wait(3), "loopback cancel timed out"

            result = {"request_id": "req-nw1-4", "result": "cancelled"}
            server.send_to_client("worker-nw1-4", result, MessageType.INFER_RESULT)
            assert result_received.wait(3), "loopback result timed out"

            assert ("worker-nw1-4", "infer_cancel", cancel) in received_by_server
            assert ("infer_result", result) in received_by_client
            assert client.is_registered is True
        finally:
            client.disconnect()
            server.stop()


# ================================================================
# build_message 测试
# ================================================================

class TestBuildMessage:
    """测试消息构建"""

    def test_build_json_message(self):
        """构建 JSON 控制消息"""
        data = {"action": "register", "node_id": "client1"}
        packet = build_message(MessageType.REGISTER, data)

        # 解包验证
        header = packet[:HEADER_LEN]
        meta_len = unpack_header(header)
        meta_bytes = packet[HEADER_LEN:HEADER_LEN + meta_len]
        meta = json.loads(meta_bytes.decode("utf-8"))

        assert meta["type"] == "register"
        # build_message 对 dict 数据会设置 format: "json"
        assert meta.get("format") == "json"

    def test_build_tensor_message(self):
        """构建张量消息（双层封包）"""
        tensor = torch.randn(4, 128)
        packet = build_message(MessageType.TENSOR, tensor)

        # 第一层: meta
        header = packet[:HEADER_LEN]
        meta_len = unpack_header(header)
        meta_bytes = packet[HEADER_LEN:HEADER_LEN + meta_len]
        meta = json.loads(meta_bytes.decode("utf-8"))

        assert meta["type"] == "tensor"
        assert meta["format"] == "tensor"

        # 第二层: tensor data
        offset = HEADER_LEN + meta_len
        tensor_header = packet[offset:offset + HEADER_LEN]
        tensor_len = unpack_header(tensor_header)
        tensor_bytes = packet[offset + HEADER_LEN:offset + HEADER_LEN + tensor_len]

        # 反序列化
        restored = deserialize_tensor(tensor_bytes)
        assert torch.equal(restored, tensor)

    def test_build_message_no_data(self):
        """无数据的消息（如心跳）"""
        packet = build_message(MessageType.HEARTBEAT)
        header = packet[:HEADER_LEN]
        meta_len = unpack_header(header)
        meta_bytes = packet[HEADER_LEN:HEADER_LEN + meta_len]
        meta = json.loads(meta_bytes.decode("utf-8"))

        assert meta["type"] == "heartbeat"

    def test_build_message_str_data(self):
        """字符串数据"""
        packet = build_message(MessageType.RESULT, "推理完成")
        header = packet[:HEADER_LEN]
        meta_len = unpack_header(header)
        # 字符串被当作普通 JSON 数据处理（嵌入在 meta 中或作为单独字段）

        # build_message 对 str 的处理：与 dict 相同路径
        assert meta_len > 0


# ================================================================
# serialize_tensor / deserialize_tensor 测试
# ================================================================

class TestTensorSerialization:
    """测试张量序列化"""

    def test_roundtrip_float32(self):
        """FP32 张量序列化往返"""
        tensor = torch.randn(3, 64, 128)
        restored = deserialize_tensor(serialize_tensor(tensor))
        assert torch.equal(restored, tensor)

    def test_roundtrip_float16(self):
        """FP16 张量序列化往返"""
        tensor = torch.randn(2, 32, 64, dtype=torch.float16)
        restored = deserialize_tensor(serialize_tensor(tensor))
        assert torch.equal(restored, tensor)

    def test_roundtrip_int64(self):
        """INT64 张量（input_ids）序列化往返"""
        tensor = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.int64)
        restored = deserialize_tensor(serialize_tensor(tensor))
        assert torch.equal(restored, tensor)

    def test_roundtrip_scalar(self):
        """标量张量"""
        tensor = torch.tensor(42.0)
        restored = deserialize_tensor(serialize_tensor(tensor))
        assert torch.equal(restored, tensor)

    def test_serialize_returns_bytes(self):
        """序列化应返回 bytes"""
        tensor = torch.randn(4, 4)
        result = serialize_tensor(tensor)
        assert isinstance(result, bytes)

    def test_large_tensor(self):
        """较大张量（模拟中间特征）"""
        tensor = torch.randn(1, 2048, 2048)  # ~16MB FP32
        serialized = serialize_tensor(tensor)
        restored = deserialize_tensor(serialized)
        assert torch.equal(restored, tensor)
        # 大小应在 MAX_PACKET_SIZE 范围内
        assert len(serialized) < MAX_PACKET_SIZE


# ================================================================
# MAX_PACKET_SIZE 验证
# ================================================================

class TestPacketSizeLimit:
    """测试包大小限制"""

    def test_max_packet_size_reasonable(self):
        """MAX_PACKET_SIZE 应足够大"""
        # 256 MB 应能容纳最大的中间特征张量
        assert MAX_PACKET_SIZE == 256 * 1024 * 1024

    def test_header_len_is_4(self):
        """长度头固定 4 字节"""
        assert HEADER_LEN == 4

    def test_client_rejects_oversized_payload_before_receiving_body(self):
        sock = _ChunkSocket([
            struct.pack(">I", MAX_PACKET_SIZE + 1),
        ])
        client = TCPClient(client_id="packet_limit_client")

        with pytest.raises(ConnectionError, match="invalid packet length"):
            client.recv_data(sock)

        assert sock.recv_sizes == [HEADER_LEN]

    def test_client_rejects_oversized_tensor_before_receiving_body(
        self, monkeypatch,
    ):
        payload = b"{}"
        sock = _ChunkSocket([
            struct.pack(">I", len(payload)),
            payload,
            struct.pack(">I", MAX_PACKET_SIZE + 1),
        ])
        client = TCPClient(client_id="tensor_limit_client")
        monkeypatch.setattr(
            tcp_comm_mod,
            "parse_message",
            lambda _payload: {"_needs_tensor": True},
        )

        with pytest.raises(
            ConnectionError, match="invalid tensor packet length",
        ):
            client.recv_data(sock)

        assert sock.recv_sizes == [HEADER_LEN, len(payload), HEADER_LEN]


class _ChunkSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.recv_sizes = []

    def recv(self, size):
        self.recv_sizes.append(size)
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if len(chunk) <= size:
            return chunk
        self.chunks.insert(0, chunk[size:])
        return chunk[:size]


# ================================================================
# serialize_tensor_fast / deserialize_tensor_fast 测试
# ================================================================

class TestFastTensorSerialization:
    """测试高速序列化路径（流水线隐藏状态传输）"""

    def test_fast_small_roundtrip_fp32(self):
        """小张量 (<1MB) FP32 往返 — 走 TNR1 路径"""
        tensor = torch.randn(3, 64, 128)  # ~98KB
        data = serialize_tensor_fast(tensor)
        assert data[:4] == b'TNR1', f"小张量应走 TNR1 路径，实际 magic: {data[:4]!r}"
        restored = deserialize_tensor_fast(data)
        assert torch.equal(restored, tensor)

    def test_fast_small_roundtrip_fp16(self):
        """小张量 FP16 往返"""
        tensor = torch.randn(2, 32, 64, dtype=torch.float16)
        data = serialize_tensor_fast(tensor)
        restored = deserialize_tensor_fast(data)
        assert torch.equal(restored, tensor)

    def test_fast_small_roundtrip_int64(self):
        """小张量 INT64（input_ids）往返"""
        tensor = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.int64)
        data = serialize_tensor_fast(tensor)
        restored = deserialize_tensor_fast(data)
        assert torch.equal(restored, tensor)

    def test_fast_large_roundtrip(self):
        """大张量 (≥1MB) 往返 — 走 TNR0 路径（numpy 零拷贝）"""
        # (1, 512, 2048) FP16 ≈ 2MB — 模拟 prefill 阶段隐藏状态
        tensor = torch.randn(1, 512, 2048, dtype=torch.float16)
        data = serialize_tensor_fast(tensor)
        assert data[:4] == b'TNR0', f"大张量应走 TNR0 路径，实际 magic: {data[:4]!r}"
        restored = deserialize_tensor_fast(data)
        assert torch.equal(restored, tensor)

    def test_fast_large_decode_phase(self):
        """解码阶段隐藏状态 (1, 1, 2048) FP16 ≈ 4KB — 走小张量路径"""
        tensor = torch.randn(1, 1, 2048, dtype=torch.float16)
        data = serialize_tensor_fast(tensor)
        assert data[:4] == b'TNR1', "decode 阶段张量小，应走 TNR1 路径"
        restored = deserialize_tensor_fast(data)
        assert torch.equal(restored, tensor)

    def test_fast_equals_slow(self):
        """fast 序列化/反序列化结果应与 slow 一致"""
        tensors = [
            torch.randn(4, 4),
            torch.randn(1, 512, 2048, dtype=torch.float16),
            torch.tensor([[1, 2, 3]], dtype=torch.int64),
            torch.tensor(42.0),
        ]
        for t in tensors:
            fast_restored = deserialize_tensor_fast(serialize_tensor_fast(t))
            slow_restored = deserialize_tensor(serialize_tensor(t))
            assert torch.equal(fast_restored, slow_restored), \
                f"fast ≠ slow for shape {t.shape}, dtype {t.dtype}"

    def test_fast_returns_bytes(self):
        """高速序列化应返回 bytes"""
        tensor = torch.randn(8, 8)
        result = serialize_tensor_fast(tensor)
        assert isinstance(result, bytes)
        assert len(result) > 4  # 至少有 magic header

    def test_fast_large_under_max_packet(self):
        """大张量序列化后应在 MAX_PACKET_SIZE 范围内"""
        tensor = torch.randn(1, 2048, 2048, dtype=torch.float16)  # ~8MB
        data = serialize_tensor_fast(tensor)
        assert len(data) < MAX_PACKET_SIZE, \
            f"序列化大小 {len(data)} 超出 MAX_PACKET_SIZE {MAX_PACKET_SIZE}"

    def test_fast_shape_preserved(self):
        """往返后 shape 和 dtype 应完整保留"""
        shapes = [
            (1, 1, 2048),
            (1, 512, 2048),
            (3, 64, 128),
            (1, 24, 32, 64),  # 4D 张量
        ]
        for shape in shapes:
            tensor = torch.randn(*shape, dtype=torch.float16)
            restored = deserialize_tensor_fast(serialize_tensor_fast(tensor))
            assert restored.shape == tensor.shape, \
                f"shape 不一致: {restored.shape} ≠ {tensor.shape}"
            assert restored.dtype == tensor.dtype, \
                f"dtype 不一致: {restored.dtype} ≠ {tensor.dtype}"


# ================================================================
# TCPServer 连接管理与日志测试
# ================================================================

class TestTCPServerConnectionManagement:
    """测试 TCPServer 注册拒绝、连接表线程安全与回调日志。"""

    def test_callback_does_not_reflectively_rewrite_control_plane_owner(self):
        """数据面事件不得通过 bound-method owner 改写控制面字段。"""
        received = threading.Event()

        class Owner:
            def __init__(self):
                self._tcp_server = "explicit-owner-binding"

            def on_message(self, _client_id, _message):
                received.set()

        owner = Owner()
        server = TCPServer(host="127.0.0.1", port=0)
        server.on_message = owner.on_message
        server._running = True
        srv_sock, cli_sock = socket.socketpair()
        thread = threading.Thread(
            target=server._handle_client,
            args=(srv_sock, ("127.0.0.1", 54320), "pending_54320"),
            daemon=True,
        )
        try:
            thread.start()
            cli_sock.sendall(build_message(MessageType.STATUS_RES, {"ok": True}))
            assert received.wait(2)
            assert owner._tcp_server == "explicit-owner-binding"
        finally:
            server._running = False
            cli_sock.close()
            thread.join(timeout=2)
            server.stop()

    def test_registration_uses_advertised_endpoint_not_peer_port(self, monkeypatch):
        """节点服务地址应使用 advertised_address，peer_addr 保留临时源端口。"""
        monkeypatch.setenv("QLH_ALLOW_UNPROTECTED_CLUSTER_TCP", "1")
        server = TCPServer(host="127.0.0.1", port=0)
        srv_sock, cli_sock = socket.socketpair()
        try:
            msg = {
                "type": "register",
                "data": {
                    "client_id": "client1",
                    "role": "client",
                    "hostname": "worker",
                    "advertised_host": "10.0.0.9",
                    "advertised_port": 8888,
                    "advertised_address": "10.0.0.9:8888",
                    "auth": tcp_comm_mod.build_auth_signature("client1"),
                },
            }
            client_id = server._handle_registration(
                srv_sock, ("10.0.0.9", 54321), "pending_54321", msg
            )
            assert client_id == "client1"
            info = server.get_client_info("client1")
            assert info["advertised_addr"] == "10.0.0.9:8888"
            assert info["addr"] == "10.0.0.9:8888"
            assert info["peer_addr"] == "10.0.0.9:54321"
        finally:
            server.stop()
            srv_sock.close()
            cli_sock.close()

    def test_legacy_registration_falls_back_to_server_port(self, monkeypatch):
        """旧客户端未上报 advertised_port 时应使用 peer_ip:SERVER_PORT，而不是临时端口。"""
        monkeypatch.setenv("QLH_ALLOW_UNPROTECTED_CLUSTER_TCP", "1")
        server = TCPServer(host="127.0.0.1", port=0)
        srv_sock, cli_sock = socket.socketpair()
        try:
            msg = {
                "type": "register",
                "data": {
                    "client_id": "client_legacy",
                    "role": "client",
                    "auth": tcp_comm_mod.build_auth_signature("client_legacy"),
                },
            }
            server._handle_registration(
                srv_sock, ("10.0.0.8", 51234), "pending_51234", msg
            )
            info = server.get_client_info("client_legacy")
            assert info["advertised_addr"] == f"10.0.0.8:{tcp_comm_mod.SERVER_PORT}"
            assert info["peer_addr"] == "10.0.0.8:51234"
        finally:
            server.stop()
            srv_sock.close()
            cli_sock.close()

    def test_heartbeat_keeps_client_reported_rtt(self):
        """服务端保存客户端测得的 RTT，避免用跨主机时钟估算。"""
        server = TCPServer(host="127.0.0.1", port=0)
        srv_sock, cli_sock = socket.socketpair()
        try:
            msg = {
                "type": "register",
                "data": {
                    "client_id": "client_rtt",
                    "role": "client",
                    "auth": tcp_comm_mod.build_auth_signature("client_rtt"),
                },
            }
            server._handle_registration(
                srv_sock, ("127.0.0.1", 54321), "pending_rtt", msg,
            )
            server._handle_heartbeat(
                "client_rtt",
                {"data": {"t_send": 1.0, "rtt_ms": 4.812}},
            )
            info = server.get_client_info("client_rtt")
            assert info["avg_rtt_ms"] == pytest.approx(4.812)
            assert info["last_rtt_ms"] == pytest.approx(4.812)
        finally:
            server.stop()
            srv_sock.close()
            cli_sock.close()

    def test_rejected_registration_returns_ack_and_raises(self):
        """认证失败的 REGISTER 应返回 rejected ACK 并抛内部拒绝异常。"""
        server = TCPServer(host="127.0.0.1", port=0)
        srv_sock, cli_sock = socket.socketpair()
        try:
            msg = {
                "type": "register",
                "data": {
                    "client_id": "client_bad",
                    "role": "client",
                    "auth": {"auth_timestamp": 1.0, "auth_signature": "bad"},
                },
            }
            with pytest.raises(tcp_comm_mod._RegistrationRejected):
                server._handle_registration(srv_sock, ("127.0.0.1", 54321), "pending_54321", msg)

            header = recv_exact(cli_sock, HEADER_LEN)
            assert header is not None
            payload = recv_exact(cli_sock, unpack_header(header))
            ack = parse_message(payload)
            assert ack["type"] == "register"
            assert ack["data"]["status"] == "rejected"
            assert server.get_client_ids() == []
        finally:
            srv_sock.close()
            cli_sock.close()

    @pytest.mark.parametrize(
        "client_id,role",
        [("master", "client"), ("client_reserved_role", "master")],
    )
    def test_registration_rejects_reserved_master_identity(self, client_id, role, monkeypatch):
        monkeypatch.setattr(tcp_comm_mod, "_get_cluster_secret", lambda: "s" * 32)
        server = TCPServer(host="127.0.0.1", port=0)
        srv_sock, cli_sock = socket.socketpair()
        try:
            msg = {
                "type": "register",
                "data": {
                    "client_id": client_id,
                    "role": role,
                    "auth": tcp_comm_mod.build_auth_signature(client_id),
                },
            }
            with pytest.raises(tcp_comm_mod._RegistrationRejected):
                server._handle_registration(
                    srv_sock, ("127.0.0.1", 54321), "pending_54321", msg,
                )

            header = recv_exact(cli_sock, HEADER_LEN)
            payload = recv_exact(cli_sock, unpack_header(header))
            ack = parse_message(payload)
            assert ack["data"]["status"] == "rejected"
            assert server.get_client_ids() == []
        finally:
            srv_sock.close()
            cli_sock.close()

    def test_registration_rejects_legacy_pipeline_peer(self, monkeypatch):
        monkeypatch.setattr(tcp_comm_mod, "_get_cluster_secret", lambda: "s" * 32)
        server = TCPServer(host="127.0.0.1", port=0)
        srv_sock, cli_sock = socket.socketpair()
        try:
            client_id = "client-peer"
            msg = {
                "type": "register",
                "data": {
                    "client_id": client_id,
                    "role": "client",
                    "node_type": "pipeline_peer",
                    "auth": tcp_comm_mod.build_auth_signature(client_id),
                },
            }
            with pytest.raises(tcp_comm_mod._RegistrationRejected):
                server._handle_registration(
                    srv_sock, ("127.0.0.1", 54321), "pending_54321", msg,
                )

            header = recv_exact(cli_sock, HEADER_LEN)
            payload = recv_exact(cli_sock, unpack_header(header))
            ack = parse_message(payload)
            assert ack["data"]["status"] == "rejected"
            assert "pipeline_peer" in ack["data"]["reason"]
            assert server.get_client_ids() == []
        finally:
            srv_sock.close()
            cli_sock.close()

    def test_duplicate_registration_replaces_confirmed_session(self):
        """重连应替换已确认旧会话，保证新 socket 能收到 ACK。"""
        server = TCPServer(host="127.0.0.1", port=0)
        old_srv, old_cli = socket.socketpair()
        new_srv, new_cli = socket.socketpair()
        try:
            client_id = "client_duplicate"
            server._set_client(
                client_id,
                ClientConn(
                    client_id=client_id,
                    sock=old_srv,
                    addr=("127.0.0.1", 1),
                    role="client",
                    registration_confirmed=True,
                    last_heartbeat=time.time(),
                ),
            )
            msg = {
                "type": "register",
                "data": {
                    "client_id": client_id,
                    "role": "client",
                    "auth": tcp_comm_mod.build_auth_signature(client_id),
                },
            }
            server._handle_registration(
                new_srv, ("127.0.0.1", 54325), "pending_54325", msg,
            )
            assert server.get_client_info(client_id)["peer_addr"] == "127.0.0.1:54325"
            assert server.confirm_registration(client_id) is True
            header = recv_exact(new_cli, HEADER_LEN)
            assert header is not None
            payload = recv_exact(new_cli, unpack_header(header))
            ack = parse_message(payload)
            assert ack["type"] == "register"
            assert ack["data"]["status"] == "registered"
            assert ack["data"]["client_id"] == client_id
        finally:
            server.stop()
            old_srv.close()
            old_cli.close()
            new_srv.close()
            new_cli.close()

    def test_broadcast_continues_after_one_client_send_failure(self):
        class RecordingSocket:
            def __init__(self):
                self.packets = []

            def sendall(self, packet):
                self.packets.append(packet)

            def close(self):
                pass

        class BrokenSocket:
            def sendall(self, packet):
                raise OSError("simulated send failure")

            def close(self):
                pass

        server = TCPServer(host="127.0.0.1", port=0)
        good = RecordingSocket()
        server._set_client(
            "good",
            ClientConn("good", good, ("127.0.0.1", 1), role="client"),
        )
        server._set_client(
            "bad",
            ClientConn("bad", BrokenSocket(), ("127.0.0.1", 2), role="client"),
        )

        server.broadcast({"event": "test"}, MessageType.TASK_START)

        assert len(good.packets) == 1
        assert parse_message(good.packets[0][HEADER_LEN:])["data"] == {"event": "test"}

    def test_rejected_registration_does_not_call_on_message_or_leak_client(self):
        """_handle_client 应拦截注册拒绝，不进入上层回调且不泄露 pending_*。"""
        server = TCPServer(host="127.0.0.1", port=0)
        called = []
        server.on_message = lambda client_id, msg: called.append((client_id, msg))
        # _handle_client 的接收循环以 _running 为条件：不置位则线程立即退出并
        # 关闭 socket，下面的 sendall 会抛 BrokenPipeError（与本用例意图无关的
        # 夹具缺陷）。对齐同类用例 test_scheduler_rejected_registration_* 的写法。
        server._running = True

        srv_sock, cli_sock = socket.socketpair()
        try:
            t = threading.Thread(
                target=server._handle_client,
                args=(srv_sock, ("127.0.0.1", 54321), "pending_54321"),
                daemon=True,
            )
            t.start()

            cli_sock.sendall(build_message(MessageType.REGISTER, {
                "client_id": "client_bad",
                "role": "client",
                "auth": {"auth_timestamp": 1.0, "auth_signature": "bad"},
            }))

            t.join(timeout=2)
            assert not t.is_alive()
            assert called == []
            assert server.get_client_ids() == []
            assert not any(cid.startswith("pending_") for cid in server.get_client_ids())
        finally:
            cli_sock.close()

    def test_scheduler_rejected_registration_returns_rejected_and_closes(self):
        """通过协议校验但被上层调度器拒绝时，应返回 rejected 且不泄漏 clients。"""
        server = TCPServer(host="127.0.0.1", port=0)

        def reject_in_scheduler(client_id, msg):
            server.reject_client(client_id, "capacity full")

        server.on_message = reject_in_scheduler
        server._running = True
        srv_sock, cli_sock = socket.socketpair()
        try:
            t = threading.Thread(
                target=server._handle_client,
                args=(srv_sock, ("127.0.0.1", 54322), "pending_54322"),
                daemon=True,
            )
            t.start()

            cli_sock.sendall(build_message(MessageType.REGISTER, {
                "client_id": "client_full",
                "role": "client",
                "auth": tcp_comm_mod.build_auth_signature("client_full"),
            }))

            header = recv_exact(cli_sock, HEADER_LEN)
            assert header is not None
            payload = recv_exact(cli_sock, unpack_header(header))
            ack = parse_message(payload)
            assert ack["type"] == "register"
            assert ack["data"]["status"] == "rejected"
            assert "capacity full" in ack["data"]["reason"]

            t.join(timeout=2)
            assert not t.is_alive()
            assert server.get_client_ids() == []
        finally:
            cli_sock.close()

    def test_disconnect_all_clients_fences_then_closes_and_notifies(self):
        class FakeSock:
            def __init__(self):
                self.shutdown_called = False
                self.closed = False

            def shutdown(self, _how):
                self.shutdown_called = True

            def close(self):
                self.closed = True

        server = TCPServer(host="127.0.0.1", port=0)
        first_sock = FakeSock()
        second_sock = FakeSock()
        first = ClientConn("client-a", first_sock, ("127.0.0.1", 1))
        second = ClientConn("client-b", second_sock, ("127.0.0.1", 2))
        first.registration_confirmed = True
        events = []
        server.clients = {"client-a": first, "client-b": second}
        server._pending_registrations = {"client-b": second}
        server.on_disconnect = lambda client_id: events.append(("disconnect", client_id))

        def rotate():
            events.append(("rotate", None))
            return "new-secret", 2

        rotation, count = server.disconnect_all_clients(
            "credential rotation", before_disconnect=rotate,
        )

        assert rotation == ("new-secret", 2)
        assert count == 2
        assert events[0] == ("rotate", None)
        assert set(events[1:]) == {
            ("disconnect", "client-a"),
            ("disconnect", "client-b"),
        }
        assert server.clients == {}
        assert server._pending_registrations == {}
        assert first_sock.shutdown_called and first_sock.closed
        assert second_sock.shutdown_called and second_sock.closed

    def test_registration_verified_before_rotation_cannot_publish_after_cutover(
        self, monkeypatch,
    ):
        class FakeSock:
            def __init__(self):
                self.packets = []

            def sendall(self, packet):
                self.packets.append(packet)

        server = TCPServer(host="127.0.0.1", port=0)
        epoch = {"value": 1}
        entered = threading.Event()
        release = threading.Event()
        errors = []
        monkeypatch.setattr(
            tcp_comm_mod, "_get_cluster_secret_epoch", lambda: epoch["value"],
        )

        def delayed_verify(*_args, **_kwargs):
            entered.set()
            assert release.wait(timeout=5)
            return True, "ok"

        monkeypatch.setattr(tcp_comm_mod, "verify_auth_signature", delayed_verify)
        sock = FakeSock()

        def register():
            try:
                server._handle_registration(
                    sock,
                    ("127.0.0.1", 10001),
                    "pending_10001",
                    {"data": {
                        "client_id": "client-race",
                        "role": "client",
                        "node_type": "pc",
                        "auth": {"auth_secret_epoch": 1},
                    }},
                )
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=register, daemon=True)
        thread.start()
        assert entered.wait(timeout=5)

        def rotate():
            epoch["value"] = 2
            return "new-secret", 2

        server.disconnect_all_clients(
            "credential rotation", before_disconnect=rotate,
        )
        release.set()
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], tcp_comm_mod._RegistrationRejected)
        assert server.get_client_ids() == []

    def test_scheduler_confirmed_registration_returns_registered(self):
        """REGISTER 只有在上层调度器确认后才返回 registered。"""
        server = TCPServer(host="127.0.0.1", port=0)

        def confirm_in_scheduler(client_id, msg):
            assert server.get_client_info(client_id) is not None
            server.confirm_registration(client_id)

        server.on_message = confirm_in_scheduler
        server._running = True
        srv_sock, cli_sock = socket.socketpair()
        try:
            t = threading.Thread(
                target=server._handle_client,
                args=(srv_sock, ("127.0.0.1", 54323), "pending_54323"),
                daemon=True,
            )
            t.start()

            cli_sock.sendall(build_message(MessageType.REGISTER, {
                "client_id": "client_ok",
                "role": "client",
                "auth": tcp_comm_mod.build_auth_signature("client_ok"),
            }))

            header = recv_exact(cli_sock, HEADER_LEN)
            assert header is not None
            payload = recv_exact(cli_sock, unpack_header(header))
            ack = parse_message(payload)
            assert ack["type"] == "register"
            assert ack["data"]["status"] == "registered"
            assert ack["data"]["client_id"] == "client_ok"
            assert server.get_client_ids() == ["client_ok"]
        finally:
            server.stop()
            cli_sock.close()

    def test_confirm_registration_send_failure_stays_unconfirmed_and_reject_cleans(self):
        """ACK 发送失败时不应把 pending 连接标记为 registered。"""
        class FailingSock:
            def __init__(self):
                self.closed = False
                self.shutdown_called = False

            def sendall(self, payload):
                raise OSError("send failed")

            def shutdown(self, how):
                self.shutdown_called = True

            def close(self):
                self.closed = True

        server = TCPServer(host="127.0.0.1", port=0)
        sock = FailingSock()
        conn = ClientConn(client_id="client_fail", sock=sock, addr=("127.0.0.1", 1))
        server._set_client("client_fail", conn)

        assert server.confirm_registration("client_fail") is False
        assert conn.registration_confirmed is False

        server.reject_client("client_fail", "confirm failed")

        assert server.get_client_ids() == []
        assert sock.shutdown_called is True
        assert sock.closed is True

    def test_pop_client_is_idempotent_and_send_missing_raises_connection_error(self):
        """重复清理连接不应 KeyError，不存在客户端发送应抛 ConnectionError。"""
        server = TCPServer(host="127.0.0.1", port=0)
        srv_sock, cli_sock = socket.socketpair()
        try:
            conn = ClientConn(client_id="client1", sock=srv_sock, addr=("127.0.0.1", 1))
            server._set_client("client1", conn)
            assert server._pop_client("client1") is conn
            assert server._pop_client("client1") is None
            with pytest.raises(ConnectionError):
                server.send_to_client("client1", {"x": 1}, MessageType.STATUS_REQ)
        finally:
            srv_sock.close()
            cli_sock.close()

    def test_on_heartbeat_exception_is_logged(self, monkeypatch, caplog):
        """on_heartbeat 回调异常应进入 DEBUG 日志且携带 exc_info。"""
        client = TCPClient(server_host="127.0.0.1", server_port=1, client_id="client1")
        client.sock = object()
        client._running = True

        def fake_send_data(data, msg_type, connection_sock=None):
            client._running = False

        def bad_callback():
            raise ValueError("boom")

        monkeypatch.setattr(client, "send_data", fake_send_data)
        monkeypatch.setattr(tcp_comm_mod.time, "sleep", lambda _s: None)
        client.on_heartbeat = bad_callback

        with caplog.at_level(logging.DEBUG, logger="tcp_comm"):
            client._heartbeat_loop()

        records = [r for r in caplog.records if "on_heartbeat 回调异常" in r.getMessage()]
        assert records
        assert records[0].exc_info is not None


# ================================================================
# Phase 7 P1: recv_exact 超时测试
# ================================================================

class TestRecvExactTimeout:
    """测试 recv_exact 在 socket 超时和对方关闭时的行为。"""

    def test_recv_exact_returns_none_on_peer_close(self):
        """对方关闭连接时 recv_exact 应返回 None。"""
        from tcp_comm import recv_exact

        srv, cli = socket.socketpair()
        try:
            cli.close()  # 写端关闭
            srv.settimeout(1.0)
            result = recv_exact(srv, 1024)
            assert result is None, "对方关闭时应返回 None"
        finally:
            srv.close()

    def test_recv_exact_timeout_raises_on_blocking(self):
        """超时时 socket 应抛出 socket.timeout（无数据到达）。"""
        from tcp_comm import recv_exact

        srv, cli = socket.socketpair()
        try:
            srv.settimeout(0.1)
            # 对方不发数据 → recv 超时
            with pytest.raises((socket.timeout, OSError)):
                recv_exact(srv, 1024)
        finally:
            srv.close()
            cli.close()

    def test_serialize_deserialize_large_tensor_roundtrip(self):
        """接近 16MB 的张量序列化→反序列化往返应正确。"""
        from tcp_comm import serialize_tensor_fast, deserialize_tensor_fast

        # 创建一个 ~4M 元素的 float32 张量 ≈ 16MB
        shape = (1024, 1024, 4)
        t = torch.randn(*shape, dtype=torch.float32)
        data = serialize_tensor_fast(t)
        assert len(data) > 0
        t2 = deserialize_tensor_fast(data)
        assert t2.shape == shape
        assert torch.allclose(t, t2, atol=1e-6)

    def test_serialize_tensor_fast_rejects_unsupported_dtype(self):
        """Phase 5.1: 不支持的 dtype 应抛 ValueError 而非静默转 float32。"""
        from tcp_comm import serialize_tensor_fast

        # 需要 >= 1MB 触发 TNR0（大张量）路径；float64 不在支持的 dtype map 中
        # float64: element_size=8, nbytes=8*250k=2MB → TNR0 路径
        t = torch.zeros(250000, dtype=torch.float64)
        with pytest.raises(ValueError, match="不支持"):
            serialize_tensor_fast(t)


# ================================================================
# C2 修复验证：_reconnect() 重连行为测试
# ================================================================

class TestReconnectBehavior:
    """测试 _reconnect() 的重连逻辑（BUG C2 修复验证）"""

    def test_reconnect_calls_connect_once(self, monkeypatch):
        """_reconnect 应仅调用 connect() 一次（connect 内部已有重试）。

        BUG C2 修复引入了双重重试问题（外层5次 × connect内部5次 = 25次），
        后续修复改为仅调用 connect() 一次。
        """
        client = TCPClient(
            server_host="127.0.0.1", server_port=1,
            client_id="test_reconnect", role="client",
        )
        connect_calls = []

        def mock_connect(on_message=None):
            connect_calls.append(1)
            return False  # 模拟连接失败

        monkeypatch.setattr(client, "connect", mock_connect)
        monkeypatch.setattr(tcp_comm_mod.time, "sleep", lambda _: None)
        monkeypatch.setattr(client, "_schedule_reconnect_retry", lambda: None)
        client.on_message = None

        client._reconnect()

        # connect() 应只被调用一次（内部已有 5 次重试）
        assert len(connect_calls) == 1, (
            f"_reconnect 应仅调用 connect() 一次，实际调用了 {len(connect_calls)} 次"
        )

    def test_reconnect_keeps_stale_heartbeat_stopped_on_failure(self, monkeypatch):
        """connect() 失败后应设置 _running=True 以便心跳循环下次重试。"""
        client = TCPClient(
            server_host="127.0.0.1", server_port=1,
            client_id="test_retry", role="client",
        )

        monkeypatch.setattr(client, "connect", lambda on_message=None: False)
        monkeypatch.setattr(tcp_comm_mod.time, "sleep", lambda _: None)
        monkeypatch.setattr(client, "_schedule_reconnect_retry", lambda: None)
        client.on_message = None

        client._reconnect()

        assert client._running is False
        assert client._registered is False

    def test_reconnect_closes_socket_before_reconnect(self, monkeypatch):
        """重连前应关闭旧 socket。"""
        client = TCPClient(
            server_host="127.0.0.1", server_port=1,
            client_id="test_close", role="client",
        )
        mock_sock = MagicMock()
        client.sock = mock_sock

        monkeypatch.setattr(client, "connect", lambda on_message=None: True)
        monkeypatch.setattr(tcp_comm_mod.time, "sleep", lambda _: None)
        client.on_message = None

        client._reconnect()

        mock_sock.close.assert_called_once()
        assert client.sock is None


class TestTCPSendSerialization:
    """同一连接上的心跳、控制消息和张量帧不得交叉写入。"""

    class DetectingSocket:
        def __init__(self):
            self.active = 0
            self.overlap = False
            self.calls = 0
            self.lock = threading.Lock()
            self.first_send = threading.Event()
            self.release_first = threading.Event()

        def sendall(self, _payload):
            with self.lock:
                if self.active:
                    self.overlap = True
                self.active += 1
                self.calls += 1
                first = self.calls == 1
                if first:
                    self.first_send.set()
            if first:
                assert self.release_first.wait(1)
            with self.lock:
                self.active -= 1

    @staticmethod
    def _run_concurrent(send, sock):
        start = threading.Barrier(3)

        def run(index):
            start.wait(timeout=1)
            send(index)

        threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        start.wait(timeout=1)
        assert sock.first_send.wait(1)
        assert sock.calls == 1, "第二个写入应被同一连接的发送锁挡住"
        sock.release_first.set()
        for thread in threads:
            thread.join(timeout=1)
            assert not thread.is_alive()

    def test_client_send_data_serializes_concurrent_writers(self):
        client = TCPClient(server_host="127.0.0.1", server_port=1, client_id="client1")
        sock = self.DetectingSocket()
        client.sock = sock

        self._run_concurrent(
            lambda index: client.send_data(
                {"index": index}, MessageType.HEARTBEAT,
            ),
            sock,
        )

        assert sock.calls == 2
        assert sock.overlap is False

    def test_server_send_to_client_serializes_concurrent_writers(self):
        server = TCPServer(host="127.0.0.1", port=0)
        sock = self.DetectingSocket()
        server._set_client(
            "client1",
            ClientConn(client_id="client1", sock=sock, addr=("127.0.0.1", 1)),
        )

        self._run_concurrent(
            lambda index: server.send_to_client(
                "client1", {"index": index}, MessageType.STATUS_RES,
            ),
            sock,
        )

        assert sock.calls == 2
        assert sock.overlap is False


class TestTCPClientRegistrationAck:
    """Registration acknowledgement handling."""

    def test_disconnect_callback_is_current_generation_and_idempotent(self):
        client = TCPClient(
            server_host="127.0.0.1", server_port=1, client_id="client1",
        )
        calls = []
        client.on_disconnect = lambda: calls.append("lost")
        client._connection_generation = 2
        client._registered = True

        client._notify_disconnect(1)
        assert calls == []
        client._notify_disconnect(2)
        client._notify_disconnect(2)
        assert calls == ["lost"]
        assert client.is_registered is False

    def test_recv_loop_is_bound_to_its_original_socket(self, monkeypatch):
        client = TCPClient(
            server_host="127.0.0.1", server_port=1, client_id="client1",
        )
        class FakeSocket:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        original_sock = FakeSocket()
        replacement_sock = object()
        received_from = []
        client.sock = original_sock
        client._running = True
        client._registered = True
        client._connection_generation = 1

        def recv_data(sock=None):
            received_from.append(sock)
            client.sock = replacement_sock
            return None

        monkeypatch.setattr(client, "recv_data", recv_data)

        client._recv_loop(1)

        assert received_from == [original_sock]
        # The replacement socket owns the current session; the stale receive
        # loop must not clear its registration state.
        assert client.is_registered is True
        assert original_sock.closed is True
        assert client.sock is replacement_sock

    def test_delayed_stale_recv_loop_cannot_close_replacement_socket(self):
        """旧线程晚于重连启动时，只能清理自己绑定的旧 socket。"""
        client = TCPClient(
            server_host="127.0.0.1", server_port=1, client_id="client1",
        )

        class FakeSocket:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        old_sock = FakeSocket()
        replacement_sock = FakeSocket()
        client.sock = replacement_sock
        client._running = True
        client._registered = True
        client._connection_generation = 2

        # Simulate a generation-1 receive thread that did not get CPU time
        # until generation 2 had already registered its replacement socket.
        client._recv_loop(1, old_sock)

        assert old_sock.closed is True
        assert replacement_sock.closed is False
        assert client.sock is replacement_sock
        assert client.is_registered is True

    def test_stale_heartbeat_generation_cannot_send_on_new_connection(
            self, monkeypatch):
        client = TCPClient(
            server_host="127.0.0.1", server_port=1, client_id="client1",
        )
        client.sock = object()
        client._running = True
        client._connection_generation = 2
        monkeypatch.setattr(
            client,
            "send_data",
            lambda *args, **kwargs: pytest.fail("旧心跳线程不能使用新连接"),
        )

        client._heartbeat_loop(1)

    def test_heartbeat_uses_its_original_socket(self, monkeypatch):
        client = TCPClient(
            server_host="127.0.0.1", server_port=1, client_id="client1",
        )
        original_sock = object()
        client.sock = original_sock
        client._running = True
        client._connection_generation = 1
        calls = []

        def send_data(data, msg_type, connection_sock=None):
            calls.append(connection_sock)
            client._running = False

        monkeypatch.setattr(client, "send_data", send_data)
        monkeypatch.setattr(tcp_comm_mod.time, "sleep", lambda _seconds: None)

        client._heartbeat_loop(1)

        assert calls == [original_sock]

    def test_heartbeat_reports_existing_rtt(self, monkeypatch):
        client = TCPClient(
            server_host="127.0.0.1", server_port=1, client_id="client1",
        )
        client.sock = object()
        client._running = True
        client._connection_generation = 1
        client.avg_rtt_ms = 4.81234
        payloads = []

        def send_data(data, msg_type, connection_sock=None):
            payloads.append(data)
            client._running = False

        monkeypatch.setattr(client, "send_data", send_data)
        monkeypatch.setattr(tcp_comm_mod.time, "sleep", lambda _seconds: None)

        client._heartbeat_loop(1)

        assert payloads[0]["rtt_ms"] == pytest.approx(4.812)

    def test_stale_socket_cannot_notify_disconnect_or_send_again(self, monkeypatch):
        client = TCPClient(
            server_host="127.0.0.1", server_port=1, client_id="client1",
        )
        old_sock = object()
        new_sock = object()
        client.sock = new_sock
        client._running = True
        client._registered = True
        client._connection_generation = 1
        disconnects = []
        client.on_disconnect = lambda: disconnects.append(True)

        client._notify_disconnect(1, connection_sock=old_sock)

        assert client._registered is True
        assert disconnects == []

        sent = []
        client.sock = old_sock
        client._running = True
        client._connection_generation = 1

        def send_data(data, msg_type, connection_sock=None):
            sent.append(connection_sock)
            client.sock = new_sock

        monkeypatch.setattr(client, "send_data", send_data)
        monkeypatch.setattr(tcp_comm_mod.time, "sleep", lambda _seconds: None)
        client._heartbeat_loop(1)

        assert sent == [old_sock]

    def test_heartbeat_ack_updates_ewma_and_bounded_snapshot(self, monkeypatch):
        client = TCPClient(
            server_host="127.0.0.1", server_port=1, client_id="client1",
        )
        client._connection_generation = 1
        client._heartbeat_quality.begin_generation(1)
        client._last_heartbeat_send = 10.0
        client._heartbeat_quality.record_send(1, 10.0)
        monkeypatch.setattr(tcp_comm_mod.time, "time", lambda: 10.125)

        client._handle_heartbeat_ack(
            {"data": {}},
            connection_generation=1,
        )

        snapshot = client.get_network_quality_snapshot()
        assert client.avg_rtt_ms == pytest.approx(125.0)
        assert snapshot["avg_rtt_ms"] == 125.0
        assert snapshot["rtt_ms_p50"] == 125.0
        assert snapshot["pending_heartbeat"] is False

    def test_stale_heartbeat_ack_cannot_change_current_generation_metrics(
            self, monkeypatch):
        client = TCPClient(
            server_host="127.0.0.1", server_port=1, client_id="client1",
        )
        client._connection_generation = 1
        client._heartbeat_quality.begin_generation(1)
        client._heartbeat_quality.record_send(1, 10.0)
        client._heartbeat_quality.begin_generation(2)
        client._connection_generation = 2
        client._heartbeat_quality.record_send(2, 20.0)
        monkeypatch.setattr(tcp_comm_mod.time, "time", lambda: 20.1)

        client._handle_heartbeat_ack(
            {"data": {"t_send": 10.0}},
            connection_generation=1,
        )

        snapshot = client.get_network_quality_snapshot()
        assert client.avg_rtt_ms == 0.0
        assert snapshot["sample_count"] == 0
        assert snapshot["generation"] == 2
        assert snapshot["pending_heartbeat"] is True

    def test_malformed_heartbeat_echo_does_not_fall_back_to_last_send(
            self, monkeypatch):
        client = TCPClient(
            server_host="127.0.0.1", server_port=1, client_id="client1",
        )
        client._connection_generation = 1
        client._heartbeat_quality.begin_generation(1)
        client._last_heartbeat_send = 10.0
        client._heartbeat_quality.record_send(1, 10.0)
        monkeypatch.setattr(tcp_comm_mod.time, "time", lambda: 10.1)

        client._handle_heartbeat_ack(
            {"data": {"t_send": -1.0}},
            connection_generation=1,
        )

        assert client.avg_rtt_ms == 0.0
        assert client.get_network_quality_snapshot()["sample_count"] == 0

    def test_registration_preparation_finishes_before_socket_connect(
            self, monkeypatch):
        packet = build_message(
            MessageType.REGISTER,
            {"status": "registered", "client_id": "worker1"},
        )

        class FakeSocket:
            def __init__(self):
                self._payload = bytearray(packet)

            def settimeout(self, _timeout):
                pass

            def connect(self, _addr):
                pass

            def getsockname(self):
                return ("100.64.0.2", 50000)

            def sendall(self, _data):
                pass

            def recv(self, size):
                chunk = self._payload[:size]
                del self._payload[:size]
                return bytes(chunk)

            def close(self):
                pass

        client = TCPClient(
            server_host="100.64.0.1", server_port=8888, client_id="worker1",
        )
        preparation_socket_states = []

        def detect_network():
            preparation_socket_states.append(client.sock)
            return "tailscale"

        def compute_model_sha():
            preparation_socket_states.append(client.sock)
            return "sha-model"

        monkeypatch.setattr(tcp_comm_mod, "detect_network_type", detect_network)
        monkeypatch.setattr(client, "_compute_local_model_sha256", compute_model_sha)
        monkeypatch.setattr(tcp_comm_mod.socket, "socket", lambda *_args, **_kwargs: FakeSocket())
        monkeypatch.setattr(
            client,
            "_heartbeat_loop",
            lambda _generation=None, _connection_sock=None: None,
        )
        monkeypatch.setattr(
            client,
            "_recv_loop",
            lambda _generation=None, _connection_sock=None: None,
        )

        assert client.connect() is True
        assert preparation_socket_states == [None, None]

    def test_connect_returns_false_when_registration_rejected(self, monkeypatch):
        packet = build_message(
            MessageType.REGISTER,
            {"status": "rejected", "reason": "bad secret"},
        )

        class FakeSocket:
            def __init__(self, payload: bytes):
                self._payload = bytearray(payload)
                self.closed = False

            def settimeout(self, _timeout):
                pass

            def connect(self, _addr):
                pass

            def getsockname(self):
                return ("100.64.0.2", 50000)

            def sendall(self, _data):
                pass

            def recv(self, size: int):
                chunk = self._payload[:size]
                del self._payload[:size]
                return bytes(chunk)

            def close(self):
                self.closed = True

        fake_socket = FakeSocket(packet)

        monkeypatch.setattr(tcp_comm_mod.socket, "socket", lambda *_args, **_kwargs: fake_socket)
        monkeypatch.setattr(tcp_comm_mod, "detect_network_type", lambda: "tailscale")
        monkeypatch.setattr(tcp_comm_mod, "detect_lan_ip", lambda: "100.64.0.2")

        client = TCPClient(
            server_host="100.64.0.1",
            server_port=8888,
            client_id="client-rejected",
            role="client",
        )

        assert client.connect() is False
        assert client.is_registered is False
        assert "bad secret" in client.last_register_error
        assert fake_socket.closed is True
        assert client.sock is None


class TestTCPLifecycleRaceMatrix:
    """T-RACE-2: deterministic TCP registration and socket ownership fences."""

    def test_registered_ack_precedes_post_registration_business_frame(
            self, monkeypatch):
        import config as cfg

        monkeypatch.setattr(cfg, "CLUSTER_SECRET", "t-race-2-order-secret")
        monkeypatch.setattr(
            TCPClient,
            "_compute_local_model_sha256",
            staticmethod(lambda: "sha-t-race-2"),
        )
        monkeypatch.setattr(
            TCPClient,
            "_heartbeat_loop",
            lambda self, connection_generation=None, connection_sock=None: None,
        )
        monkeypatch.setattr(tcp_comm_mod, "detect_network_type", lambda: "ethernet")
        monkeypatch.setattr(tcp_comm_mod, "detect_lan_ip", lambda: "127.0.0.1")

        received = []
        business_frame = threading.Event()
        server = TCPServer(host="127.0.0.1", port=0)

        def push_after_ack(client_id):
            server.send_to_client(
                client_id,
                {"event": "registered_business_frame"},
                MessageType.NODE_LIST_SYNC,
            )

        server.start(on_registration_confirmed=push_after_ack)
        port = server.sock.getsockname()[1]
        client = TCPClient(
            server_host="127.0.0.1",
            server_port=port,
            client_id="worker-t-race-2",
            role="client",
            advertise_host="127.0.0.1",
            advertise_port=port,
        )

        try:
            assert client.connect(
                on_message=lambda message: (
                    received.append(message), business_frame.set(),
                ),
            ) is True
            assert client.is_registered is True
            assert business_frame.wait(2), "post-registration frame was not delivered"
            assert len(received) == 1
            assert received[0]["type"] == MessageType.NODE_LIST_SYNC.value
            assert received[0]["data"] == {
                "event": "registered_business_frame",
            }
        finally:
            client.disconnect()
            server.stop()

    def test_send_data_rejects_stale_bound_socket_without_writing(self):
        class RecordingSocket:
            def __init__(self):
                self.packets = []

            def sendall(self, packet):
                self.packets.append(packet)

        client = TCPClient(
            server_host="127.0.0.1", server_port=1, client_id="worker-t-race-2",
        )
        stale_sock = RecordingSocket()
        current_sock = RecordingSocket()
        client.sock = current_sock

        with pytest.raises(ConnectionError, match="新代际替换"):
            client.send_data(
                {"event": "stale_write"},
                MessageType.HEARTBEAT,
                connection_sock=stale_sock,
            )

        assert stale_sock.packets == []
        assert current_sock.packets == []

    def test_send_data_classifies_close_interleaving(self):
        class ClosedSocket:
            def sendall(self, _packet):
                raise OSError(10038, "not a socket")

        client = TCPClient(
            server_host="127.0.0.1", server_port=1, client_id="worker-t-race-2",
        )
        client.sock = ClosedSocket()

        with pytest.raises(ConnectionError, match="主节点连接发送失败") as exc_info:
            client.send_data({"event": "close_race"}, MessageType.HEARTBEAT)

        assert isinstance(exc_info.value.__cause__, OSError)
        assert exc_info.value.__cause__.errno == 10038


# ================================================================
# IPv4/IPv6 双栈测试（问题 #5 回归）
# ================================================================

def _ipv6_supported() -> bool:
    """本机是否支持 IPv6 回环连接。"""
    if not socket.has_ipv6:
        return False
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        s.close()
        return True
    except OSError:
        return False


class TestDualStackUtils:
    """parse_host_port / format_host_port 对 v4/v6 字面量的处理。"""

    def test_parse_host_port_v4(self):
        assert parse_host_port("1.2.3.4:8888") == ("1.2.3.4", 8888)
        assert parse_host_port("1.2.3.4") == ("1.2.3.4", None)
        assert parse_host_port("1.2.3.4:notaport") == ("1.2.3.4", None)

    def test_parse_host_port_v6(self):
        assert parse_host_port("[fe80::1]:8888") == ("fe80::1", 8888)
        assert parse_host_port("[fe80::1]") == ("fe80::1", None)
        assert parse_host_port("fe80::1") == ("fe80::1", None)
        assert parse_host_port("[fe80::1]:abc") == ("fe80::1", None)

    def test_parse_host_port_hostname(self):
        assert parse_host_port("myhost:8000") == ("myhost", 8000)
        assert parse_host_port("myhost") == ("myhost", None)

    def test_format_host_port(self):
        assert format_host_port("1.2.3.4", 8888) == "1.2.3.4:8888"
        assert format_host_port("fe80::1", 8888) == "[fe80::1]:8888"
        assert format_host_port("[fe80::1]", 8888) == "[fe80::1]:8888"

    def test_wildcard_and_ipv6_literal_helpers(self):
        assert tcp_comm_mod._is_wildcard_host("0.0.0.0")
        assert tcp_comm_mod._is_wildcard_host("::")
        assert tcp_comm_mod._is_wildcard_host("")
        assert not tcp_comm_mod._is_wildcard_host("127.0.0.1")
        assert tcp_comm_mod._ipv6_literal("fe80::1")
        assert not tcp_comm_mod._ipv6_literal("[fe80::1]")
        assert not tcp_comm_mod._ipv6_literal("1.2.3.4")


class TestDualStackServer:
    """TCPServer 双栈监听：通配地址同时接受 v4/v6 连接。"""

    def test_wildcard_bind_creates_v4_and_v6_sockets(self):
        socks = TCPServer._create_listen_sockets("0.0.0.0", 0)
        try:
            assert len(socks) == 2
            names = sorted(s.getsockname()[0] for s in socks)
            assert "0.0.0.0" in names and "::" in names
            assert len({sock.getsockname()[1] for sock in socks}) == 1
            # IPv6 socket 必须 V6ONLY=1，避免与 v4 socket 端口冲突
            v6_sock = next(s for s in socks if s.getsockname()[0] == "::")
            assert v6_sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 1
        finally:
            for s in socks:
                s.close()

    def test_specific_v4_bind_single_socket(self):
        socks = TCPServer._create_listen_sockets("127.0.0.1", 0)
        try:
            assert len(socks) == 1
            assert socks[0].getsockname()[0] == "127.0.0.1"
        finally:
            for s in socks:
                s.close()

    @pytest.mark.skipif(not _ipv6_supported(), reason="本机不支持 IPv6")
    def test_v4_and_v6_clients_both_connect(self):
        server = TCPServer(host="0.0.0.0", port=0)
        assert server.port == 0
        server.start()
        try:
            port = server.sock.getsockname()[1]
            c4 = socket.create_connection(("127.0.0.1", port), timeout=3)
            c4.close()
            c6 = socket.create_connection(("::1", port), timeout=3)
            c6.close()
            # 服务端应已 accept 两个连接（等待 accept 线程处理）
            deadline = time.time() + 3
            while time.time() < deadline and not server._client_ids_snapshot():
                threading.Event().wait(0.05)
        finally:
            server.stop()


class TestDualStackClient:
    """TCPClient._connect_socket 自动选择 v4/v6。"""

    def test_connect_socket_v4(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            sock = TCPClient._connect_socket("127.0.0.1", port)
            conn, _ = listener.accept()
            sock.close()
            conn.close()
        finally:
            listener.close()

    @pytest.mark.skipif(not _ipv6_supported(), reason="本机不支持 IPv6")
    def test_connect_socket_v6_literal_and_bracketed(self):
        listener = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        listener.bind(("::1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            for target in ("::1", "[::1]"):
                sock = TCPClient._connect_socket(target, port)
                conn, _ = listener.accept()
                sock.close()
                conn.close()
        finally:
            listener.close()
