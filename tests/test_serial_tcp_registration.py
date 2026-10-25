"""Registration retirement checked after real socket reads, without polling races."""
import socket
import threading

import pytest

import config
import tcp_comm
from transport_port import MessageType


@pytest.mark.parametrize("outcome", ["eof", "ack", "oserror"])
@pytest.mark.parametrize("replace_socket", [False, True])
def test_retired_registration_cannot_publish_or_close_successor(
        monkeypatch, outcome, replace_socket):
    monkeypatch.setattr(config, "CLUSTER_SECRET", "serial-registration-isolated-secret")
    monkeypatch.setattr(tcp_comm, "detect_network_type", lambda: "ethernet")
    monkeypatch.setattr(tcp_comm, "detect_lan_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(tcp_comm.TCPClient, "_compute_local_model_sha256", staticmethod(lambda: ""))
    server = tcp_comm.TCPServer(host="127.0.0.1", port=0)
    register_seen, read_done, release = (threading.Event() for _ in range(3))
    errors, results, retries = [], [], []

    def receive(client_id, msg):
        if msg["type"] == MessageType.REGISTER.value:
            try:
                if outcome == "ack":
                    assert server.confirm_registration(client_id)
                else:
                    # A real remote EOF produces None instead of OSError.
                    connection = server._get_client(client_id)
                    connection.sock.shutdown(socket.SHUT_RDWR)
                register_seen.set()
            except BaseException as exc:
                errors.append(exc)
                register_seen.set()

    server.start(on_message=receive)
    client = tcp_comm.TCPClient(server_host="127.0.0.1",
                                server_port=server.sock.getsockname()[1],
                                client_id="serial-registration-owner")
    original_receive = client.recv_data

    def receive_then_pause():
        frame = original_receive()
        assert (frame is not None) == (outcome == "ack")
        read_done.set()
        assert release.wait(5)
        if outcome == "oserror":
            raise OSError("registration socket retired during recv")
        return frame

    monkeypatch.setattr(client, "recv_data", receive_then_pause)
    monkeypatch.setattr(client, "_schedule_reconnect_retry",
                        lambda: retries.append(client.last_reconnect_error))
    # Any workers started by a stale ACK are an immediate assertion failure.
    monkeypatch.setattr(client, "_heartbeat_loop", lambda *_: errors.append("stale heartbeat"))
    monkeypatch.setattr(client, "_recv_loop", lambda *_: errors.append("stale recv"))

    def reconnect():
        try:
            client._reconnect()
            results.append(client.last_reconnect_error)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=reconnect, name="serial-register-read")
    successor = peer = old_socket = None
    try:
        thread.start()
        assert register_seen.wait(5), errors
        assert read_done.wait(5), errors
        with client._disconnect_callback_lock:
            old_socket = client.sock
            client._connection_generation += 1
            if replace_socket:
                successor, peer = socket.socketpair()
                client.sock = successor
        old_socket.close()
        release.set()
        thread.join(5)
        assert not thread.is_alive()
        assert not errors, errors
        assert results == ["registration_interrupted"]
        assert client.reconnect_failure_count == 0
        assert not client.is_registered
        assert client.registration_epoch == 0
        assert retries == ["registration_interrupted"]
        assert old_socket.fileno() == -1
        if successor is not None:
            assert client.sock is successor
            assert successor.fileno() != -1, "stale registration closed its successor"
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(5)
        client.on_disconnect = None
        client.disconnect()
        if peer is not None:
            peer.close()
        server.stop()


def test_current_registration_remote_eof_is_a_real_failure(monkeypatch):
    client = tcp_comm.TCPClient(client_id="current-eof")
    monkeypatch.setattr(tcp_comm, "detect_network_type", lambda: "ethernet")
    monkeypatch.setattr(tcp_comm, "detect_lan_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(client, "_compute_local_model_sha256", lambda: "")
    left, right = socket.socketpair()
    monkeypatch.setattr(client, "_connect_socket", lambda *_: left)
    monkeypatch.setattr(client, "send_data", lambda *_: None)
    monkeypatch.setattr(client, "_schedule_reconnect_retry", lambda: None)
    right.close()
    try:
        client._reconnect()
        assert client.last_reconnect_error == "worker_reconnect_failed"
        assert client.reconnect_failure_count == 1
        assert not client._last_registration_interrupted
    finally:
        client.disconnect()


def test_registration_retired_between_receive_validation_and_publication(monkeypatch):
    """The publish lock must recheck retirement after the earlier read check."""
    client = tcp_comm.TCPClient(client_id="retired-before-publish")
    monkeypatch.setattr(tcp_comm, "detect_network_type", lambda: "ethernet")
    monkeypatch.setattr(tcp_comm, "detect_lan_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(client, "_compute_local_model_sha256", lambda: "")
    old_socket, old_peer = socket.socketpair()
    successor, successor_peer = socket.socketpair()
    monkeypatch.setattr(client, "_connect_socket", lambda *_: old_socket)
    monkeypatch.setattr(client, "send_data", lambda *_: None)
    read_finished, publishing, release = (threading.Event() for _ in range(3))
    results, workers, errors = [], [], []
    original_lock = client._reconnect_state_lock

    class PauseBeforePublish:
        def __enter__(self):
            if read_finished.is_set() and not publishing.is_set():
                publishing.set()
                assert release.wait(5)
            return original_lock.__enter__()

        def __exit__(self, *args):
            return original_lock.__exit__(*args)

    def acknowledge():
        read_finished.set()
        return {"type": "register", "data": {"status": "registered"}}

    def connect():
        try:
            results.append(client.connect())
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(client, "recv_data", acknowledge)
    monkeypatch.setattr(client, "_reconnect_state_lock", PauseBeforePublish())
    monkeypatch.setattr(client, "_heartbeat_loop", lambda *_: workers.append("heartbeat"))
    monkeypatch.setattr(client, "_recv_loop", lambda *_: workers.append("recv"))
    thread = threading.Thread(target=connect, name="registration-before-publish")
    try:
        thread.start()
        assert publishing.wait(5)
        with client._disconnect_callback_lock:
            client._connection_generation += 1
            client.sock = successor
        release.set()
        thread.join(5)
        assert not thread.is_alive()
        assert not errors, errors
        assert results == [False]
        assert client._last_registration_interrupted
        assert client.registration_epoch == 0
        assert not client.is_registered
        assert not workers
        assert client.sock is successor
        assert successor.fileno() != -1
        assert old_socket.fileno() == -1
    finally:
        release.set()
        thread.join(5)
        client.on_disconnect = None
        client.disconnect()
        old_peer.close()
        successor_peer.close()
