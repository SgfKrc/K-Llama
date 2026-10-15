"""Run unchanged production TCPClient methods with stdlib-only dependencies.

The AST seam avoids torch, scheduler, user databases and network access.
python -m unittest discover -s tests -p test_tcp_reconnect_retry.py -v
"""
from __future__ import annotations

import ast
import logging
import math
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "tcp_comm.py"
METHODS = {"__init__", "connect", "_schedule_reconnect_retry", "_reconnect", "disconnect",
           "_notify_disconnect", "_recv_loop"}


def isolated_client_type():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    client = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TCPClient")
    client.body = [n for n in client.body if isinstance(n, ast.Assign) or
                   isinstance(n, ast.FunctionDef) and n.name in METHODS]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             client], type_ignores=[])
    namespace = dict(socket=socket, threading=threading, time=time, math=math,
                     logger=logging.getLogger("isolated-tcp-reconnect"),
                     SERVER_IP="127.0.0.1", SERVER_PORT=1, HEARTBEAT_INTERVAL=15,
                     _CLIENT_SOCKET_TIMEOUT_SECONDS=45, RECONNECT_MAX_RETRIES=2, RECONNECT_DELAY=0.005,
                     HeartbeatQualityWindow=lambda: SimpleNamespace(record_disconnect=lambda *_: None,
                                                                   begin_generation=lambda *_: None),
                     detect_network_type=lambda: "test", detect_lan_ip=lambda: "127.0.0.1",
                     build_auth_signature=lambda *_: {}, format_host_port=lambda host, port: f"{host}:{port}",
                     MessageType=SimpleNamespace(REGISTER="register", HEARTBEAT_ACK=SimpleNamespace(value="heartbeat_ack")),
                     ControlFenceError=type("ControlFenceError", (Exception,), {}))
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["TCPClient"], namespace


class FakeSocket:
    def __init__(self):
        self.closed = False
    def close(self):
        self.closed = True
    def shutdown(self, *_):
        pass
    def getsockname(self):
        return ("127.0.0.1", 12345)


class TcpReconnectRetryTest(unittest.TestCase):
    def setUp(self):
        cls, self.namespace = isolated_client_type()
        self.threads = []
        self.thread_lock = threading.Lock()
        def new_thread(*args, **kwargs):
            worker = threading.Thread(*args, **kwargs)
            with self.thread_lock:
                self.threads.append(worker)
            return worker
        self.namespace["threading"] = SimpleNamespace(Thread=new_thread, Event=threading.Event,
                                                     Lock=threading.Lock, local=threading.local)
        self.client = cls(client_id="isolated-reconnect")
        self.client._transport_runtime_call = lambda *_, **__: None
        self.client._heartbeat_loop = lambda *_: None
        self.client._recv_loop_for_test = self.client._recv_loop
        self.client._recv_loop = lambda *_: None
        self.client._compute_local_model_sha256 = lambda: ""
        self.client._connect_socket = lambda *_: FakeSocket()
        self.client.send_data = lambda *_: None
        self.client.recv_data = lambda: {"type": "register", "data": {"status": "registered"}}

    def tearDown(self):
        self.client.disconnect()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with self.thread_lock:
                workers = list(self.threads)
            for worker in workers:
                if worker.ident is not None:
                    worker.join(0.05)
            if all(not worker.is_alive() for worker in workers):
                break
        self.assertTrue(all(not worker.is_alive() for worker in self.threads))

    def wait_for(self, predicate, message="condition did not become true"):
        deadline = time.monotonic() + 2
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertTrue(predicate(), message)

    def test_successful_auto_reconnect_notifies_upper_protocol_once(self):
        notifications = []
        self.client.on_reconnect = lambda: notifications.append(self.client._registration_epoch)
        self.client._reconnect()
        self.assertEqual([1], notifications)
        self.assertTrue(self.client._registered)

    def test_failed_auto_reconnect_does_not_notify_upper_protocol(self):
        notifications = []
        self.client.on_reconnect = lambda: notifications.append(True)
        self.client.connect = lambda *_: False
        self.client.MAX_RECONNECT_FAILURES = 1
        self.client._reconnect()
        self.assertEqual([], notifications)

    def test_upper_reconnect_callback_failure_does_not_retire_transport_owner(self):
        def failed_callback():
            raise RuntimeError("upper protocol inspection failed")
        self.client.on_reconnect = failed_callback
        self.client._reconnect()
        self.assertTrue(self.client._registered)
        self.assertFalse(self.client._reconnect_in_progress)
        self.assertEqual("ready", self.client.reconnect_state)

    def test_two_delayed_retry_failures_then_recovery_keep_exactly_one_owner(self):
        calls, active, maximum = [], 0, 0
        lock = threading.Lock()
        def connect(_=None):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                calls.append(len(calls) + 1)
                ok = len(calls) == 4
            time.sleep(0.002)
            if ok:
                self.client._running = self.client._registered = True
            with lock:
                active -= 1
            return ok
        self.client.connect = connect
        self.client._reconnect()
        self.wait_for(lambda: self.client._registered and not self.client._reconnect_in_progress,
                      "retry chain stopped before the fourth connect")
        self.assertEqual([1, 2, 3, 4], calls)
        self.assertEqual(1, maximum)
        self.assertFalse(self.client._reconnect_retry_scheduled)

    def test_duplicate_schedules_and_reconnect_calls_cannot_create_a_second_owner(self):
        entered, release, calls = threading.Event(), threading.Event(), []
        def connect(_=None):
            calls.append(1)
            entered.set()
            release.wait(1)
            self.client._running = self.client._registered = True
            return True
        self.client.connect = connect
        self.client._schedule_reconnect_retry()
        self.assertTrue(entered.wait(1))
        try:
            callers = [threading.Thread(target=self.client._schedule_reconnect_retry) for _ in range(8)]
            for worker in callers:
                worker.start()
            for worker in callers:
                worker.join()
            self.client._reconnect()
            self.assertEqual(1, len(self.threads))
        finally:
            release.set()
        self.wait_for(lambda: self.client._registered)
        self.assertEqual([1], calls)

    def test_manual_disconnect_cancels_queued_retry_and_cannot_be_revived(self):
        self.namespace["RECONNECT_DELAY"] = 10
        calls = []
        self.client.connect = lambda _=None: calls.append(1) or False
        self.client._reconnect()
        self.assertTrue(self.client._reconnect_retry_scheduled)
        self.client.disconnect()
        for worker in self.threads:
            worker.join(0.3)
        self.client._schedule_reconnect_retry()
        self.client._reconnect()
        self.assertEqual([1], calls)
        self.assertFalse(self.client._running)
        self.assertFalse(self.client._registered)
        self.assertTrue(all(not worker.is_alive() for worker in self.threads))

    def test_disconnect_during_reconnect_closes_late_success_and_never_reschedules(self):
        entered, release, late = threading.Event(), threading.Event(), FakeSocket()
        def connect(_=None):
            entered.set()
            release.wait(1)
            self.client.sock = late
            self.client._running = self.client._registered = True
            return True
        self.client.connect = connect
        worker = threading.Thread(target=self.client._reconnect)
        worker.start()
        self.assertTrue(entered.wait(1))
        self.client.disconnect()
        release.set()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertTrue(late.closed)
        self.assertFalse(self.client._running)
        self.assertFalse(self.client._registered)
        self.assertFalse(self.client._reconnect_retry_scheduled)

    def test_explicit_connect_after_stop_starts_new_intent_and_resets_budget(self):
        self.client.disconnect()
        self.assertTrue(self.client.connect())
        self.assertTrue(self.client._registered)
        self.assertTrue(self.client._reconnect_enabled)
        self.assertEqual(0, self.client.reconnect_failure_count)
        self.assertEqual("ready", self.client.reconnect_state)

    def test_disconnect_while_socket_connects_fences_the_unpublished_socket(self):
        entered, release, opened, outcome = threading.Event(), threading.Event(), FakeSocket(), []
        def open_socket(*_):
            entered.set()
            release.wait(1)
            return opened
        self.client._connect_socket = open_socket
        worker = threading.Thread(target=lambda: outcome.append(self.client.connect()))
        worker.start()
        self.assertTrue(entered.wait(1))
        self.client.disconnect()
        release.set()
        worker.join(1)
        self.assertEqual([False], outcome)
        self.assertTrue(opened.closed)
        self.assertFalse(self.client._registered)
        self.assertFalse(self.client._running)

    def test_persistent_failure_has_named_terminal_state_and_capped_exponential_delays(self):
        calls, waits = [], []
        class RecordingEvent:
            def __init__(self):
                self.event = threading.Event()
            def wait(self, delay):
                waits.append(delay)
                return self.event.wait(delay)
            def set(self):
                self.event.set()
        self.namespace["threading"].Event = RecordingEvent
        self.client.MAX_RECONNECT_BACKOFF_SECONDS = 0.02
        self.client.connect = lambda _=None: calls.append(1) or False
        self.client._reconnect()
        self.wait_for(lambda: self.client.reconnect_state == "retry_exhausted")
        self.assertEqual(self.client.MAX_RECONNECT_FAILURES, len(calls))
        self.assertEqual("worker_reconnect_exhausted", self.client.last_reconnect_error)
        self.assertEqual([0.005, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02], waits)
        self.assertEqual(0.0, self.client.next_reconnect_at)
        self.assertFalse(self.client._registered)

    def test_receive_eof_notifies_and_queues_reconnect_for_current_owner(self):
        current, notified, retries = FakeSocket(), [], []
        self.client.sock = current
        self.client._running = self.client._registered = True
        self.client._connection_generation = 2
        self.client.recv_data = lambda *_: None
        self.client.on_disconnect = lambda: notified.append(1)
        self.client._schedule_reconnect_retry = lambda: retries.append(1)
        self.client._recv_loop_for_test(2, current)
        self.assertEqual([1], notified)
        self.assertEqual([1], retries)
        self.assertFalse(self.client._registered)
        self.assertFalse(self.client._running)
        self.assertTrue(current.closed)

    def test_receive_eof_before_reconnect_owner_retires_keeps_a_successor(self):
        original_connect = self.client.connect
        calls = []
        def connect(on_message=None):
            calls.append(1)
            ok = original_connect(on_message)
            if len(calls) == 1:
                # REGISTER succeeded, but EOF arrives before _reconnect releases its active owner.
                original_receive = self.client.recv_data
                self.client.recv_data = lambda *_: None
                try:
                    self.client._recv_loop_for_test(self.client._connection_generation, self.client.sock)
                finally:
                    self.client.recv_data = original_receive
                self.assertFalse(self.client._registered)
                self.assertFalse(self.client._reconnect_retry_scheduled)
            return ok
        self.client.connect = connect
        self.client._reconnect()
        self.wait_for(lambda: self.client._registered and not self.client._reconnect_in_progress,
                      "EOF during reconnect publication left no successor owner")
        self.assertEqual([1, 1], calls)
        self.assertEqual("ready", self.client.reconnect_state)
        self.assertFalse(self.client._reconnect_retry_scheduled)

    def test_old_intent_cannot_retire_a_newly_registered_socket(self):
        entered, release, replacement = threading.Event(), threading.Event(), FakeSocket()
        def old_connect(_=None):
            entered.set()
            release.wait(1)
            return True
        self.client.connect = old_connect
        worker = threading.Thread(target=self.client._reconnect)
        worker.start()
        self.assertTrue(entered.wait(1))
        self.client.disconnect()
        with self.client._reconnect_state_lock:
            self.client._reconnect_intent_epoch += 1
            self.client._reconnect_enabled = True
            self.client.sock = replacement
            self.client._running = self.client._registered = True
            self.client.reconnect_state = "ready"
        release.set()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertIs(replacement, self.client.sock)
        self.assertFalse(replacement.closed)
        self.assertTrue(self.client._registered)

    def test_stop_during_register_ack_never_publishes_registration_or_starts_workers(self):
        entered, release, outcome = threading.Event(), threading.Event(), []
        def ack():
            entered.set()
            release.wait(1)
            return {"type": "register", "data": {"status": "registered"}}
        self.client.recv_data = ack
        worker = threading.Thread(target=lambda: outcome.append(self.client.connect()))
        worker.start()
        self.assertTrue(entered.wait(1))
        opened = self.client.sock
        self.client.disconnect()
        release.set()
        worker.join(1)
        self.assertEqual([False], outcome)
        self.assertTrue(opened.closed)
        self.assertFalse(self.client._registered)
        self.assertEqual([], self.threads)

    def test_late_failed_owner_cannot_schedule_into_a_new_intent(self):
        self.client._reconnect_context.epoch = self.client._reconnect_intent_epoch
        self.client.disconnect()
        with self.client._reconnect_state_lock:
            self.client._reconnect_intent_epoch += 1
            self.client._reconnect_enabled = True
        self.client._schedule_reconnect_retry()
        self.assertEqual([], self.threads)
        self.assertFalse(self.client._reconnect_retry_scheduled)
        self.client._reconnect_context.epoch = None

    def test_old_retry_token_cannot_connect_after_a_new_explicit_intent(self):
        self.namespace["RECONNECT_DELAY"] = 10
        self.client._schedule_reconnect_retry()
        owner = self.client._reconnect_retry_owner
        self.client.disconnect()
        self.assertTrue(self.client.connect())
        current = self.client.sock
        self.client._reconnect(_intent_epoch=owner[0], _retry_owner=owner)
        self.assertIs(current, self.client.sock)
        self.assertFalse(current.closed)
        self.assertTrue(self.client._registered)


if __name__ == "__main__":
    unittest.main()
