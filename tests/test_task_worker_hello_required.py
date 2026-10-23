"""#106/#114：主节点必须在 REGISTER ACK 里声明它加上的 task-worker 握手要求。

死锁链条：主节点开关打开 ⇒ REGISTER 后立刻 fence，而 fence **只能**由客户端
主动发 task-worker HELLO 解除；此前是否发 HELLO 只读客户端本地的同一开关 ⇒
没开的 PC 节点永远不发 HELLO ⇒ release 债务（outbox `state=pending,
attempts=0`）永远投递不出去 ⇒ 整集群 `/api/models/load` 恒 503。

这些回归只验证两件最小事实：
1. 声明与 fence 在同一个边沿、同一条件（声明 ⇔ fence）；
2. 客户端据 ACK 里的声明发 HELLO，且**不改变** fail-closed（不清债、不放行投递）。
"""
import threading

from scheduler import Scheduler
from scheduler_task_worker import SchedulerTaskWorkerMixin
from task_worker_adapter import TaskWorkerControlPlane
from tcp_comm import MessageType, TCPClient, TCPServer
from tests.test_task_worker_hello_owner import RecordingSocket, publish_registration
from tests.test_task_worker_hello_followup import master  # noqa: F401


class LocalOptOutWorker(SchedulerTaskWorkerMixin):
    """A client whose local task-worker experiment switch is off."""

    def __init__(self, node_id="surface"):
        self._node_id = node_id
        self._task_worker_control = TaskWorkerControlPlane()
        self._task_worker_refresh_lock = threading.Lock()
        self._task_worker_refresh_generation = 0
        self._task_worker_refresh_requested = False
        self._tcp_client = TCPClient(client_id=node_id)
        publish_registration(self._tcp_client)
        # Capability publication merges into the shared node table, which this
        # harness does not own; the gate under test runs before that work.
        self._merge_task_worker_capabilities_into_node = lambda *_, **__: None

    def get_distributed_inference_enabled(self):
        return True

    def _effective_role(self):
        return "client"

    def _runtime_setting(self, name):
        assert name == "TASK_WORKER_EXPERIMENTAL_ENABLED"
        return False

    def get_effective_node_id(self):
        return self._node_id

    def _task_worker_engine_schema(self):
        return {"schema_version": 1, "runtimes": []}

    def _task_worker_capabilities(self):
        return {"stage_types": ["aggregate"], "engines": ["pytorch"],
                "models": [], "max_concurrency": 1}


def test_local_opt_out_client_hellos_only_after_master_declaration():
    """本地开关关闭时：未声明不发 HELLO，声明后发一次 HELLO。"""
    worker = LocalOptOutWorker()
    sock = worker._tcp_client.sock

    assert worker._task_worker_handshake_required(worker._tcp_client) is False
    assert worker._send_task_worker_hello() is False
    assert sock.packets == []

    worker._tcp_client.task_worker_hello_required = True
    assert worker._task_worker_handshake_required(worker._tcp_client) is True
    assert worker._send_task_worker_hello() is True
    assert [packet["message_type"] for packet in sock.packets] == ["hello"]


def test_register_ack_carries_declared_requirement_only_when_set():
    server = TCPServer()
    sock = RecordingSocket()

    server._send_register_ack(sock, "registered", client_id="surface")
    assert "task_worker_hello_required" not in sock.packets[-1]

    server.set_task_worker_hello_required("surface", True)
    server._send_register_ack(sock, "registered", client_id="surface")
    assert sock.packets[-1]["task_worker_hello_required"] is True
    # The rejection ACK never claims a handshake requirement.
    server._send_register_ack(sock, "rejected", client_id="surface", reason="full")
    assert "task_worker_hello_required" not in sock.packets[-1]

    # A later REGISTER explicitly overwrites the declaration: no stale demand.
    server.set_task_worker_hello_required("surface", False)
    server._send_register_ack(sock, "registered", client_id="surface")
    assert "task_worker_hello_required" not in sock.packets[-1]


def test_register_edge_declares_exactly_what_it_fences(monkeypatch):
    """声明与 fence 同边沿同条件：master + 实验开关 + pc/android。"""
    import scheduler as scheduler_mod

    sched = Scheduler()
    monkeypatch.setattr(sched, "_effective_role", lambda: "master")
    server = TCPServer()
    sched._tcp_server = server

    monkeypatch.setattr(scheduler_mod, "TASK_WORKER_EXPERIMENTAL_ENABLED", False)
    sched._on_tcp_message("pixel", {
        "type": "register",
        "data": {"node_type": "android", "role": "client", "hostname": "pixel"},
    })
    assert "pixel" not in server._task_worker_hello_required
    assert sched._task_worker_control.pending_worker_ids() == set()

    monkeypatch.setattr(scheduler_mod, "TASK_WORKER_EXPERIMENTAL_ENABLED", True)
    # A real REGISTER frame carries its registration context; this direct call
    # has none, so supply the exact inbound epoch the fence is bound to.
    monkeypatch.setattr(server, "get_registration_peer_epoch", lambda _cid: 7)
    sched._on_tcp_message("surface", {
        "type": "register",
        "data": {"node_type": "pc", "role": "client", "hostname": "surface"},
    })
    assert "surface" in server._task_worker_hello_required
    assert sched._task_worker_control.pending_worker_ids() == {"surface"}

    server.set_task_worker_hello_required("surface", False)
    sched._on_tcp_message("surface", {
        "type": "register",
        "data": {"node_type": "pc", "role": "client", "hostname": "surface"},
    })
    assert "surface" in server._task_worker_hello_required


def test_real_registration_records_declared_requirement(monkeypatch):
    """端到端：REGISTER ACK 的声明落到 TCPClient，未声明时不置位。"""
    import config as cfg
    import tcp_comm as tcp_comm_mod

    monkeypatch.setattr(cfg, "CLUSTER_SECRET", "hello-required-secret")
    monkeypatch.setattr(
        TCPClient, "_compute_local_model_sha256", staticmethod(lambda: "sha-hr"),
    )
    monkeypatch.setattr(
        TCPClient, "_heartbeat_loop",
        lambda self, connection_generation=None, connection_sock=None: None,
    )
    monkeypatch.setattr(tcp_comm_mod, "detect_network_type", lambda: "ethernet")
    monkeypatch.setattr(tcp_comm_mod, "detect_lan_ip", lambda: "127.0.0.1")

    declared = {"surface"}
    server = TCPServer(host="127.0.0.1", port=0)

    def on_server_message(client_id, msg):
        if msg.get("type") == MessageType.REGISTER.value:
            server.set_task_worker_hello_required(client_id, client_id in declared)
            assert server.confirm_registration(client_id)

    server.start(on_message=on_server_message)
    port = server.sock.getsockname()[1]
    clients = []
    try:
        for node_id in ("surface", "pixel"):
            client = TCPClient(
                server_host="127.0.0.1", server_port=port,
                client_id=node_id, role="client", advertise_port=port,
            )
            clients.append(client)
            assert client.connect() is True
        assert clients[0].task_worker_hello_required is True
        assert clients[1].task_worker_hello_required is False
    finally:
        for client in clients:
            client.disconnect()
        server.stop()


def test_declared_requirement_releases_the_fence_after_hello(master):
    """整条根因链：声明 ⇒ 本地未开启的客户端发 HELLO ⇒ fence 解除。

    `master` fixture 已经注册并 fence 了 worker（generation 11）。修复前这个
    fence 只在客户端本地开关打开时才会被 hello 解除；现在声明足以驱动协商，
    所以 release 债务（`_send_pipeline_worker_config` 的屏障之后）才有投递机会
    —— 债务本身仍必须等到 identity 全量匹配的合法 ACK 才清除。
    """
    server = master._tcp_server
    assert master._task_worker_control.pending_worker_ids() == {"worker"}

    # The registration edge declares exactly the handshake it fences for.
    server.set_task_worker_hello_required("worker", True)

    worker = LocalOptOutWorker(node_id="worker")
    worker._tcp_client.task_worker_hello_required = True
    assert worker._send_task_worker_hello() is True
    hello_payload = worker._tcp_client.sock.packets[-1]

    master._handle_task_worker_message("worker", {"data": hello_payload})

    assert master._task_worker_control.pending_worker_ids() == set()
    assert master._task_worker_control.worker_snapshot("worker")["accepted"] is True
