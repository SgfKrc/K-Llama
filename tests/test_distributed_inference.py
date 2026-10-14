"""
集成测试 — 分布式推理工作流 + 主节点从节点管理
===============================================
模拟真实推理场景：加载 tiny Qwen2 模型 → 主节点调度 → 推理执行 → 验证结果。
同时覆盖节点注册、删除、注销、扩容等管理功能。

测试依赖: transformers + torch（tiny config，无需 GPU，无需下载模型）
"""

import sys
import os
import json
import logging
import threading
import time
from typing import Optional

import pytest
import torch

# ------------------------------------------------------------------
# setup
# ------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# 加载 API composition root，以配置 SchedulerCallbackSet。
import api_server  # noqa: F401,E402

from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers import AutoTokenizer

from model_module import ModelManager
from scheduler import Scheduler, PipelineQueue, NodeInfo, NodeState, NodeRole

# 用最小的 Qwen2 config，无需真实模型文件
TINY_VOCAB_SIZE = 32000


# ------------------------------------------------------------------
# Mock Tokenizer — 不依赖网络下载
# ------------------------------------------------------------------

class MockTokenizer:
    """最小化 tokenizer: 字符级编码 + decode，不访问 HuggingFace Hub。"""

    def __init__(self):
        self.eos_token_id = 0
        self.pad_token_id = 1
        self.bos_token_id = 2
        self.vocab_size = TINY_VOCAB_SIZE

    def __call__(self, text, return_tensors="pt", **kwargs):
        ids = [min(ord(c) % (TINY_VOCAB_SIZE - 1), TINY_VOCAB_SIZE - 2) + 1
               for c in text]
        if not ids:
            ids = [self.bos_token_id]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        result = []
        for i in ids:
            if skip_special_tokens and i <= self.bos_token_id:
                continue
            result.append(chr(min(i, 127)))
        return "".join(result)

    def encode(self, text, return_tensors="pt", **kwargs):
        ids = [min(ord(c) % (TINY_VOCAB_SIZE - 1), TINY_VOCAB_SIZE - 2) + 1
               for c in text]
        return torch.tensor(ids, dtype=torch.long)


# ------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------


def _make_tiny_model() -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=TINY_VOCAB_SIZE,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_model() -> Qwen2ForCausalLM:
    return _make_tiny_model()


@pytest.fixture
def scheduler():
    return Scheduler()


@pytest.fixture
def mgr(tiny_model):
    """创建 ModelManager 并注入 tiny 模型 + mock tokenizer。"""
    from model_host import model_host as _host

    _host._manager = None
    tokenizer = MockTokenizer()
    mgr = ModelManager()
    mgr.model = tiny_model
    mgr.tokenizer = tokenizer
    mgr._engine_type = "pytorch"
    _host._manager = mgr
    yield mgr
    _host._manager = None


# ================================================================
# 分布式推理工作流测试
# ================================================================


class TestDistributedInferenceWorkflow:
    """
    端到端推理流程：

    chat request → run_pipeline_safe → (回退到) _run_full_model_inference →
    model_manager.chat → 返回 response。
    """

    def test_fallback_inference_produces_response(self, scheduler, mgr):
        """无流水线节点时，回退到全模型推理，应返回非空 response。"""
        scheduler._tcp_server = None  # 确保无 TCP server，走回退
        result = scheduler._run_full_model_inference(
            prompt="Hello, how are you?",
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
        )
        assert isinstance(result, dict)
        assert "response" in result
        assert len(result["response"]) > 0
        assert "error" not in result

    def test_fallback_refuses_partial_artifact(self, scheduler, mgr, monkeypatch):
        """★ #38（2026-10-06）：master 只持有裁层工件时，整模回退必须 fail-closed。

        此前回退只看 `is_pipeline_prepared`（= `_pipeline_distributed_only and
        _pipeline_descriptor`）与 `ensure_full_model()`，二者对 **llama.cpp master 完全无效**
        —— 实测当时加载的是 `qwen25-05b-f16-head8.gguf`
        （日志 `llama.cpp 层段已加载: [0,8) embed=True lm_head=False`），LlamaCppEngine
        **既无 `ensure_full_model` 也无 `layer_range` 实例属性** ⇒ 回退按「全层」硬跑 ⇒
        产出垃圾文本**却返回 HTTP 200**（fail-open 最坏的一种）。
        现改用描述符里的**裁层痕迹**作判据（`assignment_layer_range` /
        `partial_assignment` / `loaded_artifact` / `lm_head=False`），两侧通用。
        """
        # 不真的重载整模（那要 14s+）；本用例只验「有裁层痕迹 ⇒ 拒绝」这一步
        monkeypatch.setattr(mgr, "ensure_full_model", lambda *a, **k: None, raising=False)
        monkeypatch.setattr(
            mgr, "get_pipeline_descriptor",
            lambda: {"lm_head": False, "assignment_layer_range": [0, 8]},
            raising=False,
        )
        result = scheduler._run_full_model_inference(
            prompt="Hello, how are you?",
            max_new_tokens=8,
        )
        assert result["response"] == ""
        assert "refusing_full_model_fallback_with_partial_artifact" in result["error"]

    def test_stream_fallback_refuses_partial_artifact(self, scheduler, mgr, monkeypatch):
        """#38: streaming fallback must reject a partial GGUF too."""
        monkeypatch.setattr(
            mgr,
            "get_pipeline_descriptor",
            lambda: {"lm_head": False, "assignment_layer_range": [0, 8]},
            raising=False,
        )
        events = list(
            scheduler._run_full_model_inference_stream(
                prompt="Hello, how are you?",
                max_new_tokens=8,
            )
        )
        assert events
        assert events[-1]["done"] is True
        assert "refusing_full_model_fallback_with_partial_artifact" in events[-1]["error"]

    def test_fallback_inference_with_metrics(self, scheduler, mgr):
        """回退推理应返回 metrics（包含 mode 字段）。"""
        result = scheduler._run_full_model_inference(
            prompt="What is AI?",
            max_new_tokens=16,
        )
        assert "metrics" in result
        assert "mode" in result["metrics"]

    def test_fallback_inference_empty_prompt(self, scheduler, mgr):
        """空 prompt 也应安全返回。"""
        result = scheduler._run_full_model_inference(
            prompt="",
            max_new_tokens=8,
        )
        assert "response" in result
        assert "error" not in result

    def test_fallback_inference_streaming_callback(self, scheduler, mgr):
        """流式回调应逐 token 推送并接收 done 信号。"""
        tokens = []
        done = []

        def cb(event):
            if "token" in event:
                tokens.append(event["token"])
            if "done" in event:
                done.append(event)

        scheduler._run_full_model_inference(
            prompt="Hello",
            max_new_tokens=16,
            _stream_callback=cb,
        )
        assert len(tokens) > 0
        assert len(done) == 1

    def test_run_pipeline_safe_fallback(self, scheduler, mgr):
        """run_pipeline_safe 在无节点时走回退，应成功返回。"""
        scheduler._tcp_server = None
        result = scheduler.run_pipeline_safe(
            prompt="Tell me a joke",
            max_new_tokens=32,
        )
        assert isinstance(result, dict)
        assert "response" in result or "error" in result

    def test_queue_worker_inference(self, scheduler, mgr):
        """队列 worker → _process_queued_pipeline_task → 推理 → 返回结果。"""
        scheduler._tcp_server = None
        queue = scheduler.pipeline_queue
        queue.start(process_fn=scheduler._process_queued_pipeline_task)
        try:
            tid = queue.enqueue(prompt="Hello world", max_new_tokens=16)
            result = queue.wait_for_result(tid, timeout=30)
            assert result["status"] == "done"
            assert "response" in result["result"]
            assert len(result["result"]["response"]) > 0
        finally:
            queue.stop()

    def test_multiple_queued_requests(self, scheduler, mgr):
        """连续入队多个请求，每个都应完成。"""
        scheduler._tcp_server = None
        queue = scheduler.pipeline_queue
        queue.start(process_fn=scheduler._process_queued_pipeline_task)
        tids = []
        try:
            for i in range(3):
                tids.append(queue.enqueue(
                    prompt="Quick test",
                    max_new_tokens=8 + i * 4,
                ))
            for tid in tids:
                r = queue.wait_for_result(tid, timeout=30)
                assert r["status"] == "done", f"{tid}: {r}"
                assert "response" in r["result"]
        finally:
            queue.stop()


# ================================================================
# 主节点管理功能测试
# ================================================================


class TestMasterNodeManagement:
    """测试节点注册、注销、删除、扩容等主节点管理功能。"""

    _registered_ids: set = set()

    @pytest.fixture(autouse=True)
    def _node_cleanup(self, scheduler):
        """测试后清理本测试类注册的节点（内存 + 数据库）。"""
        # 测试前快照
        before = set(scheduler.nodes.keys())
        yield
        # 测试后：清理新增节点
        after = set(scheduler.nodes.keys())
        new_ids = after - before
        for nid in new_ids:
            if nid != "master":
                try:
                    del scheduler.nodes[nid]
                except Exception:
                    pass
        if new_ids:
            import logging
            logging.getLogger(__name__).info(f"清理了 {len(new_ids)} 个测试节点: {new_ids}")

    def test_register_node_tcp(self, scheduler):
        """通过 register_node() 注册 TCP 从节点。"""
        ok = scheduler.register_node(
            node_id="client_tcp",
            role="client",
            address="10.0.0.2:8888",
            hostname="worker-pc",
            node_type="pc",
        )
        assert ok
        node = scheduler.nodes["client_tcp"]
        assert node.role == "client"
        assert node.node_type == "pc"
        assert node.state == NodeState.ONLINE  # TCP 注册默认在线

    def test_manual_register_android_node(self, scheduler):
        """手动注册 Android 薄客户端节点。"""
        result = scheduler.manual_register_node(
            "android-001",
            hostname="Pixel 7",
            address="",
            network_type="wifi",
            node_type="android",
        )
        assert result["status"] == "registered"
        node = scheduler.nodes["android-001"]
        assert node.node_type == "android"
        assert node.state == NodeState.OFFLINE

    def test_manual_register_duplicate_updates_metadata(self, scheduler):
        """重复注册同一 Android 节点应更新主机名等字段。"""
        scheduler.manual_register_node(
            "android-002", hostname="Old", node_type="android",
        )
        result = scheduler.manual_register_node(
            "android-002", hostname="New Name", node_type="android",
        )
        assert result["status"] == "updated"
        assert scheduler.nodes["android-002"].hostname == "New Name"

    def test_delete_offline_android_node(self, scheduler, monkeypatch):
        """删除离线 Android 节点应从内存移除。"""
        scheduler.manual_register_node(
            "android-del", hostname="ToDelete", node_type="android",
        )
        pushed = []
        scheduler._push_node_update_to_all_clients = lambda *a: pushed.append(a)

        result = scheduler.delete_node("android-del")
        assert result["status"] == "deleted"
        assert "android-del" not in scheduler.nodes

    def test_delete_master_rejected(self, scheduler):
        """不能删除 master 节点。"""
        assert scheduler.delete_node("master")["status"] == "invalid"

    def test_delete_online_node_rejected(self, scheduler):
        """在线节点不能直接删除，需先注销。"""
        scheduler.nodes["client_online"] = NodeInfo(
            node_id="client_online", role="client", state=NodeState.ONLINE,
        )
        assert scheduler.delete_node("client_online")["status"] == "online"

    def test_delete_missing_node(self, scheduler):
        """不存在的节点返回 not_found。"""
        assert scheduler.delete_node("does-not-exist")["status"] == "not_found"

    def test_deregister_marks_offline_but_keeps_record(self, scheduler):
        """注销节点应标记 offline 但保留在 nodes 中。"""
        scheduler.nodes["client_x"] = NodeInfo(
            node_id="client_x", role="client", state=NodeState.ONLINE,
            address="10.0.0.3:8888",
        )
        ok = scheduler.deregister_node("client_x")
        assert ok
        node = scheduler.nodes["client_x"]
        assert node.state == NodeState.OFFLINE
        assert node.address == ""

    def test_update_max_nodes_expands_capacity(self, scheduler):
        """扩容后 _max_nodes 应更新。"""
        old = scheduler._max_nodes
        result = scheduler.update_max_nodes(old + 2)
        assert result["status"] == "ok"
        assert scheduler._max_nodes == old + 2
        assert result["max_nodes"] == old + 2

    def test_update_max_nodes_shrink(self, scheduler):
        """缩容不应丢失在线/已注册节点。"""
        scheduler.update_max_nodes(5)
        scheduler.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
            address="10.0.0.1:8888",
        )
        result = scheduler.update_max_nodes(2)
        assert "client1" in scheduler.nodes  # 在线节点不受缩容影响
        assert scheduler._max_nodes == 2

    def test_get_nodes_lists_all(self, scheduler):
        """get_nodes 应返回所有节点。"""
        scheduler.nodes["master"] = NodeInfo(
            node_id="master", role="master", state=NodeState.ONLINE,
        )
        scheduler.register_node("n1", "client", address="1.2.3.4:8888")
        scheduler.manual_register_node("n2", hostname="test", node_type="android")
        nodes = scheduler.get_nodes()
        ids = {n["node_id"] for n in nodes}
        assert "master" in ids
        assert "n1" in ids
        assert "n2" in ids

    def test_get_config_reflects_max_nodes(self, scheduler):
        """get_config 应返回当前 max_nodes（非硬编码值）。"""
        scheduler.update_max_nodes(5)
        config = scheduler.get_config()
        assert config["max_nodes"] == 5

    def test_update_max_nodes_cleans_phantom_nodes(self, scheduler):
        """扩容/缩容时应清理幽灵节点（无地址、无主机名、从未连接）。"""
        # 模拟旧代码创建的幽灵节点
        scheduler.nodes["client3"] = NodeInfo(
            node_id="client3", role="client", state=NodeState.OFFLINE,
        )
        scheduler.nodes["client4"] = NodeInfo(
            node_id="client4", role="client", state=NodeState.OFFLINE,
        )
        assert "client3" in scheduler.nodes
        assert "client4" in scheduler.nodes

        result = scheduler.update_max_nodes(scheduler._max_nodes + 2)
        assert result["status"] == "ok"
        assert "client3" not in scheduler.nodes, "幽灵节点应被清理"
        assert "client4" not in scheduler.nodes, "幽灵节点应被清理"

    def test_phantom_cleanup_resets_the_entire_assignment_control_plane(
            self, scheduler, monkeypatch):
        scheduler.nodes["phantom"] = NodeInfo(
            node_id="phantom", role="client", state=NodeState.OFFLINE,
        )
        scheduler._worker_assignments.begin(
            "worker",
            expected_config={"config_id": "cfg-old", "generation": 7},
        )
        scheduler._layer_config_retry_state["worker"] = {
            "attempts": 2, "next_retry": 10.0,
        }
        scheduler._pipeline_load_transaction = {
            "config_id": "cfg-old",
            "phase": "ready",
            "plan": {
                "admitted": True,
                "plan_id": "plan-old",
                "assignments": [{"node_id": "worker"}],
            },
        }
        scheduler._active_pipeline_capacity_plan = {
            "admitted": True,
            "plan_id": "plan-old",
            "assignments": [{"node_id": "worker"}],
        }
        monkeypatch.setattr(scheduler, "_get_total_model_layers", lambda: 24)
        monkeypatch.setattr(scheduler, "compute_layer_assignment", lambda: [])

        scheduler.update_max_nodes(scheduler._max_nodes + 1)

        assert scheduler._worker_assignments.snapshot() == {}
        assert scheduler._layer_config_retry_state == {}
        assert scheduler._pipeline_load_transaction is None
        assert scheduler._active_pipeline_capacity_plan is None
        assignments = scheduler.get_layer_assignments()
        assert assignments["strategy"] != "capacity"
        assert assignments.get("plan_id") != "plan-old"

    def test_update_max_nodes_preserves_real_nodes(self, scheduler):
        """幽灵清理不应删除有连接记录的真实节点。"""
        # 真实 TCP 注册节点（有 address）
        scheduler.nodes["real_tcp"] = NodeInfo(
            node_id="real_tcp", role="client", state=NodeState.OFFLINE,
            address="10.0.0.5:8888",
        )
        # 真实手动注册节点（有 hostname）
        scheduler.nodes["real_android"] = NodeInfo(
            node_id="real_android", role="client", state=NodeState.OFFLINE,
            hostname="Samsung S21", node_type="android",
        )
        # 曾经上线过的节点（有 connected_at）
        scheduler.nodes["real_was_online"] = NodeInfo(
            node_id="real_was_online", role="client", state=NodeState.OFFLINE,
            connected_at=1700000000.0,
        )

        scheduler.update_max_nodes(scheduler._max_nodes + 3)
        assert "real_tcp" in scheduler.nodes
        assert "real_android" in scheduler.nodes
        assert "real_was_online" in scheduler.nodes

    def test_get_nodes_includes_node_type(self, scheduler):
        """get_nodes 返回的节点信息应包含 node_type 字段。"""
        scheduler.nodes["master"] = NodeInfo(
            node_id="master", role="master", state=NodeState.ONLINE,
        )
        scheduler.register_node("pc_node", "client", address="10.0.0.1:8888",
                                hostname="desktop", node_type="pc")
        scheduler.manual_register_node("android_node", hostname="Pixel",
                                       node_type="android")
        nodes = scheduler.get_nodes()
        by_id = {n["node_id"]: n for n in nodes}
        assert by_id["master"]["node_type"] == "pc"
        assert by_id["pc_node"]["node_type"] == "pc"
        assert by_id["android_node"]["node_type"] == "android"

    def test_manual_register_updates_node_type(self, scheduler):
        """重复手动注册应可更新 node_type。"""
        scheduler.manual_register_node("multi-type", hostname="Device",
                                       node_type="pc")
        assert scheduler.nodes["multi-type"].node_type == "pc"

        result = scheduler.manual_register_node("multi-type", hostname="Device",
                                                node_type="android")
        assert result["status"] == "updated"
        assert scheduler.nodes["multi-type"].node_type == "android"

    def test_update_max_nodes_expand_shrink_expand_no_phantoms(self, scheduler):
        """扩容→缩容→再扩容不应引入幽灵节点。"""
        scheduler.update_max_nodes(10)
        assert scheduler._max_nodes == 10
        phantom_count = sum(
            1 for n in scheduler.nodes.values()
            if n.role != "master" and not n.address
            and not n.hostname and not n.connected_at
        )
        assert phantom_count == 0, "扩容不应创建幽灵节点"

        scheduler.update_max_nodes(6)
        assert scheduler._max_nodes == 6

        scheduler.update_max_nodes(10)
        assert scheduler._max_nodes == 10
        phantom_count = sum(
            1 for n in scheduler.nodes.values()
            if n.role != "master" and not n.address
            and not n.hostname and not n.connected_at
        )
        assert phantom_count == 0, "缩容→再扩容不应创建幽灵节点"


# ================================================================
# 队列管理功能测试
# ================================================================


class TestQueueManagement:
    """测试 PipelineQueue 的 pause/resume/clear/cancel 操作。"""

    @pytest.fixture
    def queue(self):
        q = PipelineQueue(max_size=10, result_ttl=10)
        return q

    def test_pause_rejects_new_enqueue(self, queue):
        """暂停后入队应抛 RuntimeError。"""
        queue.pause()
        with pytest.raises(RuntimeError, match="已暂停"):
            queue.enqueue(prompt="test")

    def test_pause_resume_cycle(self, queue):
        """暂停后恢复应能继续入队。"""
        queue.pause()
        with pytest.raises(RuntimeError):
            queue.enqueue(prompt="blocked")
        queue.resume()
        tid = queue.enqueue(prompt="ok")
        assert tid

    def test_enqueue_does_not_duplicate_explicit_generation_arguments(self, queue):
        """QueueTask 的显式字段不能再次残留在 _extra_kwargs。"""
        tid = queue.enqueue(
            prompt="short", max_new_tokens=64, request_id="req-1",
        )
        task = next(task for task in queue._q0 if task.task_id == tid)
        assert "max_new_tokens" not in task._extra_kwargs
        assert "request_id" not in task._extra_kwargs
        assert task.max_new_tokens == 64
        assert task.request_id == "req-1"

    def test_clear_cancels_all_queued(self, queue):
        """clear 应取消所有排队任务。"""
        tids = [queue.enqueue(prompt=f"task{i}") for i in range(5)]
        count = queue.clear()
        assert count == 5
        for tid in tids:
            r = queue.wait_for_result(tid, timeout=1)
            assert r["status"] == "cancelled"

    def test_cancel_single_task(self, queue):
        """cancel_task 应取消指定排队任务。"""
        tid = queue.enqueue(prompt="will be cancelled")
        ok = queue.cancel_task(tid)
        assert ok
        r = queue.wait_for_result(tid, timeout=1)
        assert r["status"] == "cancelled"

    def test_cancel_running_task_signals_cooperative_cancellation(self, queue):
        """执行中任务应收到协作取消信号并立即唤醒等待方。"""
        import threading

        started = threading.Event()
        finish = threading.Event()
        seen_cancel_event = []

        def slow(**kw):
            seen_cancel_event.append(kw.get("_cancel_event"))
            started.set()
            finish.wait(5)
            return {"response": "ok"}

        queue.start(process_fn=slow)
        tid = queue.enqueue(prompt="running")
        started.wait(2)
        assert queue.cancel_task(tid)
        result = queue.wait_for_result(tid, timeout=1)
        assert result["status"] == "cancelled"
        assert seen_cancel_event and seen_cancel_event[0].is_set()
        finish.set()
        queue.stop()

    def test_get_queue_detail_includes_aging_params(self, queue):
        """get_queue_detail 应包含老化参数和抢占统计。"""
        detail = queue.get_queue_detail()
        assert "aging_params" in detail
        assert "q0_max_tokens" in detail["aging_params"]
        assert "q1_max_tokens" in detail["aging_params"]
        assert "preempt_stats" in detail


# ================================================================
# 调度策略测试
# ================================================================


class TestSchedulingStrategy:
    """MLFQ / FIFO 策略切换和老化测试。"""

    @pytest.fixture
    def queue(self):
        return PipelineQueue(max_size=20, result_ttl=10, strategy="mlfq")

    def test_mlfq_classifies_short_as_q0(self, queue):
        """短任务 (≤128 tokens) 应进入 Q0。"""
        assert queue._classify(64) == 0
        assert queue._classify(128) == 0

    def test_mlfq_classifies_medium_as_q1(self, queue):
        """中等任务 (≤512 tokens) 应进入 Q1。"""
        assert queue._classify(256) == 1
        assert queue._classify(512) == 1

    def test_mlfq_classifies_long_as_q2(self, queue):
        """长任务 (>512 tokens) 应进入 Q2。"""
        assert queue._classify(1024) == 2

    def test_fifo_all_to_q1(self, queue):
        """FIFO 模式下所有任务应进入 Q1 保持入队顺序。"""
        queue.set_strategy("fifo")
        assert queue._classify(64) == 1
        assert queue._classify(1024) == 1

    def test_set_invalid_strategy_raises(self, queue):
        """无效策略应抛 ValueError。"""
        with pytest.raises(ValueError):
            queue.set_strategy("invalid")

    def test_aging_q2_to_q1(self, queue):
        """Q2 任务等待超时应自动提升到 Q1。"""
        old = queue._aging_q2_to_q1
        queue._aging_q2_to_q1 = 0  # 立即触发老化
        queue.enqueue(prompt="long", max_new_tokens=600)
        task = queue._get_next_task()
        assert task is not None
        assert task.original_level == 2  # 原始级别为 Q2
        queue._aging_q2_to_q1 = old

    def test_absolute_aging_max_wait(self, queue):
        """超绝对上限的任务应被置顶 Q0。"""
        old = queue._aging_max_wait
        queue._aging_max_wait = 0
        queue.enqueue(prompt="long", max_new_tokens=600)
        task = queue._schedule_next()
        assert task is not None
        # 老化后任务级别变化
        queue._aging_max_wait = old


# ================================================================
# Layer assignment management (v0.1.7)
# ================================================================

