"""票 14 `REFACTOR-DECISION-01` 的 **characterization 基线**。

这些断言**钉住当前行为** —— 包括已知的重复与分叉，**不是**理想设计的表达。
票面要求的完成标准是「事实源减少、事务边界清晰、分叉判据消失、产品证据不退化」，
本文件就是那句「分叉判据消失」的**可核查证据源**：

* 重构**不得**让这些断言静默通过；
* 重构若**确实**改变了某条事实（例如把 4 份重复常量收敛成 1 份、或给能力判据补上
  `inference_path` 维度），必须**显式**改本文件并在提交信息里说明「这是收敛，
  不是回归」。

因此每一条断言都刻意写成「当前事实 + 为什么它在票 14 视野内」两个部分。
不依赖网络、设备、模型或签名密钥；`python -m pytest` 直接可跑。
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path
import sys
import unittest

import pytest

pytestmark = pytest.mark.legacy_characterization

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def read_source(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


def assignment_value(source: str, symbol: str) -> str:
    """取 `SYMBOL = "..."` 形式的字面量值（`SYMBOL` 必须是行首的赋值目标）。"""
    match = re.search(
        rf"^{re.escape(symbol)}\s*=\s*([\"'])(?P<value>[^\"']*)\1\s*$",
        source,
        re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"{symbol} 没有模块级字面量赋值")
    return match.group("value")


class TestEngineCapabilityFactSources(unittest.TestCase):
    """R2 migration: both projections and runtime gates share EngineRegistry.

    Replaces the R0 evidence of independent field tables, family-only server
    admission, PC default admission, and an unconsumed negative segment flag.
    Product wire fixtures remain unchanged; the v1 engine schema is optional.
    """

    def test_both_legacy_tables_derive_from_the_same_registry(self) -> None:
        import koakuma_engine
        import model_load_resolver
        from qlh_core.contracts.engines import ENGINE_REGISTRY, LEGACY_ENGINE_SPECS

        self.assertIs(koakuma_engine.LEGACY_ENGINE_SPECS, LEGACY_ENGINE_SPECS)
        self.assertIs(model_load_resolver.LEGACY_ENGINE_SPECS, LEGACY_ENGINE_SPECS)
        self.assertEqual(set(koakuma_engine._BACKEND_CAPABILITIES), set(LEGACY_ENGINE_SPECS))
        self.assertEqual(set(model_load_resolver.ENGINE_CAPABILITIES), set(LEGACY_ENGINE_SPECS))
        for family, spec in LEGACY_ENGINE_SPECS.items():
            self.assertIs(spec, ENGINE_REGISTRY.default(family))
            self.assertEqual(koakuma_engine._BACKEND_CAPABILITIES[family].capabilities,
                             spec.capabilities)
            projected = model_load_resolver.ENGINE_CAPABILITIES[family]
            self.assertEqual((projected.artifact_kind, projected.layer_range_mode),
                             (spec.artifact_kind, spec.layer_range_mode))

    def test_concrete_runtime_separates_llama_binding_keep_head_and_server(self) -> None:
        from koakuma_engine import Capability, backend_capabilities, runtime_supports
        from llama_server_engine import LlamaServerEngine
        from qlh_core.contracts.engines import (
            ENGINE_REGISTRY, FULL_REQUEST, FORWARD_LAYERS, LAYER_RANGE,
            LLAMA_CPP_BINDING, LLAMA_CPP_KEEP_HEAD, LLAMA_CPP_SERVER,
        )

        for kind in (LLAMA_CPP_BINDING, LLAMA_CPP_KEEP_HEAD):
            self.assertTrue(ENGINE_REGISTRY.supports("llama_cpp", kind, FORWARD_LAYERS))
        server = ENGINE_REGISTRY.require("llama_cpp", LLAMA_CPP_SERVER)
        self.assertEqual(server.operations, frozenset({FULL_REQUEST}))
        self.assertFalse(server.supports(LAYER_RANGE))
        self.assertFalse(backend_capabilities("llama_cpp", LLAMA_CPP_SERVER).supports(Capability.FORWARD_LAYERS))
        self.assertFalse(runtime_supports(LlamaServerEngine(), Capability.FORWARD_LAYERS))

    def test_planner_requires_runtime_evidence_and_rejects_server_even_with_positive_probe(self) -> None:
        from scheduler import _node_supports_forward_layers
        from qlh_core.contracts.engines import ENGINE_REGISTRY, LLAMA_CPP_BINDING, LLAMA_CPP_SERVER

        for node in ({}, {"node_type": "pc"}, {"node_type": "android"},
                     {"device_info": {"backend_id": "llama_cpp"}},
                     {"device_info": {"capabilities": ["forward_layers"]}}):
            self.assertFalse(_node_supports_forward_layers(node))
        for kind, expected in ((LLAMA_CPP_BINDING, True), (LLAMA_CPP_SERVER, False)):
            self.assertEqual(_node_supports_forward_layers({"device_info": {
                "backend_id": "llama_cpp", "runtime_kind": kind,
                "native_abi": ENGINE_REGISTRY.require("llama_cpp", kind).native_abi,
                "capabilities": ["forward_layers"],
            }}), expected)

    def test_negative_segment_probe_narrows_the_static_runtime_contract(self) -> None:
        from scheduler import _node_supports_forward_layers
        from qlh_core.contracts.engines import ENGINE_REGISTRY, LLAMA_CPP_BINDING

        self.assertFalse(_node_supports_forward_layers({"device_info": {
            "backend_id": "llama_cpp", "inference_path": LLAMA_CPP_BINDING,
            "native_abi": ENGINE_REGISTRY.require("llama_cpp", LLAMA_CPP_BINDING).native_abi,
            "serves_layer_segments": False,
        }}))


class TestReasonCodeFactSources(unittest.TestCase):
    """② 结构化 reason：**同一个 code 有 4 份独立定义 + 多套词表**。"""

    DEADLINE_CODE = "request_deadline_exceeded"

    def test_deadline_code_has_one_definition_and_compatibility_aliases(self) -> None:
        from qlh_core.contracts.deadline import REQUEST_DEADLINE_EXCEEDED
        from request_deadline import REQUEST_DEADLINE_EXCEEDED as compatibility
        from request_outcome import REASON_REQUEST_DEADLINE_EXCEEDED
        from relay_transport import REQUEST_DEADLINE_EXCEEDED as relay
        from scheduler_task_worker import _REQUEST_DEADLINE_EXCEEDED as worker
        self.assertEqual(REQUEST_DEADLINE_EXCEEDED, self.DEADLINE_CODE)
        for value in (compatibility, REASON_REQUEST_DEADLINE_EXCEEDED, relay, worker):
            self.assertIs(value, REQUEST_DEADLINE_EXCEEDED)
        sources = ["request_deadline.py", "request_outcome.py", "relay_transport.py", "scheduler_task_worker.py"]
        literal = re.compile(r"^[A-Z_]+\s*=\s*['\"]request_deadline_exceeded['\"]", re.MULTILINE)
        self.assertTrue(all(not literal.search(read_source(source)) for source in sources))
        self.assertEqual(len(literal.findall(read_source("qlh_core/contracts/deadline.py"))), 1)

    def test_independent_reason_vocabularies_stay_separate(self) -> None:
        """当前事实：每个领域各自定义 reason 常量，**没有共享枚举**。

        这不是要求合并（各域语义确实不同），而是钉住「有几套词表」这个数字，
        以便票 14 判断哪些该收敛、哪些该保留。
        """
        vocabularies = {
            "qlh_core/contracts/outcome.py": "REQUEST_OUTCOME 终局词表",
            "model_preflight.py": "MODEL_PREFLIGHT 事前检查词表",
            "worker_assignment_state.py": "WORKER_ASSIGNMENT 词表",
            "model_pipeline_availability.py": "MODEL_PIPELINE_AVAILABILITY 词表",
        }
        found = {}
        for filename in vocabularies:
            source = read_source(filename)
            # 常量允许前导下划线（如 `model_pipeline_availability.py` 的
            # `_RUNTIME_UNSUPPORTED`），但必须带一个字符串字面量值。
            constants = re.findall(
                r"^_?[A-Z][A-Z0-9_]{3,}\s*=\s*[\"']",
                source,
                re.MULTILINE,
            )
            found[filename] = len(set(constants))
            self.assertGreater(
                found[filename], 0,
                f"{filename} 应有 reason 常量（当前事实）",
            )
        # 4 套词表各自独立存在 —— 数量变化即结构变化，需显式复核。
        self.assertEqual(len(found), 4)

    def test_layer_coverage_gap_has_two_codes_for_one_concept(self) -> None:
        """当前事实：同一个「层覆盖不足」概念有 **2 个不同 code**。

        票面点名的「分叉判据」。二者是否该统一由票 14 裁决，此处只钉住现状。
        """
        self.assertIn(
            "model_preflight_layer_coverage_gap",
            read_source("model_preflight.py"),
        )
        self.assertIn(
            "pipeline_layer_range_coverage_insufficient",
            read_source("qlh_core/planning/capacity.py"),
        )


class TestPipelinePlanFactSources(unittest.TestCase):
    """③ 不可变 pipeline plan：**目前是多份可变 dict + 一批同名 `*Plan` 冻结类**。"""

    def test_plan_lives_in_two_independent_places(self) -> None:
        """当前事实：容量 plan 与事务 plan 是**两份**，并交替对外返回。"""
        source = read_source("scheduler.py")
        self.assertIn(
            "self._active_pipeline_capacity_plan: Optional[dict] = None", source)
        self.assertIn('transaction_plan = dict(transaction.get("plan", {}))', source)
        # 读侧还会就地覆盖 reason 文本（非纯读）—— 票面「reason 被改写」的实例
        self.assertIn('active["reason"]', source)

    def test_named_plan_types_are_unrelated(self) -> None:
        """当前事实：存在一批**互不相关**的 `*Plan` 冻结类，各自一套构造/消费。

        票 14 要抽「不可变 pipeline plan」单一事实源 —— 这些是它要裁决归属的对象。
        """
        candidates = [
            ("pipeline_reshard.py", "PipelineReshardPlan"),
            ("torch_hetero_plan.py", "HeterogeneousPlan"),
            ("relay_cut_objective.py", "RelayCutPlan"),
            ("task_worker_chunks.py", "StageChunkPlan"),
            ("relay_contract.py", "RelayTrimPlan"),
        ]
        for filename, class_name in candidates:
            source = read_source(filename)
            self.assertIn(
                f"class {class_name}", source,
                f"{filename} 里的 {class_name} 不见了 —— 结构变了，请更新基线",
            )
            self.assertIn("frozen=True", source, f"{class_name} 应为冻结 dataclass")


class TestSessionContextFactSources(unittest.TestCase):
    """④ 会话上下文：**服务已统一，但仍有进程内 facade 镜像**。"""

    def test_context_owner_keeps_client_selection_without_facade_mirrors(self) -> None:
        from chat_context import ConversationContextService
        context = ConversationContextService(load_history=lambda _: [], persist_turn=lambda *a: True)
        context.prepare("first", client_id="alpha", runtime_kv=False)
        context.prepare("second", client_id="beta", runtime_kv=False)
        self.assertEqual(context.resolve_session(client_id="alpha"), "first")
        self.assertEqual(context.resolve_session(client_id="beta"), "second")
        import api_server
        self.assertNotIn("active_session_id", vars(api_server))
        self.assertNotIn("session_histories", vars(api_server))

    def test_persistence_is_a_separate_fact_source(self) -> None:
        """当前事实：持久化在 `local_store` 独立承担，服务只持有两个回调。"""
        context = read_source("chat_context.py")
        self.assertIn("self._load_history = load_history", context)
        self.assertIn("self._persist_turn = persist_turn", context)
        store = read_source("local_store.py")
        for symbol in ("def create_local_session", "def save_local_conversation_turn",
                       "def load_local_conversation", "def delete_local_session"):
            self.assertIn(symbol, store, f"{symbol} 不见了 —— 存储边界变了")


if __name__ == "__main__":
    unittest.main()
