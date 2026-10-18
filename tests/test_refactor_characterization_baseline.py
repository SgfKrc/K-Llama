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
    """① `engine × runtime capability`：**两张同键不同字段的表 + 判据分叉 + 死字段**。"""

    def test_two_tables_share_keys_but_no_fields(self) -> None:
        """当前事实：两张表都用 `BackendId` 作键，但**字段完全不重叠**。

        票 14 的目标之一是抽出一份声明式的 `engine × runtime capability` 事实源 ——
        这就是它要合并的两个候选。
        """
        from koakuma_engine import _BACKEND_CAPABILITIES
        from model_load_resolver import ENGINE_CAPABILITIES

        self.assertEqual(
            set(_BACKEND_CAPABILITIES), set(ENGINE_CAPABILITIES),
            "两张表若键集不同，说明又多了第三处定义",
        )
        capability_fields = {f.name for f in dataclasses.fields(
            next(iter(_BACKEND_CAPABILITIES.values())))}
        resolver_fields = {f.name for f in dataclasses.fields(
            next(iter(ENGINE_CAPABILITIES.values())))}
        self.assertEqual(
            capability_fields & resolver_fields, set(),
            "字段一旦重叠，两张表就该合并了 —— 这是票 14 的触发信号",
        )

    def test_llama_cpp_declares_forward_layers_but_island_does_not(self) -> None:
        """当前事实：能力位表里 `llama_cpp` 与 `pytorch` 有 `FORWARD_LAYERS`，`island` 没有。

        （19 节里 python 的 `c in set` 判据是 3.11 语义；此处用等价显式写法。）
        """
        from koakuma_engine import (
            Capability,
            _BACKEND_CAPABILITIES,
            BackendId,
        )

        self.assertIn(Capability.FORWARD_LAYERS, _BACKEND_CAPABILITIES[BackendId.LLAMA_CPP].capabilities)
        self.assertIn(Capability.FORWARD_LAYERS, _BACKEND_CAPABILITIES[BackendId.PYTORCH].capabilities)
        self.assertNotIn(Capability.FORWARD_LAYERS, _BACKEND_CAPABILITIES[BackendId.ISLAND].capabilities)

    def test_node_side_judgement_ignores_the_runtime_path_dimension(self) -> None:
        """★ 分叉实证：同一个 `llama_cpp` 家族，节点侧按 `backend_id` 判 ⇒ True。

        节点侧完全没有 `inference_path` 维度，所以一台**跑 `llama-server`** 的机器
        在节点侧会被判「能承接层段」，而在本机侧 `LlamaServerEngine.supports(
        FORWARD_LAYERS)` 是 False。**两处结论相反**，这就是票面要消灭的「分叉判据」。
        """
        from koakuma_engine import Capability
        from scheduler import _node_supports_forward_layers

        # 1) 自报能力最权威
        self.assertTrue(_node_supports_forward_layers(
            {"node_type": "pc", "device_info": {"capabilities": [Capability.FORWARD_LAYERS]}}))
        # 2) 只给 backend_id 就按白名单判 —— server 路径在这里被误判为「可层段」
        self.assertTrue(_node_supports_forward_layers(
            {"node_type": "android", "device_info": {"backend_id": "llama_cpp"}}))
        self.assertTrue(_node_supports_forward_layers(
            {"node_type": "android", "device_info": {"engine": "llama_cpp"}}))
        # 3) 兜底语义：未上报能力时，仅 pc 允许（旧行为，保守向后兼容）
        self.assertTrue(_node_supports_forward_layers({"node_type": "pc"}))
        self.assertFalse(_node_supports_forward_layers({"node_type": "android"}))
        # `node_type` 缺省即 pc
        self.assertTrue(_node_supports_forward_layers({}))

    def test_serves_layer_segments_has_no_consumer(self) -> None:
        """★ 死字段实证：`serves_layer_segments` 只有定义、没有任何读取者。

        票 15a 新增的这个能力位**尚未接线** —— 调度仍走 `stage_types`。票 14 若要
        抽「单一能力事实源」，必须先把这条接上或删掉，否则抽出来的是空壳。
        """
        occurrences: list[str] = []
        for path in sorted(SRC.rglob("*.py")):
            for number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), 1):
                if "serves_layer_segments" not in line:
                    continue
                occurrences.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")

        self.assertEqual(
            len(occurrences), 4,
            "定义/注释数量变了，说明这个字段被接线或被删除 —— 请更新本条基线：\n"
            + "\n".join(occurrences),
        )
        # 4 处全是「写」（定义 + 注释）与「读自己」；没有任何跨模块消费者。
        writers = [item for item in occurrences if re.search(r"[:=]\s*(self\.path ==|\{)", item)]
        self.assertGreaterEqual(len(writers), 1)


class TestReasonCodeFactSources(unittest.TestCase):
    """② 结构化 reason：**同一个 code 有 4 份独立定义 + 多套词表**。"""

    DEADLINE_CODE = "request_deadline_exceeded"

    def test_the_same_deadline_code_is_defined_four_times(self) -> None:
        """★ 最刺眼的一处：一个 code 值有 **4 份模块级常量定义**。

        票面要求「不允许在 API、scheduler、solver、worker 各层重写真因」——
        这 4 份就是反例；票 14 应收敛成单一定义并在各层 import。
        """
        definitions = {
            "request_deadline.py": "REQUEST_DEADLINE_EXCEEDED",
            "request_outcome.py": "REASON_REQUEST_DEADLINE_EXCEEDED",
            "relay_transport.py": "REQUEST_DEADLINE_EXCEEDED",
            "scheduler_task_worker.py": "_REQUEST_DEADLINE_EXCEEDED",
        }
        values = {}
        for filename, symbol in definitions.items():
            values[f"{filename}:{symbol}"] = assignment_value(
                read_source(filename), symbol)
        self.assertEqual(
            set(values.values()), {self.DEADLINE_CODE},
            f"4 份定义的值必须一致（当前事实），实际：{values}",
        )
        self.assertEqual(len(values), 4, "定义份数变了，请更新本条基线")

    def test_independent_reason_vocabularies_stay_separate(self) -> None:
        """当前事实：每个领域各自定义 reason 常量，**没有共享枚举**。

        这不是要求合并（各域语义确实不同），而是钉住「有几套词表」这个数字，
        以便票 14 判断哪些该收敛、哪些该保留。
        """
        vocabularies = {
            "request_outcome.py": "REQUEST_OUTCOME 终局词表",
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
            read_source("pipeline_capacity.py"),
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

    def test_service_is_single_but_facade_duplicates_it(self) -> None:
        """当前事实：`ConversationContextService` 是唯一实现，但 `api_server` 又建了一层
        模块级 facade（`active_session_id` / `session_histories` 两个名字指向同一对象）。

        票 14 若抽「会话上下文服务」，这层 facade 是「事务边界」要收紧的对象。
        """
        context = read_source("chat_context.py")
        self.assertIn("class ConversationContextService", context)
        self.assertIn("self.active_session_id: Optional[str] = None", context)
        self.assertIn("self.histories: MutableMapping[str, list[dict]] = {}", context)

        api = read_source("api_server.py")
        self.assertIn("_chat_context = ConversationContextService(", api)
        self.assertIn("active_session_id = _chat_context.active_session_id", api)
        self.assertIn("session_histories = _chat_context.histories", api)

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
