"""模型托管宿主（inference-svc 进程内）。

并行共存（§1.4）：本类在 inference-svc 独立进程内新建 ModelHost 实例，
与 api_server 进程内的 model_host 单例互不共享、互不影响。

1.1（本文件）：薄委托 ModelHost——其内部 _LazyModelManager 延迟 import
model_module，未加载模型时冷启动成本 <100ms（§2.3：model_module 子树
9.7s 全部发生在首次 load 请求时）。
1.2：把 api_server 数据面执行段（_execute_chat_full 等 8 函数）与
scheduler 流水线段（_run_pipeline 等）复制为本宿主方法，源文件保持
不动（复制迁移，禁改源）。

1.2a（已复制）：
  - _execute_task_worker_stage（含嵌套 run_model 闭包）→
    EngineHost.execute_task_worker_stage
  - 辅助纯函数：_format_model_response / _parse_thinking_response /
    _strip_native_thinking_tags（api_server.py:904-1088 复制，零改动）
"""
import inspect
import json
import logging
import re
import threading
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import local_store as _local_store
from api_errors import coded_http_error
from chat_context import (
    ConversationContextConflict,
    ConversationContextService,
    PreparedChatContext,
    TurnPersistenceResult,
)
from koakuma_engine import (
    Capability,
    backend_id_for,
    registered_backends,
    runtime_supports,
)
from model_load_resolver import (
    LAYER_RANGE_DYNAMIC,
    LAYER_RANGE_PRECUT_ARTIFACT,
    ModelLoadFacts,
    ModelLoadResolution,
    ModelLoadResolutionError,
    effective_pytorch_cuda_available,
    preferred_engine_for_artifacts,
    resolve_model_load,
)
from multimodal import materialize_image_data_url
from request_deadline import (
    REQUEST_DEADLINE_EXCEEDED,
    DeadlineCancelEvent,
    RequestDeadline,
    RequestDeadlineExceeded,
    coerce_request_deadline,
    hold_lock_until_deadline,
    request_stop_reason,
)
from request_outcome import (
    REASON_GENERATION_CANCELLED,
    REASON_REQUEST_REFUSED,
)


def _call_with_optional_request_deadline(
    callback,
    *args,
    request_deadline: Optional[RequestDeadline],
):
    """Call an override while keeping pre-deadline test/plugin signatures valid."""

    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        parameters = {}
    deadline_parameter = parameters.get("request_deadline")
    if deadline_parameter is not None:
        if deadline_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            return callback(*args, request_deadline)
        return callback(*args, request_deadline=request_deadline)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return callback(*args, request_deadline=request_deadline)
    return callback(*args)


# ---- 复制自 api_server.py:887-904（常量，保真不变） ----
THINKING_START = "【思考】"
THINKING_END = "【思考结束】"
THINKING_SYSTEM_PROMPT = (
    "你是一个善于深度思考的AI助手。回答前先进行推理分析，再给出答案。\n\n"
    "严格按以下格式输出：\n"
    "【思考】\n"
    "（你的推理过程，2-3句话即可）\n"
    "【思考结束】\n"
    "（你的最终回答）\n\n"
    "注意：\n"
    "- 必须在【思考结束】之后写回答内容\n"
    "- 回答部分不要写标记符号\n"
    "- 不要重复输出【思考】或【思考结束】"
)


# ---- 复制自 api_server.py:904-941（纯函数，保真不变） ----
def _strip_native_thinking_tags(text: str) -> str:
    """Remove native thinking/answer tags and leaked ChatML sentinels."""
    import re as _re

    if not text:
        return text

    result = _re.sub(
        r'<\s*think\s*>.*?<\s*/\s*think\s*>',
        '',
        text,
        flags=_re.DOTALL | _re.IGNORECASE,
    )
    result = _re.sub(
        r'^.*?<\s*/\s*think\s*>',
        '',
        result,
        count=1,
        flags=_re.DOTALL | _re.IGNORECASE,
    )

    response_match = _re.search(
        r'<\s*(?:answer|response)\s*>(.*?)(?:<\s*/\s*(?:answer|response)\s*>|$)',
        result,
        flags=_re.DOTALL | _re.IGNORECASE,
    )
    if response_match:
        result = response_match.group(1)

    result = _re.sub(r'<\s*/?\s*(?:think|answer|response)\s*>', '', result, flags=_re.IGNORECASE)
    result = result.replace('<|im_end|>', '').replace('<|im_start|>', '')
    result = _re.sub(r'<\s*\|im_(?:start|end)\|\s*>', '', result)
    result = _re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


# ---- 复制自 api_server.py:943-1074（纯函数，保真不变） ----
def _parse_thinking_response(text: str) -> tuple:
    """
    解析模型输出，分离思考内容和最终答案。

    （复制自 api_server._parse_thinking_response，含容错逻辑：
    缺少结束标记 → 智能分割；答案为空 → 提取思考最后一段；
    重复标记 → 使用第一次出现的有效标记对。）

    Returns:
        (answer_content, thinking_content)
    """
    import re as _re

    if not text:
        return "", None

    start_idx = text.find(THINKING_START)
    end_idx = text.find(THINKING_END)

    # ---- 情况1：标记成对且顺序正确 ----
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        thinking = text[start_idx + len(THINKING_START):end_idx].strip()
        answer = text[end_idx + len(THINKING_END):].strip()

        thinking = _re.sub(r'^分析思路[：:]\s*', '', thinking)

        answer = _re.sub(r'^【最终答案】[：:]?\s*', '', answer)
        answer = _re.sub(r'^(最终答案|回答|Answer)[：:]\s*', '', answer, flags=_re.IGNORECASE)
        for _pat in [r'^\[你的最终回答[^\]]*\]\s*', r'^\[你的推理过程[^\]]*\]\s*',
                     r'^（推理内容）\s*', r'^（答案内容）\s*',
                     r'^（给用户的答案[^）]*）\s*']:
            answer = _re.sub(_pat, '', answer)

        answer = answer.replace(THINKING_START, "").replace(THINKING_END, "").strip()

        prefix = text[:start_idx].strip()
        if prefix:
            answer = prefix + ("\n" + answer if answer else "")

        if thinking:
            if not answer and thinking:
                paragraphs = thinking.split("\n")
                for p in reversed(paragraphs):
                    p = p.strip()
                    if p and len(p) > 10:
                        answer = p
                        break
                if not answer:
                    answer = thinking
            return answer, thinking

    # ---- 情况2：DeepSeek-R1 / Qwen3 本地格式 ----
    import re as _re2
    native_match = _re2.search(
        r'<\s*think\s*>(.*?)<\s*/\s*think\s*>',
        text,
        flags=_re2.DOTALL | _re2.IGNORECASE,
    )
    if native_match:
        thinking = native_match.group(1).strip()
        answer = text[:native_match.start()].strip()
        after_think = text[native_match.end():].strip()
        after_think = _re2.sub(r'<\s*/?\s*(?:response|answer)\s*>', '', after_think, flags=_re2.IGNORECASE)
        if after_think:
            answer = (answer + '\n' + after_think).strip() if answer else after_think
        response_match = _re2.search(
            r'<\s*(?:response|answer)\s*>(.*)',
            answer if answer else '',
            flags=_re2.DOTALL | _re2.IGNORECASE,
        )
        if response_match:
            answer = response_match.group(1).strip()
        answer = _re2.sub(r'<\s*/?\s*(?:think|response|answer)\s*>', '', answer, flags=_re2.IGNORECASE)
        answer = answer.replace(THINKING_START, "").replace(THINKING_END, "")
        answer = answer.replace('<|im_end|>', '').replace('<|im_start|>', '').strip()
        if thinking:
            return answer, thinking

    closing_only_match = _re2.search(
        r'^(.*?)<\s*/\s*think\s*>(.*)$',
        text,
        flags=_re2.DOTALL | _re2.IGNORECASE,
    )
    if closing_only_match:
        thinking = closing_only_match.group(1).strip()
        thinking = thinking.replace(THINKING_START, "").replace(THINKING_END, "").strip()
        answer = closing_only_match.group(2).strip()
        answer = _re2.sub(r'<\s*/?\s*(?:response|answer)\s*>', '', answer, flags=_re2.IGNORECASE)
        answer = answer.replace(THINKING_START, "").replace(THINKING_END, "")
        answer = answer.replace('<|im_end|>', '').replace('<|im_start|>', '').strip()
        if answer or thinking:
            return answer, thinking or None

    # ---- 情况3：格式未遵循 ----
    cleaned = text.replace(THINKING_START, "").replace(THINKING_END, "").strip()
    cleaned = _strip_native_thinking_tags(cleaned)
    cleaned = _re.sub(r'^分析思路[：:]\s*', '', cleaned)
    cleaned = _re.sub(r'^(最终答案|回答|Answer)[：:]\s*', '', cleaned, flags=_re.IGNORECASE)
    return cleaned, None


# ---- 复制自 api_server.py:1076-1088（纯函数，保真不变） ----
def _format_model_response(text: str, show_thinking: bool,
                           native_thinking_prompt: bool = False) -> tuple[str, Optional[str]]:
    """Format generated text without exposing unfinished native reasoning."""
    if show_thinking:
        return _parse_thinking_response(text)
    if native_thinking_prompt and "</think>" not in (text or "").lower():
        return "", None
    return _strip_native_thinking_tags(text), None


# ---- 复制自 api_server.py:2225-2232（纯函数，保真不变） ----
def _chat_origin(req: "ChatRequest") -> str:
    """根据请求上报信息推断请求来源，用于 metrics 展示。"""
    if req.client_node_type == "android":
        return "android_http"
    if req.client_node_type == "pc":
        return "pc_http"
    return "web_http"


# ---- 复制自 api_server.py:2234-2259（宿主适配：scheduler 全局 → 参数注入） ----
def _augment_chat_metrics(
    metrics: Optional[dict],
    req: "ChatRequest",
    *,
    serving_node_id: str = "",
    distributed_enabled: bool = False,
    **defaults,
) -> dict:
    """补齐统一聊天 metrics 字段，不覆盖调度器已给出的真实执行信息。

    （api_server 版从 scheduler 全局取 serving_node_id / 分布式开关；
    本进程由 EngineHost 持有，经参数注入。）
    """
    result = dict(metrics or {})
    for key, value in defaults.items():
        result.setdefault(key, value)
    origin = _chat_origin(req)
    result.setdefault("request_origin", origin)
    result.setdefault("request_origin_node_id", req.client_node_id or "")
    result.setdefault("request_origin_node_type", req.client_node_type or "")
    result.setdefault("client_mode", req.client_mode or "")
    result.setdefault("client_app_variant", req.client_app_variant or "")
    result.setdefault("serving_node_id", serving_node_id)
    result.setdefault("distributed_requested", distributed_enabled)
    result.setdefault("distributed_used", False)
    result.setdefault("fallback", False)
    result.setdefault("fallback_reason", "")
    result.setdefault("workers_used", [])
    result.setdefault("layer_assignments", [])
    result.setdefault("request_id", _request_id_ctx.get("-"))
    result.setdefault("generation_id", req.generation_id or "")
    return result


# ---- 复制自 api_server.py:828-901（纯函数，保真不变） ----
def _build_chat_prompt(
    messages: list[dict],
    system_prompt: Optional[str] = None,
    assistant_prefill: Optional[str] = None,
) -> str:
    """
    使用 Qwen 的 chat template 构建对话 prompt。
    Qwen-1.8B-Chat 使用 <|im_start|>/<|im_end|> 格式。
    """
    parts = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    if assistant_prefill:
        parts.append(assistant_prefill)
    return "\n".join(parts)


def _build_model_chat_prompt(
    tokenizer,
    messages: list[dict],
    system_prompt: Optional[str] = None,
    assistant_prefill: Optional[str] = None,
) -> str:
    """Build a prompt with the active tokenizer's native chat template."""
    chat_messages = []
    if system_prompt:
        chat_messages.append({"role": "system", "content": system_prompt})
    chat_messages.extend(messages)

    try:
        prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        native_thinking_prompt = "<think>" in prompt[-64:].lower()
        if assistant_prefill and not native_thinking_prompt:
            prompt += assistant_prefill
        return prompt
    except Exception:
        return _build_chat_prompt(
            messages,
            system_prompt=system_prompt,
            assistant_prefill=assistant_prefill,
        )


from .protocol import ChatRequest  # noqa: E402（置于辅助函数后保持文档头清晰）

logger = logging.getLogger("inference_service.engine_host")

# ---- 复制自 api_server.py:96（请求 ID 上下文，与 api_server 同机制） ----
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


# ---- 复制自 api_server.py:406-410（异常，保真不变） ----
class ChatGenerationCancelled(RuntimeError):
    def __init__(self, generation_id: str):
        self.generation_id = generation_id
        super().__init__(f"generation {generation_id} cancelled")


# ---- 复制自 api_server.py:459-464（纯函数，保真不变） ----
def _raise_if_generation_cancelled(
    cancel_event: Optional[threading.Event], generation_id: Optional[str],
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ChatGenerationCancelled(generation_id or "gen_unknown")


# ---- 复制自 api_server.py:1178-1226（纯函数，保真不变） ----
def _is_question(text: str) -> bool:
    """
    判断文本是否为真正的疑问句，而非陈述句。

    （复制自 api_server._is_question：必须以问号结尾 + 含疑问指示词
    + 拒绝陈述句式关键词 + 拒绝列举开头。）
    """
    text = text.strip()
    if not text:
        return False

    if not (text.endswith('？') or text.endswith('?')):
        return False

    question_indicators = [
        '吗', '呢',
        '什么', '怎么', '如何', '为何',
        '哪些', '哪个', '哪种', '哪位',
        '有没有', '能否', '是否', '可否',
        '能不能', '会不会', '可不可以',
        '多少', '几',
        '谁', '哪', '何时', '怎样',
        '可以', '能帮', '推荐', '介绍',
    ]
    has_indicator = any(ind in text for ind in question_indicators)
    if not has_indicator:
        return False

    statement_patterns = [
        '有以下', '包括以下', '如下',
        '例如', '比如',
        '这是', '以下是', '下面是',
        '区别在于', '不同之处', '特点有',
        '首先', '其次', '然后', '最后',
        '第一', '第二', '第三',
        '步骤', '流程', '方法有',
    ]
    if any(p in text for p in statement_patterns):
        return False

    if re.match(r'^[\d]+[\.\、\)）]', text):
        return False

    return True


# ---- 复制自 api_server.py:1438-1573（纯函数，保真不变） ----
def _fallback_followups(history: list, existing: list[str]) -> list[str]:
    """
    基于对话关键词匹配的追问模板兜底。

    （复制自 api_server._fallback_followups：关键词 → 追问模板映射，
    按优先级排序，去重后补足至 3 条。）
    """
    last_assistant = ""
    last_user = ""
    for msg in reversed(history):
        if msg["role"] == "assistant" and not last_assistant:
            last_assistant = msg["content"]
        if msg["role"] == "user" and not last_user:
            last_user = msg["content"]

    combined = (last_user + " " + last_assistant).lower()

    templates = []

    if any(kw in combined for kw in ["量化", "quant", "int4", "int8", "fp16", "精度"]):
        templates.extend([
            "INT4和INT8量化在实际应用中如何选择？",
            "量化会对模型推理能力造成多大影响？",
            "除了量化还有哪些模型压缩方法？",
        ])

    if any(kw in combined for kw in ["边缘计算", "边缘", "edge", "分布式", "推理"]):
        templates.extend([
            "边缘推理和云端推理各有什么优缺点？",
            "分布式推理中的通信开销如何优化？",
            "边缘设备的算力瓶颈通常在哪里？",
        ])

    if any(kw in combined for kw in ["python", "代码", "编程", "写一个", "函数", "算法"]):
        templates.extend([
            "这段代码的时间复杂度是多少？",
            "有没有更高效的实现方式？",
            "能解释一下这段代码的核心逻辑吗？",
        ])

    if any(kw in combined for kw in ["模型", "训练", "微调", "lora", "参数"]):
        templates.extend([
            "这个模型的训练数据来源是什么？",
            "如何在特定领域数据上微调模型？",
            "LoRA微调相比全参数微调有哪些优势？",
        ])

    if any(kw in combined for kw in ["transformer", "注意力", "attention", "架构"]):
        templates.extend([
            "Transformer相比RNN有哪些优势？",
            "自注意力机制的计算复杂度如何？",
            "多头注意力的作用是什么？",
        ])

    if any(kw in combined for kw in ["token", "tokenizer", "分词", "词表"]):
        templates.extend([
            "不同的分词方法对模型性能有影响吗？",
            "中文分词和英文分词的主要区别是什么？",
            "BPE分词算法的原理是什么？",
        ])

    if any(kw in combined for kw in ["显存", "gpu", "内存", "oom", "优化", "加速"]):
        templates.extend([
            "还有哪些降低推理显存占用的方法？",
            "CPU推理在什么场景下比GPU更合适？",
            "KV Cache的显存占用如何估算？",
        ])

    if any(kw in combined for kw in ["应用", "场景", "实际", "落地", "工业"]):
        templates.extend([
            "当前这个技术还有哪些落地挑战？",
            "业界有哪些成功的应用案例可以参考？",
            "这项技术的商业化前景如何？",
        ])

    if any(kw in combined for kw in ["hello", "你好", "介绍", "你是谁", "能做什么"]):
        templates.extend([
            "你能帮我写代码吗？",
            "你的知识截止到什么时候？",
            "你擅长哪些类型的任务？",
        ])

    if any(kw in combined for kw in ["学习", "入门", "新手", "教程", "怎么学"]):
        templates.extend([
            "有哪些推荐的学习资源或课程？",
            "学习这个需要什么前置知识？",
            "从入门到精通大概需要多久？",
        ])

    if any(kw in combined for kw in ["区别", "对比", "比较", "不同", "差异", "选择"]):
        templates.extend([
            "在选择时应该考虑哪些关键因素？",
            "有没有具体的场景举例说明？",
            "未来哪个方向更有发展前景？",
        ])

    if any(kw in combined for kw in ["安全", "隐私", "加密", "攻击", "漏洞"]):
        templates.extend([
            "这种攻击的防御措施有哪些？",
            "业界有哪些典型的安全事件？",
            "如何在性能和安全性之间平衡？",
        ])

    if any(kw in combined for kw in ["数据", "dataset", "数据集", "预处理", "清洗"]):
        templates.extend([
            "数据质量对模型效果的影响有多大？",
            "有哪些常用的数据增强方法？",
            "如何处理数据中的类别不平衡问题？",
        ])

    default_templates = [
        "能再详细解释一下吗？",
        "这个结论有什么前提条件或局限性？",
        "有没有相关的参考资料或论文推荐？",
        "实际应用中需要注意哪些细节？",
        "能举一个具体的例子说明吗？",
    ]

    result = list(existing)
    candidate_pool = templates + default_templates
    for q in candidate_pool:
        if q not in result and len(result) < 3:
            result.append(q)

    if len(result) < 2:
        for q in default_templates:
            if q not in result and len(result) < 3:
                result.append(q)

    logger.info(f"追问兜底: 模型生成了 {len(existing)} 条，模板补充至 {len(result)} 条")
    return result


class EngineHost:
    """推理宿主：进程内模型托管 + 执行段宿主 + generation 取消注册表。"""

    _environment = None

    def __init__(self, *, environment=None):
        self._environment = None if environment is None else dict(environment)
        # 延迟 import：保持模块顶层轻量（config 69ms 可接受，model_module 不在此触发）
        import config as runtime_config
        from model_host import ModelHost

        self._host: ModelHost = ModelHost(environment=self._environment)
        self._model_change_lock = threading.RLock()
        self._layers: List[str] = []  # 已加载层段（1.2 与 model 侧真实状态对齐）
        # ★ 2026-10-03：v3 层段 Stage 的 keep-head 上游（`KeepHeadUpstream`）。子进程隔离
        #   由它自理；这里只负责按 env 配置创建一次并复用（重载 500MB GGUF 代价高）。
        self._layer_upstream: Any = None
        self._worker_runtime_lock = threading.RLock()
        self._worker_runtime_managed = False
        self._worker_runtime_active = None
        self._worker_runtime_candidate = None
        self._worker_runtime_previous = None
        self._worker_runtime_identity = None
        self._worker_runtime_phase = "idle"
        # A filtered manager follows current.json; KV is per workflow/sequence.
        self._supplied_layer_manager: Any = None
        self._supplied_layer_key: Any = None
        self._supplied_layer_workflow: Any = None
        self._supplied_layer_caches: dict = {}
        self._gen_lock = threading.RLock()
        self._generations: Dict[str, threading.Event] = {}
        # 1.2c 宿主适配状态（api_server 全局 → 实例属性）
        self._conversation_stats: Dict[str, int] = {
            "total_prompt_tokens": 0,
            "total_generated_tokens": 0,
            "total_time_seconds": 0.0,
            "rounds": 0,
        }
        # 1.4 注入：scheduler-svc 侧任务完成回调（当前 no-op，进程内基线）
        self._on_task_complete = None
        # 1.4 注入：scheduler-svc 下发的节点身份/分布式开关（metrics 用）
        self._serving_node_id = ""
        self._distributed_enabled = False
        # 1.2c 会话管理状态（api_server 全局 → 实例属性）
        self._chat_context = ConversationContextService(
            load_history=self._load_conversation_history,
            persist_turn=self._persist_conversation_turn,
        )
        self._active_session_id: Optional[str] = None
        self._session_histories: Dict[str, list] = self._chat_context.histories
        self._kv_cache: Any = None  # PagedKVCache 单例（惰性，_init_kv_cache）
        self._device_profile: Optional[dict] = None  # 惰性探测（_ensure_device_profile）
        self._device_profile_started = False
        # 1.4 注入：scheduler-svc 客户端接口（None = 单机基线：分布式/流水线禁用）
        self._scheduler: Any = None
        self._run_mode: str = str(
            self._environment.get("QLH_RUN_MODE", "distributed") if self._environment is not None
            else getattr(runtime_config, "RUN_MODE", "standalone") or "standalone"
        ).strip().lower()
        self._on_task_error = None  # 1.4 注入：任务失败回调
        self.role: str = "master"  # master / client（1.3 角色感知，1.5 peer 使用）
        # 1.2d task_graph 状态（api_server 全局 → 实例属性，惰性创建）
        self._task_graph_coordinator: Any = None
        self._task_graph_execution_slot = threading.BoundedSemaphore(1)

    # ------------------------------------------------------------------
    # 模型生命周期（委托 ModelHost / ModelManager）
    # ------------------------------------------------------------------
    def _model_lifecycle_lock(self):
        return getattr(
            self._host,
            "full_chat_execution_lock",
            self._model_change_lock,
        )

    def _llm_is_loaded(self) -> bool:
        checker = getattr(self._host, "has_loaded_model", None)
        if callable(checker):
            return bool(checker())
        return bool(getattr(self._host, "model_loaded", False)) or bool(
            getattr(self._host, "is_loaded", False)
        )

    def initialize_storage(self) -> str:
        """Validate the shared main-node SQLite before serving requests."""
        return _local_store.initialize_local_store()

    def _persist_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metrics: Optional[dict] = None,
    ) -> TurnPersistenceResult:
        try:
            if not _local_store.get_local_save_history():
                return TurnPersistenceResult.SKIPPED
            operation_id = _request_id_ctx.get("")
            committed = _local_store.save_local_conversation_turn(
                session_id,
                user_message,
                assistant_message,
                metrics,
                operation_id=(
                    operation_id
                    if operation_id and operation_id != "-"
                    else None
                ),
            )
        except ValueError:
            raise
        except Exception as exc:
            logger.error("SQLite 对话提交失败: session=%s: %s", session_id, exc)
            raise RuntimeError("SQLite 对话提交失败") from exc

        return (
            TurnPersistenceResult.COMMITTED
            if committed
            else TurnPersistenceResult.REPLAY
        )

    def _load_conversation_history(self, session_id: str) -> list[dict]:
        try:
            rows = _local_store.load_local_conversation(session_id)
        except Exception:
            logger.error(
                "SQLite 会话历史加载失败: session=%s",
                session_id,
                exc_info=True,
            )
            raise
        return [
            {"role": row["role"], "content": row["content"]}
            for row in rows
        ]

    def _sync_chat_context_facade(self) -> None:
        self._chat_context.adopt_facade_state(
            self._active_session_id, self._session_histories,
        )

    def _publish_chat_context_facade(self) -> None:
        self._active_session_id = self._chat_context.active_session_id
        self._session_histories = self._chat_context.histories

    def _prepare_chat_context(
        self,
        requested_session_id: Optional[str],
        first_message: Optional[str] = None,
        *,
        force_reload: bool = False,
    ) -> PreparedChatContext:
        self._sync_chat_context_facade()
        prepared = self._chat_context.prepare(
            requested_session_id, force_reload=force_reload,
        )
        self._publish_chat_context_facade()
        if prepared.session_changed or prepared.reloaded:
            if self._kv_cache:
                self._kv_cache.clear()
            if backend_id_for(self._host) == "pytorch":
                self._init_kv_cache()
            else:
                self._kv_cache = None
        if first_message and not prepared.history:
            self._auto_title_session(prepared.session_id, first_message)
        return prepared

    def _commit_chat_context_turn(
        self,
        session_id: Optional[str],
        user_message: str,
        assistant_message: str,
        metrics: Optional[dict] = None,
        *,
        expected_revision: Optional[int] = None,
        operation_id: Optional[str] = None,
    ) -> bool:
        self._sync_chat_context_facade()
        resolved_operation_id = operation_id or _request_id_ctx.get("")
        request_token = None
        if operation_id:
            request_token = _request_id_ctx.set(operation_id)
        try:
            persisted = self._chat_context.commit_turn(
                session_id,
                user_message,
                assistant_message,
                metrics,
                operation_id=(
                    resolved_operation_id
                    if resolved_operation_id and resolved_operation_id != "-"
                    else None
                ),
                expected_revision=expected_revision,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ConversationContextConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        finally:
            if request_token is not None:
                _request_id_ctx.reset(request_token)
        self._publish_chat_context_facade()
        return persisted

    def chat_full_with_operation_id(
        self,
        req: ChatRequest,
        cancel_event: Optional[threading.Event],
        operation_id: str,
        request_deadline: Optional[RequestDeadline] = None,
    ) -> Dict[str, Any]:
        """Run one complete chat transaction with an explicit HTTP identity."""

        with self._chat_context.transaction(deadline=request_deadline):
            with hold_lock_until_deadline(
                self._model_lifecycle_lock(),
                request_deadline,
                operation="full chat execution lock",
            ):
                token = _request_id_ctx.set(operation_id or "-")
                try:
                    return _call_with_optional_request_deadline(
                        self.chat_full,
                        req,
                        cancel_event,
                        request_deadline=request_deadline,
                    )
                finally:
                    _request_id_ctx.reset(token)

    def chat_stream_events_with_operation_id(
        self,
        req: ChatRequest,
        cancel_event: Optional[threading.Event],
        operation_id: str,
        request_deadline: Optional[RequestDeadline] = None,
    ):
        """Bound stream consumption while the producer retains model ownership.

        A backend iterator can block inside ``next()`` and ignore cooperative
        cancellation.  The caller must still observe the request deadline, but
        the conversation/model locks must not be released until that backend
        actually exits.  A dedicated owner thread therefore holds both locks;
        this generator only consumes its bounded queue.
        """

        import queue
        import time as _time

        stream_cancel_event = cancel_event
        if not isinstance(stream_cancel_event, DeadlineCancelEvent):
            stream_cancel_event = DeadlineCancelEvent(
                stream_cancel_event, request_deadline,
            )
        events: queue.Queue = queue.Queue(maxsize=1)
        consumer_done = threading.Event()
        demand = threading.Semaphore(0)
        producer_done = object()

        def put_event(item) -> bool:
            while not consumer_done.is_set():
                try:
                    events.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    pass
            return False

        def produce() -> None:
            try:
                with self._chat_context.transaction(deadline=request_deadline):
                    with hold_lock_until_deadline(
                        self._model_lifecycle_lock(),
                        request_deadline,
                        operation="full chat execution lock",
                    ):
                        token = _request_id_ctx.set(operation_id or "-")
                        try:
                            iterable = _call_with_optional_request_deadline(
                                self.chat_stream_events,
                                req,
                                stream_cancel_event,
                                request_deadline=request_deadline,
                            )
                            iterator = iter(iterable)
                            while True:
                                demand.acquire()
                                if consumer_done.is_set():
                                    return
                                try:
                                    event = next(iterator)
                                except StopIteration:
                                    break
                                if not put_event((event, None)):
                                    return
                        finally:
                            _request_id_ctx.reset(token)
            except Exception as exc:
                put_event((None, exc))
            finally:
                put_event((producer_done, None))

        threading.Thread(
            target=produce,
            name="inference-svc-stream-owner",
            daemon=True,
        ).start()

        completed_normally = False
        cancel_grace_deadline = None
        demand_outstanding = False
        try:
            while True:
                stop_reason = request_stop_reason(
                    stream_cancel_event, request_deadline,
                )
                if stop_reason == REQUEST_DEADLINE_EXCEEDED:
                    yield {
                        "done": True,
                        "error": "request deadline exceeded",
                        "reason_code": REQUEST_DEADLINE_EXCEEDED,
                    }
                    return
                if stop_reason:
                    if cancel_grace_deadline is None:
                        cancel_grace_deadline = _time.monotonic() + 0.25
                    elif _time.monotonic() >= cancel_grace_deadline:
                        yield {
                            "done": True,
                            "error": "generation cancelled",
                            "cancelled": True,
                            "reason_code": REASON_GENERATION_CANCELLED,
                        }
                        return
                wait_timeout = 0.1
                if cancel_grace_deadline is not None:
                    wait_timeout = min(
                        wait_timeout,
                        max(0.001, cancel_grace_deadline - _time.monotonic()),
                    )
                if not demand_outstanding:
                    demand.release()
                    demand_outstanding = True
                if request_deadline is not None:
                    remaining = request_deadline.remaining()
                    if remaining <= 0:
                        continue
                    wait_timeout = min(wait_timeout, remaining)
                try:
                    event, error = events.get(timeout=wait_timeout)
                except queue.Empty:
                    continue
                demand_outstanding = False
                if event is producer_done:
                    completed_normally = True
                    return
                if error is not None:
                    raise error
                yield event
        finally:
            consumer_done.set()
            demand.release()
            if not completed_normally:
                stream_cancel_event.set()

    def commit_stream_event(
        self,
        req: ChatRequest,
        event: dict,
        cancel_event: Optional[threading.Event],
        operation_id: str,
    ) -> bool:
        """Commit only after the ASGI consumer accepts a terminal done event."""

        context_session_id = event.pop("_context_session_id", None)
        context_revision = event.pop("_context_revision", None)
        if context_session_id is None or context_revision is None:
            return bool(event.get("history_committed", False))
        _raise_if_generation_cancelled(cancel_event, req.generation_id)
        response_text = str(event.get("response") or "")
        if not response_text:
            return False
        committed = self._commit_chat_context_turn(
            context_session_id,
            req.message,
            response_text,
            event.get("metrics") if isinstance(event.get("metrics"), dict) else {},
            expected_revision=int(context_revision),
            operation_id=operation_id,
        )
        self._conversation_stats["rounds"] += 1
        return committed

    def select_engine(self, profile=None) -> str:
        return self._host.select_engine(profile)

    def load_model(
        self,
        engine: Optional[str] = None,
        quant_type: Optional[str] = None,
        use_compile: bool = False,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        resolution: ModelLoadResolution | None = None,
    ) -> Dict[str, Any]:
        # use_compile 走进程内 config 全局开关（对齐 api_server.py:2236
        # cfg.USE_COMPILE = req.use_compile）；ModelManager.load_model 签名
        # 无此参数（2026-08-05 真实加载复测暴露 TypeError）
        with self._model_lifecycle_lock():
            import config as _cfg

            _cfg.USE_COMPILE = bool(use_compile)
            self.release_worker_runtime()
            result = self._host.load_model(
                engine=engine,
                quant_type=quant_type,
                model_id=model_id,
                model_path=model_path,
                profile=self._ensure_device_profile(),
                resolution=resolution,
            )
            # 对齐 api_server.py:2264-2265：load 成功后显式置位运行时状态
            # （ModelHost.model_loaded 不自更新；2026-08-05 真实加载复测暴露
            # /v1/status 恒 model_loaded:false）。current_quant 优先取 manager
            # 实际生效值（对齐 api_server getattr(model_manager, 'quant_type')
            # or quant），避免请求值与引擎归一化值不一致（如 GGUF 引擎忽略 int4）
            self._host.model_loaded = True
            self._host.current_quant = (
                getattr(self._host, "quant_type", None) or quant_type or "int4"
            )
            self._reset_runtime_conversation_state(clear_histories=True)
            return result if isinstance(result, dict) else {"success": True, "data": result}

    def unload_model(self) -> Dict[str, Any]:
        with self._model_lifecycle_lock():
            self.release_worker_runtime()
            if self._llm_is_loaded():
                self._host.unload_model()
            self._layers.clear()
            self._host.model_loaded = False
            self._reset_runtime_conversation_state(clear_histories=True)
            return {"success": True, "message": "模型已卸载"}

    def switch_model(
        self,
        model_id: str,
        engine: Optional[str] = None,
        quant_type: Optional[str] = None,
        model_path: Optional[str] = None,
        resolution: ModelLoadResolution | None = None,
    ) -> Dict[str, Any]:
        with self._model_lifecycle_lock():
            self.release_worker_runtime()
            result = self._host.switch_model(
                model_id=model_id,
                engine=engine,
                quant_type=quant_type,
                model_path=model_path,
                profile=self._ensure_device_profile(),
                resolution=resolution,
            )
            # 对齐 api_server.py:5146-5147：切换成功置位运行时状态
            # （api_server 在 result['success'] 分支设置）
            if isinstance(result, dict) and result.get("success"):
                self._host.model_loaded = True
                self._host.current_quant = (
                    getattr(self._host, "quant_type", None)
                    or result.get("quant_type")
                    or getattr(self._host, "current_quant", "int4")
                )
                self._reset_runtime_conversation_state(clear_histories=True)
            return result if isinstance(result, dict) else {"success": True, "data": result}

    def close(self) -> None:
        with self._model_lifecycle_lock():
            self.release_worker_runtime()
        coordinator = self._task_graph_coordinator
        self._task_graph_coordinator = None
        if coordinator is not None:
            coordinator.close()
        unload = getattr(self._host, "unload_model", None)
        if callable(unload):
            unload()

    def current_model(self) -> Dict[str, Any]:
        """当前模型信息（/v1/models/current，对齐 api_server.py:2188-2207 完整字段）。"""
        loaded = bool(getattr(self._host, "model_loaded", False))
        mgr = self._host
        if not loaded:
            return {"loaded": False, "quant_type": None, "model_id": None}
        info: Dict[str, Any] = {}
        mem: Dict[str, Any] = {}
        try:
            info = mgr.get_model_info()
        except Exception:
            pass
        try:
            mem = mgr.get_memory_usage()
        except Exception:
            pass
        return {
            "loaded": True,
            "model_id": getattr(mgr, "active_model_id", None),
            "quant_type": getattr(self._host, "current_quant", None),
            "model_name": info.get("model_name", getattr(mgr, "active_model_id", "")),
            "model_path": info.get("model_path", ""),
            "engine": info.get("engine", ""),
            "total_params": info.get("total_params", "N/A"),
            "device": info.get("device", "N/A"),
            "gpu_allocated_gb": mem.get("gpu_allocated_gb", 0),
            "gpu_reserved_gb": mem.get("gpu_reserved_gb", 0),
        }

    def list_models(self) -> Dict[str, Any]:
        """模型注册表 + 文件状态（/v1/models，对齐 api_server.py:5088-5099）。

        DB 注册的实验模型由 control-svc /models/registry 承载（阶段 3.2），
        此处仅返回内置模型；active_model_id 取当前加载状态。
        """
        import model_config as mc

        models = [self._model_api_payload(m) for m in mc.get_builtin_models()]
        active = None
        if bool(getattr(self._host, "model_loaded", False)):
            try:
                active = getattr(self._host, "active_model_id", None)
            except Exception:
                active = None
        return {"models": models, "active_model_id": active}

    def list_local_model_assets(self) -> Dict[str, Any]:
        """Expose discovered local packages without claiming legacy loadability."""
        from local_model_assets import discover_local_model_assets

        return discover_local_model_assets()

    def preflight_local_model_asset(self, model_id: str) -> Dict[str, Any]:
        """Run a read-only isolated runtime preflight for one inventory asset."""
        from local_model_assets import preflight_local_model_asset

        return preflight_local_model_asset(model_id)

    def available_models(self) -> Dict[str, Any]:
        """可选模型配置 + 可用引擎（/v1/models/available，对齐 api_server.py:4727-4879）。

        torch 延迟 import（保持冷启动不加载 model_module 的优化，仅首次
        调用本端点时支付 ~1.8s）。"""
        import model_config as mc

        model_payloads = [self._model_api_payload(m) for m in mc.get_builtin_models()]
        engine_ids = {
            engine
            for payload in model_payloads
            for engine in payload.get("supported_engines", [])
        }
        available_engines: List[Dict[str, Any]] = []

        if "llama_cpp" in engine_ids:
            available_engines.append({
                "id": "llama_cpp",
                "name": "llama.cpp + GGUF",
                "description": "GGUF 量化模型，适合 CPU/集显或轻量试水",
                "model_size_gb": None,
                "requires_cuda": False,
            })

        has_cuda = False
        pytorch_quants: List[Dict[str, Any]] = []
        if "pytorch" in engine_ids:
            try:
                import torch  # 延迟：保持冷启动优化
                has_cuda = torch.cuda.is_available()
            except Exception:
                has_cuda = False
            available_engines.append({
                "id": "pytorch",
                "name": "PyTorch + Safetensors" + (" (CUDA)" if has_cuda else " (CPU)"),
                "description": "Safetensors 格式，支持 INT4/INT8/FP16 量化" + ("，GPU 加速" if has_cuda else "，CPU 模式较慢"),
                "model_size_gb": None,
                "requires_cuda": has_cuda,
            })
            pytorch_quants = [
                {
                    "id": "int4",
                    "name": "INT4 量化 ⭐",
                    "description": "4-bit 量化，显存 ~1.8 GB，速度 ~29 tok/s（推荐边缘设备）",
                    "memory_gb": 1.8,
                    "speed_tok_s": 29,
                    "compile_support": False,
                    "engine": "pytorch",
                    "is_available": True,
                },
                {
                    "id": "int8",
                    "name": "INT8 量化",
                    "description": "8-bit 量化，显存 ~2.3 GB，速度 ~10 tok/s",
                    "memory_gb": 2.3,
                    "speed_tok_s": 10,
                    "compile_support": False,
                    "engine": "pytorch",
                    "is_available": True,
                },
            ]
            if has_cuda:
                pytorch_quants.insert(0, {
                    "id": "fp16",
                    "name": "FP16 原版",
                    "description": "原始精度，显存 ~3.5 GB，速度最快 (~53 tok/s)",
                    "memory_gb": 3.5,
                    "speed_tok_s": 53,
                    "compile_support": True,
                    "engine": "pytorch",
                    "is_available": True,
                })

        gguf_quants: List[Dict[str, Any]] = []
        if "llama_cpp" in engine_ids:
            gguf_quants = [{
                "id": "gguf",
                "name": "GGUF 量化",
                "description": "量化精度由 GGUF 文件决定（Q4_K_M / Q5_K_M 等），适合 CPU/集显",
                "memory_gb": None,
                "speed_tok_s": None,
                "compile_support": False,
                "engine": "llama_cpp",
                "is_available": True,
            }]

        # ---- TP 孤岛引擎（配置启用即可选，延迟 import 保持冷启动优化） ----
        island_quants: List[Dict[str, Any]] = []
        try:
            import config as _cfg
            if getattr(_cfg, "ISLAND_ENABLED", False) and getattr(_cfg, "ISLAND_BASE_URL", ""):
                from island_engine import mask_island_url
                _island_url_masked = mask_island_url(getattr(_cfg, "ISLAND_BASE_URL", ""))
                available_engines.append({
                    "id": "island",
                    "name": f"TP 孤岛 ({getattr(_cfg, 'ISLAND_BACKEND', 'openai-compatible')})",
                    "description": f"整请求转发到孤岛端点 {_island_url_masked}，量化/并行由孤岛后端决定",
                    "model_size_gb": None,
                    "requires_cuda": False,
                })
                island_quants = [{
                    "id": "island",
                    "name": "孤岛后端",
                    "description": f"由孤岛后端决定（{getattr(_cfg, 'ISLAND_BACKEND', '')}，TP={getattr(_cfg, 'ISLAND_TP_SIZE', 1)}）",
                    "memory_gb": None,
                    "speed_tok_s": None,
                    "compile_support": False,
                    "engine": "island",
                    "is_available": True,
                }]
        except Exception:
            pass

        # ---- 外部推理服务（路线 B） ----
        external_quants: List[Dict[str, Any]] = []
        try:
            import config as _cfg2
            if getattr(_cfg2, "EXTERNAL_ENABLED", False) and getattr(_cfg2, "EXTERNAL_BASE_URL", ""):
                from external_provider import mask_external_url
                _external_url_masked = mask_external_url(getattr(_cfg2, "EXTERNAL_BASE_URL", ""))
                _external_scope = getattr(_cfg2, "EXTERNAL_DATA_SCOPE", "opt_in")
                available_engines.append({
                    "id": "external_api",
                    "name": f"外部推理服务 ({getattr(_cfg2, 'EXTERNAL_LABEL', '')})",
                    "description": (
                        f"整请求路由到外部端点 {_external_url_masked}"
                        f"（数据作用域: {_external_scope}，按请求 allow_external 授权，"
                        f"非本地引擎，无需加载）"
                    ),
                    "model_size_gb": None,
                    "requires_cuda": False,
                })
                external_quants = [{
                    "id": "external_api",
                    "name": "外部服务后端",
                    "description": (
                        f"由外部端点决定（{getattr(_cfg2, 'EXTERNAL_LABEL', '')}，"
                        f"作用域 {_external_scope}）"
                    ),
                    "memory_gb": None,
                    "speed_tok_s": None,
                    "compile_support": False,
                    "engine": "external_api",
                    "is_available": True,
                }]
        except Exception:
            pass

        loaded = bool(getattr(self._host, "model_loaded", False))
        current = getattr(self._host, "current_quant", None) if loaded else None
        current_engine = None
        if loaded:
            try:
                mgr = self._host
                if bool(getattr(mgr, "is_loaded", False)):
                    current_engine = backend_id_for(mgr) or None
            except Exception:
                current_engine = None
        return {
            "models": pytorch_quants + gguf_quants + island_quants + external_quants,
            "current": current,
            "current_engine": current_engine,
            "available_engines": available_engines,
        }

    def _model_api_payload(self, model) -> Dict[str, Any]:
        """序列化模型配置 + 落盘可用性（对齐 api_server.py:4960-5010）。"""
        import model_config as mc

        file_status = mc.get_model_file_status(model)
        supported_engines: List[str] = []
        if file_status["has_gguf"]:
            supported_engines.append("llama_cpp")
        if file_status["has_safetensors"]:
            supported_engines.append("pytorch")
        is_available = bool(supported_engines)
        unavailable_reason = file_status["unavailable_reason"]
        if file_status["is_available"] and not supported_engines:
            unavailable_reason = "模型文件已存在，但当前设备缺少可用推理后端。"
        from torch_runtime import cuda_available

        preferred_engine = preferred_engine_for_artifacts(
            has_safetensors=bool(file_status["has_safetensors"]),
            has_gguf=bool(file_status["has_gguf"]),
            cuda_available=effective_pytorch_cuda_available(
                system_cuda_available=cuda_available(load=True),
                profile=self._ensure_device_profile(),
            ),
        )
        default_quant = "Q4_K_M" if preferred_engine == "llama_cpp" else "int4"
        return {
            "model_id": model.model_id,
            "name": model.name,
            "is_builtin": mc.get_builtin_model(model.model_id) is not None,
            "model_type": model.model_type,
            "is_experimental": model.is_experimental,
            "recommended_vram_gb": model.recommended_vram_gb,
            "max_context": model.max_context,
            "quant_types": model.quant_types,
            "description": model.description,
            "huggingface_id": model.huggingface_id,
            "location": model.location,
            "model_path": model.model_path,
            "gguf_path": model.gguf_path,
            "is_available": is_available,
            "unavailable_reason": unavailable_reason,
            "available_formats": file_status["available_formats"],
            "has_safetensors": file_status["has_safetensors"],
            "has_gguf": file_status["has_gguf"],
            "expected_paths": file_status["expected_paths"],
            "supported_engines": supported_engines,
            "preferred_engine": preferred_engine,
            "default_quant_type": default_quant,
            "requires_cuda": bool(
                model.is_experimental
                and file_status["has_safetensors"]
                and "pytorch" not in supported_engines
            ),
        }

    # ------------------------------------------------------------------
    # 层段接口（client 角色；master 角色本地层段同用）
    # ------------------------------------------------------------------
    def load_layer_range(
        self,
        layer_range: str,
        embed: bool = False,
        lm_head: bool = False,
        *,
        model_id: str | None = None,
        model_path: str | None = None,
        quant_type: str | None = None,
        engine: str | None = None,
        layer_range_mode: str | None = None,
        model_sha256: str | None = None,
        resolution: ModelLoadResolution | None = None,
    ) -> Dict[str, Any]:
        """加载层段。layer_range 形如 "0-12"（[start, end)，对齐
        ModelManager.load_layer_range(start_layer, end_layer,
        has_embedding, has_lm_head) 语义。"""
        try:
            start_str, end_str = layer_range.split("-", 1)
            start_layer, end_layer = int(start_str), int(end_str)
        except (ValueError, AttributeError):
            raise ValueError(f"非法 layer_range: {layer_range!r}（期望如 '0-12'）")
        if resolution is not None:
            model_id = resolution.model_id
            model_path = resolution.model_path
            quant_type = resolution.quant_type
            engine = resolution.engine
            layer_range_mode = resolution.layer_range_mode

        with self._model_lifecycle_lock():
            if layer_range_mode not in {
                None, LAYER_RANGE_DYNAMIC, LAYER_RANGE_PRECUT_ARTIFACT,
            }:
                raise ValueError(
                    f"引擎 {engine or 'unknown'} 不支持层段模式 {layer_range_mode!r}"
                )
            transaction_id = f"layer_{uuid4().hex}"
            candidate_prepared = False
            try:
                self._host.prepare_pipeline_candidate(
                    transaction_id,
                    model_id=model_id or "",
                    model_path=model_path or "",
                    quant_type=quant_type,
                    layer_range=(start_layer, end_layer),
                    model_sha256=model_sha256,
                )
                candidate_prepared = True
                result = self._host.materialize_pipeline_candidate(
                    transaction_id,
                    start_layer=start_layer,
                    end_layer=end_layer,
                    has_embedding=embed,
                    has_lm_head=lm_head,
                    model_id=model_id,
                )
                self._host.commit_pipeline_candidate(transaction_id)
                self._host.finalize_pipeline_candidate(transaction_id)
            except Exception:
                if candidate_prepared:
                    try:
                        self._host.abort_pipeline_candidate(transaction_id)
                    except Exception:
                        logger.warning(
                            "层段候选事务回滚失败: transaction=%s",
                            transaction_id,
                            exc_info=True,
                        )
                raise
            self._host.model_loaded = False
            if quant_type:
                self._host.current_quant = quant_type
            if layer_range not in self._layers:
                self._layers.append(layer_range)
            return result if isinstance(result, dict) else {
                "success": True,
                "engine": engine,
                "model_id": model_id,
                "quant_type": quant_type,
                "layer_range": layer_range,
            }

    def unload_layer_range(self, layer_range: str) -> Dict[str, Any]:
        if layer_range in self._layers:
            self._layers.remove(layer_range)
        return {"success": True, "unloaded": layer_range}

    def forward_layers(
        self,
        layer_range: str,
        hidden,
        past_key_values=None,
        **kwargs,
    ):
        return self._host.forward_layers(
            layer_range=layer_range,
            hidden=hidden,
            past_key_values=past_key_values,
            **kwargs,
        )

    def embedding(self, input_ids):
        """Embedding 段（1.2 从 scheduler 流水线段复制真实实现）。"""
        raise NotImplementedError("embedding 段在 1.2 随流水线执行段复制接入")

    def lm_head(self, hidden):
        """LM Head 段（1.2 从 scheduler._run_master_lm_head 复制真实实现）。"""
        raise NotImplementedError("lm_head 段在 1.2 随流水线执行段复制接入")

    def ready(self) -> Dict[str, Any]:
        """/v1/ready：模型或层段就绪即 ready（冷启动方案 §5.4 语义）。"""
        loaded = bool(getattr(self._host, "model_loaded", False))
        return {
            "ready": loaded or bool(self._layers),
            "model_loaded": loaded,
            "layers": list(self._layers),
        }

    def status(self) -> Dict[str, Any]:
        """/v1/status：引擎、当前模型、显存、层段。"""
        mgr = self._host
        loaded = bool(getattr(self._host, "model_loaded", False))
        model = getattr(mgr, "model", None)
        device = str(getattr(model, "device", "")) if model is not None else None
        try:
            model_id = mgr.active_model_id() if callable(getattr(mgr, "active_model_id", None)) else None
        except Exception:
            model_id = None
        return {
            "engine": backend_id_for(mgr) or None,
            "model_id": model_id,
            "device": device,
            "model_loaded": loaded,
            "quant_type": getattr(self._host, "current_quant", None),
            "layers": list(self._layers),
        }

    def kv_state_kind(self) -> str:
        """★ #31 M6：本进程 KV 状态的**载体类型** —— `"tuple"` / `"hybrid"` / `"unknown"`。

        为什么要有它：`/layers/forward` 的 `past_key_values_ref` 走 `KVHost` 的
        **`PagedKVCache`（纯 KV 分页）**，而 hybrid（如 `qwen3_5`）的 `linear_attention` 层
        用的是 **recurrent state**（**不在** KV 分页里）⇒ 分页载体对它**根本不成立**：
        拿它当 `past_key_values` 会**静默丢状态**（同一轮不报错、数值却错，事后极难归因）。
        所以这里如实报出载体类型，让端点**明确 fail-closed**，而不是"看起来能跑"。

        判据用**层类型**（`model_module._is_hybrid_layer_types`），**不**按 `model_type`
        白名单 —— 后者会随新架构漂移，而"存在 linear_attention 层"才是"存在 recurrent state"
        的本质（同一个判定也被 `forward_layers` 用来决定 mask/cache 形态，口径一致）。
        """
        try:
            from model_module import _is_hybrid_layer_types
        except ImportError:  # pragma: no cover - 环境相关
            return "unknown"
        model = getattr(self._host, "model", None)
        config = getattr(model, "config", None)
        if config is None:
            return "unknown"
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            # 没有 `layer_types` ⇒ 纯 full-attention（老架构）⇒ 沿用既有 KV 契约
            return "tuple"
        return "hybrid" if _is_hybrid_layer_types(layer_types) else "tuple"

    def generation_status(self) -> Dict[str, Any]:
        """Return active generation IDs for cancellation and leak diagnostics."""

        with self._gen_lock:
            generation_ids = sorted(self._generations)
        return {
            "active_generation_count": len(generation_ids),
            "active_generation_ids": generation_ids,
        }

    def conversation_status(self) -> Dict[str, Any]:
        """Return aggregate conversation state without exposing message content."""

        with self._gen_lock:
            return {
                "active_session_id": self._active_session_id,
                "session_message_counts": {
                    session_id: len(history)
                    for session_id, history in self._session_histories.items()
                },
                "stats": dict(self._conversation_stats),
            }

    def session_history(self, session_id: str) -> list[dict]:
        """Return a defensive copy of one in-memory session history."""

        with self._gen_lock:
            return [dict(message) for message in self._session_histories.get(session_id, [])]

    def kv_cache_status(self) -> Dict[str, Any]:
        """Return non-content metadata for the local paged KV cache."""

        cache = self._kv_cache
        if cache is None:
            return {"initialized": False}
        return {
            "initialized": True,
            "page_size": getattr(cache, "page_size", None),
            "max_pages": getattr(cache, "max_pages", None),
            "device": str(getattr(cache, "device", "")),
        }

    # ------------------------------------------------------------------
    # 对话（1.1 薄实现：本地模型 chat/chat_stream；
    # 1.2 替换为 _execute_chat_full / fast 模式副本，含历史/追问/持久化）
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 完整聊天流程（1.2c 复制自 api_server._execute_chat_full
    # api_server.py:3323-3872；宿主适配：kv_cache/conversation_stats/
    # 会话状态 → 实例属性、model_manager/model_host → self._host、
    # scheduler → self._scheduler 注入点（None=单机基线）、
    # RUN_MODE → self._run_mode）
    # ------------------------------------------------------------------
    def chat_full(
        self,
        req: ChatRequest,
        cancel_event: Optional[threading.Event] = None,
        request_deadline: Optional[RequestDeadline] = None,
    ) -> Dict[str, Any]:
        """
        执行完整聊天流程 — 从 /api/chat 提取的共用核心逻辑。

        处理: 会话切换、自动标题、客户端转发、流水线推理、
              llama.cpp、PyTorch、历史维护、DB 持久化、追问生成。

        Returns:
            {"content": str, "thinking_content": str|None,
             "metrics": dict, "followups": list[str]}

        Raises:
            HTTPException: 模型未加载、OOM、推理失败
        """
        import time as _time
        from fastapi import HTTPException
        import config as _cfg

        request_deadline = coerce_request_deadline(
            request_deadline,
            timeout_seconds=_cfg.PIPELINE_TIMEOUT,
        )
        request_deadline.require_remaining()

        # ---- task_graph 分支（1.2d：对齐 api_server._execute_requested_chat）----
        if req.execution_mode == "task_graph":
            try:
                return self.execute_task_graph_chat(
                    req,
                    cancel_event,
                    request_deadline=request_deadline,
                )
            except TypeError as exc:
                if "request_deadline" not in str(exc):
                    raise
                return self.execute_task_graph_chat(req, cancel_event)

        # T9.5：distributed_required 无分布式路径时明确失败（full 模式）
        routing_gate = self._routing_gate_error(req)
        if routing_gate:
            from fastapi import HTTPException as _HE
            raise _HE(400, routing_gate)

        _raise_if_generation_cancelled(cancel_event, req.generation_id)

        # ---- 多会话支持 ----
        prepared_context = self._prepare_chat_context(req.session_id, req.message)
        request_deadline.require_remaining()
        target_session_id = prepared_context.session_id
        history = prepared_context.history

        # ---- 路线 B：外部推理服务整请求路由（数据作用域门控，默认不出集群）----
        external_fallback_reason = ""
        _ext_decision = self._external_route_decision(req)
        image_data_urls = getattr(req, "image_data_urls", [])
        local_native_image = bool(
            image_data_urls and self._local_native_vision_available()
        )
        if (
            local_native_image
            and not _ext_decision.use_external
            and len(image_data_urls) != 1
        ):
            raise HTTPException(
                400,
                "本地 MTMD 图像回退仅支持一张图片；多图请求不能静默丢弃其余图片",
            )
        if image_data_urls and not _ext_decision.use_external and not local_native_image:
            raise HTTPException(
                400,
                "当前本地模型没有原生视觉能力，且 external_api 不可用或未获授权："
                f"{_ext_decision.reason}",
            )
        if _ext_decision.use_external:
            try:
                return self.execute_external_chat(
                    req,
                    history,
                    target_session_id,
                    cancel_event,
                    context_revision=prepared_context.revision,
                )
            except ChatGenerationCancelled:
                raise
            except HTTPException:
                raise
            except Exception as exc:
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                if image_data_urls and not local_native_image:
                    raise HTTPException(
                        502,
                        "多模态外部推理服务调用失败，禁止丢弃图片后回退："
                        f"{exc}",
                    ) from exc
                if not self._host.model_loaded or not getattr(self._host, "is_loaded", False):
                    if req.prefer_external:
                        raise HTTPException(
                            502,
                            f"外部推理服务调用失败，且本地无可用推理引擎：{exc}",
                        ) from exc
                    try:
                        self._ensure_chat_model_or_forwarding()
                    except HTTPException:
                        raise
                    except Exception as load_exc:
                        raise HTTPException(
                            502,
                            f"外部推理服务调用失败（{exc}），"
                            f"且本地模型加载失败：{load_exc}",
                        ) from exc
                external_fallback_reason = f"external_api_failed: {exc}"
                logger.warning(f"外部推理服务调用失败: {exc}，回退到本地推理路径")

        # ---- 分布式推理路由：从节点转发给主节点（local_only 强制本地）----
        sched = self._scheduler
        pipeline_attempted = False
        pipeline_failure_reason = ""
        distributed_enabled = bool(
            sched is not None and sched.get_distributed_inference_enabled()
        )
        if (req.routing_preference != "local_only"
                and distributed_enabled
                and self._run_mode == "distributed"
                and sched._effective_role() == "client"):
            try:
                result = sched.forward_inference_to_master(
                    message=req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    show_thinking=req.show_thinking,
                    routing_preference=req.routing_preference,
                    session_id=req.session_id,
                    messages=list(history) + [{"role": "user", "content": req.message}],
                    request_id=_request_id_ctx.get("-"),
                    _cancel_event=cancel_event,
                    _request_deadline=request_deadline,
                )
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                if result.get("reason_code") == REQUEST_DEADLINE_EXCEEDED:
                    raise coded_http_error(
                        504,
                        REQUEST_DEADLINE_EXCEEDED,
                        result.get("error") or "request deadline exceeded",
                    )
                if result.get("status") == "ok":
                    response_text = result.get("content", "")
                    forward_metrics = _augment_chat_metrics(
                        result.get("metrics", {}),
                        req,
                        serving_node_id=self._serving_node_id,
                        distributed_enabled=distributed_enabled,
                        engine="distributed_forward",
                        execution_mode="forwarded_to_master",
                        route="pc_client_forward_to_master",
                    )
                    if external_fallback_reason and not forward_metrics.get(
                        "fallback_reason",
                    ):
                        forward_metrics["fallback"] = True
                        forward_metrics["fallback_reason"] = external_fallback_reason

                    self._enforce_distributed_required(
                        req,
                        forward_metrics,
                        detail="从节点转发结果未标记为分布式执行",
                    )

                    self._commit_chat_context_turn(
                        target_session_id, req.message, response_text, forward_metrics,
                        expected_revision=prepared_context.revision,
                    )

                    self._conversation_stats["rounds"] += 1
                    self._record_task_complete()

                    master_followups = result.get("followups", [])
                    if master_followups:
                        followups = master_followups[:3]
                    else:
                        followups = _fallback_followups(history, [])

                    return {
                        "content": response_text,
                        "thinking_content": result.get("thinking_content"),
                        "metrics": forward_metrics,
                        "followups": followups,
                    }
                elif result.get("status") == "disconnected":
                    logger.warning("分布式推理转发失败（未连接主节点），回退到本地推理")
                    self._enforce_distributed_required(req, detail="从节点未连接主节点")
                elif result.get("status") == "timeout":
                    logger.warning("分布式推理转发超时，回退到本地推理")
                    self._enforce_distributed_required(req, detail="分布式转发超时")
                else:
                    logger.warning(f"分布式推理转发失败: {result.get('error', 'unknown')}，回退到本地推理")
                    self._enforce_distributed_required(
                        req, detail=str(result.get("error", "分布式转发失败")),
                    )
            except ChatGenerationCancelled:
                raise
            except HTTPException:
                raise
            except Exception as e:
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                logger.warning(f"分布式推理转发异常: {e}，回退到本地推理")
                self._enforce_distributed_required(req, detail=str(e))

            if self._pipeline_worker_is_reserved():
                raise HTTPException(
                    503,
                    "本设备正作为 PyTorch 分层从节点，"
                    "当前无法转发到主节点，已拒绝覆盖分层模型。",
                )
            if not self._host.model_loaded or not getattr(self._host, "is_loaded", False):
                self._auto_load_default_model()

        if self._pipeline_worker_is_reserved():
            raise HTTPException(
                503,
                "本设备正作为 PyTorch 分层从节点，"
                "请先断开主节点或明确切换本地模型。",
            )

        # ---- 分布式流水线推理路径（主节点 + PyTorch 引擎 + 从节点可用；local_only 跳过）----
        if (req.routing_preference != "local_only"
                and distributed_enabled
                and self._run_mode == "distributed"
                and sched._effective_role() == "master"
                and runtime_supports(self._host, Capability.FORWARD_LAYERS)):
            pipeline_attempted = True
            try:
                pipeline_result = sched.run_pipeline_safe(
                    req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    session_id=req.session_id,
                    messages=list(history) + [{"role": "user", "content": req.message}],
                    show_thinking=req.show_thinking,
                    _require_distributed=(req.routing_preference == "distributed_required"),
                    _force_distributed_assignment=True,
                    _cancel_event=cancel_event,
                    _request_deadline=request_deadline,
                )
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                if pipeline_result.get("error"):
                    if (
                        pipeline_result.get("reason_code")
                        == REQUEST_DEADLINE_EXCEEDED
                    ):
                        raise coded_http_error(
                            504,
                            REQUEST_DEADLINE_EXCEEDED,
                            pipeline_result.get("error")
                            or "request deadline exceeded",
                        )
                    pipeline_failure_reason = str(pipeline_result["error"])
                    logger.warning(f"流水线推理失败: {pipeline_result['error']}，回退到本地推理")
                    self._enforce_distributed_required(
                        req, detail=str(pipeline_result["error"]),
                    )
                else:
                    response_text = pipeline_result.get("response", "")
                    if not response_text:
                        logger.warning("流水线返回空响应，回退到本地推理")
                        self._enforce_distributed_required(
                            req, detail="流水线返回空响应",
                        )
                    else:
                        pipeline_metrics = _augment_chat_metrics(
                            pipeline_result.get("metrics", {}),
                            req,
                            serving_node_id=self._serving_node_id,
                            distributed_enabled=distributed_enabled,
                            engine="distributed_pipeline",
                            execution_mode="distributed_pipeline",
                            route="master_pipeline",
                        )
                        if external_fallback_reason and not pipeline_metrics.get(
                            "fallback_reason",
                        ):
                            pipeline_metrics["fallback"] = True
                            pipeline_metrics["fallback_reason"] = (
                                external_fallback_reason
                            )
                        self._enforce_distributed_required(
                            req,
                            pipeline_metrics,
                            detail="流水线结果未标记为分布式执行",
                        )
                        self._commit_chat_context_turn(
                            target_session_id, req.message, response_text, pipeline_metrics,
                            expected_revision=prepared_context.revision,
                        )

                        self._conversation_stats["rounds"] += 1
                        if not pipeline_metrics.get("distributed_used"):
                            self._record_task_complete()

                        if backend_id_for(self._host) in ("llama_cpp", "island"):
                            followups = self.generate_followups_llama(history)
                        elif backend_id_for(self._host) == "pytorch":
                            followups = _fallback_followups(history, [])
                        else:
                            followups = _fallback_followups(history, [])

                        return {
                            "content": response_text,
                            "thinking_content": pipeline_result.get("thinking"),
                            "metrics": pipeline_metrics,
                            "followups": followups,
                        }
            except ChatGenerationCancelled:
                raise
            except HTTPException:
                raise
            except Exception as e:
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                pipeline_failure_reason = str(e)
                logger.warning(f"流水线推理异常: {e}，回退到本地推理")
                self._enforce_distributed_required(req, detail=str(e))

        if req.routing_preference == "distributed_required":
            raise HTTPException(
                503,
                "distributed_required 执行失败：当前引擎未完成分布式流水线",
            )

        if self._pipeline_model_is_prepared() and not pipeline_attempted:
            raise HTTPException(
                409 if req.routing_preference == "local_only" else 503,
                "当前模型仅以分布式流水线模式准备，禁止作为本地完整模型执行；"
                "请等待流水线就绪或显式加载完整模型。",
            )

        model_manager = self._host
        # ---- llama.cpp / 孤岛引擎路径（整请求推理，不参与层拆分）----
        if backend_id_for(model_manager) in ("llama_cpp", "island"):
            if pipeline_attempted:
                # A1 relay + A3 Route-A mixed failures must not bypass the
                # full-model guard and run a partial GGUF as a chat model.
                ensure_full = getattr(model_manager, "ensure_full_model", None)
                if not callable(ensure_full):
                    raise HTTPException(
                        503,
                        "分布式流水线失败且当前引擎不提供整模回退校验",
                    )
                try:
                    ensure_full()
                except Exception as exc:
                    raise HTTPException(
                        503,
                        f"分布式流水线失败，整模回退已拒绝: {exc}",
                    ) from exc
            try:
                engine_name = backend_id_for(model_manager)
                request_history = [
                    *history,
                    {"role": "user", "content": req.message},
                ]
                if local_native_image:
                    if engine_name != "llama_cpp":
                        raise RuntimeError("本地原生图像请求要求 llama.cpp MTMD 引擎")
                    with materialize_image_data_url(image_data_urls[0]) as image_path:
                        result = model_manager.chat_image(
                            image_path=image_path,
                            prompt=req.message,
                            max_tokens=min(req.max_new_tokens + 96, 512),
                            max_answer_tokens=min(req.max_new_tokens, 256),
                            temperature=req.temperature,
                            top_p=req.top_p,
                            _cancel_event=cancel_event,
                        )
                else:
                    result = model_manager.chat(
                        messages=request_history,
                        max_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        _cancel_event=cancel_event,
                    )
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                response_text = result.get("content", "")
                if not req.show_thinking:
                    response_text = _strip_native_thinking_tags(response_text)
                completed_history = [
                    *request_history,
                    {"role": "assistant", "content": response_text},
                ]
                tokens_per_sec = result.get("tokens_per_second", 0)
                usage = result.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                route_suffix = f"{engine_name}_mtmd" if local_native_image else engine_name
                local_route = f"{_chat_origin(req)}_to_master_local_{route_suffix}"
                fallback_reason = ""
                if external_fallback_reason:
                    fallback_reason = external_fallback_reason
                elif pipeline_failure_reason:
                    fallback_reason = pipeline_failure_reason
                elif distributed_enabled and self._run_mode == "distributed":
                    if engine_name == "island":
                        fallback_reason = "island engine delegates whole-request inference to the TP island"
                    else:
                        fallback_reason = "llama.cpp engine does not support layer-split pipeline"
                metrics = _augment_chat_metrics(
                    {
                        "engine": engine_name,
                        "execution_mode": f"local_{route_suffix}",
                        "route": local_route,
                        "native_mtmd": local_native_image,
                        "tokens_per_second": round(tokens_per_sec, 1) if tokens_per_sec else 0,
                        "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else 0,
                        "generated_tokens": completion_tokens,
                        "completion_tokens": completion_tokens,
                        "usage": usage,
                        "fallback": bool(fallback_reason),
                        "fallback_reason": fallback_reason,
                    },
                    req,
                    serving_node_id=self._serving_node_id,
                    distributed_enabled=distributed_enabled,
                )

                followups = self.generate_followups_llama(
                    completed_history, cancel_event,
                )
                _raise_if_generation_cancelled(cancel_event, req.generation_id)

                save_metrics = dict(metrics)
                save_metrics["followups"] = followups
                self._commit_chat_context_turn(
                    target_session_id, req.message, response_text, save_metrics,
                    expected_revision=prepared_context.revision,
                )

                self._conversation_stats["total_generated_tokens"] += completion_tokens
                self._conversation_stats["rounds"] += 1
                self._record_task_complete()

                return {
                    "content": response_text,
                    "thinking_content": None,
                    "metrics": metrics,
                    "followups": followups,
                }
            except ChatGenerationCancelled:
                raise
            except HTTPException:
                raise
            except Exception as e:
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                self._record_task_error()
                _engine_label = (
                    "孤岛引擎"
                    if backend_id_for(model_manager) == "island"
                    else "llama.cpp"
                )
                logger.error(f"{_engine_label} 推理失败: {e}", exc_info=True)
                raise HTTPException(500, f"推理失败: {str(e)}")

        # ---- PyTorch 引擎路径（CUDA/独显）----
        # Torch is only needed after routing has selected the PyTorch backend;
        # keep the GGUF path usable in the torch-free distribution.
        import torch as _torch

        try:
            model_manager.ensure_full_model()
            tier_max = self._host.generation_config.get("tier_max_new_tokens", self._host.generation_config["max_new_tokens"])
            thinking_budget = 384 if req.show_thinking else 0
            effective_max = min(req.max_new_tokens + thinking_budget,
                                tier_max + thinking_budget,
                                4096)
            self._host.generation_config["max_new_tokens"] = effective_max
            self._host.generation_config["temperature"] = req.temperature
            self._host.generation_config["top_p"] = req.top_p

            request_history = [
                *history,
                {"role": "user", "content": req.message},
            ]

            tokenizer = model_manager.tokenizer
            thinking_prompt = THINKING_SYSTEM_PROMPT if req.show_thinking else None
            thinking_prefill = "【思考】\n" if req.show_thinking else None
            prompt = _build_model_chat_prompt(
                tokenizer,
                request_history,
                system_prompt=thinking_prompt,
                assistant_prefill=thinking_prefill,
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model_manager.get_device())
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_manager.get_device())
            prompt_len = input_ids.shape[1]
            stop_sequences = model_manager._merge_stop_sequences(None)
            generation_kwargs = {}
            eos_token_ids = model_manager._get_generation_eos_token_ids(stop_sequences)
            if eos_token_ids is not None:
                generation_kwargs["eos_token_id"] = eos_token_ids
            stop_criteria_kwargs = (
                {"cancel_event": cancel_event} if cancel_event is not None else {}
            )
            stop_criteria = model_manager._build_stop_criteria(
                stop_sequences, prompt_len, **stop_criteria_kwargs,
            )
            if stop_criteria is not None:
                generation_kwargs["stopping_criteria"] = stop_criteria

            t0 = _time.time()
            with _torch.no_grad():
                outputs = model_manager.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=effective_max,
                    temperature=req.temperature if req.temperature > 0 else 1.0,
                    top_p=req.top_p,
                    do_sample=req.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    **generation_kwargs,
                )
            _raise_if_generation_cancelled(cancel_event, req.generation_id)
            elapsed = _time.time() - t0

            generated_ids = outputs[0][prompt_len:]
            raw_text = model_manager._decode_generated_ids(generated_ids, stop_sequences).strip()

            native_thinking_prompt = "<think>" in prompt[-128:].lower()
            parsed_text = raw_text
            if req.show_thinking and not native_thinking_prompt and "<think" not in raw_text.lower():
                parsed_text = "【思考】\n" + raw_text
            response_text, thinking_content = _format_model_response(
                parsed_text,
                req.show_thinking,
                native_thinking_prompt=native_thinking_prompt,
            )

            completed_history = [
                *request_history,
                {"role": "assistant", "content": response_text},
            ]

            new_tokens = len(generated_ids)
            tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0
            metrics = _augment_chat_metrics(
                {
                    "engine": "pytorch",
                    "execution_mode": "local_pytorch",
                    "route": f"{_chat_origin(req)}_to_master_local_pytorch",
                    "prompt_tokens": prompt_len,
                    "new_tokens": new_tokens,
                    "generated_tokens": new_tokens,
                    "total_tokens": prompt_len + new_tokens,
                    "elapsed_seconds": round(elapsed, 3),
                    "tokens_per_second": round(tokens_per_sec, 1),
                    "gpu_memory_mb": round(_torch.cuda.memory_allocated() / (1024**2), 1)
                    if _torch.cuda.is_available()
                    else 0,
                },
                req,
                serving_node_id=self._serving_node_id,
                distributed_enabled=distributed_enabled,
                # ★ 2026-10-05（DIST-4）：与 `api_server.py` 的同名路径对齐 ——
                #   此前只用 `external_fallback_reason`，丢掉了 `:1296` / `:1356`
                #   写入的 `pipeline_failure_reason` ⇒ 「pipeline 失败 ⇒ 本地整模回退」
                #   会报成 `fallback=False` + 空 reason，与「从没请求分布式」无法区分。
                #   注意 `:1422-1423` 那处本地回退的用法本来就是对的，只有这里漏了。
                fallback=bool(
                    external_fallback_reason or pipeline_failure_reason
                ),
                fallback_reason=(
                    external_fallback_reason
                    or (
                        "pipeline_failed_then_local_pytorch: "
                        f"{pipeline_failure_reason}"
                        if pipeline_failure_reason else ""
                    )
                ),
            )

            followups = self._generate_followups(
                completed_history,
                tokenizer,
                model_manager.model,
                model_manager.get_device(),
                cancel_event,
            )
            _raise_if_generation_cancelled(cancel_event, req.generation_id)

            save_metrics = dict(metrics)
            save_metrics["followups"] = followups
            self._commit_chat_context_turn(
                target_session_id, req.message, response_text, save_metrics,
                expected_revision=prepared_context.revision,
            )

            self._conversation_stats["total_prompt_tokens"] += prompt_len
            self._conversation_stats["total_generated_tokens"] += new_tokens
            self._conversation_stats["total_time_seconds"] += elapsed
            self._conversation_stats["rounds"] += 1
            self._record_task_complete()

            logger.info(
                f"推理完成: {new_tokens} tokens / {elapsed:.2f}s = {tokens_per_sec:.1f} tok/s"
            )

            return {
                "content": response_text,
                "thinking_content": thinking_content,
                "metrics": metrics,
                "followups": followups,
            }

        except ChatGenerationCancelled:
            raise
        except RequestDeadlineExceeded as exc:
            raise coded_http_error(
                504, REQUEST_DEADLINE_EXCEEDED, str(exc),
            ) from exc
        except HTTPException:
            raise
        except _torch.cuda.OutOfMemoryError:
            self._record_task_error()
            if self._kv_cache:
                self._kv_cache.clear()
            self._get_active_history().clear()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            raise HTTPException(507, "GPU 显存不足（OOM），已自动清空对话历史。请缩短消息后重试。")

        except Exception as e:
            self._record_task_error()
            logger.error(f"推理异常: {e}", exc_info=True)
            raise HTTPException(500, f"推理失败: {str(e)}")

    def _record_task_complete(self) -> None:
        if self._on_task_complete is not None:
            try:
                self._on_task_complete(success=True)
            except Exception:
                pass

    def _record_task_error(self) -> None:
        if self._on_task_error is not None:
            try:
                self._on_task_error()
            except Exception:
                pass

    def chat_stream_events(
        self,
        req: ChatRequest,
        cancel_event: Optional[threading.Event],
        request_deadline: Optional[RequestDeadline] = None,
    ):
        """SSE 事件序列（1.1 薄实现；1.2 替换为 fast 模式副本）。

        Yields:
            dict 事件：{"token": ...} 或 {"done": True, "response": ..., "metrics": ...}
        """
        import config as _cfg

        request_deadline = coerce_request_deadline(
            request_deadline,
            timeout_seconds=_cfg.PIPELINE_TIMEOUT,
        )
        if not isinstance(cancel_event, DeadlineCancelEvent):
            cancel_event = DeadlineCancelEvent(
                cancel_event, request_deadline,
            )
        request_deadline.require_remaining()
        self._enforce_distributed_required(
            req, detail="inference-service interactive 仅实现本地流式执行",
        )
        if getattr(req, "image_data_urls", []):
            raise RuntimeError("图像请求仅支持 full 响应模式，禁止忽略图片后执行文本流式推理")
        prepared_context = self._prepare_chat_context(req.session_id, req.message)
        request_deadline.require_remaining()
        messages = [
            *prepared_context.history,
            {"role": "user", "content": req.message},
        ]
        response_parts: list[str] = []
        try:
            chunks = self._host.chat_stream(
                messages=messages,
                max_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                _cancel_event=cancel_event,
            )
        except TypeError as exc:
            if "_cancel_event" not in str(exc):
                raise
            chunks = self._host.chat_stream(
                messages=messages,
                max_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
        for chunk in chunks:
            stop_reason = request_stop_reason(
                cancel_event, request_deadline,
            )
            if stop_reason == REQUEST_DEADLINE_EXCEEDED:
                yield {
                    "done": True,
                    "error": "request deadline exceeded",
                    "reason_code": REQUEST_DEADLINE_EXCEEDED,
                }
                return
            if stop_reason:
                raise ChatGenerationCancelled(req.generation_id or "gen_unknown")
            response_parts.append(str(chunk))
            yield {"token": chunk}
        if request_deadline.expired():
            yield {
                "done": True,
                "error": "request deadline exceeded",
                "reason_code": REQUEST_DEADLINE_EXCEEDED,
            }
            return
        _raise_if_generation_cancelled(cancel_event, req.generation_id)
        response_text = "".join(response_parts)
        metrics = {
            "engine": backend_id_for(self._host) or "unknown",
            "execution_mode": "streaming",
            "request_id": _request_id_ctx.get("-"),
        }
        yield {
            "done": True,
            "response": response_text,
            "followups": [],
            "metrics": metrics,
            "request_id": "-",
            "session_id": prepared_context.session_id,
            "history_committed": False,
            # Private handoff consumed by routes.py before SSE serialization.
            # The producer thread must never commit before the client-side
            # consumer accepts this terminal event.
            "_context_session_id": prepared_context.session_id,
            "_context_revision": prepared_context.revision,
        }

    def speculative_run(self, req) -> Dict[str, Any]:
        """投机解码实验端点（1.2b 复制自 api_server._run_speculative_experiment
        api_server.py:2506-2519；SPEC_ENABLED 门控与异常映射在 routes 层）。"""
        from speculative import run_speculative_chat

        return run_speculative_chat(
            req.message,
            allow_external=bool(req.allow_external),
            max_new_tokens=int(req.max_new_tokens),
            gamma=(int(req.gamma) if req.gamma > 0 else None),
            max_rounds=(int(req.max_rounds) if req.max_rounds > 0 else None),
            temperature=(float(req.temperature) if req.temperature >= 0 else None),
            seed=int(req.seed),
            draft_hint=req.draft_hint or "",
        )

    # ------------------------------------------------------------------
    # 追问生成（1.2b 复制自 api_server._generate_followups_llama
    # api_server.py:1356-1435；宿主适配：model_manager → self._host）
    # ------------------------------------------------------------------
    def generate_followups_llama(
        self,
        history: list,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[str]:
        """
        使用 llama.cpp 引擎生成追问建议。

        通过 model_host.chat() 调用（llama.cpp 路径），失败时回退到
        关键词模板兜底（_fallback_followups）。
        """
        if not history or len(history) < 2:
            return []
        if cancel_event is not None and cancel_event.is_set():
            return []

        system_prompt = (
            "根据对话内容，生成2-3个你会追问的问题。每个问题一行，以？结尾。"
        )
        followup_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"根据以下对话，生成我想追问的问题：\n"
             f"用户：{history[-2]['content'][:200]}\n"
             f"助手：{history[-1]['content'][:300]}"},
        ]

        questions = []
        try:
            result = self._host.chat(
                messages=followup_messages,
                max_tokens=128,
                temperature=0.8,
                top_p=0.9,
                _cancel_event=cancel_event,
            )
            if cancel_event is not None and cancel_event.is_set():
                return []
            text = result.get("content", "").strip()

            for line in text.split("\n"):
                line = line.strip()
                line = re.sub(r'^[\d]+[\.\、\)）\s\-]+', '', line).strip()
                if line.upper().startswith("Q:") or line.upper().startswith("Q："):
                    line = line.split(":", 1)[-1].split("：", 1)[-1].strip()
                if line and len(line) >= 5 and len(line) <= 80 and _is_question(line):
                    questions.append(line)

            hallucination_patterns = [
                "通义千问", "千问", "ChatGPT", "Claude", "GPT-", "文心一言",
                "讯飞星火", "豆包", "Kimi", "Copilot", "Bard", "Gemini",
                "百川", "智谱", "ChatGLM", "混元",
            ]
            questions = [q for q in questions if not any(p in q for p in hallucination_patterns)]

            filtered = []
            seen = set()
            for q in questions:
                key = q[:15]
                if key not in seen:
                    seen.add(key)
                    filtered.append(q)
            questions = filtered

            logger.info(f"llama.cpp 追问生成: {len(questions)} 条 → {questions}")

        except Exception as e:
            logger.warning(f"llama.cpp 追问生成失败（非致命）: {e}")
            questions = []

        if len(questions) < 2:
            fallback = _fallback_followups(history, questions)
            questions = fallback

        return questions[:3]

    # ------------------------------------------------------------------
    # task-worker 数据面执行段（1.2a 复制自 api_server._execute_task_worker_stage
    # api_server.py:2734-2872，源文件保持不动；宿主适配：model_manager → self._host）
    # ------------------------------------------------------------------
    def execute_task_worker_stage(
        self,
        stage_request,
        provider_cancel_event: threading.Event,
    ) -> Dict[str, Any]:
        """Execute the shared local/remote Stage contract on a full model."""
        from task_graph import DEPENDENCY_FAILURES_KEY, TaskGraphError
        from task_provider import ProviderError, ProviderExecutionError

        # Every public Route-A execution enters the committed prepare gate.
        # The internal helper remains usable for legacy native unit fixtures.
        if stage_request.stage_type == "layer_forward":
            with self._worker_layer_lock():
                self._check_worker_runtime_stage(stage_request, required=True)
                return self._execute_layer_forward_stage_locked(stage_request, provider_cancel_event)
        root_input = stage_request.root_input
        options = root_input.get("task_options", {})
        if not isinstance(options, dict):
            raise TaskGraphError("任务 Stage 缺少有效执行参数")
        try:
            candidate_budget = max(
                1, min(int(options.get("candidate_max_tokens", 512)), 512)
            )
            final_budget = max(
                1, min(int(options.get("final_max_tokens", 1024)), 1024)
            )
            temperature = max(
                0.0, min(float(options.get("temperature", 0.7)), 2.0)
            )
            top_p = max(0.0, min(float(options.get("top_p", 0.9)), 1.0))
        except (TypeError, ValueError) as exc:
            raise TaskGraphError("任务 Stage 执行参数无效") from exc
        show_thinking = bool(options.get("show_thinking", False))
        model_manager = self._host

        def run_model(
            messages: list[dict],
            max_tokens: int,
            *,
            retry_empty_on_same_provider: bool = False,
        ) -> dict:
            if provider_cancel_event.is_set():
                return {
                    "content": "",
                    "usage": {},
                    "tokens_per_second": 0,
                    "model": model_manager.active_model_id,
                }
            result = model_manager.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                _cancel_event=provider_cancel_event,
            )
            raw_content = str(result.get("content", "") or "").strip()
            thinking_content = None
            if show_thinking:
                content, thinking_content = _format_model_response(
                    raw_content, show_thinking=True,
                )
            else:
                content = _strip_native_thinking_tags(raw_content)
            if not content and not provider_cancel_event.is_set():
                if retry_empty_on_same_provider:
                    raise ProviderExecutionError(
                        "complete model returned an empty aggregate result",
                        code="empty_provider_output",
                        provider_id=stage_request.provider_id,
                        same_provider_retryable=True,
                    )
                raise TaskGraphError("完整模型返回空 Stage 结果")
            return {
                "content": content,
                "thinking_content": thinking_content,
                "usage": dict(result.get("usage", {}) or {}),
                "tokens_per_second": result.get("tokens_per_second", 0),
                "model": result.get("model", model_manager.active_model_id),
                "usage_estimated": bool(result.get("usage_estimated", False)),
            }

        if stage_request.stage_type == "full_inference":
            candidate_instructions = {
                "candidate_a": (
                    "独立分析用户问题，给出准确、可验证且简洁的候选答案。"
                    "不要提及其他候选或任务链。"
                ),
                "candidate_b": (
                    "从不同角度独立解决用户问题，重点检查遗漏、反例和不确定性。"
                    "输出可直接供后续汇总的候选答案。"
                ),
            }
            instruction = candidate_instructions.get(stage_request.stage_id)
            messages = root_input.get("messages")
            if instruction is None or not isinstance(messages, list):
                raise TaskGraphError("完整推理 Stage 输入无效")
            if show_thinking:
                instruction = f"{instruction}\n\n{THINKING_SYSTEM_PROMPT}"
            return run_model(
                [{"role": "system", "content": instruction}, *messages],
                candidate_budget,
            )
        if stage_request.stage_type == "aggregate":
            message = str(root_input.get("message", "") or "")
            if not message:
                raise TaskGraphError("聚合 Stage 缺少原始问题")
            candidate_payload = {
                stage_id: value.get("content", "")
                for stage_id, value in stage_request.dependencies.items()
                if stage_id != DEPENDENCY_FAILURES_KEY
                and isinstance(value, dict)
                and str(value.get("content", "") or "").strip()
            }
            if not candidate_payload:
                raise TaskGraphError("聚合 Stage 没有可用候选")
            failure_payload = stage_request.dependencies.get(
                DEPENDENCY_FAILURES_KEY, {},
            )
            aggregation_prompt = (
                "请根据原始问题和可用的独立候选，输出一个最终答案。"
                "纠正冲突和明显错误；没有证据时明确不确定性。"
                "只输出最终答案，不描述内部任务链。\n\n"
                f"原始问题：{message}\n\n候选：\n"
                + json.dumps(candidate_payload, ensure_ascii=False)
                + (
                    "\n\n未完成候选摘要：\n"
                    + json.dumps(failure_payload, ensure_ascii=False)
                    if isinstance(failure_payload, dict) and failure_payload
                    else ""
                )
            )
            try:
                return run_model(
                    ([{"role": "system", "content": THINKING_SYSTEM_PROMPT}]
                     if show_thinking else [])
                    + [{"role": "user", "content": aggregation_prompt}],
                    final_budget,
                    retry_empty_on_same_provider=True,
                )
            except ProviderError:
                raise
            except (TimeoutError, ConnectionError) as exc:
                raise ProviderExecutionError(
                    "transient aggregate model execution failed",
                    code="provider_execution_failed",
                    provider_id=stage_request.provider_id,
                    same_provider_retryable=True,
                ) from exc
        if stage_request.stage_type == "layer_forward":
            return self._execute_layer_forward_stage(
                stage_request, provider_cancel_event,
            )
        raise TaskGraphError(f"不支持的 Stage 类型: {stage_request.stage_type}")

    # ------------------------------------------------------------------
    # v3 层段 Stage（`layer_forward`）—— PC 侧执行
    # ------------------------------------------------------------------

    def _worker_layer_lock(self):
        lock = getattr(self, "_worker_runtime_lock", None)
        if lock is None:
            lock = self._worker_runtime_lock = threading.RLock()
        return lock

    def fence_worker_runtime(self):
        # Admission/disconnect must fence immediately, without waiting for a
        # native forward or unload that holds the runtime lock.
        event = getattr(self, "_worker_runtime_fenced", None)
        if event is None:
            event = self._worker_runtime_fenced = threading.Event()
        event.set()

    def _resume_worker_runtime(self):
        event = getattr(self, "_worker_runtime_fenced", None)
        if event is not None:
            event.clear()

    def _capture_worker_runtime(self):
        from worker_prepare import LayerRuntime
        active = getattr(self, "_worker_runtime_active", None)
        if active is None:
            active = LayerRuntime(None)
        # Preserve ownership on partial close failures and exact KV rollback.
        active.upstream = getattr(self, "_layer_upstream", None)
        active.manager = getattr(self, "_supplied_layer_manager", None)
        active.workflow = getattr(self, "_supplied_layer_workflow", None)
        active.caches = getattr(self, "_supplied_layer_caches", {})
        return active

    def _install_worker_runtime(self, runtime):
        self._worker_runtime_active = runtime
        self._layer_upstream = runtime.upstream if runtime is not None else None
        self._layer_upstream_model_path = (
            str(runtime.upstream.model_path) if runtime is not None and runtime.upstream is not None else ""
        )
        self._supplied_layer_manager = runtime.manager if runtime is not None else None
        self._supplied_layer_key = (
            self._supplied_layer_selection_key(runtime.selected)
            if runtime is not None and runtime.manager is not None else None
        )
        self._supplied_layer_workflow = runtime.workflow if runtime is not None else None
        self._supplied_layer_caches = runtime.caches if runtime is not None else {}

    def _owned_worker_runtimes(self):
        items = (self._capture_worker_runtime(), getattr(self, "_worker_runtime_candidate", None),
                 getattr(self, "_worker_runtime_previous", None),
                 *getattr(self, "_worker_runtime_orphans", []))
        result = []
        for runtime in items:
            if runtime is not None and runtime.owned and all(runtime is not old for old in result):
                result.append(runtime)
        return result

    def worker_runtime_owned_identities(self):
        from copy import deepcopy
        with self._worker_layer_lock():
            return [deepcopy(runtime.identity) for runtime in self._owned_worker_runtimes()
                    if runtime.identity is not None]

    def _retain_failed_worker_runtime(self, runtime):
        with self._worker_layer_lock():
            orphans = getattr(self, "_worker_runtime_orphans", None)
            if orphans is None:
                orphans = self._worker_runtime_orphans = []
            if all(runtime is not old for old in orphans):
                orphans.append(runtime)

    def _build_supplied_layer_manager(self, selected, provider_cancel_event, runtime_owner=None,
                                      *, load_spec=None):
        from worker_prepare_contract import WorkerPrepareError, validate_load_spec
        from model_module import ModelManager
        manifest = selected["manifest"]
        start, end = manifest["layer_range"]
        model_id = "supply-" + manifest["artifact_sha256"][:24]
        spec = validate_load_spec(load_spec) if load_spec is not None else None
        if spec is None:
            # Legacy commands use one deterministic compatibility policy.
            # New deployments negotiate and require the complete frozen spec.
            quant_type = "fp32"
        else:
            expected_components = {"layers", "final_norm"}
            if manifest.get("has_embedding", start == 0):
                expected_components.add("embedding")
            if manifest.get("has_lm_head", end == manifest["total_layers"]):
                expected_components.add("lm_head")
            if (spec.materialization != "safetensors_key_filtered"
                    or set(spec.components) != expected_components
                    or spec.kv_policy != "dynamic" or spec.sequence_count != 1
                    or spec.kv_device != spec.device or spec.kv_dtype != spec.dtype
                    or spec.quant_type not in {"fp16", "fp32"}):
                raise WorkerPrepareError("worker_prepare_load_spec_unsupported")
            quant_type = spec.quant_type
        manager = ModelManager()
        if runtime_owner is not None:
            runtime_owner.manager = manager
        try:
            if provider_cancel_event.is_set():
                raise WorkerPrepareError("worker_prepare_cancelled")
            manager.prepare_pipeline_model(
                model_id=model_id, model_path=selected["directory"],
                quant_type=quant_type, layer_range=(start, end),
                model_sha256=manifest["artifact_sha256"],
            )
            if provider_cancel_event.is_set():
                raise WorkerPrepareError("worker_prepare_cancelled")
            manager.load_layer_range(
                start_layer=start, end_layer=end,
                has_embedding=manifest.get("has_embedding", start == 0),
                has_lm_head=manifest.get("has_lm_head", end == manifest["total_layers"]),
                model_path=selected["directory"], quant_type=quant_type,
                total_layers=manifest["total_layers"], model_id=model_id,
                **({"load_spec": spec.to_dict()} if spec is not None else {}),
            )
            if provider_cancel_event.is_set():
                raise WorkerPrepareError("worker_prepare_cancelled")
            if tuple(manager.layer_range or ()) != (start, end) or not manager.is_loaded:
                raise WorkerPrepareError("worker_prepare_layer_range_mismatch")
            if spec is not None:
                model = getattr(manager, "model", None)
                parameters = (list(model.parameters())
                              if callable(getattr(model, "parameters", None)) else [])
                floating = [parameter for parameter in parameters if parameter.is_floating_point()]
                if (not floating or manager.quant_type != spec.quant_type
                        or any(str(parameter.device) != spec.device
                               or str(parameter.dtype).removeprefix("torch.") != spec.dtype
                               for parameter in floating)):
                    raise WorkerPrepareError("worker_prepare_load_spec_runtime_mismatch")
                # Internal policy snapshot only; actual materialization receipts
                # and assignment budget reconciliation are a deployment concern.
                manager.runtime_load_spec = spec
            return manager
        except Exception:
            if runtime_owner is None:
                from worker_prepare import LayerRuntime, close_layer_runtime
                partial = LayerRuntime(None, selected, manager=manager)
                try:
                    close_layer_runtime(partial)
                except Exception:
                    self._retain_failed_worker_runtime(partial)
                    raise
            raise

    def _build_worker_runtime(self, command, selected, cancel_event):
        from copy import deepcopy
        from worker_prepare import LayerRuntime, close_layer_runtime
        from worker_prepare_contract import WorkerPrepareError, validate_command, validate_load_spec
        command = validate_command(command)
        identity = command["identity"]
        spec = (validate_load_spec(command["load_spec"], identity=identity,
                                   context_size=command["context_size"])
                if "load_spec" in command else None)
        runtime = LayerRuntime(deepcopy(identity), deepcopy(selected))
        if spec is not None:
            runtime.load_spec = spec
        try:
            if cancel_event.is_set():
                raise WorkerPrepareError("worker_prepare_cancelled")
            if identity["stage_model_identity"]["format"] == "safetensors":
                runtime.manager = self._build_supplied_layer_manager(
                    selected, cancel_event, runtime, **({"load_spec": spec} if spec is not None else {}),
                )
            else:
                if spec is not None:
                    expected_components = {"layers"}
                    if identity["segment_mode"] == "head":
                        expected_components.add("embedding")
                    if identity["segment_mode"] == "tail":
                        expected_components.update(("final_norm", "lm_head"))
                    if (spec.materialization != "gguf_segment" or spec.device != "cpu"
                            or spec.dtype != "artifact" or spec.quant_type != "artifact"
                            or spec.kv_device != "cpu" or spec.kv_dtype != "float16"
                            or spec.kv_policy != "preallocated"
                            or set(spec.components) != expected_components):
                        raise WorkerPrepareError("worker_prepare_load_spec_unsupported")
                from keep_head_shim import resolve_keep_head_shim
                from llama_keep_head import KeepHeadUpstream
                import numpy as np
                shim = resolve_keep_head_shim(**({"environment": self._environment} if self._environment is not None else {}))
                if not shim:
                    raise WorkerPrepareError("worker_prepare_shim_unavailable")
                runtime.upstream = KeepHeadUpstream(
                    shim, selected["model_path"], mode="nextn",
                    n_ctx=spec.context_size if spec is not None else command["context_size"],
                    n_batch=spec.batch_size if spec is not None else min(command["context_size"], 2048),
                    n_threads=8, n_seq_max=spec.sequence_count if spec is not None else 1,
                    **({"environment": self._environment} if self._environment is not None else {}),
                )
                upstream = runtime.upstream
                if (not upstream.is_available or upstream.n_embd != identity["hidden_size"]
                        or upstream.n_layer != identity["layer_range"][1] - identity["layer_range"][0]):
                    raise WorkerPrepareError("worker_prepare_layer_range_mismatch")
                if cancel_event.is_set():
                    raise WorkerPrepareError("worker_prepare_cancelled")
                # Probe the optional tail entry point in the child, then clear probe KV.
                hidden = np.zeros((1, identity["hidden_size"]), dtype=np.float32)
                if identity["segment_mode"] == "tail":
                    token = upstream.forward_hidden_to_token(hidden, n_past=0)
                    if type(token) is not int or token < 0:
                        raise WorkerPrepareError("worker_prepare_probe_failed")
                else:
                    output = upstream.forward_hidden_to_hidden(hidden, n_past=0)
                    if output.shape != hidden.shape or not np.isfinite(output).all():
                        raise WorkerPrepareError("worker_prepare_probe_failed")
                upstream.reset()
            if cancel_event.is_set() or not runtime.loaded:
                raise WorkerPrepareError("worker_prepare_cancelled")
            if spec is not None:
                if runtime.manager is not None:
                    parameters = list(runtime.manager.model.parameters())
                    buffers = list(runtime.manager.model.buffers()) if callable(getattr(runtime.manager.model, "buffers", None)) else []
                    materialized = sum(p.numel() * p.element_size() for p in parameters + buffers)
                else:
                    observation = runtime.upstream.materialization_observation()
                    for field in ("context_size", "batch_size", "device", "dtype", "quant_type", "kv_device", "kv_dtype", "sequence_count"):
                        if observation.get(field) != getattr(spec, field):
                            raise WorkerPrepareError("worker_prepare_load_spec_runtime_mismatch")
                    materialized = observation["materialized_bytes"]
                runtime.materialization_receipt = {
                    "load_spec": spec.to_dict(), "materialized_bytes": materialized,
                    "memory_scope": "weight_tensors", "layer_range": deepcopy(identity["layer_range"]),
                    "artifact_sha256": identity["artifact_sha256"],
                }
            return runtime
        except Exception as exc:
            if runtime.upstream is None:
                runtime.upstream = getattr(exc, "cleanup_owner", None)
            try:
                close_layer_runtime(runtime)
            except Exception:
                self._retain_failed_worker_runtime(runtime)
                raise
            raise

    def worker_runtime_receipt(self, identity):
        from copy import deepcopy
        for runtime in self._owned_worker_runtimes():
            if runtime.identity == identity and runtime.loaded:
                receipt = getattr(runtime, "materialization_receipt", None)
                if receipt is not None:
                    return deepcopy(receipt)
        return None

    def prepare_worker_runtime(self, command, selected, cancel_event):
        from copy import deepcopy
        from worker_prepare_contract import WorkerPrepareError, validate_command
        command = validate_command(command)
        if command["action"] != "prepare":
            raise WorkerPrepareError("invalid_prepare_action")
        identity = command["identity"]
        self.fence_worker_runtime()
        with self._worker_layer_lock():
            self._worker_runtime_managed = True
            if getattr(self, "_worker_runtime_phase", "idle") == "committed" or getattr(self, "_worker_runtime_candidate", None) is not None:
                raise WorkerPrepareError("worker_prepare_cleanup_pending")
            self._worker_runtime_identity = deepcopy(identity)
            self._worker_runtime_phase = "preparing"
        try:
            runtime = self._build_worker_runtime(command, selected, cancel_event)
        except Exception:
            with self._worker_layer_lock():
                self._worker_runtime_phase = "failed"
            raise
        with self._worker_layer_lock():
            self._worker_runtime_candidate = runtime
            if cancel_event.is_set():
                from worker_prepare import close_layer_runtime
                close_layer_runtime(runtime)
                self._worker_runtime_candidate = None
                self._worker_runtime_phase = "failed"
                raise WorkerPrepareError("worker_prepare_cancelled")
            self._worker_runtime_candidate = runtime
            self._worker_runtime_identity = deepcopy(identity)
            self._worker_runtime_phase = "prepared"

    def worker_runtime_loaded(self, identity=None):
        with self._worker_layer_lock():
            for runtime in (getattr(self, "_worker_runtime_candidate", None), getattr(self, "_worker_runtime_active", None)):
                if runtime is not None and (identity is None or runtime.identity == identity) and runtime.loaded:
                    return True
            return False

    def worker_runtime_ready(self, identity):
        with self._worker_layer_lock():
            fenced = getattr(self, "_worker_runtime_fenced", None)
            active = getattr(self, "_worker_runtime_active", None)
            return bool(active is not None and active.identity == identity and active.loaded
                        and (fenced is None or not fenced.is_set())
                        and getattr(self, "_worker_runtime_phase", "idle") in {"committed", "finalized", "aborted"})

    def worker_runtime_present(self):
        with self._worker_layer_lock():
            return bool(self._owned_worker_runtimes()) or getattr(self, "_worker_runtime_phase", "idle") == "preparing"

    def retire_previous_worker_runtime(self, identity):
        from worker_prepare import close_layer_runtime
        from worker_prepare_contract import WorkerPrepareError
        with self._worker_layer_lock():
            previous = getattr(self, "_worker_runtime_previous", None)
            if previous is None or previous.identity != identity:
                raise WorkerPrepareError("worker_prepare_identity_mismatch")
            close_layer_runtime(previous)
            self._worker_runtime_previous = None

    def commit_worker_runtime(self, identity):
        from worker_prepare_contract import WorkerPrepareError, validate_identity
        identity = validate_identity(identity)
        with self._worker_layer_lock():
            if getattr(self, "_worker_runtime_identity", None) != identity:
                raise WorkerPrepareError("worker_prepare_identity_mismatch")
            if self._worker_runtime_phase in {"committed", "finalized"}:
                if not self.worker_runtime_loaded(identity):
                    raise WorkerPrepareError("worker_prepare_runtime_unavailable")
                return
            runtime = getattr(self, "_worker_runtime_candidate", None)
            if self._worker_runtime_phase != "prepared" or runtime is None or not runtime.loaded:
                raise WorkerPrepareError("worker_prepare_not_prepared")
            self._worker_runtime_previous = self._capture_worker_runtime()
            self._install_worker_runtime(runtime)
            self._worker_runtime_candidate = None
            self._worker_runtime_phase = "committed"
            self._resume_worker_runtime()

    def abort_worker_runtime(self, identity, *, restore_previous=True):
        from worker_prepare import close_layer_runtime
        from worker_prepare_contract import WorkerPrepareError, validate_identity
        identity = validate_identity(identity)
        with self._worker_layer_lock():
            active = self._capture_worker_runtime()
            candidate = getattr(self, "_worker_runtime_candidate", None)
            phase = getattr(self, "_worker_runtime_phase", "idle")
            matches = [runtime for runtime in self._owned_worker_runtimes() if runtime.identity == identity]
            if getattr(self, "_worker_runtime_identity", None) != identity:
                # Selection can fail before native preparation starts. Only
                # that empty candidate may restore the still-owned old active.
                if candidate is not None or matches or phase == "preparing":
                    raise WorkerPrepareError("worker_prepare_identity_mismatch")
            elif phase == "finalized" and active.identity == identity:
                raise WorkerPrepareError("worker_prepare_already_finalized")
            self.fence_worker_runtime()
            for runtime in matches:
                close_layer_runtime(runtime)
            if active.identity == identity:
                previous = getattr(self, "_worker_runtime_previous", None)
                if not restore_previous:
                    close_layer_runtime(previous)
                self._install_worker_runtime(previous if restore_previous else None)
                self._worker_runtime_previous = None
            if candidate is not None and candidate.identity == identity:
                self._worker_runtime_candidate = None
            self._worker_runtime_orphans = [runtime for runtime in getattr(self, "_worker_runtime_orphans", []) if runtime.owned]
            self._worker_runtime_phase = "aborted"
            if restore_previous and self._capture_worker_runtime().loaded:
                self._resume_worker_runtime()

    def finalize_worker_runtime(self, identity):
        from worker_prepare import close_layer_runtime
        from worker_prepare_contract import WorkerPrepareError, validate_identity
        identity = validate_identity(identity)
        with self._worker_layer_lock():
            if getattr(self, "_worker_runtime_identity", None) != identity or self._worker_runtime_phase not in {"committed", "finalized"}:
                raise WorkerPrepareError("worker_prepare_not_committed")
            if not self.worker_runtime_loaded(identity):
                raise WorkerPrepareError("worker_prepare_runtime_unavailable")
            close_layer_runtime(getattr(self, "_worker_runtime_previous", None))
            self._worker_runtime_previous = None
            self._worker_runtime_phase = "finalized"

    def release_worker_runtime(self, identity=None):
        from worker_prepare import close_layer_runtime
        from worker_prepare_contract import WorkerPrepareError, validate_identity
        with self._worker_layer_lock():
            active = self._capture_worker_runtime()
            candidate = getattr(self, "_worker_runtime_candidate", None)
            previous = getattr(self, "_worker_runtime_previous", None)
            owned = self._owned_worker_runtimes()
            if identity is not None:
                identity = validate_identity(identity)
                targets = [runtime for runtime in owned if runtime.identity == identity]
                if not targets and owned:
                    raise WorkerPrepareError("worker_prepare_identity_mismatch")
            else:
                targets = owned
            # Releasing the committed candidate retires its retained previous
            # as well. A different prepared candidate/active remains untouched.
            retire_previous = identity is None or (active.identity == identity and active.identity is not None)
            if retire_previous and previous is not None and previous.owned and all(previous is not target for target in targets):
                targets.append(previous)
            # An old cleanup must never close a different new candidate/active.
            if identity is None or active.identity == identity or (candidate is not None and candidate.identity == identity):
                self.fence_worker_runtime()
            for runtime in targets:
                close_layer_runtime(runtime)
            if identity is None or active.identity == identity:
                self._install_worker_runtime(None)
            if identity is None or (candidate is not None and candidate.identity == identity):
                self._worker_runtime_candidate = None
            if retire_previous or (previous is not None and previous.identity == identity):
                self._worker_runtime_previous = None
            self._worker_runtime_orphans = [runtime for runtime in getattr(self, "_worker_runtime_orphans", []) if runtime.owned]
            if identity is None or getattr(self, "_worker_runtime_identity", None) == identity:
                self._worker_runtime_phase = "released"

    def _check_worker_runtime_stage(self, stage_request, *, required=False):
        from worker_prepare_contract import WorkerPrepareError, stage_identity, validate_stage_identity
        fields = stage_request.stage_fields or {}
        if not required and not getattr(self, "_worker_runtime_managed", False) and "prepare_identity" not in fields:
            return
        actual = validate_stage_identity(fields.get("prepare_identity"))
        fenced = getattr(self, "_worker_runtime_fenced", None)
        if ((fenced is not None and fenced.is_set())
                or getattr(self, "_worker_runtime_phase", "idle") not in {"committed", "finalized", "aborted"}):
            raise WorkerPrepareError("worker_prepare_stage_fenced")
        runtime = getattr(self, "_worker_runtime_active", None)
        if runtime is None or runtime.identity is None or actual != stage_identity(runtime.identity) or not runtime.loaded:
            raise WorkerPrepareError("worker_prepare_stage_identity_mismatch")
        identity = stage_request.model_identity
        if hasattr(identity, "snapshot"):
            identity = identity.snapshot()
        if identity != runtime.identity["stage_model_identity"] or fields.get("layer_range") != runtime.identity["layer_range"]:
            raise WorkerPrepareError("worker_prepare_stage_contract_mismatch")

    def _layer_forward_upstream(self):
        """按 env 配置创建并复用 keep-head 上游（进程隔离由 `KeepHeadUpstream` 自理）。

        env（PC 侧没有像 Android 那样的工件目录扫描，这是唯一的配置入口）：
          * `QLH_LAYER_GGUF`         —— 本节点负责的**裁层 GGUF** 路径（必填）
          * `QLH_KEEP_HEAD_SHIM`     —— `qlh_keep_head.dll` 路径（必填）
          * `QLH_LAYER_N_CTX` / `QLH_LAYER_N_BATCH` / `QLH_LAYER_THREADS` /
            `QLH_LAYER_N_SEQ_MAX`   —— 可选，缺省 4096 / 2048 / 8 / 1

        ⚠️ 不要图省事绕开隔离：`qlh_keep_head.dll` 经 `libllama.dll` 依赖**按 basename**
        解析的 `ggml-base.dll` / `ggml.dll`，与 pip `llama_cpp` 是**同名不同版本**；
        Windows loader 在进程生命周期内只认第一个加载的，绑定不可撤销（`WinError 127`
        的根因，见 `llama_keep_head._shim_abi_collides_with_llama_cpp`）。
        """
        import os

        if getattr(self, "_worker_runtime_managed", False):
            from worker_prepare_contract import WorkerPrepareError
            active = getattr(self, "_worker_runtime_active", None)
            if active is None or active.upstream is None or not active.loaded:
                raise WorkerPrepareError("worker_prepare_runtime_unavailable")
            return active.upstream

        # 本函数体要用 TaskGraphError，而该文件惯用函数内 import（同 :1802 / :2016 /
        # :2140），此前漏了这一处 ⇒ 层段 keep-head 上游缺 env 或绑定失败时抛的是
        # `NameError: name 'TaskGraphError' is not defined`，而不是真正的错误信息
        # （2026-10-06 用 Edge 包当 layer worker 入网时实测踩到）。
        from task_graph import TaskGraphError
        # ★ 2026-10-08（#53）：与 master 侧共用 shim 定位（含"仓库默认构建产物"一档）。
        from keep_head_shim import missing_shim_hint, resolve_keep_head_shim

        from model_supply_worker import selected_gguf_path
        environment = os.environ if self._environment is None else self._environment
        if self._environment is None:
            model_path = selected_gguf_path() or environment.get("QLH_LAYER_GGUF", "").strip()
        else:
            model_path = environment.get("QLH_LAYER_GGUF", "").strip()
        if self._layer_upstream is not None:
            previous_path = str(getattr(self, "_layer_upstream_model_path", "") or "")
            if (not previous_path or previous_path == model_path) and self._layer_upstream.is_available:
                return self._layer_upstream
            self._layer_upstream.close()
            self._layer_upstream = None
        # ★ 2026-10-08（#53）：shim 走**共用的定位入口** —— 环境变量之后还有"仓库默认
        #   构建产物"这一档，与 master 侧（`llama_engine`）完全一致。此前这里只读环境
        #   变量 ⇒ 同一台机器上"master 能用、worker 报缺环境变量"。
        shim_path = resolve_keep_head_shim(**({"environment": self._environment} if self._environment is not None else {}))
        if not model_path or not shim_path:
            # 具名错误：分别指出**缺哪一个**，并列出 shim 已尝试过的路径。
            missing = []
            if not model_path:
                missing.append("QLH_LAYER_GGUF（裁层 GGUF 路径）")
            if not shim_path:
                missing.append(missing_shim_hint())
            raise TaskGraphError("层段 Stage 缺少必要配置：" + "；".join(missing))
        from llama_keep_head import KeepHeadUnavailable, KeepHeadUpstream

        def _env_int(name: str, default: int) -> int:
            raw = environment.get(name, "").strip()
            try:
                return max(1, int(raw)) if raw else default
            except ValueError:
                return default

        try:
            self._layer_upstream_model_path = model_path
            self._layer_upstream = KeepHeadUpstream(
                shim_path,
                model_path,
                mode="nextn",
                n_ctx=_env_int("QLH_LAYER_N_CTX", 4096),
                n_batch=_env_int("QLH_LAYER_N_BATCH", 2048),
                n_threads=_env_int("QLH_LAYER_THREADS", 8),
                n_seq_max=_env_int("QLH_LAYER_N_SEQ_MAX", 1),
                **({"environment": self._environment} if self._environment is not None else {}),
            )
        except KeepHeadUnavailable as exc:
            owner = getattr(exc, "cleanup_owner", None)
            if owner is not None:
                self._layer_upstream = owner
            raise TaskGraphError(f"层段 keep-head 上游不可用：{exc}") from exc
        return self._layer_upstream

    def _unload_supplied_layer_manager(self) -> None:
        """Release segment weights and every KV reference before replacing them."""
        manager = getattr(self, "_supplied_layer_manager", None)
        self._supplied_layer_manager = None
        self._supplied_layer_key = None
        self._supplied_layer_workflow = None
        self._supplied_layer_caches = {}
        if manager is not None:
            manager.unload_model()

    @staticmethod
    def _supplied_layer_selection_key(selected: dict) -> tuple:
        from pathlib import Path

        return (selected["manifest_sha256"], str(Path(selected["directory"]).resolve()))

    def _supplied_layer_manager_for(self, selected: dict, provider_cancel_event):
        """Materialize only the registered range, never a complete-model load."""
        from task_graph import TaskGraphError as _StageError

        if provider_cancel_event.is_set():
            raise _StageError("supplied_stage_cancelled")
        key = self._supplied_layer_selection_key(selected)
        if getattr(self, "_supplied_layer_manager", None) is not None:
            return self._supplied_layer_manager

        if getattr(self, "_worker_runtime_managed", False):
            raise _StageError("worker_prepare_runtime_unavailable")

        upstream = getattr(self, "_layer_upstream", None)
        if upstream is not None:
            upstream.close()
            self._layer_upstream = None
        try:
            manager = self._build_supplied_layer_manager(selected, provider_cancel_event)
        except Exception as exc:
            if getattr(exc, "reason_code", "") == "worker_prepare_cancelled":
                raise _StageError("supplied_stage_cancelled") from exc
            raise
        self._supplied_layer_manager = manager
        self._supplied_layer_key = key
        return manager

    def _execute_supplied_layer_forward(
        self, selected, stage_request, hidden, *, want_hidden, n_past,
        positions, seq_ids, provider_cancel_event,
    ):
        """Adapt v3 hidden/KV to ModelManager's existing batched forward API."""
        import torch
        from task_graph import TaskGraphError as _StageError

        manifest = selected["manifest"]
        fields = stage_request.stage_fields or {}
        if fields.get("layer_range") != manifest["layer_range"]:
            raise _StageError("supplied_layer_range_mismatch")
        if hidden.shape[1] != manifest["hidden_size"]:
            raise _StageError("supplied_hidden_size_mismatch")
        identity = getattr(stage_request, "model_identity", None)
        if identity is not None:
            identity = identity.snapshot() if hasattr(identity, "snapshot") else identity
            if identity != {
                "model_id": "supply-" + manifest["artifact_sha256"][:24],
                "engine": "pytorch", "format": "safetensors", "revision": "1",
                "sha256": manifest["artifact_sha256"],
            }:
                raise _StageError("supplied_model_identity_mismatch")
        if not want_hidden and not manifest.get(
            "has_lm_head", manifest["layer_range"][1] == manifest["total_layers"],
        ):
            raise _StageError("supplied_segment_has_no_lm_head")

        n_tokens = hidden.shape[0]
        seq_ids = [0] * n_tokens if seq_ids is None else seq_ids
        if (not isinstance(seq_ids, (list, tuple)) or len(seq_ids) != n_tokens
                or any(type(value) is not int or value < 0 for value in seq_ids)):
            raise _StageError("supplied_seq_ids_invalid")
        if positions is None:
            counters = {}
            positions = []
            for seq in seq_ids:
                position = counters.get(seq, n_past)
                positions.append(position)
                counters[seq] = position + 1
        if (not isinstance(positions, (list, tuple)) or len(positions) != n_tokens
                or any(type(value) is not int or value < 0 for value in positions)):
            raise _StageError("supplied_positions_invalid")
        groups = {}
        for index, seq in enumerate(seq_ids):
            groups.setdefault(seq, []).append(index)
        workflow = stage_request.workflow_id
        if getattr(self, "_supplied_layer_workflow", None) != workflow:
            self._supplied_layer_caches = {}
            self._supplied_layer_workflow = workflow
        # Validate a working mapping first: one sequence's reset must not
        # erase committed KV when a later sequence fails input validation.
        caches = dict(getattr(self, "_supplied_layer_caches", {}))
        for seq, indices in groups.items():
            seq_positions = [positions[index] for index in indices]
            if seq_positions != list(range(seq_positions[0], seq_positions[0] + len(indices))):
                raise _StageError("supplied_positions_must_be_contiguous")
            if seq_positions[0] == 0:
                caches.pop(seq, None)
            elif seq not in caches or caches[seq][1] != seq_positions[0]:
                raise _StageError("supplied_decode_requires_matching_prefill")

        try:
            manager = self._supplied_layer_manager_for(selected, provider_cancel_event)
            states = torch.from_numpy(hidden.copy()).unsqueeze(0)
            output = torch.empty((n_tokens, hidden.shape[1]), dtype=torch.float32) if want_hidden else None
            token = None
            for seq, indices in groups.items():
                if provider_cancel_event.is_set():
                    raise _StageError("supplied_stage_cancelled")
                index_tensor = torch.tensor(indices, dtype=torch.long)
                result = manager.forward_layers(
                    hidden_states=states.index_select(1, index_tensor),
                    position_ids=torch.tensor([[positions[index] for index in indices]], dtype=torch.long),
                    past_key_values=caches[seq][0] if seq in caches else None,
                    use_cache=True,
                    apply_lm_head=False,
                )
                if provider_cancel_event.is_set():
                    raise _StageError("supplied_stage_cancelled")
                # Preserve hybrid recurrent state: the tuple is a lossy fallback.
                cache = result.get("cache")
                if cache is None:
                    cache = result.get("past_key_values")
                if cache is None:
                    raise _StageError("supplied_forward_missing_kv_cache")
                caches[seq] = (cache, positions[indices[-1]] + 1)
                out = result["hidden_states"]
                if want_hidden:
                    output.index_copy_(0, index_tensor, out[0].detach().to(device="cpu", dtype=torch.float32))
                elif seq == seq_ids[-1]:
                    logits = manager.forward_lm_head(out[:, -1:, :])
                    token = int(logits[0, -1].argmax().item())
            if provider_cancel_event.is_set():
                raise _StageError("supplied_stage_cancelled")
        except Exception:
            # Cache objects mutate during forward; partial failures cannot resume.
            self._supplied_layer_caches = {}
            raise
        self._supplied_layer_caches = caches
        return output.numpy() if want_hidden else token

    def _execute_layer_forward_stage(self, stage_request, provider_cancel_event):
        with self._worker_layer_lock():
            self._check_worker_runtime_stage(stage_request)
            return self._execute_layer_forward_stage_locked(stage_request, provider_cancel_event)

    def _execute_layer_forward_stage_locked(
        self,
        stage_request,
        provider_cancel_event: threading.Event,
    ) -> Dict[str, Any]:
        """执行一次 v3 层段 Stage：吃上游 hidden，产 hidden（中间段）或 token（尾段）。

        结果形状对齐 `scheduler_pipeline._layer_stage_result_to_pipeline_value`：
          * 中间段（`root_input.want_hidden == true`）⇒
            `{"hidden_out_f32": b64, "hidden_out_sha256": sha256(原始 f32 字节), "token_argmax": None}`
          * 尾段 ⇒ `{"token_argmax": int}`

        `metadata` 由 `scheduler_task_worker` 侧用 `sanitize_result_metadata` 统一净化，
        它只取 `usage` / `usage_estimated` / `tokens_per_second` / `model` 四项，
        不会动这两个字段。
        """
        import base64
        import hashlib

        import numpy as np

        from task_graph import TaskGraphError as _StageError

        if provider_cancel_event.is_set():
            raise _StageError("层段 Stage 在开始前已被取消")

        root_input = stage_request.root_input or {}
        encoded = root_input.get("hidden_f32")
        if not isinstance(encoded, str):
            raise _StageError("层段 Stage 缺少 root_input.hidden_f32")
        stage_fields = stage_request.stage_fields or {}
        spec = stage_fields.get("hidden_spec") or {}
        n_tokens = int(spec.get("n_tokens", 0) or 0)
        n_embd = int(spec.get("n_embd", 0) or 0)
        if n_tokens < 1 or n_embd < 1:
            raise _StageError("层段 Stage 的 hidden_spec 维度必须为正")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise _StageError("层段 Stage 的 hidden_f32 不是合法 base64") from exc
        if len(raw) != n_tokens * n_embd * 4:
            raise _StageError(
                f"层段 Stage 的 hidden 长度不符：{len(raw)} != {n_tokens * n_embd * 4}"
            )

        want_hidden = bool(root_input.get("want_hidden", False))
        seq_ids = stage_fields.get("seq_ids")
        positions = stage_fields.get("positions")
        # `positions` 给定时 `n_past` 被忽略；缺省退化为从 `pos_base` 起递增。
        n_past = int(root_input.get("pos_base", 0) or 0)

        hidden = np.frombuffer(raw, dtype=np.float32).reshape(n_tokens, n_embd)
        from model_supply_worker import get_selected_model

        if getattr(self, "_worker_runtime_managed", False):
            selected = self._worker_runtime_active.selected
        else:
            selected = get_selected_model()
        supplied = selected is not None and selected["manifest"]["format"] == "safetensors"
        if (not supplied or getattr(self, "_supplied_layer_key", None)
                != self._supplied_layer_selection_key(selected)):
            self._unload_supplied_layer_manager()
        identity = getattr(stage_request, "model_identity", None)
        if hasattr(identity, "snapshot"):
            identity = identity.snapshot()
        if (not supplied and isinstance(identity, dict)
                and identity.get("engine") == "pytorch"
                and identity.get("format") == "safetensors"):
            raise _StageError("supplied_registration_unavailable")
        if supplied:
            out = self._execute_supplied_layer_forward(
                selected, stage_request, hidden, want_hidden=want_hidden,
                n_past=n_past, positions=positions, seq_ids=seq_ids,
                provider_cancel_event=provider_cancel_event,
            )
            if not want_hidden:
                return {"token_argmax": int(out)}
            out_raw = np.ascontiguousarray(out, dtype=np.float32).tobytes()
            return {
                "hidden_out_f32": base64.b64encode(out_raw).decode("ascii"),
                "hidden_out_sha256": hashlib.sha256(out_raw).hexdigest(),
                "token_argmax": None,
            }
        upstream = self._layer_forward_upstream()

        # ★ 2026-10-03：批次**从位置 0 开始**就说明这是新 prompt 的 prefill ⇒ 必须先清
        #   子进程的 KV。否则下一个请求会带着上一个请求残留的位置（实测：
        #   `the last position stored in the KV cache for sequence 0 is X = 45`，而 batch
        #   的起始位置是 0 ⇒ `llama_decode rc=-1`、`decode: failed to initialize batch`）。
        #   与 JNI 侧 `qlh_layer_batch_starts_at_zero` 同语义 —— 那里是每步判一次，
        #   这里按 Stage 判（一个 Stage = 一整段 prefill 或一次 decode）。
        starts_at_zero = (
            positions[0] == 0 if positions else n_past == 0
        )
        if starts_at_zero:
            upstream.reset()

        if want_hidden:
            out = upstream.forward_hidden_to_hidden(
                hidden, n_past=n_past, seq_ids=seq_ids, positions=positions,
            )
            out_raw = np.ascontiguousarray(out, dtype=np.float32).tobytes()
            return {
                "hidden_out_f32": base64.b64encode(out_raw).decode("ascii"),
                "hidden_out_sha256": hashlib.sha256(out_raw).hexdigest(),
                "token_argmax": None,
            }
        token = upstream.forward_hidden_to_token(
            hidden, n_past=n_past, seq_ids=seq_ids, positions=positions,
        )
        return {"token_argmax": int(token)}

    # ------------------------------------------------------------------
    # 1.2d task_graph 执行段（复制自 api_server.py:2571-3321，源文件不动）
    # 入口：execute_task_graph_chat（slot/journal/角色门控）→
    #       execute_task_graph_chat_with_slot（主体 448 行）
    # ------------------------------------------------------------------
    def _ensure_task_graph_coordinator(self) -> Any:
        """1.2d 惰性创建任务图协调器（复制自 api_server._create_task_graph_coordinator
        api_server.py:354-395；全局实例 → 实例属性，首次访问时创建并缓存）。"""
        if self._task_graph_coordinator is not None:
            return self._task_graph_coordinator
        import config as _cfg
        from task_graph import TaskGraphCoordinator, TaskGraphUnavailable
        from task_journal import SQLiteTaskJournal, TaskJournalError

        if not _cfg.TASK_GRAPH_ENABLED:
            coordinator = TaskGraphCoordinator(
                max_records=_cfg.TASK_GRAPH_MAX_RECORDS,
                max_parallel_stages=_cfg.TASK_GRAPH_MAX_PARALLEL_STAGES,
            )
            self._task_graph_coordinator = coordinator
            return coordinator
        journal = None
        try:
            journal = SQLiteTaskJournal(_cfg.TASK_GRAPH_JOURNAL_PATH)
            coordinator = TaskGraphCoordinator(
                max_records=_cfg.TASK_GRAPH_MAX_RECORDS,
                journal=journal,
                max_parallel_stages=_cfg.TASK_GRAPH_MAX_PARALLEL_STAGES,
            )
            recovery = coordinator.recover_persisted_workflows()
            cleanup = coordinator.cleanup_journal(
                max_age_days=_cfg.TASK_GRAPH_RETENTION_DAYS,
                max_records=_cfg.TASK_GRAPH_RETENTION_MAX_RECORDS,
            )
            if recovery.get("recovered_workflows", 0):
                logger.warning("任务图启动恢复完成: %s", recovery)
            if cleanup.get("deleted_workflows", 0):
                logger.info("任务图 journal 保留清理完成: %s", cleanup)
            self._task_graph_coordinator = coordinator
            return coordinator
        except (TaskJournalError, TaskGraphUnavailable) as exc:
            if journal is not None:
                try:
                    journal.close()
                except Exception:
                    pass
            coordinator = TaskGraphCoordinator(
                max_records=_cfg.TASK_GRAPH_MAX_RECORDS,
                max_parallel_stages=_cfg.TASK_GRAPH_MAX_PARALLEL_STAGES,
                availability_error=f"task journal unavailable: {exc}",
            )
            logger.error(
                "任务图 journal 初始化失败，任务图已禁用: %s",
                exc,
                exc_info=True,
            )
            self._task_graph_coordinator = coordinator
            return coordinator


    def _dispatch_local_task_provider(
        self, request, cancel_event: threading.Event,
    ) -> Dict[str, Any]:
        """1.2d 复制自 api_server._dispatch_local_task_provider（api_server.py:2603-2612）。"""
        from task_graph import TaskGraphError
        from task_provider import ProviderExecutor
        from typing import cast
        executor = request.runtime_context.get("local_provider_executor")
        if not callable(executor):
            raise TaskGraphError("本地任务 Provider 缺少请求执行上下文")
        result = cast(ProviderExecutor, executor)(request, cancel_event)
        if not isinstance(result, dict):
            raise TaskGraphError("本地任务 Provider 返回值必须是 dict")
        return result



    def _ensure_local_task_provider(
        self,
    ) -> Dict[str, Any]:
        """1.2d 复制自 api_server._ensure_local_task_provider（api_server.py:2616-2629）；
        宿主适配：self._ensure_task_graph_coordinator() → 实例、scheduler → self._scheduler。"""
        from task_provider import LocalFullModelProvider, ProviderError
        if self._ensure_task_graph_coordinator().has_provider("local_full_model"):
            return
        try:
            self._ensure_task_graph_coordinator().register_provider(LocalFullModelProvider(
                self._dispatch_local_task_provider,
                provider_id="local_full_model",
                node_id=(self._scheduler.get_effective_node_id() if self._scheduler is not None else ""),
                max_concurrency=1,
            ))
        except ProviderError:
            if not self._ensure_task_graph_coordinator().has_provider("local_full_model"):
                raise



    def _prepared_pipeline_model_identity(self):
        """distributed-only 已准备模型的身份（未物化权重也能给出）。

        `prepare-pipeline` 刻意不加载权重，但它把 inspect 出的描述符（`model_sha256`
        / 模型 id / 路径 / 引擎）留在了宿主上 ⇒ 身份从那里取。否则 Route A 会陷入
        "要活动身份就必须先物化权重、而 distributed-only 路径按设计不加载"的死锁。
        """
        from task_provider import ModelIdentity

        host = self._host
        if not getattr(host, "is_pipeline_prepared", False):
            return None
        try:
            descriptor = host.get_pipeline_descriptor() or {}
        except Exception:
            logger.warning("读取已准备的流水线描述符失败", exc_info=True)
            return None
        model_id = str(
            getattr(host, "active_model_id", "") or descriptor.get("model_id", "")
        )
        sha256 = str(descriptor.get("model_sha256", "") or "").lower()
        if not model_id or not sha256:
            return None
        try:
            return ModelIdentity(
                model_id=model_id,
                engine="pytorch",
                format="safetensors",
                revision="prepared",
                sha256=sha256,
            )
        except ValueError:
            logger.warning("已准备的流水线模型身份非法", exc_info=True)
            return None

    def _active_task_graph_model_identity(
        self,
    ) -> Dict[str, Any]:
        """1.2d 复制自 api_server._active_task_graph_model_identity（api_server.py:2631-2679）；
        宿主适配：model_host/model_manager → self._host。"""
        import hashlib
        from task_provider import ModelIdentity
        if not self._host.model_loaded or not self._host.is_loaded:
            return self._prepared_pipeline_model_identity()
        engine = backend_id_for(self._host)
        model_path = str(getattr(self._host, "_model_path", "") or "")
        model_id = str(getattr(self._host, "active_model_id", "") or "")
        if engine not in registered_backends() or not model_id or not model_path:
            return None
        if engine == "island":
            # 孤岛模型无本地 artifact：以"端点指纹 + 后端模型名"替代文件摘要，
            # 统计中如实标注为外部端点（不伪装成本地文件，见调研方案 §2.2）。
            island_engine = getattr(self._host, "_island_engine", None)
            backend_model = str(getattr(island_engine, "model_name", "") or "")
            masked_url = str(getattr(island_engine, "masked_base_url", "") or model_path)
            if not backend_model:
                return None
            digest = hashlib.sha256(
                f"{masked_url}::{backend_model}".encode("utf-8")
            ).hexdigest()
            return ModelIdentity(
                model_id=model_id,
                engine="island",
                format="openai_api",
                revision=f"island-{digest[:12]}",
                sha256=digest,
            )
        try:
            from model_sync import compute_file_sha256, compute_model_sha256

            if engine == "pytorch":
                digest = compute_model_sha256(model_path)
                model_format = "safetensors"
            else:
                digest = compute_file_sha256(model_path)
                model_format = "gguf"
        except Exception:
            logger.warning("无法计算任务链当前模型摘要", exc_info=True)
            return None
        if len(digest) != 64:
            return None
        return ModelIdentity(
            model_id=model_id,
            engine=engine,
            format=model_format,
            revision=f"local-{digest[:12]}",
            sha256=digest,
        )



    def _sync_remote_task_worker_providers(
        self,
    ) -> list[str]:
        """1.2d 复制自 api_server._sync_remote_task_worker_providers（api_server.py:2680-2692）；
        宿主适配：scheduler → self._scheduler（None 时无远端 provider）。"""
        from task_provider import ProviderError
        registered = []
        if self._scheduler is None:
            return []
        for provider in self._scheduler.remote_task_worker_providers():
            if not self._ensure_task_graph_coordinator().has_provider(provider.provider_id):
                try:
                    self._ensure_task_graph_coordinator().register_provider(provider)
                except ProviderError:
                    if not self._ensure_task_graph_coordinator().has_provider(provider.provider_id):
                        raise
            registered.append(provider.provider_id)
        return sorted(registered)



    def _eligible_remote_task_worker_provider_ids(
        self, model_identity, stage_type: str, *, limit: int = 4,
    ) -> list[str]:
        """1.2d 复制自 api_server._eligible_remote_task_worker_provider_ids
        （api_server.py:2694-2718）；宿主适配：_cfg.TASK_WORKER_EXPERIMENTAL_ENABLED
        → config、scheduler → self._scheduler。"""
        import config as _cfg
        if not _cfg.TASK_WORKER_EXPERIMENTAL_ENABLED:
            return []
        if self._scheduler is None:
            return []
        providers = {
            provider.provider_id: provider
            for provider in self._scheduler.remote_task_worker_providers()
        }
        self._sync_remote_task_worker_providers()
        statuses = {
            str(item.get("provider_id", "")): item
            for item in self._ensure_task_graph_coordinator().provider_status()
        }
        eligible = []
        for provider_id, provider in providers.items():
            status = statuses.get(provider_id, {})
            if (
                status.get("provider_kind") != "remote_full_worker"
                or not status.get("healthy")
                or not status.get("available")
                or stage_type not in status.get("supported_stage_types", [])
                or not provider.supports_model_identity(model_identity, stage_type)
            ):
                continue
            max_concurrency = max(1, int(status.get("max_concurrency", 1) or 1))
            active = max(0, int(status.get("active_reservations", 0) or 0))
            eligible.append((
                active / max_concurrency,
                active,
                provider_id,
            ))
        eligible.sort()
        return [item[2] for item in eligible[:max(0, int(limit))]]



    def execute_task_graph_chat(
        self,
        req: ChatRequest,
        cancel_event: Optional[threading.Event] = None,
        request_deadline: Optional[RequestDeadline] = None,
    ) -> Dict[str, Any]:
        """Run the fixed local task graph without claiming multi-device execution.
        1.2d 复制自 api_server._execute_task_graph_chat（api_server.py:2571-2614）；
        宿主适配：TASK_GRAPH_ENABLED → config、task_graph_coordinator/
        self._task_graph_execution_slot/model_host → 实例状态、scheduler →
        self._scheduler（None = 单机基线视为 master）。"""
        import config as _cfg
        from fastapi import HTTPException
        from api_errors import coded_http_error
        if not _cfg.TASK_GRAPH_ENABLED:
            raise coded_http_error(
                409,
                "TASK_GRAPH_DISABLED",
                "任务链实验未启用。请设置 QLH_TASK_GRAPH_ENABLED=true 后重启。",
            )
        journal = self._ensure_task_graph_coordinator().journal_status()
        if not journal.get("available", False):
            raise HTTPException(
                503,
                {
                    "message": "任务链 journal 不可用，已拒绝不可恢复执行。",
                    "reason": journal.get("error", "journal health check failed"),
                },
            )
        if self._scheduler is not None and self._scheduler._effective_role() != "master":
            raise HTTPException(409, "任务链协调器当前只允许在主节点运行。")

        if not self._task_graph_execution_slot.acquire(blocking=False):
            raise HTTPException(429, "已有任务链正在执行，请稍后重试。")
        try:
            with hold_lock_until_deadline(
                self._model_lifecycle_lock(),
                request_deadline,
                operation="full chat execution lock",
            ):
                return self.execute_task_graph_chat_with_slot(
                    req, cancel_event, request_deadline,
                )
        finally:
            self._task_graph_execution_slot.release()



    def execute_task_graph_chat_with_slot(
        self,
        req: ChatRequest,
        cancel_event: Optional[threading.Event] = None,
        request_deadline: Optional[RequestDeadline] = None,
    ) -> Dict[str, Any]:
        """Execute one workflow while the process-wide task-graph slot is held.
        1.2d 复制自 api_server._execute_task_graph_chat_with_slot
        （api_server.py:2874-3321）；宿主适配：scheduler → self._scheduler、
        task_graph_coordinator → self._ensure_task_graph_coordinator()、
        model_host/model_manager → self._host、self._gen_lock →
        self._gen_lock、conversation_stats/self._active_session_id → 实例属性、
        record_task_complete → self._record_task_complete()。"""
        import config as _cfg
        import local_store as _local_store
        from api_errors import coded_http_error
        from dataclasses import replace
        from distributed_completion import (
            DistributedCompletionError,
            validate_distributed_completion,
        )
        from fastapi import HTTPException
        from task_graph import (
            StageSpec,
            TaskGraphError,
            TaskGraphUnavailable,
            WorkflowCancelled,
            WorkflowDeadlineExceeded,
            WorkflowExecutionError,
            dual_candidate_template,
        )
        from task_provider import ProviderError
        request_deadline = coerce_request_deadline(
            request_deadline,
            timeout_seconds=_cfg.PIPELINE_TIMEOUT,
        )
        request_deadline.require_remaining()

        remote_stage_id = str(req.task_graph_remote_stage or "")
        remote_provider_id = str(req.task_graph_remote_provider_id or "")
        distributed_task_required = (
            req.routing_preference == "distributed_required"
        )
        auto_remote = bool(
            req.task_graph_auto_remote
            or (
                not remote_stage_id
                and req.routing_preference in {
                    "distributed_preferred", "distributed_required",
                }
            )
        )
        if bool(remote_stage_id) != bool(remote_provider_id):
            raise coded_http_error(
                400,
                "TASK_GRAPH_MANUAL_REMOTE_FIELDS_INCOMPLETE",
                "N2.1 手动远端执行必须同时指定 Stage 和 Provider ID。",
            )
        if auto_remote and remote_stage_id:
            raise coded_http_error(
                400,
                "TASK_GRAPH_REMOTE_POLICY_CONFLICT",
                "N2.3 自动 Worker 选择不能与 N2.1 手动远端 Stage 同时启用。",
            )
        if remote_stage_id and not _cfg.TASK_WORKER_EXPERIMENTAL_ENABLED:
            raise coded_http_error(
                409,
                "TASK_WORKER_EXPERIMENT_DISABLED",
                "PC Full Worker 实验调度未启用。请设置 "
                "QLH_TASK_WORKER_EXPERIMENTAL_ENABLED=true 后重启。",
            )
        if auto_remote and not _cfg.TASK_WORKER_EXPERIMENTAL_ENABLED:
            if distributed_task_required:
                raise coded_http_error(
                    503,
                    "TASK_WORKER_EXPERIMENT_DISABLED",
                    "distributed_required 需要已启用的 PC Full Worker 实验调度。",
                )

        prepared_context = self._prepare_chat_context(req.session_id, req.message)
        target_session_id = prepared_context.session_id
        history = prepared_context.history
        request_deadline.require_remaining()

        base_messages = list(history) + [{"role": "user", "content": req.message}]
        root_input = {
            "message": req.message,
            "messages": base_messages,
            "task_options": {
                "candidate_max_tokens": max(1, min(req.max_new_tokens, 512)),
                "final_max_tokens": max(1, min(req.max_new_tokens, 1024)),
                "temperature": req.temperature,
                "top_p": req.top_p,
                "show_thinking": req.show_thinking,
            },
        }

        try:
            self._ensure_local_task_provider()
        except ProviderError as exc:
            raise HTTPException(
                503,
                {
                    "message": "本地任务 Provider 注册失败。",
                    "reason": f"{exc.code}: {exc}",
                },
            ) from exc

        model_identity = None
        stages: Optional[list[StageSpec]] = None
        final_stage_id = ""
        auto_provider_ids: list[str] = []
        auto_fallback_reason = ""
        if remote_stage_id:
            remote_providers = {
                provider.provider_id: provider
                for provider in (self._scheduler.remote_task_worker_providers() if self._scheduler is not None else [])
            }
            self._sync_remote_task_worker_providers()
            remote_provider = remote_providers.get(remote_provider_id)
            if remote_provider is None:
                raise HTTPException(
                    404, "The selected remote PC Full Worker does not exist."
                )
            remote_status = next((
                item for item in self._ensure_task_graph_coordinator().provider_status()
                if item.get("provider_id") == remote_provider_id
                and item.get("provider_kind") == "remote_full_worker"
            ), None)
            if remote_status is None:
                raise HTTPException(404, "指定的远端 PC Full Worker Provider 不存在。")
            if not remote_status.get("healthy") or not remote_status.get("available"):
                raise HTTPException(503, "指定的远端 PC Full Worker 当前不可用或正忙。")
            model_identity = self._active_task_graph_model_identity()
            if model_identity is None:
                raise HTTPException(409, "手动远端 Stage 要求主节点先加载完整模型并生成精确身份。")
            template_stages, final_stage_id = dual_candidate_template()
            selected_stage = next((
                stage for stage in template_stages
                if stage.stage_id == remote_stage_id
            ), None)
            if (
                selected_stage is None
                or not remote_provider.supports_model_identity(
                    model_identity, selected_stage.stage_type,
                )
            ):
                raise HTTPException(
                    409,
                    {
                        "message": (
                            "The selected remote PC Full Worker does not have "
                            "the exact active model required by this Stage."
                        ),
                        "reason_code": "model_identity_mismatch",
                        "provider_id": remote_provider_id,
                        "stage_id": remote_stage_id,
                    },
                )
            stages = [
                replace(
                    stage,
                    provider=remote_provider_id,
                    fallback_providers=(),
                    pure=False,
                    max_same_provider_retries=0,
                )
                if stage.stage_id == remote_stage_id else stage
                for stage in template_stages
            ]
        elif auto_remote:
            if not _cfg.TASK_WORKER_EXPERIMENTAL_ENABLED:
                auto_fallback_reason = "task_worker_experiment_disabled"
            else:
                model_identity = self._active_task_graph_model_identity()
            if _cfg.TASK_WORKER_EXPERIMENTAL_ENABLED and model_identity is None:
                auto_fallback_reason = "model_identity_unavailable"
                if distributed_task_required:
                    raise coded_http_error(
                        409,
                        "TASK_WORKER_MODEL_IDENTITY_UNAVAILABLE",
                        "distributed_required 需要主节点已加载完整模型并生成精确身份。",
                    )
            elif _cfg.TASK_WORKER_EXPERIMENTAL_ENABLED:
                auto_provider_ids = self._eligible_remote_task_worker_provider_ids(
                    model_identity,
                    "full_inference",
                )
                if not auto_provider_ids:
                    auto_fallback_reason = "no_eligible_remote_provider"
                    if distributed_task_required:
                        raise coded_http_error(
                            503,
                            "TASK_WORKER_NO_ELIGIBLE_REMOTE_PROVIDER",
                            "distributed_required 没有可用且模型身份精确匹配的 PC Full Worker。",
                        )
                else:
                    template_stages, final_stage_id = dual_candidate_template()
                    candidate_index = 0
                    planned_stages = []
                    for stage in template_stages:
                        if stage.stage_type != "full_inference":
                            planned_stages.append(stage)
                            continue
                        if candidate_index >= len(auto_provider_ids):
                            planned_stages.append(replace(
                                stage,
                                provider="local_full_model",
                                fallback_providers=(),
                                pure=True,
                            ))
                            candidate_index += 1
                            continue
                        primary = auto_provider_ids[candidate_index]
                        other_remotes = [
                            provider_id for provider_id in auto_provider_ids
                            if provider_id != primary
                        ][:3]
                        planned_stages.append(replace(
                            stage,
                            provider=primary,
                            fallback_providers=tuple(
                                [
                                    *other_remotes,
                                    *(
                                        [] if distributed_task_required
                                        else ["local_full_model"]
                                    ),
                                ]
                            ),
                            pure=True,
                        ))
                        candidate_index += 1
                    stages = planned_stages

        request_id = str(_request_id_ctx.get("-") or "-")
        runtime_context = {
            "local_provider_executor": self.execute_task_worker_stage,
            "task_graph_remote_policy": (
                "manual" if remote_stage_id else "auto" if auto_remote else "local"
            ),
        }
        try:
            if stages is None:
                final_output, workflow = self._ensure_task_graph_coordinator().run_template(
                    template=req.task_graph_template,
                    root_input=root_input,
                    request_id=request_id,
                    session_id=target_session_id or "default",
                    model_identity=model_identity,
                    runtime_context=runtime_context,
                    workflow_id=req.workflow_id,
                    cancel_event=cancel_event,
                    request_deadline=request_deadline,
                )
            else:
                final_output, workflow = self._ensure_task_graph_coordinator().run(
                    stages=stages,
                    final_stage_id=final_stage_id,
                    template=req.task_graph_template,
                    root_input=root_input,
                    request_id=request_id,
                    session_id=target_session_id or "default",
                    model_identity=model_identity,
                    runtime_context=runtime_context,
                    workflow_id=req.workflow_id,
                    cancel_event=cancel_event,
                    request_deadline=request_deadline,
                )
        except WorkflowCancelled as exc:
            raise HTTPException(
                409,
                {"message": "任务链已取消", "workflow_id": exc.workflow_id},
            ) from exc
        except WorkflowExecutionError as exc:
            status_code = 504 if exc.code == "request_deadline_exceeded" else 500
            raise coded_http_error(
                status_code,
                exc.code,
                {
                    "message": str(exc),
                    "workflow_id": exc.workflow_id,
                    "stage_id": exc.stage_id,
                },
            ) from exc
        except WorkflowDeadlineExceeded as exc:
            raise coded_http_error(
                504,
                exc.code,
                {
                    "message": str(exc),
                    "workflow_id": exc.workflow_id,
                },
            ) from exc
        except TaskGraphUnavailable as exc:
            raise HTTPException(
                503,
                {
                    "message": "任务链 journal 写入失败，执行已停止。",
                    "reason": str(exc),
                },
            ) from exc
        except TaskGraphError as exc:
            raise HTTPException(400, f"任务链请求无效: {exc}") from exc

        response_text = str(final_output.get("content", "") or "").strip()
        thinking_content = final_output.get("thinking_content")
        if not response_text:
            raise HTTPException(500, "任务链最终聚合结果为空。")

        if distributed_task_required:
            precommit_attempts = [
                attempt
                for stage in workflow.get("stages", [])
                for attempt in stage.get("attempts", [])
                if attempt.get("state") == "completed"
            ]
            precommit_remote_attempts = [
                attempt
                for attempt in precommit_attempts
                if attempt.get("provider_kind") == "remote_full_worker"
            ]
            if not precommit_remote_attempts:
                try:
                    self._ensure_task_graph_coordinator().discard_result(
                        workflow["workflow_id"],
                        terminal_reason_code=REASON_REQUEST_REFUSED,
                    )
                except TaskGraphUnavailable as exc:
                    raise HTTPException(
                        503,
                        {
                            "message": "distributed_required 校验失败且结果无法丢弃。",
                            "reason": str(exc),
                        },
                    ) from exc
                raise coded_http_error(
                    503,
                    "TASK_WORKER_REMOTE_STAGE_NOT_EXECUTED",
                    "distributed_required 未执行任何远端 Full Worker Stage。",
                )
            try:
                validate_distributed_completion(
                    req.routing_preference,
                    {
                        "distributed_used": True,
                        "execution_mode": "task_graph",
                        "workers_used": [
                            str(attempt.get("provider_node_id", "") or "")
                            for attempt in precommit_remote_attempts
                            if attempt.get("provider_node_id")
                        ],
                    },
                )
            except DistributedCompletionError as exc:
                try:
                    self._ensure_task_graph_coordinator().discard_result(
                        workflow["workflow_id"],
                        terminal_reason_code=REASON_REQUEST_REFUSED,
                    )
                except TaskGraphUnavailable as discard_exc:
                    raise HTTPException(
                        503,
                        {
                            "message": "发行 profile 拒绝任务图结果且结果无法丢弃。",
                            "reason": str(discard_exc),
                        },
                    ) from discard_exc
                raise coded_http_error(
                    503,
                    "DISTRIBUTED_REQUIRED_RELEASE_ROUTE_FORBIDDEN",
                    f"distributed_required 发行路线校验失败：{exc}",
                ) from exc

        try:
            with hold_lock_until_deadline(
                self._gen_lock,
                request_deadline,
                operation="generation commit lock",
            ):
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                if cancel_event is not None and cancel_event.is_set():
                    _raise_if_generation_cancelled(cancel_event, req.generation_id)
                try:
                    workflow = self._ensure_task_graph_coordinator().commit_result(
                        workflow["workflow_id"],
                    )
                except WorkflowCancelled as exc:
                    raise ChatGenerationCancelled(
                        req.generation_id or "gen_unknown",
                    ) from exc
                except WorkflowDeadlineExceeded as exc:
                    raise coded_http_error(
                        504, REQUEST_DEADLINE_EXCEEDED, str(exc),
                    ) from exc
                except TaskGraphUnavailable:
                    raise
        except ChatGenerationCancelled:
            try:
                self._ensure_task_graph_coordinator().discard_result(workflow["workflow_id"])
            except TaskGraphUnavailable as exc:
                raise HTTPException(
                    503,
                    {
                        "message": "任务链取消无法写入 journal。",
                        "reason": str(exc),
                    },
                ) from exc
            raise
        except TaskGraphUnavailable as exc:
            raise HTTPException(
                503,
                {
                    "message": "任务链 journal 写入失败，结果未提交。",
                    "reason": str(exc),
                },
            ) from exc

        attempts = [
            attempt
            for stage in workflow.get("stages", [])
            for attempt in stage.get("attempts", [])
            if attempt.get("state") == "completed"
        ]
        usages = [
            dict(attempt.get("result_metadata", {}).get("usage", {}) or {})
            for attempt in attempts
        ]
        prompt_tokens = sum(int(usage.get("prompt_tokens", 0) or 0) for usage in usages)
        completion_tokens = sum(
            int(usage.get("completion_tokens", 0) or 0) for usage in usages
        )
        providers = sorted({
            str(attempt.get("provider", "") or "")
            for attempt in attempts
            if attempt.get("provider")
        })
        serving_node_id = (self._scheduler.get_effective_node_id() if self._scheduler is not None else "")
        participating_nodes = sorted({
            str(attempt.get("provider_node_id", "") or serving_node_id)
            for attempt in attempts
        }) or [serving_node_id]
        remote_attempts = [
            attempt for attempt in attempts
            if attempt.get("provider_kind") == "remote_full_worker"
        ]
        remote_nodes = sorted({
            str(attempt.get("provider_node_id", "") or "")
            for attempt in remote_attempts
            if attempt.get("provider_node_id")
        })
        remote_used = bool(remote_attempts)
        if distributed_task_required and not remote_used:
            raise coded_http_error(
                503,
                "TASK_WORKER_REMOTE_STAGE_NOT_EXECUTED",
                "distributed_required 未执行任何远端 Full Worker Stage。",
            )
        provider_status_by_id = {
            str(item.get("provider_id", "")): item
            for item in self._ensure_task_graph_coordinator().provider_status()
        }
        planned_remote_nodes = sorted({
            str(provider_status_by_id.get(provider_id, {}).get("node_id", "") or "")
            for provider_id in auto_provider_ids
            if provider_status_by_id.get(provider_id, {}).get("node_id")
        })
        retried_stages = [
            stage for stage in workflow.get("stages", [])
            if int(stage.get("retry_count", 0) or 0) > 0
        ]
        retry_error_codes = [
            str(stage.get("last_retry_error_code", "") or "")
            for stage in retried_stages
            if stage.get("last_retry_error_code")
        ]
        same_provider_retry_count = sum(
            int(stage.get("same_provider_retry_count", 0) or 0)
            for stage in retried_stages
        )
        total_retry_count = sum(
            int(stage.get("retry_count", 0) or 0)
            for stage in retried_stages
        )
        reassignment_count = max(
            0, total_retry_count - same_provider_retry_count,
        )
        fallback_used = reassignment_count > 0 or bool(
            auto_remote and not auto_provider_ids
        )
        retry_reason = retry_error_codes[0] if retry_error_codes else ""
        fallback_reason = (
            retry_reason if reassignment_count > 0 else auto_fallback_reason
        )
        metrics = _augment_chat_metrics(
            {
                "engine": backend_id_for(self._host),
                "execution_mode": "task_graph",
                "provider": (
                    providers[0] if len(providers) == 1 else "task_graph"
                ),
                "orchestrator": "task_graph",
                "subproviders": providers,
                "workflow_id": workflow["workflow_id"],
                "workflow_template": workflow["template"],
                "workflow_state": workflow["state"],
                "partial_result": bool(workflow.get("partial_result", False)),
                "stage_retry_count": total_retry_count,
                "same_provider_retry_count": same_provider_retry_count,
                "reassignment_count": reassignment_count,
                "retry_reason": retry_reason,
                "stage_count": workflow["stage_count"],
                "stage_attempt_count": workflow["attempt_count"],
                "nodes_planned": (
                    1 + len(planned_remote_nodes)
                    if auto_remote else len(participating_nodes)
                ),
                "nodes_participated": len(participating_nodes),
                "participating_nodes": participating_nodes,
                "distributed_requested": bool(remote_stage_id or auto_remote),
                "distributed_used": remote_used,
                "distributed_kind": (
                    "task_graph_remote_manual"
                    if remote_used and remote_stage_id
                    else "task_graph_remote_auto"
                    if remote_used and auto_remote
                    else "task_graph_local_fallback"
                    if auto_remote
                    else "task_graph_local_poc"
                ),
                "workers_used": remote_nodes,
                "manual_remote_stage": remote_stage_id,
                "manual_remote_provider": remote_provider_id,
                "auto_remote_enabled": auto_remote,
                "auto_remote_providers": auto_provider_ids,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "usage_estimated": any(
                    bool(attempt.get("result_metadata", {}).get(
                        "usage_estimated", False,
                    ))
                    for attempt in attempts
                ),
                "elapsed_seconds": workflow["duration_seconds"],
                "fallback": fallback_used,
                "fallback_reason": fallback_reason,
            },
            req,
            route=(
                f"{_chat_origin(req)}_to_task_graph_manual_remote"
                if remote_stage_id
                else f"{_chat_origin(req)}_to_task_graph_auto_remote"
                if auto_remote
                else f"{_chat_origin(req)}_to_local_task_graph"
            ),
        )
        completed_history = [
            *history,
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": response_text},
        ]
        followups = _fallback_followups(completed_history, [])
        save_metrics = dict(metrics)
        save_metrics["followups"] = followups
        self._commit_chat_context_turn(
            target_session_id, req.message, response_text, save_metrics,
            expected_revision=prepared_context.revision,
        )

        self._conversation_stats["total_prompt_tokens"] += prompt_tokens
        self._conversation_stats["total_generated_tokens"] += completion_tokens
        self._conversation_stats["total_time_seconds"] += workflow["duration_seconds"]
        self._conversation_stats["rounds"] += 1
        try:
            self._record_task_complete()
        except Exception:
            pass

        return {
            "content": response_text,
            "thinking_content": thinking_content,
            "metrics": metrics,
            "followups": followups,
        }



    # ------------------------------------------------------------------
    # 会话管理（1.2c 复制自 api_server.py:1090-1175，全局 → 实例属性）
    # ------------------------------------------------------------------
    def _get_active_history(self) -> list:
        """
        获取当前活跃会话的对话历史列表。

        如果没有活跃会话，返回空列表（不自动创建会话）。
        返回的列表对象可被原地修改（append、clear 等）。
        """
        self._sync_chat_context_facade()
        history = self._chat_context.active_history()
        self._publish_chat_context_facade()
        return history

    def _ensure_device_profile(self) -> Optional[dict]:
        """惰性设备画像探测（api_server 启动事件等价物，同步版）。"""
        if self._device_profile_started:
            return self._device_profile
        self._device_profile_started = True
        try:
            from device_profiler import get_profile
            profiler = get_profile()
            self._device_profile = profiler.to_dict()
            logger.info(
                f"🚀 设备检测完成: tier={profiler.tier.value} "
                f"score={profiler.score:.1f}/100"
            )
        except Exception as e:
            logger.error(f"设备检测失败: {e}")
            self._device_profile = None
        return self._device_profile

    def _init_kv_cache(self):
        """初始化分页 KV 缓存（根据设备画像自适应大小）。"""
        import torch as _torch
        from paged_kv_cache import PagedKVCache

        num_heads = 16      # Qwen-1.8B: 16 attention heads
        head_dim = 64
        mgr = self._host
        if getattr(mgr, "model", None) is not None:
            try:
                cfg = mgr.model.config
                num_heads = cfg.num_attention_heads
                head_dim = cfg.hidden_size // num_heads
            except Exception:
                pass

        device = str(mgr.get_device()) if callable(getattr(mgr, "get_device", None)) else "cpu"
        self._ensure_device_profile()
        if self._device_profile:
            self._kv_cache = PagedKVCache.from_profile(
                profile=self._device_profile,
                device=device,
                dtype=_torch.float16,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            logger.info(
                f"🧠 KV 缓存已初始化 (profile): num_heads={num_heads}, "
                f"head_dim={head_dim}, device={device}"
            )
        else:
            self._kv_cache = PagedKVCache(
                device=device,
                dtype=_torch.float16,
            )
            logger.info(
                f"🧠 KV 缓存已初始化 (default): device={device}"
            )
        return self._kv_cache

    def _switch_session(self, target_id: str, *, force_reload: bool = False) -> None:
        """
        切换到目标会话：暂存当前历史 → 加载目标历史 → 清 KV Cache。
        """
        prepared = self._prepare_chat_context(
            target_id, force_reload=force_reload,
        )
        logger.info(
            "已切换到会话: %s (reloaded=%s)",
            prepared.session_id,
            prepared.reloaded,
        )

    def _reset_runtime_conversation_state(self, clear_histories: bool = True) -> None:
        """Clear in-memory conversation/KV state after a model change."""
        if self._kv_cache:
            self._kv_cache.clear()
        self._kv_cache = None
        self._sync_chat_context_facade()
        self._chat_context.invalidate(clear_histories=clear_histories)
        self._publish_chat_context_facade()
        self._conversation_stats = {
            "total_prompt_tokens": 0,
            "total_generated_tokens": 0,
            "total_time_seconds": 0.0,
            "rounds": 0,
        }

    def _auto_title_session(self, session_id: str, first_message: str) -> None:
        """用首条用户消息自动生成会话标题（截取前30字）"""
        title = first_message.strip()[:30]
        if len(first_message.strip()) > 30:
            title += "..."
        try:
            _local_store.update_local_session_title(session_id, title)
        except Exception as exc:
            logger.warning("SQLite 自动标题更新失败: session=%s: %s", session_id, exc)

    # ------------------------------------------------------------------
    # 追问生成（1.2c 复制自 api_server.py:1228-1354，PyTorch 路径）
    # ------------------------------------------------------------------
    def _generate_followups(
        self,
        history: list,
        tokenizer,
        model,
        device,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[str]:
        """
        根据对话上下文，让模型生成 2-3 个追问建议。
        """
        import torch as _torch

        if not history or len(history) < 2:
            return []
        if cancel_event is not None and cancel_event.is_set():
            return []

        system_prompt = (
            "根据对话历史，生成3个用户可能追问的疑问句。\n"
            "严格规则：\n"
            "1. 每个输出必须以 Q: 开头，单独一行\n"
            "2. 每个输出必须是疑问句（以？结尾），严禁输出陈述句\n"
            "3. 不要输出解释、列举、定义等陈述性内容\n"
            "正确示例:\n"
            "Q: 深度学习与机器学习有什么区别？\n"
            "Q: 能推荐一些入门学习资源吗？\n"
            "Q: 这个概念在实际中有哪些应用？\n"
            "错误示例（严禁输出）:\n"
            "Q: 机器学习和深度学习有以下几点区别：\n"
            "Q: 深度学习是机器学习的一个分支\n"
            "Q: 1. 监督学习 2. 无监督学习"
        )
        followup_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        recent = history[-6:]
        for msg in recent:
            followup_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        followup_prompt += "<|im_start|>assistant\n"

        questions = []

        try:
            inputs = tokenizer(followup_prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            stop_criteria_kwargs = (
                {"cancel_event": cancel_event} if cancel_event is not None else {}
            )
            stop_criteria = self._host._build_stop_criteria(
                [], input_ids.shape[1], **stop_criteria_kwargs,
            )
            generation_kwargs = {}
            if stop_criteria is not None:
                generation_kwargs["stopping_criteria"] = stop_criteria

            with _torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=80,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **generation_kwargs,
                )

            if cancel_event is not None and cancel_event.is_set():
                return []

            generated = outputs[0][input_ids.shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.upper().startswith("Q:") or line.upper().startswith("Q：") or line.startswith("问："):
                    q = line.split(":", 1)[-1].split("：", 1)[-1].strip()
                else:
                    q = re.sub(r'^[\d]+[\.\、\)）\s\-]+', '', line).strip()
                if q and len(q) >= 5 and len(q) <= 80 and _is_question(q):
                    questions.append(q)

            hallucination_patterns = [
                "通义千问", "千问", "ChatGPT", "Claude", "GPT-", "文心一言",
                "讯飞星火", "豆包", "Kimi", "Copilot", "Bard", "Gemini",
                "百川", "智谱", "ChatGLM", "混元",
            ]
            questions = [
                q for q in questions
                if not any(p in q for p in hallucination_patterns)
            ]

            filtered = []
            seen_words = set()
            for q in questions:
                words = frozenset(q[:10])
                if words not in seen_words:
                    seen_words.add(words)
                    filtered.append(q)
            questions = filtered

            logger.info(f"模型追问生成: {len(questions)} 条 → {questions}")

        except Exception as e:
            logger.warning(f"追问生成失败（非致命）: {e}")
            questions = []

        if len(questions) < 2:
            fallback = _fallback_followups(history, questions)
            questions = fallback

        return questions[:3]

    # ------------------------------------------------------------------
    # 路由决策与模型加载（1.2c 复制自 api_server.py:2261-2293 / 3882-3911 /
    # 3913-4011；scheduler 依赖 → self._scheduler 注入点，None=单机基线）
    # ------------------------------------------------------------------
    def _distributed_path_available(self) -> bool:
        """当前是否有可用的分布式执行路径（T9.5 同步 api_server 口径）。

        主节点可协调流水线/Worker；从节点可转发主节点；两者都计为合格路径。
        """
        sched = self._scheduler
        try:
            if not (sched is not None
                    and sched.get_distributed_inference_enabled()
                    and self._run_mode == "distributed"):
                return False
            return sched._effective_role() in ("master", "client")
        except Exception:
            return False

    def _routing_gate_error(self, req: "ChatRequest") -> Optional[str]:
        """路由偏好预检（T9.5）：返回拒绝原因，None 表示允许继续。"""
        if req.routing_preference == "distributed_required":
            if not self._distributed_path_available():
                return (
                    "distributed_required 但当前没有可用的分布式路径"
                    "（分布式未启用或本节点不是主节点）"
                )
        return None

    def _enforce_distributed_required(
        self,
        req: "ChatRequest",
        metrics: Optional[dict] = None,
        *,
        detail: str = "分布式执行未完成",
    ) -> None:
        """Reject the EngineHost's local fallback for an explicit route."""
        from distributed_completion import (
            DistributedCompletionError,
            validate_distributed_completion,
        )
        from fastapi import HTTPException
        try:
            validate_distributed_completion(
                req.routing_preference, metrics, detail=detail,
            )
        except DistributedCompletionError as exc:
            raise HTTPException(
                503, f"distributed_required 执行失败：{exc}",
            ) from exc

    def _external_route_decision(self, req: "ChatRequest"):
        """按当前配置 + 请求 flag 计算外部路由决策（纯函数包装，读实时配置）。"""
        if req.routing_preference in {"local_only", "distributed_required"}:
            # T9.5：local_only 覆盖一切外部路由意图，数据不出集群
            from types import SimpleNamespace as _SN
            return _SN(
                use_external=False,
                eligible=False,
                reason=f"{req.routing_preference}_override",
            )
        import config as _cfg
        from external_provider import decide_external_route

        return decide_external_route(
            enabled=bool(getattr(_cfg, "EXTERNAL_ENABLED", False)),
            base_url=str(getattr(_cfg, "EXTERNAL_BASE_URL", "") or ""),
            data_scope=str(getattr(_cfg, "EXTERNAL_DATA_SCOPE", "opt_in")),
            allow_external=bool(req.allow_external),
            prefer_external=bool(req.prefer_external),
            prompt_chars=len(req.message or ""),
            min_prompt_chars=int(getattr(_cfg, "EXTERNAL_MIN_PROMPT_CHARS", 0) or 0),
        )

    def _local_native_vision_available(self) -> bool:
        """Check the active local engine without importing or loading a model."""
        check = getattr(self._host, "native_vision_available", None)
        if not callable(check):
            return False
        try:
            return bool(check())
        except Exception:
            return False

    def _should_forward_chat_to_master(self) -> bool:
        sched = self._scheduler
        return bool(
            sched is not None
            and sched.get_distributed_inference_enabled()
            and self._run_mode == "distributed"
            and sched._effective_role() == "client"
        )

    def _pipeline_worker_is_reserved(self) -> bool:
        sched = self._scheduler
        check = getattr(sched, "has_pipeline_worker_reservation", None) if sched else None
        return bool(callable(check) and check())

    def _pipeline_model_is_prepared(self) -> bool:
        return bool(getattr(self._host, "is_pipeline_prepared", False))

    def _run_exclusive_model_change(
        self, change, prepare=None, *, release_worker_reservation: bool = False,
    ):
        """Invalidate distributed state only after the model change succeeds.

        （api_server 版嵌套 scheduler 锁；本进程无 scheduler 时仅持
        ModelHost 执行锁，等价单机拓扑。）
        """
        with self._host.full_chat_execution_lock:
            self.release_worker_runtime()
            sched = self._scheduler
            if sched is None:
                result = change()
                if not (
                    isinstance(result, dict)
                    and result.get("success") is False
                ) and prepare is not None:
                    prepare()
                return result

            with sched._inference_lock:
                with sched._layer_execution_lock:
                    result = change()
                    if (
                        isinstance(result, dict)
                        and result.get("success") is False
                    ):
                        return result

                    begin_transition = getattr(
                        sched, "_begin_layer_config_model_change", None,
                    )
                    transition_started = callable(begin_transition)
                    if transition_started:
                        begin_transition()
                    try:
                        if prepare is not None:
                            prepare()
                        invalidate_transaction = getattr(
                            sched, "_invalidate_pipeline_load_transaction", None,
                        )
                        if callable(invalidate_transaction):
                            invalidate_transaction(
                                reason_code="pipeline_model_changed",
                                reason="local model replacement committed",
                            )
                        with sched._layer_config_lock:
                            sched._worker_assignments.clear()
                            sched._layer_config_retry_state.clear()
                            sched._pipeline_load_transaction = None
                            sched._active_pipeline_capacity_plan = None
                            sched._active_layer_config = None
                            sched._last_layer_config_ack_payload = None
                            sched._local_pipeline_steps.clear()
                        if release_worker_reservation:
                            release = getattr(
                                sched,
                                "release_pipeline_worker_for_local_model",
                                None,
                            )
                            if callable(release):
                                release()
                            else:
                                sched._pipeline_worker_reserved = False
                        self._refresh_pipeline_layer_config(sched)
                    finally:
                        if transition_started:
                            end_transition = getattr(
                                sched, "_end_layer_config_model_change", None,
                            )
                            if callable(end_transition):
                                end_transition()
                    return result

    def _refresh_pipeline_layer_config(self, sched) -> None:
        """主节点模型变化后重新下发层配置，并使旧 ACK 失效。"""
        try:
            if sched._effective_role() == "master":
                request_sync = getattr(
                    sched, "request_authoritative_layer_sync", None,
                )
                if callable(request_sync):
                    request_sync()
                else:
                    sched.push_layer_config_to_clients()
        except Exception:
            pass

    def _auto_load_default_model(self):
        """自动加载默认模型（thin client / 首次请求时服务端无模型的情况）。"""
        import config as cfg
        import glob
        import os as _os
        import time as _time

        # 0. TP 孤岛引擎优先（启用即为孤岛网关节点，无本地文件依赖）
        if getattr(cfg, "ISLAND_ENABLED", False) and getattr(cfg, "ISLAND_BASE_URL", ""):
            from island_engine import mask_island_url

            resolution = resolve_model_load(
                ModelLoadFacts(
                    model_id=None,
                    island_enabled=True,
                    island_base_url=str(cfg.ISLAND_BASE_URL),
                ),
                requested_engine="island",
                requested_quant="island",
            )

            logger.info(
                f"自动加载孤岛引擎: endpoint={mask_island_url(cfg.ISLAND_BASE_URL)}"
            )
            t0 = _time.time()
            cfg.INFERENCE_ENGINE = resolution.engine
            cfg.QUANT_TYPE = resolution.quant_type
            cfg.USE_COMPILE = False
            self._run_exclusive_model_change(
                lambda: self._host.load_model(
                    profile=self._ensure_device_profile(),
                    engine=resolution.engine,
                    quant_type=resolution.quant_type,
                    resolution=resolution,
                )
            )
            self._reset_runtime_conversation_state(clear_histories=False)
            self._init_kv_cache()
            self._conversation_stats = {
                "total_prompt_tokens": 0,
                "total_generated_tokens": 0,
                "total_time_seconds": 0.0,
                "rounds": 0,
            }
            self._host.model_loaded = True
            self._host.current_quant = resolution.quant_type
            if self._scheduler is not None:
                try:
                    self._scheduler.refresh_task_worker_capabilities()
                except Exception:
                    pass
            logger.info(f"✅ 孤岛引擎自动连接完成 ({_time.time() - t0:.1f}s)")
            return

        # ★ 2026-09-19：**按设备画像**选择默认模型（用户裁定：边缘/轻薄本 <1B，PC ~2B）。
        #   此前直接读 `cfg.GGUF_MODEL_PATH` / `cfg.MODEL_PATH`（静态常量，且原指向已退役的
        #   `qwen-1_8b`）⇒ 无论什么设备都加载同一个模型。
        _active = cfg.get_active_model_paths() if hasattr(cfg, "get_active_model_paths") else {}
        _active_id = str(_active.get("model_id") or "")
        _active_gguf = str(_active.get("gguf_path") or "")
        _active_safetensors = str(_active.get("model_path") or "")

        # 1. 优先查找 GGUF 文件（llama.cpp 引擎，不依赖 transformers/bitsandbytes）
        gguf_candidates = []
        gguf_configured = _active_gguf if _active_id else cfg.GGUF_MODEL_PATH
        if _os.path.isfile(gguf_configured):
            gguf_candidates.append(gguf_configured)
        models_dir = _os.path.dirname(gguf_configured)
        if not _active_id and _os.path.isdir(models_dir):
            for f in sorted(glob.glob(_os.path.join(models_dir, "*.gguf"))):
                if f not in gguf_candidates:
                    gguf_candidates.append(f)

        if _active_id:
            logger.info(f"默认模型按设备画像选择: {_active_id}")

        safetensors_path = (
            _active_safetensors
            if _active_safetensors and _os.path.isdir(_active_safetensors)
            else "" if _active_id
            else cfg.MODEL_PATH if _os.path.isdir(cfg.MODEL_PATH) else ""
        )
        gguf_path = gguf_candidates[0] if gguf_candidates else ""
        prefer_pytorch = bool(
            getattr(cfg, "PREFER_PYTORCH", False) and safetensors_path
        )
        try:
            from torch_runtime import cuda_available

            profile = self._ensure_device_profile()
            cuda_ok = effective_pytorch_cuda_available(
                system_cuda_available=cuda_available(load=True),
                profile=profile,
            )

            resolution = resolve_model_load(
                ModelLoadFacts(
                    model_id=_active_id or None,
                    model_name=_active_id or "默认模型",
                    registered=True,
                    has_safetensors=bool(safetensors_path),
                    has_gguf=bool(gguf_path),
                    safetensors_path=safetensors_path or None,
                    gguf_path=gguf_path or None,
                    preferred_engine="pytorch" if prefer_pytorch else preferred_engine_for_artifacts(
                        has_safetensors=bool(safetensors_path),
                        has_gguf=bool(gguf_path),
                        cuda_available=cuda_ok,
                    ),
                    cuda_available=cuda_ok,
                ),
                requested_engine="pytorch" if prefer_pytorch else "auto",
                requested_quant=getattr(cfg, "QUANT_TYPE", "int4"),
            )
        except ModelLoadResolutionError as exc:
            raise FileNotFoundError(
                f"未找到可自动加载的模型文件（{exc.code}: {exc.message}）。已检查:\n"
                f"  GGUF 配置路径: {gguf_configured}\n"
                f"  Safetensors 路径: {cfg.MODEL_PATH}\n"
                f"  models 目录: {models_dir}"
            ) from exc

        engine = resolution.engine
        model_path = resolution.model_path
        quant = resolution.quant_type

        logger.info(f"自动加载默认模型: path={model_path}, engine={engine}")

        t0 = _time.time()
        cfg.INFERENCE_ENGINE = engine
        cfg.QUANT_TYPE = quant
        cfg.USE_COMPILE = False

        self._run_exclusive_model_change(
            lambda: self._host.load_model(
                model_path=model_path,
                quant_type=quant,
                profile=self._ensure_device_profile(),
                model_id=resolution.model_id,
                engine=engine,
                resolution=resolution,
            )
        )

        self._reset_runtime_conversation_state(clear_histories=False)
        self._init_kv_cache()
        self._conversation_stats = {
            "total_prompt_tokens": 0,
            "total_generated_tokens": 0,
            "total_time_seconds": 0.0,
            "rounds": 0,
        }
        self._host.model_loaded = True
        self._host.current_quant = getattr(self._host, "quant_type", None) or quant
        if self._scheduler is not None:
            try:
                self._scheduler.refresh_task_worker_capabilities()
            except Exception:
                pass
        elapsed = _time.time() - t0
        logger.info(f"默认模型自动加载完成 ({elapsed:.1f}s)")

    def _ensure_chat_model_or_forwarding(self, req: Optional["ChatRequest"] = None) -> None:
        """Load a local model only when this request cannot be master-forwarded."""
        if req is not None and self._external_route_decision(req).use_external:
            return
        if req is not None and req.routing_preference == "local_only":
            # T9.5：local_only 强制本地执行，从节点也不转发
            pass
        elif self._should_forward_chat_to_master():
            return
        if self._pipeline_worker_is_reserved():
            from fastapi import HTTPException
            raise HTTPException(
                503,
                "本设备正作为 PyTorch 分层从节点，不能加载本地完整模型。",
            )
        if self._pipeline_model_is_prepared():
            if req is not None and req.routing_preference == "local_only":
                from fastapi import HTTPException
                raise HTTPException(
                    409,
                    "当前模型仅以分布式流水线模式准备；"
                    "local_only 请求需要先显式加载完整模型。",
                )
            return
        if self._host.model_loaded and getattr(self._host, "is_loaded", False):
            return
        self._auto_load_default_model()

    # ------------------------------------------------------------------
    # 外部推理整请求路由（1.2c 复制自 api_server._execute_external_chat
    # api_server.py:2294-2400；宿主适配：conversation_stats → 实例属性、
    # scheduler.record_task_complete → self._on_task_complete 回调（1.4 注入））
    # ------------------------------------------------------------------
    def execute_external_chat(
        self,
        req: ChatRequest,
        history: list,
        target_session_id: Optional[str],
        cancel_event: Optional[threading.Event] = None,
        *,
        context_revision: Optional[int] = None,
    ) -> Dict[str, Any]:
        """整请求路由到外部推理服务（与 llama.cpp/孤岛整请求路径同构）。"""
        import config as _cfg
        from external_provider import get_external_chat_client
        from multimodal import build_openai_user_content

        client = get_external_chat_client()
        client.ensure_connected()
        request_history = [
            *history,
            {
                "role": "user",
                "content": build_openai_user_content(
                    req.message, getattr(req, "image_data_urls", []),
                ),
            },
        ]
        result = client.chat(
            request_history,
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            allow_external=req.allow_external,
            cancel_event=cancel_event,
        )
        _raise_if_generation_cancelled(cancel_event, req.generation_id)
        response_text = result.get("content", "")
        if not req.show_thinking:
            response_text = _strip_native_thinking_tags(response_text)
        completed_history = [
            *history,
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": response_text},
        ]
        tokens_per_sec = result.get("tokens_per_second", 0)
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        metrics = _augment_chat_metrics(
            {
                "engine": "external_api",
                "execution_mode": "external_api",
                "route": f"{_chat_origin(req)}_to_external_api",
                "provider": "external_openai",
                "external_label": getattr(_cfg, "EXTERNAL_LABEL", ""),
                "external_base_url": client.masked_base_url,
                "data_scope": getattr(_cfg, "EXTERNAL_DATA_SCOPE", ""),
                "model": result.get("model", "") or client.model_name,
                "tokens_per_second": round(tokens_per_sec, 1) if tokens_per_sec else 0,
                "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else 0,
                "generated_tokens": completion_tokens,
                "completion_tokens": completion_tokens,
                "usage": usage,
                "usage_estimated": bool(result.get("usage_estimated", False)),
                "fallback": False,
                "fallback_reason": "",
            },
            req,
            serving_node_id=self._serving_node_id,
            distributed_enabled=self._distributed_enabled,
        )

        followups = _fallback_followups(completed_history, [])

        save_metrics = dict(metrics)
        save_metrics["followups"] = followups
        self._commit_chat_context_turn(
            target_session_id, req.message, response_text, save_metrics,
            expected_revision=context_revision,
        )

        self._conversation_stats["total_generated_tokens"] += completion_tokens
        self._conversation_stats["rounds"] += 1
        if self._on_task_complete is not None:
            try:
                self._on_task_complete(success=True)
            except Exception:
                pass

        logger.info(
            f"外部推理完成: {completion_tokens} tokens, "
            f"endpoint={client.masked_base_url}"
        )
        return {
            "content": response_text,
            "thinking_content": None,
            "metrics": metrics,
            "followups": followups,
        }

    # ------------------------------------------------------------------
    # generation 注册表（取消语义，对齐 api_server._register_generation）
    # ------------------------------------------------------------------
    def register_generation(
        self, generation_id: Optional[str] = None
    ) -> Tuple[str, threading.Event]:
        with self._gen_lock:
            gid = generation_id or f"gen_{uuid4().hex[:12]}"
            ev = self._generations.get(gid)
            if ev is None:
                ev = threading.Event()
                self._generations[gid] = ev
            return gid, ev

    def unregister_generation(self, generation_id: str) -> None:
        with self._gen_lock:
            self._generations.pop(generation_id, None)

    def cancel_generation(self, generation_id: str) -> bool:
        """置取消事件；返回 False 表示 generation_id 未知（404 语义）。"""
        with self._gen_lock:
            ev = self._generations.get(generation_id)
            if ev is None:
                return False
            ev.set()
            return True
