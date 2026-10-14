"""inference-svc 契约模型（微服务架构改造计划 §4.1，冻结契约）。

本文件定义的 pydantic 模型是 inference-svc 与其他服务
（scheduler-svc / api-gateway）之间的边界，契约测试锁定字段与语义。
字段名/类型/语义对齐 api_server.py 现有对外行为
（并行共存期间旧后端保持可运行基线，见 §1.4）。

版本基线：2026-08-03（对齐 api_server.py ChatRequest 核心字段）
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from . import __contract_version__, __version__

try:
    from multimodal import validate_image_data_urls
except ImportError:  # package import: src.inference_service.protocol
    from ..multimodal import validate_image_data_urls


class HealthResponse(BaseModel):
    """进程存活探活（不涉及模型加载）。"""

    status: str = "ok"
    service: str = "inference-svc"
    version: str = __version__
    contract_version: str = __contract_version__


class ReadyResponse(BaseModel):
    """模型/层段已就绪（区分 /health 与 /ready，冷启动方案 §5.4 要求）。"""

    ready: bool
    model_loaded: bool
    layers: List[str] = Field(default_factory=list)


class StatusResponse(BaseModel):
    """引擎、当前模型、显存、层段、KV 缓存状态。"""

    engine: Optional[str] = None
    model_id: Optional[str] = None
    device: Optional[str] = None
    model_loaded: bool = False
    quant_type: Optional[str] = None
    kv_cache: Dict[str, Any] = Field(default_factory=dict)
    layers: List[str] = Field(default_factory=list)


class LoadModelRequest(BaseModel):
    engine: Optional[str] = Field(
        default=None, description="引擎：llama_cpp | pytorch | island；缺省按画像选择"
    )
    quant_type: Optional[str] = Field(
        default=None, description="量化类型（int4/int8/fp16 等），缺省用配置默认"
    )
    use_compile: bool = Field(default=False, description="PyTorch 引擎是否 apply_compile")
    model_id: Optional[str] = Field(
        default=None, description="模型标识（配置别名或路径）"
    )
    layer_range: Optional[str] = Field(
        default=None, description="client 角色可在此一并加载层段（等价 /v1/layers/load）"
    )


class UnloadModelRequest(BaseModel):
    pass


class SwitchModelRequest(BaseModel):
    model_id: str = Field(..., description="目标模型标识")
    engine: Optional[str] = None


class ChatRequest(BaseModel):
    """对齐 api_server.ChatRequest 核心字段（2026-08-03 基线）。

    1.2 复制 api_server 执行段时按实际使用字段扩展（task_graph 系列等）。
    """

    message: str = Field(..., description="用户消息", min_length=1)
    image_data_urls: List[str] = Field(
        default_factory=list,
        max_length=4,
        description="PNG/JPEG/WebP base64 data URL；external_api 最多四张，本地 MTMD 仅一张",
    )
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    max_new_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    show_thinking: bool = Field(default=False, description="启用深度思考展示")
    streaming_mode: str = Field(
        default="full",
        pattern="^(full|fast|interactive)$",
        description="full=完整功能 | fast=真流式逐 token | interactive=真流式逐 token + T9 事件契约；三者共用历史、裁剪和完成时事务提交",
    )
    routing_preference: Literal[
        "auto", "local_only", "distributed_preferred", "distributed_required"
    ] = Field(
        default="auto",
        description="请求级路由偏好（T9 契约）: auto=沿用集群配置 | local_only=仅本地 | distributed_preferred=优先分布式可回退 | distributed_required=无分布式路径则失败",
    )
    client_node_id: Optional[str] = None
    client_node_type: Optional[str] = None
    client_mode: Optional[str] = None
    client_app_variant: Optional[str] = None
    execution_mode: Literal["auto", "task_graph"] = Field(default="auto")
    task_graph_template: Literal["dual_candidate"] = Field(
        default="dual_candidate",
        description="任务链模板；首期仅支持双候选校验（对齐 api_server.ChatRequest）",
    )
    task_graph_remote_stage: Literal[
        "", "candidate_a", "candidate_b", "aggregate"
    ] = Field(
        default="",
        description="N2.1 手动指定唯一远端 Stage；为空时全部本地执行",
    )
    task_graph_remote_provider_id: str = Field(
        default="",
        max_length=64,
        description="N2.1 显式远端 Provider ID；不参与自动选择",
    )
    task_graph_auto_remote: bool = Field(
        default=False,
        description="N2.1 自动选择远端 Stage（与显式指定互斥）",
    )
    workflow_id: Optional[str] = None
    generation_id: Optional[str] = None
    allow_external: bool = Field(
        default=False,
        description=(
            "路线 B 数据作用域按请求授权：允许本请求路由到外部推理服务"
            "（QLH_EXTERNAL_*）。缺省 False——旧客户端行为不变，数据不出集群。"
        ),
    )
    prefer_external: bool = Field(
        default=False,
        description=(
            "路线 B：优先使用外部推理服务（仍受 QLH_EXTERNAL_DATA_SCOPE "
            "作用域门控约束；deny 档位下即使置 true 也不外发）。"
        ),
    )

    @model_validator(mode="after")
    def validate_multimodal_route(self):
        self.image_data_urls = validate_image_data_urls(self.image_data_urls)
        if not self.image_data_urls:
            return self
        if self.execution_mode != "auto":
            raise ValueError("图像请求暂不支持 task_graph 执行模式")
        if self.routing_preference == "local_only":
            if len(self.image_data_urls) != 1:
                raise ValueError("本地 MTMD 图像请求仅支持一张图片")
            if self.streaming_mode != "full":
                raise ValueError("本地 MTMD 图像请求仅支持 full 响应模式")
            if self.allow_external or self.prefer_external:
                raise ValueError("local_only 图像请求不能同时授权外部路由")
            return self
        if not self.allow_external or not self.prefer_external:
            raise ValueError("图像请求必须显式设置 allow_external 和 prefer_external")
        return self


class ChatCancelRequest(BaseModel):
    generation_id: str = Field(..., description="由 /v1/chat(/stream) 返回的生成 ID")


class WorkerStageRequest(BaseModel):
    """task-worker Stage 请求（ProviderStageRequest 的 JSON 传输形，
    1.4 InferenceClient 经 /v1/worker/stage 远程执行时使用）。"""

    workflow_id: str = ""
    request_id: str = ""
    stage_id: str = ""
    stage_type: str = ""
    provider_id: str = ""
    dependencies: Dict[str, Any] = Field(default_factory=dict)
    root_input: Dict[str, Any] = Field(default_factory=dict)
    model_identity: Optional[Dict[str, Any]] = None
    runtime_context: Dict[str, Any] = Field(default_factory=dict)


class SpeculativeRunRequest(BaseModel):
    """投机解码实验端点（QLH_SPEC_ENABLED 门控沿用，§4.1）。
    字段对齐 api_server.SpeculativeExperimentRequest（2026-08-03 修复：
    此前只有 prompt/max_new_tokens/temperature/top_p，engine_host.speculative_run
    访问 message/gamma/max_rounds/seed/draft_hint 时必然 AttributeError→500）。"""

    message: str = Field(default="", max_length=8192)
    allow_external: bool = Field(
        default=False,
        description=(
            "数据作用域按请求授权：草稿 token 由用户内容派生，本路径确实"
            "把数据送出集群，与路线 B 共用 QLH_EXTERNAL_DATA_SCOPE 门控。"
        ),
    )
    max_new_tokens: int = Field(default=64, ge=1, le=1024)
    gamma: int = Field(default=0, ge=0, le=16, description="每轮草稿数；0=用 QLH_SPEC_GAMMA")
    max_rounds: int = Field(default=0, ge=0, le=1024, description="0=用 QLH_SPEC_MAX_ROUNDS")
    temperature: float = Field(
        default=-1.0, ge=-1.0, le=2.0,
        description="<0 = 用 QLH_SPEC_TEMPERATURE；0 = 贪心模式",
    )
    seed: int = Field(default=0, ge=0, le=2147483647, description="RNG 种子，保证可复现")
    draft_hint: str = Field(
        default="", max_length=2048,
        description="PoC 假 draft 模型的提示序列：命中则接受率高，用于演示接受率对比",
    )


class LayerLoadRequest(BaseModel):
    layer_range: str = Field(..., description="层段范围，如 \"0-12\"")
    embed: bool = Field(default=False, description="是否包含 embedding 段")
    lm_head: bool = Field(default=False, description="是否包含 lm_head 段")


class LayerUnloadRequest(BaseModel):
    layer_range: str = Field(...)


class LayerForwardRequest(BaseModel):
    """层段前向：hidden states 经 tensor_transport 序列化后 base64 传输；
    KV 缓存永不跨进程，只传任务级引用 past_key_values_ref（§4.1）。"""

    layer_range: str = Field(...)
    tensor_ref: str = Field(
        ..., description="serialize_tensor_fast 序列化后的字节（base64）"
    )
    past_key_values_ref: Optional[str] = Field(
        default=None, description="KV 任务引用（task_id），不传 KV 数据"
    )
    task_id: Optional[str] = None


class EmbeddingRequest(BaseModel):
    tensor_ref: str = Field(..., description="输入 token ids 张量（base64）")


class LMHeadRequest(BaseModel):
    tensor_ref: str = Field(..., description="hidden states 张量（base64）")


class KVInitRequest(BaseModel):
    task_id: Optional[str] = Field(
        default=None, description="调用方预生成的任务 ID；缺省服务端生成"
    )
    device: Optional[str] = None
    page_size: Optional[int] = Field(default=None, ge=1, le=4096)
    max_pages: Optional[int] = Field(default=None, ge=1, le=65536)
    cold_cache_dir: Optional[str] = Field(default=None, min_length=1, max_length=4096)
    cold_max_pages: Optional[int] = Field(default=None, ge=1, le=65536)
    cache_unit_size: Optional[int] = Field(default=None, ge=1, le=4096)


class KVFreeRequest(BaseModel):
    task_id: str = Field(...)


class OkResponse(BaseModel):
    """统一成功响应（内部契约形态；对外映射由 api-gateway 负责，§4.2）。"""

    success: bool = True
    message: str = ""
    data: Optional[Dict[str, Any]] = None
