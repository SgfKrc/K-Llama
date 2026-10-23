"""
配置默认值与旧消费者兼容投影
============================================
AppRuntime 通过 build_settings(environment) 解析独立快照。
模块常量供尚待迁移的旧入口读取，导入阶段保持无资源启动。

量化模式对照表（已验证，Qwen-1.8B-Chat + RTX GPU + PyTorch 2.12，统一50 token基准）:
    fp16:            显存 3.47 GB | 推理 53.2 tok/s | 精度无损
    fp16 + compile:  显存 3.47 GB | 推理 55.1 tok/s | 算子融合 +3.6%
    int8:            显存 2.30 GB | 推理  9.8 tok/s | 精度微损 (compile 不兼容)
    int4:            显存 1.75 GB | 推理 28.7 tok/s | 精度略降 (compile 不兼容，慢13%)

算子融合兼容性:
    - FP16: compile 有效 (+3.6%)，合并 LayerNorm→Linear→GELU 等连续算子
    - INT4/INT8: compile 无效 (-13%)，bitsandbytes CUDA kernel 绕过原生算子
    - 推荐: INT4 不加 compile（边缘设备优先省显存，1.75 GB vs 3.47 GB）
"""

import os
import sys
import socket
import hashlib

from release_contract import apply_release_profile_to_env, release_profile_enforced

def build_settings(environment=None, *, probe_device=False) -> dict:
    """Resolve compatibility defaults from one supplied environment, without IO.

    Device probing is explicit; application startup normally delegates it to the
    profiler. No node configuration, key, model or journal is opened here.
    """
    environment = dict(os.environ if environment is None else environment)
    apply_release_profile_to_env(environment)
    # Import is a read-only compatibility view. Explicit SettingsStore startup
    # owns .env, identity migration, encrypted credentials and release overrides.
    _NODE_CONFIG = {}
    resolve_initial_node_role = None


    def _get_app_root() -> str:
        """
        获取应用根目录（模型、日志等文件的基准路径）。

        PyInstaller 打包: exe 所在目录（models/ 与 exe 同级）
        开发模式:        src/../ 即项目根目录
        """
        if getattr(sys, 'frozen', False):
            return os.path.dirname(os.path.abspath(sys.executable))
        else:
            # 开发模式：此文件在 src/ 下 → 项目根 = src/../
            return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    _APP_ROOT = _get_app_root()


    def _get_state_dir() -> str:
        """Return a user-writable state directory, never the install directory."""
        configured = environment.get("QLH_STATE_DIR", "").strip()
        if configured:
            return os.path.abspath(os.path.expanduser(configured))
        if os.name == "nt":
            base = environment.get("LOCALAPPDATA", "").strip()
            if not base:
                base = os.path.join(os.path.expanduser("~"), "AppData", "Local")
            return os.path.join(base, "QLH-Edge-Inference", "state")
        base = environment.get("XDG_STATE_HOME", "").strip()
        if not base:
            base = os.path.join(os.path.expanduser("~"), ".local", "state")
        return os.path.join(base, "qlh-edge-inference")


    STATE_DIR = _get_state_dir()


    def _env_int(name: str, default: int, min_val: int = 1, max_val: int = 65535) -> int:
        raw = environment.get(name)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        if not (min_val <= value <= max_val):
            return default
        return value


    def _env_first(*names: str, default: str = "") -> str:
        for name in names:
            value = environment.get(name)
            if value is not None and value.strip():
                return value.strip()
        return default


    def _env_bool(name: str, default: bool = False) -> bool:
        raw = environment.get(name)
        if raw is None or str(raw).strip() == "":
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}


    def _env_float(name: str, default: float, min_val: float = 0.0,
                   max_val: float = 1e9) -> float:
        raw = environment.get(name)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        if not (min_val <= value <= max_val):
            return default
        return value


    # The inference service owns this directory; HTTP callers may only opt into
    # the cold tier, never choose its filesystem location.
    KV_COLD_CACHE_ROOT = os.path.abspath(os.path.expanduser(
        _env_first("QLH_KV_COLD_CACHE_ROOT", default=os.path.join(STATE_DIR, "kv-cold")),
    ))
    KV_COLD_MAX_BYTES = _env_int(
        "QLH_KV_COLD_MAX_BYTES", 4 * 1024 * 1024 * 1024,
        min_val=1, max_val=1024 * 1024 * 1024 * 1024,
    )


    def _normalize_node_role(value: str) -> str:
        role = (value or "master").strip().lower()
        if role == "auto":
            return "auto"
        if role in {"slave", "worker", "client"}:
            return "client"
        return "master"


    # ============================================================
    # 1. 网络配置
    # ============================================================
    SERVER_IP = _env_first("QLH_SERVER_IP", default="0.0.0.0")  # 主节点监听IP
    SERVER_PORT = _env_int("QLH_SERVER_PORT", 8888)              # TCP 主节点监听端口
    API_PORT = _env_int("QLH_API_PORT", 8000)                    # FastAPI HTTP 端口
    HEARTBEAT_INTERVAL = 3          # 心跳包间隔（秒）
    RECONNECT_MAX_RETRIES = 5       # 断线重连最大尝试次数
    RECONNECT_DELAY = 2             # 重连间隔（秒）

    # ============================================================
    # 2. 模型配置（绝对路径，兼容 PyInstaller 打包 + 开发模式）
    # ============================================================
    # ★ 2026-09-19：以下三项原指向**已退役**的 `qwen-1_8b`（导致「自动加载默认模型」
    #   实际加载的就是 1.8B）。现改为指向**兜底默认模型** `qwen3-0.6b`；
    #   **真正的默认模型按设备画像选择**，见文件末尾的 `get_active_model_paths()`
    #   （用法：`_auto_load_default_model` 等消费点应改用它）。
    #   ⚠️ 这里保持**静态常量**而非顶层调用画像：`device_profiler` 会 import 本模块，
    #   顶层触发画像会形成循环 import；消费点调用时 config 已初始化完毕，才安全。
    MODEL_NAME = "Qwen/Qwen3-0.6B"               # HuggingFace 模型标识（兜底默认）
    MODEL_PATH = os.path.join(_APP_ROOT, "models", "qwen3-0.6b")            # Safetensors 格式
    GGUF_MODEL_PATH = os.path.join(_APP_ROOT, "models", "qwen3-0.6b-q8_0.gguf")  # GGUF 格式
    QUANT_TYPE = "int4"                          # 量化精度: "fp16" | "int8" | "int4"
    USE_COMPILE = True                           # 算子融合（仅 FP16+CUDA 有效；INT4/CPU 自动跳过）

    #: A4：让 `forward_layers()` 的**层循环**走 torch.compile 编译版。
    #: 前置（embedding / cache / mask / RoPE）与后置（norm / lm_head）逻辑**保持不变** ⇒ 与逐层版语义一致。
    #: 默认 False ⇒ 既有路径（含分布式层流水线）行为完全不变。
    #: ⚠️ 不要改成「编译整个 Qwen2Model 再用于分段」：它的 forward 会 apply `self.norm`，与分段语义不符
    #:    （实测 B vs A 的逐 token argmax 从 decode 第 1 步就分叉）。
    USE_MONOLITHIC_FORWARD = False

    #: ★ R-R7 rank3（2026-09-24 实测）：把各层 `RMSNorm` 的
    #:   `.to(f32) → pow(2) → mean → rsqrt → weight* → .to(f16)` **六个算子**换成**一次 `F.rms_norm`**。
    #: 依据（`build/cross-framework-layer-poc/rank_ablation.py`，qwen2.5-0.5b / 12 层 / gen=40 / CUDA-f16）：
    #:   `ms/step` **13.389 → 12.358（−7.7 %）**；cProfile 计数 `Tensor.to` **2173 → 205（−91 %）**、
    #:   `Tensor.pow`/`mean`/`rsqrt` 各 984 → **0**。
    #: ⚠️ **收益主要来自「顺带消掉那 1968 次 `.to`」**，而不是融合 kernel 本身 —— 该路径是
    #:   CPU/launch-bound（cProfile 真算力仅 31 %），`.to` 本应是 no-op 却仍走 dispatch。
    #: **判据**：带 `lm_head` 的真 logits 下 **per-token argmax 完全一致**（含与「RoPE 免 cat」叠加）。
    #: 默认 **False** ⇒ 既有路径行为完全不变；置 True 或 `QLH_USE_FUSED_RMSNORM=1` 启用。
    #: ⚠️ 只按**实例**替换（不 patch transformers 的类）⇒ 不波及其他模型/用途。
    USE_FUSED_RMSNORM = _env_bool("QLH_USE_FUSED_RMSNORM", default=False)
    #: torch.compile 的序列长度上限（2026-09-18 实测）：compile 在短序列有收益，但随生成步数
    #: 变长而劣化 —— gen=24 1.904× → 64 1.372× → **141 步 0.672×（反而慢 1.49×）**，
    #: 根因是 KV 增长使形状反复变化、`torch._dynamo` 的 `recompile_limit` 被 hybrid KV 的
    #: 不稳定守卫打满。MAX_SEQ_LEN 超过本值时跳过 compile（拐点在 64–141 之间，故保守取 128）。
    COMPILE_RECOMPILE_LIMIT = 64   # torch._dynamo.config.recompile_limit / cache_size_limit
    # 2026-09-18 实测（Qwen2.5-0.5B 非 hybrid，gen=141 decode）：
    #   compile 相对 eager 快 1.79×（limit=8 默认）→ **2.57×（limit=64）** ⇒ 调大 limit 有 +44% 收益。
    # transformers 自己在 chunked prefill 场景也用 64（generation/utils.py:4113-4115）。
    # ⚠️ torch >= 2.12 起该配置是 **thread-local**：须在使用线程内设置才生效。

    # ★ 规模门（2026-09-19 用户裁定）：编译的收益只在 **≥1.5B** 上稳定为正。
    # 依据：D→L 端到端实测里 0.5B/12 层开编译层循环**反而慢约 6×**（上游 84.2 vs 13.9 ms/step），
    #       而 2B/24 层同一开关**快 2.698×**（逐 token 一致）。
    # 1B 以内的小模型即便真能优化也会碰到边际效应 ⇒ 低于该参数量阈值时自动跳过 compile。
    COMPILE_MIN_PARAMS = int(1.5e9)  # 1.5B
    # 注：曾短暂存在过 COMPILE_MAX_SEQ_LEN（按序列长度跳过），依据是「长序列 compile 反而慢」，
    #     但那个结论来自取错模块的错误测量，已撤销（正确答案是长序列也快，且调大 limit 更有效）。
    # 2026-09-18 改为默认 True：装了 triton-windows 后 Windows 下才真正可用
    #（此前 PyPI 无 Windows triton ⇒ 报 TritonMissing，该开关等于是死的）。
    # 实测：mode='default' 使 12 层分段前向 1.0443 → 0.3258 ms/层，且 24 步 argmax 与 eager 一致。
    # ⚠️ compile **不是 bit-exact**（hidden max|diff|≈7.8e-2）；有严格「逐 token 一致」判据的
    #    验收（如跨框架接力）必须自行复验。triton 缺失时会自动告警并回退，不影响启动。
    INFERENCE_ENGINE = _env_first("QLH_INFERENCE_ENGINE", default="llama_cpp").strip().lower()

    # The default edge path is llama.cpp and must not import PyTorch merely to
    # discover that CUDA is unavailable. Explicit PyTorch/auto mode retains the
    # device probe for the full-model runtime.
    try:
        if not probe_device or INFERENCE_ENGINE in {"llama_cpp", "gguf", "island"}:
            DEVICE = "cpu"
        else:
            import torch as _torch
            DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
    except ImportError:
        DEVICE = "cpu"  # 推理设备: torch 未安装时默认 CPU

    # --- 推理引擎选择 ---
    # "auto": 自动选择 — CUDA 可用 → PyTorch + bitsandbytes, 否则 → llama.cpp + GGUF
    # "pytorch": 强制 PyTorch + Transformers（需要 CUDA 或大内存 CPU）
    # "llama_cpp": 强制 llama.cpp + GGUF（推荐 CPU / 集显设备）
    # GGUF 量化文件推荐 (RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf on HuggingFace):
    #   Q4_K_M (1.16 GB) — 推荐，速度/质量最佳平衡
    #   Q5_K_M (1.31 GB) — 更高质量
    #   Q8_0   (1.82 GB) — 近无损

    TRUST_REMOTE_CODE = True                     # Qwen 模型需要自定义代码

    # --- TP 孤岛接入（路线 A PoC，详见 docs/TP孤岛接入指南.md）---
    # 网关节点将同构 GPU 子集群（vLLM/SGLang/llama.cpp rpc-server 张量并行孤岛）
    # 的 OpenAI 兼容端点封装为 "island" 引擎，对集群呈现为单个逻辑高算力节点。
    # 孤岛节点承担整请求推理，不参与 PyTorch 层拆分（行为与 llama_cpp 节点一致）。
    ISLAND_ENABLED = _env_bool("QLH_ISLAND_ENABLED", False)     # 孤岛引擎总开关
    ISLAND_BASE_URL = _env_first("QLH_ISLAND_BASE_URL", default="")  # 如 http://10.0.0.2:8000
    ISLAND_API_KEY = environment.get("QLH_ISLAND_API_KEY", "")   # 可选 Bearer 凭据（日志脱敏）
    ISLAND_MODEL = _env_first("QLH_ISLAND_MODEL", default="")   # 为空时自动取 /v1/models 首个模型
    ISLAND_TIMEOUT = _env_int("QLH_ISLAND_TIMEOUT", 120, min_val=1, max_val=3600)  # 请求超时（秒）
    ISLAND_CONNECT_TIMEOUT = _env_int("QLH_ISLAND_CONNECT_TIMEOUT", 5, min_val=1, max_val=300)  # 连接超时（秒）
    # 聚合能力画像（网关无法探测孤岛内部硬件，由部署者显式声明；
    # 用于设备画像 island 段与主节点排序加权，防止孤岛节点被低估）
    ISLAND_GPU_COUNT = _env_int("QLH_ISLAND_GPU_COUNT", 1, min_val=1, max_val=64)   # 孤岛 GPU 总数
    ISLAND_VRAM_GB = _env_float("QLH_ISLAND_VRAM_GB", 0.0, min_val=0.0, max_val=8192.0)  # 聚合显存（GB）
    ISLAND_TP_SIZE = _env_int("QLH_ISLAND_TP_SIZE", 1, min_val=1, max_val=64)       # 张量并行度
    ISLAND_BACKEND = _env_first("QLH_ISLAND_BACKEND", default="openai-compatible")  # 后端标签，如 "vllm-tp2"


    def _normalize_external_data_scope(value: str) -> str:
        """数据作用域档位归一化。

        未设置（空值）→ 默认 opt_in；设置了但取值非法（如误写 "denied"/"off"）
        → 一律回落 deny 而不是 opt_in。作用域是安全边界，配置写错时必须
        fail-closed：宁可挡住合法请求，也不能让本想禁用出网的配置反而放行。
        """
        scope = (value or "").strip().lower()
        if not scope:
            return "opt_in"
        if scope in {"deny", "opt_in", "allow_all"}:
            return scope
        # 非法取值：打印到 stderr（config 在 logging 初始化前导入，不能用 logger）
        sys.stderr.write(
            "[QLH][WARN] QLH_EXTERNAL_DATA_SCOPE 取值非法: "
            + repr(value)
            + "，已回落为 deny（禁止一切外部路由）。合法取值: deny / opt_in / allow_all\n"
        )
        return "deny"


    # --- 外部推理服务接入（路线 B PoC，详见 docs/外部推理服务Provider接入指南.md）---
    # 集群把"整条请求"作为路由决策交给一个外部 OpenAI 兼容端点（租用 GPU 盒子、
    # 实验室服务器、云 API 等），不要求对方运行任何 QLH 代码，也无专职网关节点。
    # 与 TP 孤岛（路线 A，集群内的逻辑节点）不同：外部端点在集群信任域之外，
    # 因此数据作用域门控是硬约束——默认 opt_in，未显式授权的请求绝不离开集群。
    EXTERNAL_ENABLED = _env_bool("QLH_EXTERNAL_ENABLED", False)      # 外部推理服务总开关
    EXTERNAL_BASE_URL = _env_first("QLH_EXTERNAL_BASE_URL", default="")  # 如 https://gpu-box.example.com:8000
    EXTERNAL_API_KEY = environment.get("QLH_EXTERNAL_API_KEY", "")    # 可选 Bearer 凭据（日志脱敏）
    EXTERNAL_MODEL = _env_first("QLH_EXTERNAL_MODEL", default="")    # 为空时自动取 /v1/models 首个模型
    EXTERNAL_TIMEOUT = _env_int("QLH_EXTERNAL_TIMEOUT", 120, min_val=1, max_val=3600)  # 请求超时（秒）
    EXTERNAL_CONNECT_TIMEOUT = _env_int("QLH_EXTERNAL_CONNECT_TIMEOUT", 5, min_val=1, max_val=300)  # 连接超时（秒）
    # 可选 OpenAI reasoning_effort；留空时不发送，避免不支持该字段的端点报错。
    # Gemma 4 经 Ollama 使用 "none" 可避免 thinking 消耗全部短回复预算。
    EXTERNAL_REASONING_EFFORT = _env_first(
        "QLH_EXTERNAL_REASONING_EFFORT", default="",
    ).lower()
    if EXTERNAL_REASONING_EFFORT not in {"", "none", "minimal", "low", "medium", "high"}:
        EXTERNAL_REASONING_EFFORT = ""
    # 数据作用域门控（安全边界，不是性能开关，见调研方案 §2.2）:
    #   "deny"      硬禁用——即使请求携带 allow_external 也拒绝出集群
    #   "opt_in"    默认——仅携带 allow_external=true 的请求可路由到外部端点
    #   "allow_all" 全部请求均可外部路由（仅限自有/可信端点，慎用）
    EXTERNAL_DATA_SCOPE = _normalize_external_data_scope(
        environment.get("QLH_EXTERNAL_DATA_SCOPE", "opt_in")
    )
    # 长上下文卸载启发式：作用域放行后，提示词字符数 ≥ 此阈值的请求优先走外部
    # （0 = 关闭，仅 prefer_external 显式触发）
    EXTERNAL_MIN_PROMPT_CHARS = _env_int(
        "QLH_EXTERNAL_MIN_PROMPT_CHARS", 0, min_val=0, max_val=1000000,
    )
    EXTERNAL_LABEL = _env_first("QLH_EXTERNAL_LABEL", default="外部推理服务")  # 展示名

    # --- n-gram 自投机（绑定路径 / 无 CUDA 边缘）---
    # ★ 2026-10-13 实测（Qwen3.6-35B-A3B-UD-Q4_K_XL，`.venv-test` 绑定）：
    #   `draft_model=LlamaPromptLookupDecoding` 在**绑定路径**上有效 —— 重复文本
    #   1.30～1.40×，自由文本不劣化（不命中时也不掉速）。它与上面 `SPEC_*`
    #   「外部 verify 模型」那套是**两条不同的路**：这里不引入任何额外模型、
    #   不占额外带宽与显存，纯粹在 token 历史里做 n-gram 匹配。
    # ⚠️ 代价：绑定一旦收到 `draft_model`，`llama_cpp/llama.py` 会把 `logits_all`
    #   强制为 True（每个位置都算 logits，内存/显存开销上升）⇒ 因此**默认关闭**，
    #   按场景显式启用；只在文本重复场景（结构化输出 / 代码 / 长文回述）有收益。
    # 对应裁决：有 CUDA 走 server 的 `--spec-type`，无 CUDA 才走这里。
    NGRAM_SPEC_ENABLED = _env_bool("QLH_NGRAM_SPEC_ENABLED", False)
    NGRAM_SPEC_MAX_NGRAM = _env_int("QLH_NGRAM_SPEC_MAX_NGRAM", 2, min_val=1, max_val=16)
    NGRAM_SPEC_NUM_PRED = _env_int("QLH_NGRAM_SPEC_NUM_PRED", 10, min_val=1, max_val=64)

    # --- 本机 llama-server 子进程路径（票 15a/15b）---
    # 裁决（2026-10-12）：有 CUDA 走 server 子进程（拿得到 --spec-type 投机族、
    # `--cpu-moe`、`--cache-reuse`），无 CUDA 老实走绑定。留空时由
    # `llama_server_engine.default_server_exe()` 在 `runtime/llama-cpp/<build>/` 下自动挑选。
    LLAMA_SERVER_EXE = _env_first("QLH_LLAMA_SERVER_EXE", default="")

    # --- 整模推理路径选择（票 15a/15b 接线）---
    # 取值 = `src/inference_path_selector.py` 的 AUTO / SERVER / BINDING。
    # **默认 auto**，且它与 `select_backend` 的 auto 是**正交**的两层：
    #   * `select_backend` 的 auto 选的是**引擎档位**（pytorch / llama_cpp / island）；
    #   * 本项只在 **llama.cpp 家族内**挑传输 —— 有 CUDA 走 server 子进程（绑定是
    #     CPU-only，拿不到 GPU），无 CUDA 老实走绑定。
    # 因此 auto 不会顶掉 D 档：走 `_load_llama_cpp` 才轮到本项。server 可执行文件
    # 缺失时 auto **可观测地**退回绑定（reason 写明原因）；**显式**请求 server 则
    # 永远 fail-closed，不降级。
    INFER_PATH = _env_first("QLH_INFER_PATH", default="auto")

    # --- 多模型实验支持 (P3) ---
    # ★ 2026-09-19：`qwen-1_8b` 已退役并从内置列表移除 ⇒ 活跃模型 ID 同步为默认模型
    #   （须与 `model_config.DEFAULT_MODEL_ID` 保持一致）。
    ACTIVE_MODEL_ID = "qwen3-0.6b"           # 当前活跃的模型 ID
    EXPERIMENTAL_MODELS_ENABLED = False      # 运行时由 CUDA 检测设置（torch.cuda.is_available()）

    # ============================================================
    # 3. 分页KV缓存配置
    # ============================================================
    PAGE_SIZE = 128                  # 单页容纳 Token 数量
    MAX_PAGE_NUM = 256               # 最大内存页数
    MAX_SEQ_LEN = 4096               # 最大序列长度（Qwen-1.8B 原生 8K，laptop 档用 4K）

    # ============================================================
    # 4. 模型分层配置（动态分配，运行时由 scheduler 根据设备画像计算）
    #    Qwen-1.8B-Chat 共 24 层 Transformer
    # ============================================================
    TOTAL_MODEL_LAYERS = 24                  # Qwen-1.8B-Chat Transformer 层总数
    LAYER_STRATEGY = "dynamic"              # 分层策略: "dynamic" 动态 | "manual" 手动覆盖
    DISTRIBUTED_INFERENCE_ENABLED = _env_bool("QLH_DISTRIBUTED_INFERENCE_ENABLED", True)    # 分布式推理开关（主节点默认开启，从节点默认关闭）
    TASK_WORKER_EXPERIMENTAL_ENABLED = _env_bool(
        "QLH_TASK_WORKER_EXPERIMENTAL_ENABLED", False,
    )

    # ------------------------------------------------------- journal 复制（R-R11，默认全关）
    # `src/journal_replication_service.py` 是这套配置的**真实消费者**。注意该模块的
    # `from_env()` 刻意直接读**同一批 env**（config 常量在 import 时求值，测试里改 env 不会生效）；
    # 两处的 env 名**必须保持同名**，此处登记是为了让配置集中可见、并让配置清单类检查扫得到。
    JOURNAL_REPLICATION_ENABLED = _env_bool("QLH_JOURNAL_REPLICATION_ENABLED", False)
    JOURNAL_REPLICATION_SOURCE_NODE_ID = _env_first(
        "QLH_JOURNAL_REPLICATION_SOURCE_NODE_ID", "QLH_NODE_ID", default="",
    )
    JOURNAL_REPLICATION_STREAM_ID = _env_first(
        "QLH_JOURNAL_REPLICATION_STREAM_ID", default="task-graph-journal",
    )
    JOURNAL_REPLICATION_REPLICA_PATH = _env_first(
        "QLH_JOURNAL_REPLICATION_REPLICA_PATH", default="",
    )
    JOURNAL_REPLICATION_BIND_HOST = _env_first(
        "QLH_JOURNAL_REPLICATION_BIND_HOST", default="127.0.0.1",
    )
    JOURNAL_REPLICATION_BIND_PORT = _env_int(
        "QLH_JOURNAL_REPLICATION_BIND_PORT", 0, min_val=0, max_val=65535,
    )
    JOURNAL_REPLICATION_PEER_HOST = _env_first(
        "QLH_JOURNAL_REPLICATION_PEER_HOST", default="",
    )
    JOURNAL_REPLICATION_PEER_PORT = _env_int(
        "QLH_JOURNAL_REPLICATION_PEER_PORT", 0, min_val=0, max_val=65535,
    )
    JOURNAL_REPLICATION_TIMEOUT_S = _env_float(
        "QLH_JOURNAL_REPLICATION_TIMEOUT_S", 5.0, min_val=0.1, max_val=600.0,
    )

    # 回退分层配置（当没有从节点注册时使用）
    # 注意：分层现在由 compute_layer_assignment() 动态计算，此配置仅供文档参考。
    # 实际部署中不依赖硬编码的 client1/client2 槽位。
    DEFAULT_LAYER_CONFIG = {}

    # 显存约束：单层 Transformer 最低显存 + 量化修正系数
    # Qwen-1.8B 每层 ~70MB (FP16)，Embedding + LM Head 各 ~580MB
    MIN_VRAM_PER_LAYER_MB = 70               # 单层最低显存（FP16 基准）
    EMBEDDING_VRAM_MB = 580                  # Token Embedding 层
    LM_HEAD_VRAM_MB = 580                    # LM Head 输出层
    SAFE_VRAM_MARGIN = 1.1                   # 10% 安全余量
    LAYER_VRAM_FACTOR = {                    # 量化精度修正系数
        "fp16": 1.0,
        "int8": 0.55,
        "int4": 0.35,
    }

    # Metadata-first pipeline capacity admission.  The fixed reserve is charged to
    # every execution node for allocator workspace, activations and KV cache.  It is
    # intentionally independent from the legacy average-layer VRAM estimator.
    PIPELINE_CAPACITY_SAFETY_MARGIN = _env_float(
        "QLH_PIPELINE_CAPACITY_SAFETY_MARGIN", 1.2, min_val=1.0, max_val=3.0,
    )
    PIPELINE_CAPACITY_RESERVE_MB = _env_float(
        "QLH_PIPELINE_CAPACITY_RESERVE_MB", 512.0, min_val=0.0, max_val=32768.0,
    )
    # ★ 2026-10-23（#117）：**主节点主机内存的使用上限**。`safetensors_layer_offload`
    #   让 master 用主机内存承载层段，而主节点同时还要跑别的程序（编辑器、浏览器…），
    #   所以不能把"当前可用内存"整块当作可分配容量 —— 此前 `host_capacity_bytes`
    #   直接等于可用内存（实测一次供给就吃掉 ~14 GB / 31.6 GB）。
    #   约定：K-Llama 最多使用**物理总内存的这个比例**，其余留给系统与其他程序。
    #   上限取 `min(当前可用, 物理总内存 × 该比例)`。
    HOST_RAM_MAX_FRACTION = _env_float(
        "QLH_HOST_RAM_MAX_FRACTION", 0.6, min_val=0.05, max_val=0.95,
    )
    # ★ 2026-10-03：容量求解器的默认准则是「节点数最少」（省网络开销，对普通请求合理），
    #   因此多段拓扑（如 master + 中间段 + 尾段）永远不会被选中 —— 两段总是更少。
    #   打开这个开关后，可行解之间改为**优先用尽可用节点**。验证多段链路时需要它。
    # ★ 2026-10-24（#117）：**容量求解器模式**。此前"节点数最少 vs 用尽节点""是否用
    #   主机内存承载"是散落的布尔开关，语义不成体系。收敛成一个模式：
    #     balanced（默认）：资源优先 —— 允许 `safetensors_layer_offload` 用主机内存
    #                        承载，节点数最少（省网络往返），适合常驻/均衡负载。
    #     fastest          ：速度优先 —— 只用显存（resident，避免每步跨 PCIe 从主机
    #                        内存搬运权重），优先用尽节点并偏向算力最强的节点。
    #   实测依据：`balanced` 下 master 以 offload 扛 28 层时每 token 约 5 秒（master
    #   成了串行瓶颈）；17-18 点三机很快的那次是 resident（master 只占 8 层）。
    CAPACITY_PLANNING_MODE = _env_first(
        "QLH_CAPACITY_PLANNING_MODE", default="balanced",
    ).strip().lower()
    if CAPACITY_PLANNING_MODE not in {"balanced", "fastest"}:
        CAPACITY_PLANNING_MODE = "balanced"
    #: 主节点是否允许用主机内存承载层段（`fastest` 下关闭，换取每步全显存前向）。
    HOST_RAM_OFFLOAD_ENABLED = _env_bool(
        "QLH_HOST_RAM_OFFLOAD_ENABLED", CAPACITY_PLANNING_MODE != "fastest",
    )
    PIPELINE_PREFER_ALL_WORKERS = _env_bool(
        "QLH_PIPELINE_PREFER_ALL_WORKERS", CAPACITY_PLANNING_MODE == "fastest",
    )

    # ★ 2026-10-24（#117）：**清点阶段的等待上限**（秒）。此前 `model_supply.py` 里
    #   硬编码 `min(90, ...)`，而"未登记候选"的本地源核对需要对整模做一次全量
    #   sha256（5.78 GB，实测 90~120 s）⇒ 首次清点必然 `SUPPLY_WORKER_TIMEOUT`。
    #   做成设置，默认放宽到 300 秒；缓存命中后后续清点只需毫秒级。
    MODEL_SUPPLY_INVENTORY_TIMEOUT_S = _env_float(
        "QLH_MODEL_SUPPLY_INVENTORY_TIMEOUT_S", 300.0, min_val=10.0, max_val=3600.0,
    )

    def capacity_planning_mode() -> str:
        """★ 2026-10-24（#117）：**运行时**读取求解器模式。

        `node_config.json` 的 features 是在 `config` 模块**之后**才写进 env 的
        （`api_server.py:70` 先 `import config`，`:195` 才 `load_node_config`），所以
        模块级的 `CAPACITY_PLANNING_MODE` 看不到节点级设置。这里按需重读 env。

        注意：本函数与各设置同处 `build_settings()` 内，文件末尾才
        `globals().update(build_settings())` 把它们投影成模块属性。因此这里必须经
        `globals()` 读**模块值**，否则读到的是函数局部旧值，`monkeypatch.setattr(config,
        ...)` 之类的覆盖不会生效。
        """
        mode = (os.environ.get("QLH_CAPACITY_PLANNING_MODE") or "").strip().lower()
        if mode in {"balanced", "fastest"}:
            return mode
        return str(globals().get("CAPACITY_PLANNING_MODE", CAPACITY_PLANNING_MODE)).strip().lower()

    def host_ram_offload_enabled() -> bool:
        """主节点是否允许用主机内存承载层段（`fastest` 下关闭，速度优先）。

        优先级：**显式 env > 运行时模式**。

        注意这里**不能**回落到模块值 `HOST_RAM_OFFLOAD_ENABLED`：它在 config 加载时
        就按当时的 `CAPACITY_PLANNING_MODE` 固化，而节点级 mode 是之后才写进 env 的
        ⇒ 模块值会一直停在旧结论（实测 env mode 已是 `fastest`，offload 仍返回 True）。
        """
        explicit = os.environ.get("QLH_HOST_RAM_OFFLOAD_ENABLED")
        if explicit is not None and explicit.strip():
            return explicit.strip().lower() in {"1", "true", "yes", "on"}
        return capacity_planning_mode() != "fastest"

    def prefer_all_workers_enabled() -> bool:
        """可行解之间是否优先用尽可用节点（`fastest` 下开启）。

        优先级同 `host_ram_offload_enabled()`：显式 env > 运行时模式（理由相同 ——
        模块值会停在 config 加载时的旧结论）。测试通过 patch 模块值来驱动时，走的是
        `PIPELINE_PREFER_ALL_WORKERS` 的读取点（`plan_segment_supply` 显式传参），
        这里保持"模式优先"即可。
        """
        explicit = os.environ.get("QLH_PIPELINE_PREFER_ALL_WORKERS")
        if explicit is not None and explicit.strip():
            return explicit.strip().lower() in {"1", "true", "yes", "on"}
        module_value = globals().get("PIPELINE_PREFER_ALL_WORKERS")
        if module_value is True and capacity_planning_mode() == "balanced":
            # 显式开关被打开（含测试 patch 与既有配置），尊重它。
            return True
        return capacity_planning_mode() == "fastest"
    # Evictable supply inventories; live owners and leases remain protected.
    MODEL_SUPPLY_CACHE_KEEP = _env_int("QLH_MODEL_SUPPLY_CACHE_KEEP", 2, min_val=0, max_val=128)
    MODEL_SUPPLY_CACHE_MASTER_KEEP = _env_int("QLH_MODEL_SUPPLY_CACHE_MASTER_KEEP", 6, min_val=0, max_val=128)
    MODEL_SUPPLY_CACHE_MAX_BYTES = _env_int(
        "QLH_MODEL_SUPPLY_CACHE_MAX_BYTES", 8 * 1024**3, min_val=0, max_val=1024**4,
    )
    PIPELINE_ASSIGNMENT_CACHE_MAX_MB = _env_float(
        "QLH_PIPELINE_ASSIGNMENT_CACHE_MAX_MB", 4096.0, min_val=128.0, max_val=1048576.0,
    )
    PIPELINE_ASSIGNMENT_MIN_FREE_MB = _env_float(
        "QLH_PIPELINE_ASSIGNMENT_MIN_FREE_MB", 512.0, min_val=0.0, max_val=1048576.0,
    )
    PIPELINE_ASSIGNMENT_STALE_SECONDS = _env_float(
        "QLH_PIPELINE_ASSIGNMENT_STALE_SECONDS", 86400.0, min_val=60.0, max_val=31536000.0,
    )

    # ★ A1 / X 档（2026-09-24）：**Relay 段委托总开关，默认关**。
    # 关闭时调度层完全走既有路径（pytorch-only 层流水线），Relay 调用次数为 0。
    # 开启后也只对"在 LAYER_FORWARD 里显式携带合法 `relay_segment` 规格（middle 角色）"
    # 的节点生效 —— 见 `scheduler_pipeline._normalize_relay_segment` 与
    # `_handle_layer_forward_via_relay`。跨机通道仍只允许 loopback / 本地 SSH 隧道端点。
    PIPELINE_RELAY_ENABLED = _env_bool("QLH_RELAY_ENABLED", False)

    # ★ A1 / X 档（2026-09-24）：**哪些节点由远端 relay 段代跑本段**（主节点侧配置）。
    # 格式：`<node_id>=<role>@<host>:<port>#<n_embd>`，多条用 `;` 或 `,` 分隔，例如
    #   QLH_RELAY_SEGMENTS="worker-2=middle@127.0.0.1:50183#896;worker-3=middle@127.0.0.1:50184#896"
    # 只对 `QLH_RELAY_ENABLED=1` 生效；端点必须是 loopback（跨机走本地 SSH 隧道）。
    # 默认空 ⇒ 不下发任何 `relay_segment`，行为与接线前**完全一致**。不合法的条目整条丢弃。
    PIPELINE_RELAY_SEGMENTS = _env_first("QLH_RELAY_SEGMENTS", default="")

    # ★ 2026-10-04 产品裁定（分票规划 DIST-0）：**A1 relay 从产品调度入口剔除**。
    #   A1 依赖 probe/SSH 隧道与 loopback 段服务，只保留为技术探针与历史证据，
    #   不再是生产候选、容量来源或可用性证明（产品主线是 A3 `stage_offer_v3`）。
    #   本开关默认 **1（探针专用）**：置 1 时上面的 `QLH_RELAY_ENABLED` /
    #   `QLH_RELAY_SEGMENTS` 不再作为生产能力（`_relay_segment_for_worker` 直接
    #   不供给 relay 段），生产请求继续走 A3，只记一份具名诊断、不做静默切换。
    #   探针/实验需要恢复 A1 行为时显式设 `QLH_RELAY_PROBE_ONLY=0`。
    PIPELINE_RELAY_PROBE_ONLY = _env_bool("QLH_RELAY_PROBE_ONLY", True)

    # ★ 2026-10-05（DIST-4「发布闸门」）：**A3（Route A / `stage_offer_v3`）的独立开关**。
    #
    # 此前 A3 **没有开关** —— 路径由 assignment 是否被打上 `execution="stage_offer_v3"`
    # 自动决定（打标见 `scheduler_pipeline._push_layer_config_to_clients_locked`），
    # 而总闸 `DISTRIBUTED_INFERENCE_ENABLED` 默认开 ⇒ 现状是「实验线路默认关，但生产
    # 主线默认开且发布脚本无法区分」。
    #
    #   - `1`（默认）：A3 可用 —— **保持现状行为不变**；
    #   - `0`：不给任何 assignment 打 `stage_offer_v3` 标记 ⇒ 层段链整体不参与
    #     （`stage_offer_nodes` 为空，请求按既有 legacy/回退路径处理）。
    #
    # 默认取 `1` 是**刻意的**：本开关的用途是让**发布脚本/部署侧可以显式关掉实验
    # 线路**（`DIST-4` 的「只允许显式开启 A3」），而不是改变当前已验收通过的行为。
    PIPELINE_ROUTE_A_STAGE_OFFER_ENABLED = _env_bool(
        "QLH_ROUTE_A_STAGE_OFFER", True,
    )

    # ★ 2026-10-07：强制本机（master）走 PyTorch 上游。
    #   动机：qwen3.5 的 keep-head 上游挂点在 `output_norm` **之后**（多一次 RMSNorm，
    #   实测会让接力首步分叉）⇒ `LlamaCppEngine` 上游被 fail-closed 拒绝
    #   （docs/已知问题记录.md #35）。而 `_auto_load_default_model()` 里 GGUF **永远
    #   优先**（api_server.py 的 `if gguf_candidates:`），画像模型又自带 gguf ⇒ 无法
    #   自动落到 PyTorch。置 1 且画像模型有 safetensors 目录时，优先选该目录
    #   （`engine=pytorch`），让本机用 `model_module.forward_layers` 出 hidden。
    #   默认 0：不改变既有行为（llama 优先）。
    PREFER_PYTORCH = _env_bool("QLH_PREFER_PYTORCH", False)

    # 流水线推理超时与并发控制
    #: ★ 2026-10-11（ROG+Y700+本机三机实测）：此前是硬编码 `120`，而三机层接力每步约
    #:   0.2s ⇒ 长思考请求（`show_thinking=true` + 大 `max_new_tokens`）跑到 500+ 步就撞上，
    #:   表现为 `request_deadline_exceeded`，现场只能改源码。现在可用 `QLH_PIPELINE_TIMEOUT`
    #:   调整（默认仍是 120，既有行为不变）。
    PIPELINE_TIMEOUT = _env_int("QLH_PIPELINE_TIMEOUT", 120)   # 请求端到端预算（秒）；queue/prepare/step/fallback 共用
    PIPELINE_MODEL_SYNC_TIMEOUT = _env_float(
        "QLH_PIPELINE_MODEL_SYNC_TIMEOUT", 180.0, min_val=1.0, max_val=600.0,
    )                                           # 等待从节点同步模型和分层 ACK
    # ⚠️ **未接线**（全仓仅此一处定义、**无消费者**）：它只说明「D 档层流水线是串行的」，
    # 真正的串行由 `src/scheduler.py` 的 `_pipeline_lock` 保证。多请求交叠落地后再接线
    # （并行组 P1「多请求/多序列交叠」），在那之前**不要**把它当生效配置读。
    # 见 docs/未完成工作备忘-2026-09-23.md 的 A3。
    PIPELINE_MAX_CONCURRENT = 1              # 最大并发流水线任务数（当前仅支持 1，串行执行；**未接线**）
    # CPU-only workers may need tens of seconds for the first layer forward over
    # a DERP/Tailscale path.  Keep this configurable so deployments can tune it
    # to their slowest participating node without changing source code.
    PIPELINE_STEP_TIMEOUT = _env_float(
        "QLH_PIPELINE_STEP_TIMEOUT", 120.0, min_val=5.0, max_val=600.0,
    )                                           # 单个节点前向传播超时（秒）
    PIPELINE_QUEUE_MAX_SIZE = 100            # 请求队列最大容量（超出返回 503）
    PIPELINE_QUEUE_RESULT_TTL = 300          # 已完成任务结果保留时间（秒），超时清理
    PIPELINE_QUEUE_POLL_INTERVAL = 0.5       # 排队请求轮询间隔（秒）

    # ============================================================
    # 5. 节点身份配置
    # ============================================================
    if resolve_initial_node_role is not None:
        _initial_role = resolve_initial_node_role()
    else:
        # Keep a safe source-checkout fallback even when optional config imports
        # are unavailable during a minimal diagnostic import.
        _initial_role = environment.get(
            "QLH_NODE_ROLE", "auto" if getattr(sys, "frozen", False) else "client"
        )
    NODE_ROLE = _normalize_node_role(_initial_role)
    _default_node_id = "master" if NODE_ROLE == "master" else f"client_{socket.gethostname()}"
    NODE_ID = _env_first("QLH_NODE_ID", default=_default_node_id)
    _default_app_variant = (
        "cuda" if "cuda" in os.path.basename(_APP_ROOT).lower() else "cpu"
    )
    APP_VARIANT = _env_first("QLH_APP_VARIANT", default=_default_app_variant)
    LEGACY_RECOVERY_INSTANCE_ID = _env_first(
        "QLH_LEGACY_RECOVERY_INSTANCE_ID",
        default=f"{APP_VARIANT}-{NODE_ID}",
    )
    _task_graph_instance_segment = "".join(
        character
        if character.isascii()
        and (character.isalnum() or character in {"-", "_", "."})
        else "_"
        for character in LEGACY_RECOVERY_INSTANCE_ID
    ).strip("._")[:48] or "default"
    _task_graph_instance_digest = hashlib.sha256(
        LEGACY_RECOVERY_INSTANCE_ID.encode("utf-8")
    ).hexdigest()[:12]
    LEGACY_RECOVERY_JOURNAL_PATH = os.path.join(
        STATE_DIR,
        "task_graph",
        f"instance-{_task_graph_instance_segment}-{_task_graph_instance_digest}.sqlite3",
    )
    MAX_NODES = 3                    # 最大节点数上限（主节点可动态调整，仅限已注册节点，不含空位）
                                      # 从节点通过 TCP 注册后自动加入列表，不再预创建空槽位

    # 从节点连接主节点配置（仅 NODE_ROLE="client" 时生效）
    # 主节点启动后自动检测 Tailscale/ZeroTier 组网 IP 并写入共享数据库，
    # 从节点通过 discover_master() 自动发现即可，通常无需手动配置。
    # 仅当数据库不可用时才回退到此配置值。
    CLIENT_MASTER_HOST = _env_first("QLH_CLIENT_MASTER_HOST", "QLH_MASTER_HOST", default="")
    CLIENT_MASTER_PORT = _env_int("QLH_CLIENT_MASTER_PORT", _env_int("QLH_MASTER_PORT", 8888))

    # P3: 主节点转让审查配置（详见 src/review.py）
    REVIEW_TIMEOUT_HOURS = 48            # 审查工单超时时间（小时）
    REVIEW_APPROVE_THRESHOLD = 2         # 通过阈值: score >= +2
    REVIEW_REJECT_THRESHOLD = -2         # 阻止阈值: score <= -2

    # ============================================================
    # 6. 集群安全
    # ============================================================
    # 集群共享密钥 — 所有节点必须使用相同密钥才能加入集群。
    # node_config 启动阶段会让加密存储中的 (secret, epoch) 覆盖旧环境投影；
    # QLH_CLUSTER_SECRET 只保留首次导入/开发兼容语义，不能回滚已轮换凭据。
    CLUSTER_SECRET = environment.get("QLH_CLUSTER_SECRET", "")
    try:
        CLUSTER_SECRET_EPOCH = max(1, int(environment.get("QLH_CLUSTER_SECRET_EPOCH", "1") or 1))
    except ValueError:
        CLUSTER_SECRET_EPOCH = 1
    # HMAC 签名时间窗口（秒）— 防重放攻击
    AUTH_TIMESTAMP_WINDOW = 300  # ±5 分钟

    # ============================================================
    # 7. 运行模式
    # ============================================================
    # 运行模式：`QLH_RUN_MODE`（`single` / `distributed`）可覆盖，默认 `distributed` **不变**。
    # 用途：单机仿真/测试进程需要 `single`，否则 master 会进「强制分布式分层」并等从节点
    # 加载 ACK 直到 60 s 超时（实测 D4）。
    RUN_MODE = environment.get("QLH_RUN_MODE", "").strip().lower() or "distributed"
    if RUN_MODE not in ("single", "distributed"):
        RUN_MODE = "distributed"
    LOG_LEVEL = "INFO"               # 日志级别: DEBUG | INFO | WARNING | ERROR
    LOG_DIR = os.path.join(_APP_ROOT, "logs")  # 日志文件目录（绝对路径）

    # L5: 日志保留策略（双维度控制 — 天数 + 总空间）
    LOG_MAX_AGE_DAYS = 30                # 单个日志文件最大保留天数（0 = 不限）
    LOG_MAX_TOTAL_SIZE_MB = 200          # 日志目录总大小上限（MB，0 = 不限）
    LOG_RETENTION_CHECK_INTERVAL = 3600  # 清理检查间隔（秒），默认每小时检查一次

    # ============================================================
    # 8. 可视化配置（Streamlit 已移除，统一使用 FastAPI + React 前端）
    # ============================================================
    # 前端静态文件由 FastAPI 在端口 8000 直接提供服务（生产模式）
    # 开发模式：Vite dev server 在 5173 端口，通过 proxy 转发 /api 到 8000


    # ============================================================
    # 9. 自适应配置（由 device_profiler 运行时生成）
    # ============================================================

    def auto_config(profile: dict = None) -> dict:
        """
        根据设备画像生成自适应推理配置。

        传入 device_profiler.DeviceProfiler.to_dict() 返回的设备画像 dict，
        或传入 None 使用保守默认值（laptop 档）。

        返回的 dict 字段与上方硬编码常量一一对应，可直接用于覆盖。

        Example:
            from device_profiler import DeviceProfiler
            p = DeviceProfiler()
            cfg = auto_config(p.to_dict())
            QUANT_TYPE = cfg["quant_type"]
        """
        if profile is None:
            # 保守默认：laptop 档
            return {
                "quant_type": "int4",
                "page_size": 128,
                "max_page_num": 256,
                "max_seq_len": 4096,
                "max_new_tokens": 1024,
                "use_compile": False,
                "device": "cuda",
                "description": "默认配置 (laptop 档)，未检测到设备画像",
            }

        tier = profile.get("tier", "laptop")
        gpu = profile.get("gpu", {})
        ram = profile.get("ram", {})
        platform_info = profile.get("platform", {})

        has_cuda = gpu.get("cuda_available", False) if gpu else False
        vram_gb = gpu.get("vram_total_gb", 0) if gpu else 0
        ram_gb = ram.get("total_gb", 8) if ram else 8
        is_arm = (platform_info.get("machine", "") in ("aarch64", "armv7l", "arm64")
                  if platform_info else False)

        # 按档位返回配置
        if tier == "workstation":
            return {
                "quant_type": "fp16",
                "page_size": 128,
                "max_page_num": 512,
                "max_seq_len": 8192,      # 原生 8K 上下文
                "max_new_tokens": 2048,
                "use_compile": True,
                "device": "cuda" if has_cuda else "cpu",
                "description": "桌面工作站 — FP16 原版 + compile 融合",
            }
        elif tier == "laptop":
            return {
                "quant_type": "int4",
                "page_size": 128,
                "max_page_num": 256,
                "max_seq_len": 4096,      # 8GB VRAM 可容纳约 7K token KV 缓存
                "max_new_tokens": 1024,
                "use_compile": False,
                "device": "cuda" if has_cuda else "cpu",
                "description": "游戏本 / 独显本 — INT4 量化",
            }
        elif tier == "ultrabook":
            return {
                "quant_type": "int4",
                "page_size": 64,
                "max_page_num": 128,
                "max_seq_len": 2048,
                "max_new_tokens": 512,
                "use_compile": False,
                "device": "cuda" if has_cuda else "cpu",
                "description": "轻薄本 / 集显本 — INT4 + 缩减 KV 缓存",
            }
        elif tier == "edge":
            return {
                "quant_type": "int4",
                "page_size": 64,
                "max_page_num": 64,
                "max_seq_len": 1024,
                "max_new_tokens": 256,
                "use_compile": False,
                "device": "cpu",
                "description": "边缘设备 — CPU-only + 最小 KV 缓存",
            }
        elif tier == "mobile":
            return {
                "quant_type": "int4",
                "page_size": 32,
                "max_page_num": 32,
                "max_seq_len": 512,
                "max_new_tokens": 128,
                "use_compile": False,
                "device": "cpu",
                "description": "移动设备 — 极限压缩（建议导出 ONNX/GGUF）",
            }
        else:
            # 兜底
            return {
                "quant_type": "int4",
                "page_size": 128,
                "max_page_num": 256,
                "max_seq_len": 4096,
                "max_new_tokens": 1024,
                "use_compile": False,
                "device": "cuda" if has_cuda else "cpu",
                "description": "未知设备 — 保守配置",
            }


    def get_active_model_paths() -> dict:
        """★ 2026-09-19：按**设备画像**解析默认模型路径（供自动加载消费点使用）。

        背景：`MODEL_PATH` / `GGUF_MODEL_PATH` 是模块级**静态**常量，无法按画像变化；
        而「自动加载默认模型」（`api_server._auto_load_default_model` /
        `engine_host._auto_load_default_model`）此前直接读它们 ⇒ 无论什么设备都加载同一个模型。

        用户裁定：**边缘设备（含轻薄本/集显）用 <1B 小模型，PC 用 ~2B**（贴合原 1.8B 体量）。
        分档表与选型实测依据见 `model_config.DEFAULT_MODEL_BY_TIER`。

        ⚠️ 必须**在这里**（模块加载完成后）调用，不能在顶层求值：`device_profiler`
        会 import `config`，顶层触发会形成循环 import。

        Returns:
            dict：``model_id`` / ``name`` / ``model_path`` / ``gguf_path`` / ``model_type``；
            任何异常都退回静态常量（保证无人值守自动加载不因画像问题中断）。
        """
        try:
            import model_config as _mc

            return _mc.get_profile_default_model_paths()
        except Exception:  # noqa: BLE001
            return {
                "model_id": ACTIVE_MODEL_ID,
                "name": MODEL_NAME,
                "model_path": MODEL_PATH,
                "gguf_path": GGUF_MODEL_PATH,
                "model_type": "both",
            }

    return {name: value for name, value in locals().items()
            if name != "environment" and name != "probe_device"}


def warn_if_cluster_secret_unavailable(environment) -> None:
    """Warn at runtime startup when identity resolution found no credential."""
    if str((environment or {}).get("QLH_CLUSTER_SECRET") or "").strip():
        return
    import warnings
    warnings.warn(
        "集群凭据未加载！集群通信认证将失败。请完成可信的集群引导或凭据恢复。",
        RuntimeWarning,
        stacklevel=2,
    )


# Legacy consumers receive a read-only startup view. Explicit AppRuntime startup
# projects a reviewed snapshot once; domain migrations remove these aliases.
globals().update(build_settings())
