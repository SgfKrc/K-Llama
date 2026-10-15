# Android 版本远期计划

> 更新日期：2026-08-23
>
> 状态：历史架构基线 + 远期技术预研（Android Full Worker 与 TaskGraph Stage 接线已完成本机开发门；真机完整推理、认证、长时与设备资源验收仍后置）
>
> 适用范围：保留 Android Full/Lite、完整 Worker、任务链、GPU 后端与层段拆分的架构边界和远期路线；本文不再作为 Android 当前功能状态的唯一来源，现行排期以[安卓与PC功能差距清单](../安卓与PC功能差距清单.md)和[总体下一步计划](../distributed/总体下一步计划.md)为准
>
> 当前事实：Android Full 已具备本地 GGUF/llama.cpp 推理，以及受模型身份、资源门和租约围栏约束的完整 Worker/TaskGraph Stage 代码路径；Lite 保持薄客户端；Android 仍不能加入现有 PC PyTorch 层间流水线
>
> 远期目标：完整推理 Worker -> 任务链拆分与分布式执行 -> 同一 llama.cpp/GGML 运行时的实验性跨设备拆分
>
> 2026-08-23 状态映射：下文 P1/P2 的原始设计已完成本机开发门，保留以说明协议和运行时边界；P3-P5 仍为未实施的候选路线，不得因 P1/P2 完成而推断 Android 可参与 PyTorch 层拆分或已通过真机验收。
>
> 关联文档：[Android SAF模型存储方案](../Android%20SAF模型存储方案.md)、[PC与Android端交互体验优化计划](PC与Android端交互体验优化计划.md)、[分布式推理流水线实施计划](../distributed/分布式推理流水线实施计划.md)、[混合分布式推理体系规划](../distributed/混合分布式推理体系规划.md)、[三种分布式拆分细化实施方案](../distributed/三种分布式拆分细化实施方案.md)

---

## 1. 结论摘要

Android 后续不再限定为“全有或全无”，但必须区分三个完全不同的分布式层级：

| 层级 | 传输边界 | Android 可行性 | 规划结论 |
|---|---|---:|---|
| 完整推理 Worker | prompt/messages -> 完整文本或结构化结果 | 高 | 近期主路线 |
| 任务链/任务图拆分 | 上一阶段的文本、JSON、检索结果 -> 下一阶段 | 高 | 中期主路线 |
| Transformer 层间拆分 | input IDs/hidden states/KV cache/logits | 当前低 | 仅限同运行时预研，不接入现有 PC PyTorch 流水线 |

核心判断：

1. **当前 Android 不能做层间拆分。** 注册画像明确上报 `pipeline_worker=false`，主节点也会排除 `node_type=android`；JNI 使用 `model_params.n_gpu_layers=0`，当前仅 CPU 完整模型推理。
2. **完整 Worker 已完成本机开发接线。** Android 已具备模型管理、前台 Service、llama.cpp JNI、完整生成和取消基础，并已接入任务协议、租约/幂等/迟到结果围栏、调度准入和 Stage executor；真实认证、Android 端实际推理、网络切换、温控/电量和长时设备验收仍未完成。
3. **任务链拆分比层间拆分更适合异构设备。** 阶段之间传递文本或结构化数据，不要求 Android 与 PC 使用同一模型格式、张量布局或执行引擎。
4. **Android 参与“分布式推理”可以有两种含义。** 一种是完整任务在多个 Worker 间并发/串行编排；另一种才是单次生成的模型层跨设备计算。前者可产品化，后者仍是实验。
5. **Android GPU 加速应用可以落地，但应叫 GPU 加速版，不应泛称独显版。** 主流 Android 平板使用 SoC 集成 GPU 并共享系统内存。上游 llama.cpp 已明确验证部分 Qualcomm Adreno OpenCL 设备，但这不等于所有 Android GPU 都可稳定运行。
6. **真正带独立显存 GPU 的 Android 平板不能作为通用产品前提。** 此类设备少、Android 驱动栈和 NDK 可用计算后端不统一，只能按指定硬件/BSP 做定向版本。

---

## 2. 当前实现边界

### 2.1 已实现能力

| 能力 | Full | Lite |
|---|:---:|:---:|
| Android 原生 UI、会话和 Room 存储 | 是 | 是 |
| Material 3 主题体系与共享组件（2026-07-26 重构、2026-08-17 UX-04 中性 token 映射，见 §2.3） | 是 | 是 |
| 连接 PC 主节点并转发聊天 | 是 | 是 |
| SAF 模型目录和 GGUF 模型选择 | 是 | 否 |
| llama.cpp JNI 完整模型生成 | 是 | 否 |
| 前台 `InferenceService` | 是 | 仅保留必要壳层 |
| 集群 presence 注册 | 是 | 是 |
| 接收主节点完整推理任务 | 开发门完成；真机未验 | 否 |
| PC PyTorch 层间流水线 Worker | 否 | 否 |
| Android GPU 计算后端 | 否 | 否 |

代码层面的当前约束：

- `MainViewModel.buildAndroidPresenceDeviceInfo()` 上报 `pipeline_worker=false`。
- `api_server.py` 和 `scheduler.py` 继续将 Android 排除在 PC PyTorch 层分配之外；仅在精确模型身份、单并发和 `resource_gate` 均满足时，才允许 `android_full_worker` 承接完整推理 Stage。
- `qlh_llama_jni.cpp` 固定 `n_gpu_layers=0`。
- Android CMake 没有启用 `GGML_OPENCL`、`GGML_VULKAN` 或 `GGML_RPC`。
- 当前集群层间协议由 PC PyTorch 模型执行，传递 PyTorch hidden states，不是 llama.cpp/GGML RPC 协议。

因此前端出现“Android 已连接”不代表 Android 参与了层间推理，也不应把 Android 的 presence 或普通聊天请求统计为流水线 Worker 任务。

### 2.2 为什么不能直接接入 PC 层间流水线

PC 当前的层间拆分假设各计算节点具备同一套 PyTorch 模型语义：

- 相同模型架构、权重版本、tokenizer 和配置。
- 相同 hidden size、层编号、RoPE/position 规则和 attention mask 语义。
- 可加载明确的 Transformer 层范围，并输出下一段可继续执行的 hidden states。
- KV cache 能按当前协议创建、定位和清理。

Android Full 使用 GGUF + llama.cpp。其公开 API 面向完整 decode，量化权重布局、计算图、KV cache 和 backend tensor 均由 GGML/llama.cpp 管理。即使模型名称相同，也不能把 PyTorch hidden states 直接送入任意 llama.cpp 层段继续计算。

要跨引擎强行拆层，至少需要解决：

- 在 llama.cpp 内暴露稳定的层段入口和出口。
- 统一 hidden states 的 dtype、shape、布局、量化/反量化和端序。
- 对齐 embedding、LM head、归一化、RoPE、attention mask 和采样状态。
- 定义跨设备 KV cache 所有权、位置和回收协议。
- 每个 token 传输中间张量，并处理超时、重试、取消和迟到结果。
- 长期维护 llama.cpp fork，并跟踪上游计算图和 backend API 变化。

结论是：**现有 PC PyTorch 层间协议与 Android llama.cpp 不兼容，不能仅靠放开 `pipeline_worker` 标志实现。**

### 2.3 UI 重构（2026-07-28，已实施并编译验证）

本次改动**只涉及 UI 层**，ViewModel、网络层与服务层零改动，Full/Lite 两个 flavor 通用；它不改变本章任何分布式边界结论。

| 范围 | 内容 |
|---|---|
| 主题 | `ui/theme/{Color,Theme,Type}.kt` 使用中性 Material 3 色调系统：浅色白底黑字、深色黑底白字，灰阶 `surfaceContainer` 层次；状态色独立，全部排版 token 字距为零 |
| 共享组件 | 新增 `ui/components/Common.kt`：`QlhTopBar`、`StatusChip`、`SettingsGroup`、`SettingRow`、`EmptyState`；普通控件/卡片统一 `8dp`，状态角标保留胶囊形状 |
| 对话页 | 气泡重做（用户/助手容器色 + 对齐 + 85% 限宽、8dp 圆角带 4dp 尾角）；顶栏加本地/远程状态角标；8dp 输入栏 |
| 会话页 | `SessionListScreen` 改为卡片式 |
| 设置页 | `SettingsScreen` 重排为 8 个分组卡片；36 个参数、回调、`testTag` 与极简版分支全部保留 |

详细逐页说明见 [Android UI 改进说明](../../android/UI改进说明.md)。

**验收状态：已验收（2026-08-13 真机实测）。** 2026-07-28 已执行 `gradlew.bat :app:compileFullDebugKotlin` 并成功通过资源处理、KSP 与 Kotlin 编译；Full Release APK 已在真机多轮安装运行，暗色模式对比度与长消息气泡限宽均已目视核对通过。安全前缀默认暗色（0xFF171D23）与隐藏 token 的 `无明文` 占位、StatusChip "本地推理/远程推理" 显示均已核验。

新增的 `StatusChip` 用于展示“本地推理 / 远程推理”，这是**执行位置**的展示，不是分布式参与度的展示；§2.1 与 §9 关于“不能把 presence 或普通聊天请求统计为流水线 Worker 任务”的要求不受本次 UI 改动影响。

---

## 3. 目标架构

### 3.1 节点能力模型

后续调度不能只按 `node_type` 判断，需上报明确能力：

```json
{
  "node_type": "android",
  "capabilities": {
    "thin_client": true,
    "full_inference_worker": true,
    "task_chain_worker": true,
    "pytorch_layer_worker": false,
    "ggml_rpc_worker": false,
    "engine": "llama.cpp",
    "model_format": "gguf",
    "model_sha256": "...",
    "backend": "cpu|opencl|vulkan",
    "max_concurrency": 1
  },
  "runtime": {
    "model_loaded": true,
    "charging": true,
    "battery_percent": 83,
    "thermal_status": 1,
    "available_memory_bytes": 4294967296
  }
}
```

能力必须由运行时探测和实际模型状态决定，不能仅凭设备型号推断。`full_inference_worker=true` 也不能推导出 `pytorch_layer_worker=true`。

### 3.2 三层调度结构

```text
请求 / 工作流
    |
    v
任务图调度器：拆成可检查点恢复的 Stage / DAG
    |
    +-- Stage A -> Android 完整 Worker（llama.cpp CPU/GPU）
    +-- Stage B -> PC 单机完整推理
    +-- Stage C -> PC PyTorch 层间流水线
    +-- Stage D -> 另一 Android/PC 完整 Worker
    |
    v
聚合、校验、重试和最终响应
```

每个 Stage 只能选择一种执行后端：

- `android_full_worker`
- `pc_full_worker`
- `pc_pytorch_pipeline`
- 远期实验性的 `ggml_rpc_pool`

禁止在同一 Stage 内把 PyTorch 层段和 Android GGUF 层段混在一条 hidden-state 链中。

---

## 4. 路线 A：Android 完整推理 Worker

### 4.1 定义

完整 Worker 接收一份自包含推理任务，在 Android 本机跑完整模型，再返回文本、usage 和运行指标：

```text
Master -> prompt/messages + sampling + model requirement
Android -> accepted -> progress/heartbeat -> complete result
```

这不是层间拆分，但它是有效的分布式推理：主节点可以把不同用户请求、批量任务、工作流分支或重试分配给不同设备。

### 4.2 连接方向

Android 应主动连接主节点，维持受控 WebSocket 或现有集群协议的移动端变体。不要要求主节点主动访问手机监听端口，原因包括：

- Android 后台和厂商 ROM 会限制常驻监听服务。
- 手机网络可能处于 NAT、移动网络或地址变化状态。
- 主动长连接更容易复用现有 Tailscale 网络和集群认证。
- 断线检测、重连和任务租约可以统一处理。

### 4.3 最小协议

| 消息 | 方向 | 作用 |
|---|---|---|
| `WORKER_REGISTER` | Android -> Master | 能力、模型摘要、后端、温控和并发度 |
| `TASK_OFFER` | Master -> Android | 任务摘要、模型要求、预计 token、租约 |
| `TASK_ACCEPT/REJECT` | Android -> Master | 接受或说明电量、温度、模型不匹配等原因 |
| `TASK_START` | Android -> Master | 已获得执行槽并开始 |
| `TASK_PROGRESS` | Android -> Master | 心跳、token 数、速度和温度 |
| `TASK_RESULT` | Android -> Master | 文本、metrics、模型/引擎摘要 |
| `TASK_FAILED` | Android -> Master | 可重试错误与不可重试错误分类 |
| `TASK_CANCEL` | Master -> Android | 协作取消 |
| `TASK_CANCELLED` | Android -> Master | 取消确认和资源清理完成 |

每条任务必须包含全局唯一 `task_id`、`attempt_id` 和租约到期时间。主节点只能接收当前 attempt 的结果，迟到结果必须丢弃。

### 4.4 调度准入与评分

完整 Worker 使用独立评分，不复用 Transformer 层数分配权重：

- 活跃模型是否匹配 `model_id + model_sha256 + tokenizer/config 摘要`。
- 模型是否已加载，避免频繁冷加载。
- 最近实测 prompt/decode tok/s，而不是只看 SoC 名称。
- 当前可用 RAM、系统 low-memory 状态和上下文上限。
- 充电状态、电量、thermal status 和连续运行时间。
- RTT、断线率、任务成功率和队列长度。
- CPU/OpenCL/Vulkan backend 及其设备白名单状态。

默认规则：单台 Android `max_concurrency=1`；未充电、低电量、高温、内存不足或应用不在允许的前台 Worker 模式时拒绝新任务。

### 4.5 故障隔离

- 主节点在租约超时后把任务重新派给其他 Worker，旧 attempt 的结果不再生效。
- Android 断线只失败当前 Stage，不应令整条工作流永久失败。
- 任务输入需可重放；产生外部副作用的 Stage 必须使用幂等键。
- 模型加载错误、OOM、温控暂停、用户取消和网络断开必须使用不同错误码。
- 任务结果应包含实际 `model_sha256` 和 backend，防止错误模型静默完成。

---

## 5. 路线 B：任务链/任务图拆分后的分布式推理

### 5.1 含义

任务链拆分不是把 Transformer 第 0-11 层和第 12-23 层拆开，而是把一个复杂请求拆成有明确输入输出契约的阶段：

```text
输入规范化
  -> 检索/上下文选择
  -> 候选答案生成（可并行多个 Worker）
  -> 事实核查/评分
  -> 修订
  -> 最终格式化
```

阶段边界使用 UTF-8 文本、JSON、文档引用或其他稳定业务对象。这样 Android llama.cpp、PC PyTorch 和外部工具可以在同一工作流中协作，而不交换引擎内部张量。

### 5.2 调度语义

任务图至少要支持：

- DAG 依赖和就绪队列。
- 串行 Stage、并行分支、`fan-out/fan-in` 聚合。
- 每阶段超时、租约、重试、取消和检查点。
- Stage 级模型/能力约束。
- 中间结果大小限制、摘要和持久化。
- 幂等 attempt 与迟到结果过滤。
- 失败分支降级，不因单节点异常让整条任务链失去恢复机会。

当前 `graph_orchestrator.py` 是**通信拓扑和层分配编排器**，不是业务任务 DAG 执行器。任务链需要新的 `TaskGraphScheduler` 或等价模块，不能直接把现有 `GraphOrchestrator` 改名复用。

### 5.3 Stage 内的分布式选择

任务链拆分后，每个模型 Stage 可独立选择：

1. Android/PC 单节点完整推理。
2. 多个完整 Worker 并行生成候选答案。
3. PC 节点使用现有 PyTorch 层间流水线执行一个 Stage。
4. 远期使用同一 llama.cpp/GGML RPC 池执行一个 Stage。

因此最终架构允许“任务链拆分后再分布式推理”，但分布式边界必须显式：业务 Stage 边界可以跨引擎；hidden-state 边界只能在兼容运行时内使用。

### 5.4 首个验证工作流

建议先实现无外部副作用的双候选评审流：

```text
用户问题
  +-> Android Worker A 生成候选 A --+
  +-> PC/Android Worker B 生成候选 B --+-> PC 主节点评审/合并 -> 最终回复
```

验收重点不是回答质量，而是验证：并行派发、租约、断线重派、部分失败降级、结果归属、统计和取消传播。

---

## 6. 路线 C：Android 层间拆分可行性调研

### 6.1 路径比较

| 路径 | 技术可行性 | 与现有系统兼容性 | 产品成熟度 | 结论 |
|---|---:|---:|---:|---|
| Android 加入现有 PC PyTorch hidden-state 流水线 | 低 | 低 | 低 | 不采用 |
| 在 Android 打包 PyTorch/ExecuTorch 并实现同模型层段 | 中低 | 理论上较高 | 低 | 仅做小模型原型评估 |
| llama.cpp/GGML RPC 跨设备 backend | 中 | 需要独立执行池 | 上游标记 POC | 隔离实验路线 |
| Fork llama.cpp 暴露层段和 KV cache | 中 | 需自建新协议 | 很低 | 除非 RPC 不满足且收益明确，否则不做 |
| 完整 Worker + 任务图 | 高 | 高 | 高 | 主路线 |

### 6.2 llama.cpp/GGML RPC 能做什么

当前锁定的 llama.cpp 上游文档说明：

- `ggml-rpc-server` 可以暴露远端 CPU/GPU backend device。
- 主运行时可以连接一个或多个远端 RPC device。
- 默认会按可用内存把模型权重和 KV cache 分配到本地及远端设备。
- 可通过 `tensor-split` 调整设备间比例。
- 远端可使用 tensor cache 避免每次重新传输大权重。

这证明 Android 作为 GGML 远端计算设备在架构上**不是不可能**。如果能为 Android NDK 编译 RPC server 与 CPU/OpenCL/Vulkan backend，并由前台 Service 安全托管，PC 或另一 Android 的 llama.cpp 主运行时可以尝试使用其算力。

但上游同时明确写明 RPC 是 proof-of-concept、功能脆弱且不安全，不得暴露到开放或敏感网络。它目前不应直接进入正式集群协议，原因包括：

- 没有本项目要求的任务身份、租约、取消和结果幂等语义。
- 安全模型不足，不能只因位于 Tailnet 就忽略输入校验和服务暴露范围。
- Android 生命周期可能在远端计算图执行期间暂停或杀死进程。
- 跨 Wi-Fi/Tailscale 每 token 的 tensor 传输延迟可能抵消移动 GPU 收益。
- PC PyTorch 主节点不能直接把 GGML RPC device 插入现有 PyTorch pipeline。
- 上游 API 和协议处于实验状态，版本升级风险高。

### 6.3 RPC 实验的边界

若开展实验，应采用完全隔离的 `ggml_rpc_lab` 模式：

- 只允许同一锁定 llama.cpp commit 和相同 GGUF 模型摘要。
- 只在受控局域网/Tailnet 测试，RPC 端口不监听公网接口。
- Android 使用前台 Service、充电和温控门槛。
- 先验证 CPU RPC，再验证 OpenCL/Vulkan backend。
- 与正式 `scheduler.py` 的 PC PyTorch 层分配、任务统计和 fallback 分开。
- 任何 RPC 失败都回退到完整 Worker，不把未完成 tensor 任务转交不同引擎继续。

进入产品路线的门槛：连续运行、断线恢复、安全封装和端到端收益全部通过；仅“成功生成一次”不算可落地。

### 6.4 自定义 llama.cpp 层段 fork

只有同时满足以下条件才考虑 fork：

- GGML RPC 无法满足目标拓扑或资源控制。
- 目标设备间网络实测足以承载 hidden states/KV 流量。
- 跨设备端到端吞吐显著优于最快单节点完整推理。
- 团队能承担固定上游版本、补丁重放和模型架构适配成本。

否则自定义层段会成为长期维护负担，并可能只得到更慢、更不稳定的结果。

---

## 7. Android GPU 平板版可落地性

### 7.1 硬件术语

主流 Android 手机和平板的 Adreno、Mali、Immortalis GPU 位于 SoC 内，与 CPU 共享系统内存。它们不是 PC 语境中带独立显存的独显。

所以产品命名建议为：

- `Lite`：薄客户端。
- `Full CPU`：本地 llama.cpp CPU 完整推理。
- `Full GPU Experimental`：白名单设备的 GPU 加速完整推理/Worker。

只有拿到具体硬件、Android BSP 和可用驱动栈后，才为真正独立 GPU 设备建立专用 flavor；不规划一个面向任意“独显 Android 平板”的通用 APK。

### 7.2 后端调研结果

调研基线为仓库内 vendored llama.cpp commit `47e1de77aa0f06bf73cfd8c5281d95979f89fcbe`：

| 后端 | 上游证据 | 适合范围 | 判断 |
|---|---|---|---|
| CPU/Arm | 官方 Android binding 和 NDK 交叉编译文档；运行时检测 Arm 指令 | 所有当前 Full 支持设备 | 已落地 |
| OpenCL/Adreno | 上游文档明确列出 Android 支持 Snapdragon 8 Gen 3、8 Elite，并验证 Adreno 750/830/840 等 | Qualcomm 白名单设备 | 最现实的 GPU 原型路线 |
| Vulkan | llama.cpp 支持 Vulkan backend；Android 提供 Vulkan NDK 图形/计算接口 | 需逐设备验证驱动、算子覆盖和稳定性 | 第二实验路线，不能宣称全 Android 通用 |
| Hexagon | 上游 Snapdragon/Hexagon backend 有 Android 构建说明，但日志标为 experimental | 指定 Snapdragon 开发平台 | 独立预研，不先进入消费版 |
| QNN | 需要 Qualcomm SDK、许可和单独集成，当前 llama.cpp 接入链未验证 | 指定 Qualcomm 产品 | 暂不作为首选 |
| NNAPI | 当前项目和 llama.cpp 路线没有可直接复用的层段/backend 接入 | 不明确 | 从推荐路线移除 |

OpenCL 当前有最明确的上游 Android/Adreno 验证记录，因此建议优先于泛化 Vulkan 承诺。但它仍需要目标平板的驱动和 SoC 白名单。

### 7.3 GPU 加速不等于显存扩容

集成 GPU 与 CPU 共享 RAM。GPU offload 可能提高算力，但不会像 PC 独显那样凭空增加独立显存，甚至可能因 backend buffer、kernel 和 staging 产生额外占用。

评估必须同时测量：

- 模型加载峰值和稳定 RAM。
- prefill tok/s、decode tok/s 和首 token 延迟。
- 连续 10/30 分钟后的温控降频。
- GPU backend 崩溃、驱动 reset 和应用恢复。
- 不同 context size 的 KV cache 增长。
- CPU 与 GPU 后端的功耗、充电状态和电池温升。
- 同时作为完整 Worker 时的网络、心跳和取消响应。

### 7.4 产品化门槛

`Full GPU Experimental` 只面向设备白名单，并满足：

- CMake 为独立 flavor 开启对应 backend，不污染 Lite。
- 运行时列举到真实 GGML GPU backend device。
- `llama_supports_gpu_offload()` 为真，并实际配置非零 offload。
- 同模型、上下文和参数下，相比 CPU 有稳定端到端收益，而不只是在短基准中更快。
- 连续 30 分钟无 native crash、无不可恢复 driver error。
- 超温时能停止接单、完成当前安全点后降级或取消。
- GPU 不可用时可回退 CPU Full 或 Thin，不循环崩溃。
- 每个白名单型号保存 SoC、GPU、驱动/Android 版本和基准结果。

结论：**高端 Snapdragon Android 平板上的 GPU 加速 Full/Worker 版具备落地可能；通用 Android GPU 版和所谓通用独显平板版目前不具备发布依据。**

---

## 8. 分阶段实施计划

### P0：保持当前边界并修正文档/状态

- Android 继续上报 `pipeline_worker=false`。
- 主节点继续排除 Android 的 PyTorch 层分配。
- UI 明确区分“已连接”“完整 Worker 可用”“参与层间流水线”。
- 不再把全有/全无描述为永久架构限制。

验收：Android 注册不会改变 PC 层配置，任务统计不误报 Android 参与层间推理。

进度（2026-08-23）：P0 的边界约束继续成立（`pipeline_worker=false`、主节点仍排除 Android 层分配）。§2.3 UI 已完成既有编译/真机验收记录；P1 已完成 Android `qlh.task_worker` v2 协议、前台 Worker Service、连接退避、取消/迟到结果 fencing 和 scheduler `android_full_worker` 准入的本机开发门；P2 已完成可注入 Full Worker Stage executor、TaskGraph/SQLite journal fake 回归。真实认证、Android offer→本地推理、网络切换、温控/电量和长时设备验收后置；P3–P5 尚未开始。

### P1：完整 Worker 协议原型（本机开发门已完成）

- Android 主动长连接和能力注册。
- `TASK_OFFER/ACCEPT/RESULT/CANCEL` 最小协议。
- 模型摘要、单并发、租约和迟到结果过滤。
- 前台 Service、充电/温控/内存准入。
- 主节点独立 `full_worker_pool`，不进入 `compute_layer_assignment()`。

验收：主节点派发一项完整推理；Android 返回结果；断网后任务能重派且旧结果被丢弃。

### P2：任务链/DAG 调度（本机开发门已完成；真机执行验收后置）

- 新建业务 `TaskGraphScheduler`，与层拓扑 `GraphOrchestrator` 分离。
- 支持 Stage 依赖、并行分支、checkpoint、重试和聚合。
- 支持 Android Full Worker、PC Full 和 PC PyTorch pipeline 三类 Stage executor。
- 先完成双候选评审工作流。

验收：任一 Android 分支失败时仍可由另一分支或重派完成，不导致整条链不可恢复。

### P3：Android GPU 加速 Worker

- 首选上游已验证的 Snapdragon/Adreno OpenCL 白名单设备。
- 建立独立 GPU experimental flavor。
- JNI 支持 backend 枚举、offload 配置和真实指标上报。
- 与 CPU Full 做固定模型、固定上下文的持续基准。
- 再评估 Vulkan 设备矩阵，不预设其一定优于 OpenCL。

验收：至少一款目标平板达到产品化门槛并稳定执行完整 Worker 任务。

### P4：GGML RPC 隔离实验

- Android NDK 构建 `GGML_RPC` + CPU backend。
- 局域网验证相同 commit、相同 GGUF 的远端设备发现和模型执行。
- 再组合 OpenCL/Vulkan backend。
- 测量权重缓存、首轮加载、per-token 流量、断线和恢复。

验收：只有端到端性能优于单节点且安全/稳定性可封装，才提出正式集成设计。

### P5：层段 fork 决策门

根据 P4 数据做 Go/No-Go。默认 No-Go；没有量化收益和维护资源时不实现自定义 llama.cpp 层间协议。

---

## 9. 测试矩阵

| 维度 | 最低覆盖 |
|---|---|
| 版本 | Lite、Full CPU、GPU Experimental |
| SoC | 一款中端 CPU 设备、一款上游 OpenCL 白名单 Snapdragon 平板 |
| 网络 | 局域网、Tailscale、断网/重连、高 RTT/丢包 |
| 电源 | 充电、低电量、拔电 |
| 温控 | 正常、升温降频、系统严重温控 |
| 内存 | 正常、low-memory、模型加载 OOM、长 context |
| 任务 | 成功、拒绝、取消、超时、租约过期、迟到结果、应用被杀 |
| 模型 | 摘要一致、模型缺失、摘要不一致、加载中、已热加载 |
| 工作流 | 串行、并行分支、部分失败、聚合失败、重试耗尽 |

所有分布式统计必须分别展示：完整 Worker 任务数、任务链 Stage 数、PC 层间流水线任务数。不能用一个“分布式：是/否”覆盖三种语义。

---

## 10. 风险与停止条件

| 风险 | 对策/停止条件 |
|---|---|
| Android 后台被杀 | 前台 Service + 租约；仍频繁中断则只允许用户显式开启 Worker |
| 温控导致吞吐崩溃 | 基于持续基准评分；严重温控立即停止接单 |
| 模型不一致 | 强制摘要匹配；Stage 边界跨模型只能传业务数据 |
| 任务链单点失败 | checkpoint、attempt、重派和部分结果降级 |
| RPC 不安全/脆弱 | 隔离实验、限制监听、版本锁定；未满足门槛不进正式版 |
| GPU 驱动碎片化 | 设备白名单；未验证设备自动回退 CPU/Thin |
| GPU 加速收益低 | 以端到端持续性能为准，不因 backend 能加载就发布 |
| 自定义 fork 维护过重 | P5 默认 No-Go，优先完整 Worker 和任务图 |
| 把集成 GPU 当独显估算容量 | 统一按共享 RAM 实测，不使用 PC VRAM 评分公式 |

---

## 11. 调研依据与待验证项

本次结论以项目当前源码和仓库内锁定的 llama.cpp 上游快照为主要依据：

- [Android presence 能力上报](../../android/app/src/main/java/com/qlh/inference/MainViewModel.kt)：当前固定上报 `pipeline_worker=false`。
- [Android JNI 模型加载](../../android/app/src/main/cpp/qlh_llama_jni.cpp)：当前固定 `n_gpu_layers=0`。
- [Android CMake 配置](../../android/app/src/main/cpp/CMakeLists.txt)：当前未启用 OpenCL、Vulkan 或 RPC backend。
- [锁定快照的 Android 文档](../../android/app/src/main/cpp/llama.cpp/docs/android.md)：Android Studio binding、NDK 交叉编译与 Arm CPU 能力。
- [锁定快照的 OpenCL 文档](../../android/app/src/main/cpp/llama.cpp/docs/backend/OPENCL.md)：Android/Adreno 支持、已验证 Snapdragon/Adreno 型号和构建方法。
- [锁定快照的 RPC 文档](../../android/app/src/main/cpp/llama.cpp/tools/rpc/README.md)：远端 GGML device、权重/KV 分布、tensor cache，以及 POC/不安全警告。
- [Android 官方 NDK Vulkan 文档](https://developer.android.com/ndk/guides/graphics/getting-started)。
- [Android 官方 Vulkan 设备兼容文档](https://developer.android.com/games/develop/vulkan/device-compatibility)。
- [llama.cpp 上游 OpenCL 文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md)。
- [llama.cpp 上游 RPC 文档](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)。

本地调研基线 commit：`47e1de77aa0f06bf73cfd8c5281d95979f89fcbe`。实施 P3/P4 前必须联网复核上游最新文档和目标设备驱动，因为 GPU backend 与 RPC 仍在快速变化。

仍需通过真实设备回答的问题：

1. 目标 Snapdragon 平板的 OpenCL driver/compiler 是否满足上游 backend。
2. OpenCL 与 Vulkan 在 Qwen 1.8B GGUF 上的 prefill、decode、温控和稳定性谁更优。
3. Android 前台 Service 被切后台、锁屏和厂商省电策略影响的中断率。
4. GGML RPC 在 Wi-Fi/Tailscale 下每 token 的实际流量和延迟。
5. RPC tensor cache 在 Android 存储沙箱中的路径、空间和清理策略。
6. 真正带独立 GPU 的目标 Android 硬件是否提供 NDK 可调用且可分发的驱动/API；没有具体硬件和 BSP 前不做可落地承诺。

---

## 12. 最终路线

```text
当前：Lite 薄客户端 + Full CPU 本机完整推理
  -> P1：Android 完整推理 Worker
  -> P2：任务链/任务图拆分，Stage 选择异构执行后端
  -> P3：白名单 Android 平板 GPU 加速完整 Worker
  -> P4：同一 llama.cpp/GGML 运行时 RPC 隔离实验
  -> P5：仅在收益明确时决定是否维护自定义层段 fork
```

近期目标不是让 Android 假装成 PC PyTorch 层节点，而是先让它成为可靠、可取消、可重派、可统计的完整 Worker。任务链拆分随后提供真正可维护的异构分布式推理；层间拆分则保留为有严格准入门槛的研究方向。PC、Android、GGUF stage 和张量并行执行池的统一选择规则见[混合分布式推理体系规划](../distributed/混合分布式推理体系规划.md)。
