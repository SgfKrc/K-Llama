# PyTorch 层流水线加载峰值优化方案

> 状态：**现行**
>
> 更新日期：2026-09-15

> 文档生命周期：**现行（Active）**
>
> 创建日期：2026-08-13
> 更新日期：2026-08-16
> 适用范围：分布式层间流水线（layer-wise pipeline）从节点的模型加载瞬时显存/内存峰值治理；直接约束 Surface 7 级轻量从节点在重模型（DeepSeek 7B / gemma-4 12B / Qwen 9B 等）下的可用性
>
> 目标：**加载瞬间峰值 ≈ 分配层段权重 + 常量开销**，与稳态占用同量级，杜绝"全模型峰值压垮从节点"；一期进一步把模型身份/容量发现从权重物化中剥离，向"集群总容量加载"演进。

> **实施复核（2026-08-13）**：P0 已完成，P2 的本机开发基线已完成。`ModelManager.load_layer_range` 现在强制 Qwen/Qwen2 架构加载器返回 `_LayerRangeLoadTracker`；每个 safetensors tensor 在 `get_tensor` 前先校验 key 是否属于分配层段，越界物化、缺失分配层或绕过守卫均 fail-closed。加载过程在 meta 骨架完成、tensor 读取、目标设备安装、裁剪和清理边界采样进程 RSS 与 CUDA allocated，并通过 `get_model_info().layer_load_metrics` 暴露层索引、tensor 数、源/目标字节和峰值增量，不记录文件路径或权重 key。定向回归 `95 passed / 1 skipped`，Python 全量 `2105 passed / 8 skipped`。
>
> **双机验收补充（2026-08-20）**：QW1.8B 分层数据面真机通过——主节点 RTX 4060（Layer 0-21）+ Surface 从节点（Layer 21-24，无 CUDA），双方本地选择性加载、网络只传 hidden/KV，三次 `distributed_required` 均 `distributed_used=true`/`fallback=false`、RTT 6-12ms、64 心跳窗口无重连（见[项目进展与下一步计划](../distributed/项目进展与下一步计划.md) 2026-08-20 条目）。**本方案在 QW1.8B 上的真机边界已闭合**；真实 7B/12B 重量级模型三节点分段与峰值实测仍后置（见 §"六期"）。

---

## 1. 背景与问题

- 从节点形态：Surface 7 级轻量设备（RAM/显存有限），是分布式的主要物理从节点形态。
- 担忧：重模型（≥7B）若在加载瞬间出现**全模型权重峰值**（如 fp16 7B ≈ 14GB、fp16 12B ≈ 24GB），会直接压垮从节点——即使稳态只占层段。
- 项目现有契约：`EngineHost.load_layer_range` / `ModelManager.load_layer_range` 按主节点下发的 `layer_assignments` 只加载对应层段（见 `scheduler.compute_layer_assignment` + `_check_node_vram_for_layers`）。

## 2. 现状实现（2026-08-13 代码实测）

| 路径 | 实现 | 瞬时峰值 |
|---|---|---|
| **Qwen / Qwen2 系**（含 DeepSeek-R1-Distill-Qwen，`model_type=qwen2`） | `_load_qwen2_layer_range` / `_load_qwen_layer_range`：`safetensors.safe_open` **按 key 过滤**；`_LayerRangeLoadTracker` 在读取前强制校验并在结束时核对实际层集合 | ✅ 代码路径只允许分配层段物化；真实峰值上界待目标硬件标定 |
| Embedding / LM Head | 按 `has_embedding` / `has_lm_head` 决定保留；过滤路径下未分配端不物化 | ✅ 常量级 |
| 加载后裁剪 | `ModuleList(kept)` + `del` 释放，并断言裁剪后层数等于 assignment | 防御性二次保险（过滤路径下未分配层仍为 meta） |
| 加载观测 | tensor/materialization 边界采样 RSS + CUDA allocated；统计源权重字节与目标 dtype 字节 | ✅ 本机可回归；不宣称捕获两个边界间的瞬时尖峰 |
| **Gemma 4 Unified**（`gemma4_unified`） | KIP-11A/11B 开发门已完成：官方嵌套 text layout、隔离 assignment、分段前向、shared-KV/mask、sidecar 与 Scheduler 本地链路已落地；主运行时仍 fail-closed | ✅ 官方 Transformers 5.10.1 CPU tiny prefill/decode 与整模 logits 等价；真实工件/CUDA/双机/生产准入仍关闭 |
| **其他架构**（`qwen3`、`qwen3_vl` 等） | 主 `load_layer_range` fail-closed；已实现的 Qwen3 filtered sidecar 路径单独登记，不能回退整模加载 | 按各架构 sidecar 状态判定 |

**结论**：Qwen/Qwen2 系**现状已无全模型峰值问题**；真正的风险在**未来新架构接入层流水线时**（若实现者图省事走"全加载 → 截断"，就会引入峰值）。本方案的任务是：**把"按层 key 过滤物化"固化为唯一允许的接入路径**，并治理其余加载缓冲。

## 3. 峰值构成分析（重模型接入时的四个来源）

| 来源 | 说明 | 治理 |
|---|---|---|
| ① 全模型物化 | `from_pretrained` 或全量 `state_dict` 加载后再裁剪 | **禁止**——必须按层 key 过滤（safetensors `safe_open` 或 `state_dict` 预过滤） |
| ② 未分配端 Embedding/LM Head | 首/末节点外加载双端 | 按 `has_embedding` / `has_lm_head` 过滤 key |
| ③ 加载缓冲 | 权重读取页、转置临时张量、量化中间张量（fp16→int8/Q4 转换） | 逐层处理 + 及时 `del`/释放；量化在过滤后按层进行 |
| ④ 峰值叠加 | 加载 + 优化器/计算图同时存在 | 加载期禁止并发推理（现有加载锁已覆盖） |

## 4. 优化方案（分层执行）

### 4.1 固化现状为强制模式（P0，零代码或小改）
- **状态：Completed（2026-08-13）**。
- 将 `_load_qwen2_layer_range` 的"按层 key 过滤物化"确认为**层流水线唯一加载模式**，写入本方案为架构契约：
  - 任何新架构的 `load_layer_range` 实现必须从 safetensors 按 key 过滤，**禁止**"全加载后截断"路径。
  - 现有防御性裁剪代码保留（不删除，作为兜底断言：过滤后实际物化层数 == 分配层数）。
- `load_layer_range` 要求加载器返回有效 `_LayerRangeLoadTracker`；未返回即拒绝进入后置裁剪。
- 在 `load_layer_range` 的 `else` 分支错误消息中补充指引：新架构接入须按本方案 §4.2 模板实现。

### 4.2 新架构接入模板（P1，未来架构必用）
**状态：KIP-11A/11B Development Gate Completed（2026-08-16，见《已知问题记录》#11）；Qwen3 走独立 adapter。** KIP-11A 已按官方 Transformers 定义纠正 Gemma 4 身份和布局：顶层 `model_type=gemma4_unified`，文本层位于 `model.language_model.layers.*`，首尾权重位于 `model.language_model.embed_tokens.*`、`model.language_model.norm.*` 和 `lm_head.*`；视觉与音频权重位于 `model.embed_vision.*`、`model.embed_audio.*`，不得混入文本 assignment。KIP-11B 在独立 `.venv-gemma4-pipeline`（Transformers 5.10.1）中完成官方层段前向、连续 KV/shared-KV、显式 mask/位置、path-free sidecar 合同和 Scheduler 本地 2/3 段链；官方 CPU tiny prefill/decode 与整模 logits 最大绝对误差为 0。`.venv-gemma4-native` 仅保留原生 MTMD/llama.cpp，不得复用。当前 `pipeline_runtime_supported=false`，真实 23.9GB 工件、CUDA/双机、网络传输、恢复和生产准入仍必须后验收。**Qwen3 系继续使用 `qwen3_pipeline_adapter` + C3 manifest**，不经过主运行时 `load_layer_range`。

1. 解析 config 的 `num_hidden_layers`，校验与 assignment 一致（现有逻辑复用）；
2. 确定层 key 前缀（`model.layers.` / `transformer.h.` / `model.h.`——按架构）；
3. `safe_open` 过滤：仅物化 `{prefix}{i}.`（i ∈ [start, end)）+ 按 `has_embedding`/`has_lm_head` 过滤首尾 key；
4. 构建模型骨架（`AutoConfig` + 空模型或 meta device），用过滤后的 state_dict 加载到目标设备；
5. 量化（若启用）在层段内逐层执行，中间张量即时释放；
6. 加载后断言：`model.{layers}.__len__() == end - start`，失败 fail-closed。

### 4.3 加载缓冲治理（P2，可选增强）
- **状态：Development baseline completed / hardware validation pending（2026-08-13）**。
- safetensors `mmap` 按 tensor 读取，安装到目标模块后立即 `del tensor`；加载结束已有 `gc.collect()` 与 `torch.cuda.empty_cache()`。
- 在 meta 骨架完成、每次 tensor 读取与目标设备安装、模型裁剪和最终清理边界，以 `psutil` + `torch.cuda.memory_allocated()` 采样；指标直接进入模型信息，不另生成报告文档。
- 边界采样用于发现整模量级回归和支撑真机标定，但不等同高频 profiler；两个采样点之间的极短瞬时尖峰属于最终验收残余风险。

## 5. 验收标准

| 验收项 | 标准 | 方法 |
|---|---|---|
| 峰值上界 | 加载瞬时峰值 ≤ `层段权重 × 1.2 + 常量`（常量 = embedding/head/缓冲，实测标定） | 采样工具记录峰值，与理论层段权重比对 |
| 无全模型峰值 | 峰值 **不** 出现"全模型权重"量级（如 7B fp16 ≈ 14GB） | 峰值采样 + 断言 |
| Surface 7 可承载 | DeepSeek 7B 分 3 节点时，从节点峰值 ≤ Surface 7 可用内存 | 真机（下月从节点）实测 |
| 新架构接入 | 按 §4.2 模板实现，无"全加载截断"代码路径 | 代码评审 + 峰值回归 |

开发门与验收门分开处理：P0/P2 代码与合成模型测试完成后可以继续其他开发；上述真实模型、目标硬件和多机结论保持 `Validating`，不阻塞工作流，也不得提前标为通过。

## 6. 测试与回归

- 已完成：`tests/test_model_module.py` 共 `95 passed / 1 skipped`，相邻模型 API/分布式前向 `91 passed`、scheduler `268 passed`，Python 全量 `2105 passed / 8 skipped`；覆盖真实 tiny Qwen2 safetensors 选择性加载、双层段与全模型 logits 等价、守卫在 `get_tensor` 前拒绝未分配 key、缺层 fail-closed、加载器不可绕过守卫、未知架构接入指引、指标结构和源/目标 dtype 字节核算。
- 新架构接入时必须新增：该架构的 key 过滤等价性测试（过滤加载 vs 全量加载逐层权重一致）及分段前向等价测试。
- 峰值数值阈值不能用 tiny 模型或当前开发机常量代替；真实工件到位后按 3 轮标定冻结常量，再启用 `层段权重 × 1.2 + 常量` 自动判定。
- 真机：Surface 从节点已参与 2026-08-20 的 QW1.8B 双机分层验收（Layer 21-24）；DeepSeek 7B / Gemma 4 12B 等重量级模型的三层段分配峰值实测仍待真机标定（验收清单 A4 关联后续票）。

## 7. 关联与登记

- 关联：`docs/archive/models/Gemma-4-12B多模态支持方案.md` §2.4（PyTorch 候选 S2 接入层流水线时按本方案模板）；`docs/验收清单与资源限制登记.md` A4（双机验收含峰值实测）；`docs/archive/relay/张量并行外部辅助与混合拆分调研方案.md`（层间拆分 vs TP 的显存语义区分）。
- 优先级：P0、P2 本机开发基线与 Gemma 4 KIP-11A/11B 开发门已完成；KIP-12 受管 native binding 冻结身份门也已关闭。当前下一票转 KIP-13（P6 完整加载复用指纹）；硬件到位后再执行真实工件三轮峰值标定与阈值冻结，KIP-11B 不因 tiny smoke 进入生产 allow-list。
- 本方案不改变任何已验收行为；当前 Qwen 系分布式加载行为不变（已是目标形态）。

## 8. 从“单机先整模”迁移到“集群总容量加载”的分期计划

### 8.1 目标不变量与本机工件核验

最终目标不是“主节点勉强能先装下模型，再把层切给其他节点”，而是：**任一节点都不需要先物化完整模型；只要通过准入的节点集合对各组件的可用 RAM/VRAM 总量足够，集群就能从冷态按最终分工加载。** 控制面只允许读取配置、索引、文件头和摘要，不得以 `from_pretrained`、完整 `state_dict` 或临时整模实例换取模型层数和容量信息。

2026-08-13 对现有 PyTorch/Safetensors 工件的核验如下。数值来自 Safetensors 文件头，不读取 tensor 内容：

| 本地工件 | 架构 | 解码层 | 权重字节 | 组件拆分 | 当前流水线执行状态 |
|---|---:|---:|---:|---|---|
| `models/qwen3-4b` | `qwen3` | 36 | 8,044,936,192 | 解码层 7,267,018,752；Embedding 777,912,320；Norm 5,120 | 可发现；adapter 未实现，拒绝准入 |
| `models/qwen3-vl-4b-instruct` | `qwen3_vl` | 36 | 8,875,631,616 | 文本同上；视觉塔 830,695,424 | 可发现；文本/视觉执行器未实现，拒绝准入 |
| `models/qwen3-5-2b` | `qwen3_5` | 24 | 4,548,144,832 | 解码层 2,746,532,544；Embedding 1,017,118,720；视觉塔 662,833,152；MTP 121,656,320 | 可发现；混合注意力/视觉/MTP 执行器未实现，拒绝准入 |

这三套都是官方 PyTorch/Safetensors 工件，不是 GGUF。主运行时 `transformers==4.47.x` 仍不能直接识别这些架构；Qwen3 sidecar 当前只有 config/tokenizer 预检能力，没有 torch/accelerate 分层执行环境。因此“已下载 PyTorch 工件”与“已能参与分布式 PyTorch 流水线”必须继续分开登记。

### 8.2 分期与依赖

| 阶段 | 开发票 | 内容 | 完成门 |
|---|---|---|---|
| **一期：清单优先冷启动控制面** | `PT-PIPE-C1` | 不加载权重地解析 config、Safetensors 索引与文件头；形成逐层/Embedding/LM Head/视觉/MTP 字节账本；增加显式 distributed-only 准备状态；调度器不再要求主节点 `is_loaded` 才能取得模型身份、层数和摘要 | 已准备状态下 `model is None`、`tokenizer is None`、`is_loaded=false`；Qwen/Qwen2 可下发；Qwen3 系仅发现且 fail-closed |
| **二期：集群总容量准入与两阶段提交** | `PT-PIPE-C2` | 以逐层真实字节、运行 dtype、常量开销和安全余量计算每节点容量；先生成完整可执行分配，再提交 worker；所有节点 ACK 后主节点只加载自身层段；任一节点不足则整轮不物化。取消“主节点至少一层”的硬假设，允许独立控制面/Tokenizer 节点 | 本机 RAM/VRAM 可低于整模，集群容量足够时仍可形成方案；失败时所有节点保持未加载或释放；无半提交代际 |
| **三期：Qwen3 文本分层执行器** | `PT-PIPE-QW3` | 在隔离运行时补齐 torch/accelerate/safetensors；实现 `model.layers.N` 按 key 物化、分段前向、KV cache、LM Head 与 `enable_thinking=false` tokenizer 契约；主运行时 4.47.x 不升级 | Qwen3-4B 两段/三段 logits 等价；无完整权重峰值；sidecar 断开时 fail-closed |
| **四期：按需工件同步** | `PT-PIPE-C3` | 将现有“每个 worker 先镜像全部 Safetensors”改为 assignment manifest；只同步本节点需要的 shard/重打包 layer bundle，并对文件、key 集和模型 revision 分层校验 | RAM/VRAM 与磁盘/网络都不再要求每节点持有整模；重连和 Range 续传不改变模型代际 |
| **五期：多模态组件编排** | `PT-PIPE-MM1` | 为 Qwen3-VL/Qwen3.5 将视觉塔、文本层、Embedding、LM Head、MTP 作为不同可放置组件；固定跨节点张量契约和视觉特征生命周期 | 视觉组件可以与文本层分置；纯文本请求不错误加载视觉塔；Qwen3.5 混合层按类型执行 |
| **六期：真机标定与默认启用** | `PT-PIPE-V1` | 在多台 CUDA PC/Surface 目标节点执行三轮冷启动、峰值、掉线、重连和结果等价验收；冻结安全余量 | 通过 §5 全部门；三轮稳定后才允许 UI 默认推荐 distributed-only 模式 |

顺序约束：`C2` 先解决架构无关的总容量事务，`QW3` 再接入新执行器；`C3` 解决磁盘和传输冗余，不能用它替代内存准入；多模态必须在文本 Qwen3 契约稳定后开始。没有外部真机不阻塞 C2/QW3/C3 的合成开发，但 V1 结论必须延后。

### 8.3 一期实施记录（`PT-PIPE-C1`，Completed，2026-08-13）

- 新增 `src/pipeline_model_descriptor.py`：仅读取 `config.json`、`model.safetensors.index.json` 和 `safe_open.get_slice()` 文件头；禁止 `get_tensor`，输出逐层和组件字节、架构、层前缀、工件总字节及执行器准入结果。
- `ModelManager.prepare_pipeline_model()` 新增显式 distributed-only 状态：在摘要和描述器验证成功后记录模型身份，但保持 `model/tokenizer=None`、`is_loaded=false`；普通卸载和普通模型切换均能清理该状态。
- 新增 `POST /api/models/prepare-pipeline`。它不把元数据准备伪装成完整模型加载；响应明确区分 `loaded=false` 与 `pipeline_prepared=true`。
- Scheduler 的层数、容量估算、模型身份和模型摘要可以来自描述器，不再要求主节点已经整模加载；现有 Qwen/Qwen2 下发白名单不变。distributed-only 请求只允许流水线执行，worker 未就绪或运行失败时禁止自动整模回退。
- Qwen3、Qwen3-VL、Qwen3.5 已登记元数据布局并通过真实本地工件检查，但 `pipeline_runtime_supported=false`；一期 API 不允许把它们准备成可执行流水线，避免“能识别 config”被误报成“能运行模型”。
- 回归：描述器 + 模型管理器 `99 passed / 1 skipped`；Scheduler `270 passed`；Python 全量 `2105 passed / 8 skipped`（约 5 分 43 秒）。真实多机、峰值与集群总容量验收继续标为 `Validating`。

### 8.4 二期实施记录（`PT-PIPE-C2`，开发门 Completed，2026-08-13）

- 新增 `src/pipeline_capacity.py` 纯函数求解器：消费描述器逐层字节、Embedding/Final Norm/LM Head 字节和节点当前可用容量，不打开权重、不导入 torch；每个参与节点收取运行时 reserve，CUDA 按 FP16/源字节，CPU 按 FP32 扩张，统一套用可配置安全余量 `QLH_PIPELINE_CAPACITY_SAFETY_MARGIN`（默认 `1.2`）。Final Norm 按每个实际层段节点计费，LM Head 只归属末节点。
- 求解结果是 all-or-nothing：连续覆盖 `[0,total_layers)`、Embedding/LM Head 各有唯一归属才返回 `admitted=true`；允许 master 没有层而作为控制面/Tokenizer 节点，返回 `control_only_nodes`。单机均不足但总容量足够时返回 `aggregate_only=true`；不足、容量未知、视觉/MTP 等未实现组件均返回结构化 `reason_code` 和空 assignment。
- Scheduler 新增 `get_pipeline_capacity_plan()` 与 `GET /api/cluster/pipeline-capacity`。规划使用实时 `gpu.vram_free_gb` 或 `ram.available_gb`，没有空闲容量证据时保守拒绝，不把物理总显存/RAM 冒充可用容量；`/api/cluster/layers` 在事务进行中复用同一份 capacity assignment。
- distributed-only 模式下，下发固定 `config_id + plan_id` 的 `phase=prepare`。worker 仅做本地摘要、描述器、层范围和容量复核，返回 `prepared`，不调用 `load_layer_range`；全部 prepared ACK 后主节点只加载自己的层段/Tokenizer 元数据，再下发 `phase=commit`。commit 必须命中同代 prepared 记录，才允许首次物化。
- 任一 prepare/commit error、worker 断连或主节点本地 commit 失败都会进入 abort/release 代际；worker 清理已物化层段但保留 distributed-only 描述器，禁止自动回退到主节点整模加载。事务状态和失败原因可通过容量 API 读取。
- 新增合成回归覆盖：单机都不足但聚合足够、总容量不足、CPU 扩张、主节点零层、全员 prepared 后 commit、阶段错配、worker 掉线和 abort 清理；本票定向 `279 passed` Scheduler、`102 passed / 1 skipped` ModelManager+capacity，API 模型回归 `55 passed`。

本票仍不宣称三项能力：按需 shard/bundle 尚未实现、Qwen3/Gemma4 等新架构执行器尚未准入、真实多机峰值/掉线/结果等价仍需 PT-PIPE-V1 三轮硬件验收。

### 8.5 三期实施记录（`PT-PIPE-C3`，开发门 Completed，2026-08-13）

- 新增 `src/pipeline_assignment_manifest.py`：主节点按 `config_id + plan_id + node_id + layer_range` 生成 assignment manifest，仅包含配置、Tokenizer 支持文件、过滤后的 Safetensors `weight_map` 和被分配层段实际所在 shard；manifest 同时绑定完整模型 revision `model_sha256`，并以 `manifest_sha256` 锁定清单内容。
- 新增 `/api/models/pipeline-assignment/{model_id}`，只接受可信模型 peer，且必须命中当前 `preparing` 事务、活动 `plan_id` 和精确节点/层段/Embedding/LM Head 合同；旧 `/api/models/downloadable` 全量清单接口保留给非事务兼容路径。
- `ensure_pipeline_assignment_available()` 将 worker 缓存放在 `.pipeline_assignments/<config>-<node>/` 代际目录，复用 `.part`、HTTP Range、单文件大小/SHA 校验和原子提交。prepare 阶段通过 assignment manifest 下载，不再要求 worker 先拥有整模目录；commit 继续只使用该代准备目录。
- Safetensors 描述器支持 partial assignment 检查；C3 不允许通过缺层目录冒充完整模型，层段范围、模型类型和 config 层数仍需一致。
- 回归：C3 manifest/model-sync/descriptor `12 passed`；Scheduler + API + 安全边界 + 同步专项 `357 passed`；Python 全量 `2126 passed / 8 skipped`（约 3 分 28 秒）。真实大模型传输、跨节点断点恢复和多机结果等价仍留在后续硬件门。

本票的边界：assignment manifest 目前只准入已经支持的 Qwen/Qwen2 PyTorch 执行器；视觉/MTP 组件仍 fail-closed；worker 目录中暂保留代际 assignment 缓存，空间回收策略和 Range 中断后的真实网络验收后置。

### 8.6 `PT-PIPE-C3.1` 缓存治理与恢复实施记录（开发门 Completed，2026-08-13）

- 新增 `QLH_PIPELINE_ASSIGNMENT_CACHE_MAX_MB`、`QLH_PIPELINE_ASSIGNMENT_MIN_FREE_MB` 和 `QLH_PIPELINE_ASSIGNMENT_STALE_SECONDS` 配置；下载前按 manifest 预计字节数检查磁盘保留空间及 assignment 缓存预算。
- 新增 `reconcile_pipeline_assignment_cache()`：只扫描模型目录下的 `.pipeline_assignments`，保护当前 config/正在下载目录，清理无 marker 或带 `.part` 的废弃代际，并按最旧优先回收超预算缓存；启动 Scheduler 时对当前模型执行一次 reconcile。
- 新增 `remove_pipeline_assignment_cache()`；C2 abort/release 将 `model_id + aborted_config_id + node_id` 传给 worker，失败代际立即清理，不影响旧代或用户完整模型目录。
- 过滤后的 `model.safetensors.index.json` 写入前校验实际大小/SHA；断点下载继续保留 `.part` 和 Range，清单/key 集/文件摘要任何不一致均拒绝提交。
- 回归：model sync/cache `11 passed`；C3 manifest/model-sync/descriptor `12 passed`；Python 全量基线仍为 `2126 passed / 8 skipped`。本机未执行真实大文件跨节点断线，真实网络矩阵与磁盘权限仍需后续外部门。

### 8.7 下一票（`PT-PIPE-QW3`）

1. 在隔离 Transformers sidecar 中实现 Qwen3 文本分层 adapter，保持主运行时 `transformers==4.47.x` 不变。
2. 复用 C1 描述器、C2 容量事务和 C3 assignment manifest，先完成 `enable_thinking=false` 的单机两段合成前向，再进入真实工件 smoke。
3. Qwen3-VL/Qwen3.5 视觉、MTP 和真实多机结果等价继续后置，不能因为文本 adapter 完成而放开多模态准入。

### 8.8 `PT-PIPE-QW3` Qwen3 文本 adapter 开发记录（开发门 Completed，2026-08-13）

- 新增 `scripts/model_tools/qwen3_pipeline_adapter.py` 与隔离 sidecar worker：主运行时 `transformers==4.47.x` 不变，sidecar 只接受 `transformers>=4.51`，并以 `local_files_only=true`、`trust_remote_code=false`、network-disabled、read-only 契约运行。
- adapter 在读取权重前校验 `model.layers.N` assignment key 集合、连续层范围、config 层数和 unknown/视觉/MTP key；真实 loader 使用 meta 骨架 + 过滤 index + `safe_open.get_tensor`，不调用 `from_pretrained` 整模物化，加载后断言非分配层未被物化。
- 已实现分段前向、RoPE/causal mask、官方 Transformers `DynamicCache` 与 legacy cache 兼容、LM Head；`tie_word_embeddings=true` 时末段显式复用 `model.embed_tokens.weight`，容量账本和 assignment manifest 同步计费/下发。
- tokenizer 通过 `apply_chat_template(..., enable_thinking=false)` 硬关闭 thinking；不接受该参数、产生非空 thinking 标记或 sidecar 未隔离时 fail-closed。
- 合成 Qwen3-like 模型已验证两段 logits 等价、KV cache 段内长度和 assignment 守卫；定向回归 `18 passed`。当前 `.venv-qwen3-sidecar` 实测 `transformers 4.57.6` 已就绪但缺 `torch/accelerate`，因此真实 Qwen3-4B 权重 smoke 尚未执行，不能宣称运行时准入。

### 8.9 下一票：`PT-PIPE-QW3.1` 执行 sidecar 与真实工件 smoke

1. 以 `scripts/setup_qwen3_sidecar_env.py --pipeline` 安装平台匹配的 torch/accelerate，并对 `models/qwen3-4b` 生成 C3 assignment manifest；先做单段 CPU/meta 预检，再做可用资源门控下的真实两段/三段前向。
2. 记录 full-model materialization 断言、每段 source/materialized bytes、RSS/VRAM 峰值、prefill/decode KV cache 和 `enable_thinking=false` 最终答案格式；任何资源不足或 sidecar 异常返回结构化 `resource_rejected/sidecar_failed`，不得回退主运行时整模加载。
3. 真实 smoke 通过后再接 scheduler prepare/commit；Qwen3-VL、Qwen3.5、跨节点结果等价和 PT-PIPE-V1 三轮硬件标定仍不在本票范围。

### 8.10 `PT-PIPE-QW3.1` 执行 sidecar 资源门开发记录（开发门 Completed，2026-08-13）

- 新增 `scripts/model_tools/qwen3_pipeline_smoke.py` 和隔离 `qwen3_pipeline_smoke_worker.py`，CLI 入口为 `qwen3-pipeline-smoke`。默认只读预检；只有显式 `--execute` 且资源/依赖门均通过时才允许调用 Qwen3 filtered-assignment loader，不会把 full index 当作 assignment 绕过。
- worker 使用 Safetensors `framework=np` header 路径计算所选层段的 tensor 数量、源字节、shard 数和 CPU/CUDA 预算；即使 sidecar 尚无 Torch，也能先完成资源门。资源不足返回 `resource_rejected`，执行依赖缺失返回 `runtime_unavailable`，所有结果均为结构化、read-only、network-disabled。
- 当前 `models/qwen3-4b` 实测 `layer_range=[0,1] + embedding` 选中 13 个 tensor、`979,779,072` bytes、2 个 shard；本机可用 RAM 约 `4,177,612,800` bytes，CPU 预算约 `2,888,340,685` bytes，资源门通过。隔离 venv 为 `transformers 4.57.6`，但缺 `torch/accelerate`，故本次真实物化未执行并正确返回 `runtime_unavailable`；没有整模 fallback。
- 新增回归覆盖依赖缺失仍能计量 assignment、容量不足先于依赖门拒绝、控制器结构化转发和 CLI 退出码；QW3 smoke/adapter/sidecar 专项 `13 passed`，`py_compile` 与 `git diff --check` 通过。全量 Python 回归 `2140 passed / 8 skipped`（约 218 秒）。

### 8.11 下一票：`PT-PIPE-QW3.2` sidecar 执行依赖与真实 assignment smoke

1. 仅在用户明确允许且存在平台匹配 wheel 时，通过 `scripts/setup_qwen3_sidecar_env.py --pipeline` 安装 Torch/Accelerate；主运行时 `transformers==4.47.x` 不变，禁止 sidecar 自动联网升级或回退主进程。
2. 复用 C3 assignment manifest 生成的 filtered index，在资源门通过后执行 Qwen3-4B 单段 CPU/CUDA prefill，再扩展两段/三段 decode、KV cache、峰值和 `enable_thinking=false` 格式门；完整索引、缺层、未知 key 或峰值超限继续 fail-closed。
3. 真实 smoke 通过后才接 Scheduler prepare/commit 和跨节点传输；Qwen3-VL/Qwen3.5、跨节点结果等价与 PT-PIPE-V1 三轮硬件标定继续后置。

### 8.12 `PT-PIPE-QW3.2` 真实 sidecar assignment smoke（开发门 Completed，2026-08-14）

- `.venv-qwen3-sidecar` 已按隔离原则补齐 `torch 2.13.0+cu126`、`accelerate 1.14.0`、`psutil 7.2.2`；主运行时仍为 `transformers 4.47.1`，未被升级。`setup_qwen3_sidecar_env.py` 新增 `--torch-index-url`、`--torch-spec` 及 `QLH_QWEN3_TORCH_INDEX_URL/QLH_QWEN3_TORCH_SPEC`，避免 `--pipeline` 无意安装 CPU Torch。
- worker 在执行前创建临时 filtered assignment：只复制 config/index，权重 shard 优先硬链接，跨卷才复制；执行结束自动清理，不修改 `models/qwen3-4b`，不把 full index 交给 loader。
- 真实 CPU 首段 `layer_range=[0,1] + embedding` 通过：13 tensor、`979,779,072` source bytes、FP32 materialized `1,959,558,144` bytes、RSS delta 约 `1.98 GB`，prefill 18 tokens + decode 1 token，KV cache 长度 `19`，`thinking_disabled=true`，`full_model_materialized=false`。
- 真实 CPU 末段 `[35,36] + lm_head` 通过，tied embedding 的 logits 路径和 BF16 物化通过：source `979,779,072` bytes、materialized `979,779,072` bytes、`full_model_materialized=false`。
- 真实 CUDA 中间段 `[1,2]` 通过：12 tensor、source/materialized `201,866,752` bytes，RTX 4060 Laptop 上 CUDA peak allocated `212,327,936`、reserved `216,006,656` bytes，未触发整模加载。首段 CUDA 在可用 RAM 约 `1.41 GB` 时因预算约 `1.71 GB` 返回 `resource_rejected`，没有物化；这是预期资源门行为。
- 回归新增 tokenizer/真实 staging/KV/RSS-CUDA 峰值/缓存长度和安装参数覆盖；QW3 smoke + adapter `9 passed`，真实命令均通过或按资源结构化拒绝。QW3.2 仍不宣称跨节点结果等价或生产准入。

### 8.13 下一票：`PT-PIPE-QW3.3` 两段/三段串联与跨节点 KV 合同

1. 复用 C3 assignment manifest，在同一隔离 worker 内加载首段和末段/中段 filtered assignment，验证 hidden-state handoff、两段/三段 logits 与单模型参考的等价性；继续断言每段 `full_model_materialized=false`。
2. 为跨节点协议固定 KV cache 的段索引、序列长度、dtype/device、prefill/decode 代际和失效清理字段；覆盖缓存缺失、长度不一致、跨段设备不匹配和中途 abort 的结构化拒绝。
3. 只有串联 smoke 通过后才接 Scheduler prepare/commit；真实双机传输、断线恢复、Qwen3-VL/Qwen3.5 和 PT-PIPE-V1 三轮硬件标定继续后置。

### 8.14 `PT-PIPE-QW3.3` 两段/三段串联与 KV 合同（开发门 Completed，2026-08-14）

- 新增 `qwen3_pipeline_chain.py` 与 `qwen3_pipeline_chain_smoke.py`。链路只接受 2 或 3 个连续段，首段唯一拥有 Embedding，末段唯一拥有 LM Head；段间缺层、重叠、空洞、组件归属错误均在物化前拒绝。
- 每段先独立通过 Safetensors header/本机 RAM-CUDA 预算门，再在隔离 sidecar 内按 `prefill -> release -> decode` 串行加载。权重段用临时 filtered assignment，硬链接优先，阶段结束清理；6 次真实段加载均报告 `full_model_materialized=false`。
- 新增 hidden handoff 合同：`chain_id`、段索引、`[batch, sequence, hidden]`、序列长度、dtype、device；新增 KV 合同：段层范围、batch、序列长度、prefill/decode phase、generation 和设备类型。字段不一致、缓存长度不符或跨段设备不匹配 fail-closed，不传输 prompt/权重/整模 KV。
- 真实 `models/qwen3-4b` 三段 CPU smoke：`[0,12]+embedding / [12,24] / [24,36]+lm_head`，tokenizer `enable_thinking=false`，prefill `18` tokens，decode 后每段 KV `19` tokens；两个 hidden handoff 均 `[1,18,2560]`，prefill/decode logits shape 分别 `[1,18,151936]` / `[1,1,151936]`，执行与清理通过。
- QW3 chain/adapter/smoke/sidecar 专项 `18 passed`，`py_compile` 和 `git diff --check` 通过；本票仍不宣称跨节点传输等价、Scheduler 已接入或生产准入。

### 8.15 下一票：`PT-PIPE-QW3.4` Scheduler prepare/commit 协议模拟与跨节点故障矩阵

1. 将 chain plan、filtered assignment revision、hidden/KV contract 摘要接入现有 C2 `prepare -> prepared ACK -> commit -> ready` 事务，但先保持 dry-run/仿真，不触发真实远端模型加载。
2. 增加跨节点合同签名、序列化上限、缓存代际和 abort/release 清理矩阵；覆盖缺段、重复 ACK、合同篡改、KV 长度/设备不一致、超时和断线重试，全部结构化拒绝且不整模回退。
3. 协议模拟稳定后再做单机 loopback HTTP/Range 传输；真实双机、异构设备结果等价、Qwen3-VL/Qwen3.5 和 PT-PIPE-V1 三轮标定继续后置。

### 8.16 `PT-PIPE-QW3.4` Scheduler dry-run 与故障矩阵（开发门 Completed，2026-08-14）

- 新增 `src/qwen3_pipeline_transaction.py`，冻结 Qwen3 专用 `prepare -> prepared ACK -> commit -> ready` 与 `abort -> release` 状态机。Scheduler 仅保存独立 `_qwen3_pipeline_dry_run`，提供 begin/ACK/retry/expire/abort/release/status 方法；不复用生产 `_pipeline_load_transaction` 准入，不调用 TCP、`load_layer_range` 或 sidecar，因此 Qwen3 生产入口仍 fail-closed。
- canonical 合同锁定 `config_id/plan_id/generation`、模型 revision SHA-256、2/3 个连续段、每段 C3 `assignment_manifest_sha256`、segment SHA-256、hidden handoff SHA-256 和 node-local KV contract SHA-256；合同上限 `64 KiB`、ACK 上限 `8 KiB`，禁止 prompt/messages/input IDs/hidden tensor/KV tensor/logits/weights 字段进入控制面。
- prepare ACK 绑定节点、层段、模型/合同/manifest/hidden/KV 摘要和实时可用容量；commit ACK 还必须证明本段空缓存基线：正确 segment/layer range/generation/dtype/device、`sequence_length=0`、`phase=empty`、`cleared=true`、`full_model_materialized=false`。
- 故障矩阵覆盖：缺段/层段空洞、合同篡改、未知节点、manifest/hidden/KV 摘要不符、KV 长度/设备/代际不符、容量漂移、超限 ACK、同 ACK 幂等、改变后的重复 ACK、retry、timeout、worker disconnect 和逐节点 release ACK 清理；任一不一致全体 abort，不产生整模回退。
- 真实 `models/qwen3-4b`（revision `2c54d5a09e7e92d4f5126b92a5a457448c9593e6`）生成三段 C3 manifest 并建 dry-run 合同：`[0,12] e7d344cd... / [12,24] acd10827... / [24,36] a5b24b9f...`，合同 SHA-256 `c7bc9f48...`、大小 `2,959` bytes、3 条 prepare 消息；只做本机 header/hash 工作，未传输、未物化。
- QW3 transaction + chain + smoke + adapter + sidecar + manifest + capacity + descriptor + sync + API + Scheduler 扩展回归 `396 passed`，协议专项 `20 passed`；Python 全量 `2165 passed / 8 skipped`。本票不声称合同摘要替代 TCP HMAC/节点认证，也不声称真实双机、Range 传输或生产准入。

### 8.17 `PT-PIPE-QW3.5` 单机认证 loopback 与 Range 故障矩阵（开发门 Completed，2026-08-14）

- 新增 Qwen3 专用 TCP 消息路由，只允许完成集群 HMAC 注册且实际 socket peer 为 loopback 的连接。每个控制帧再绑定 peer、contract/generation/phase、payload SHA-256、时间窗口和 nonce；ACK 同样签名，变更后的 nonce 重放和非已认证 peer 均 fail-closed。
- worker 在 prepare 中先获取与签名合同完全一致的 C3 assignment manifest，再以无代理、禁止跳转的 loopback HTTP 请求只读 Safetensors 头部。Range 强制 `206 + Content-Range + Content-Length + SHA-256`，最多 3 次断点续传，单次合同上限 8 MiB；真实 `models/qwen3-4b` 第一 shard 仅读取 20,008 bytes 头部，未读取 tensor payload。
- Scheduler 已能驱动 `prepare -> commit -> ready -> release`，并在部分下发失败、超时、ready 后断线、重连与 release ACK 丢失时保留 abort/release 清理代际。CPU 只使用可用 RAM 容量池，CUDA 只使用 `mem_get_info()` 空闲 VRAM，不跨池借容量。全路径仍锁定 `dry_run=true / weight_materialization=false / full_model_fallback=false`。
- 故障矩阵覆盖 401/403/416、跳转、错误 Range、截断/续传、超时、SHA 篡改、manifest 拒绝/篡改、基址变更、容量不足、幂等/重放、断线和重连清理。QW3.5 专项 `48 passed`；Scheduler/TCP/API/manifest 扩大回归 `481 passed`。最终在 `.venv-test` 内以固定 4 worker 运行 unit 通道：`2196 passed / 6 skipped` （约 88 秒）。
- 本票不包含权重物化、hidden tensor 网络传输、真实双机结果等价或生产准入；Qwen3-VL/Qwen3.5 也不因此放开。

### 8.18 `PT-PIPE-QW3.6` 隔离 sidecar 的 node-local 执行生命周期（开发门 Completed，2026-08-14）

- 新增 `Qwen3PipelineSidecarSession`：主进程使用有界 JSONL 控制帧管理独立 Python 子进程，强制 Transformers/Torch 离线环境、取消系统代理、请求/响应 256 KiB 上限和真实读取超时。返回只包含资源、账本和执行汇总，不回显模型路径、权重或 hidden tensor。
- 新增持久 `qwen3_pipeline_runtime_worker.py`：prepare 先重算本地 assignment manifest 、资源门和 `enable_thinking=false` tokenizer，再以硬链接优先创建 filtered assignment；commit 只调用 `load_qwen3_layer_assignment` 物化已分配段；release/abort 释放 adapter、清空 CUDA 缓存并移除临时目录。不调用 `from_pretrained` 整模路径，不会回退主运行时整模加载。
- Qwen3 合同增加显式 `execution_mode=node_local_sidecar`；默认 `metadata_only` 路径保持 QW3.5 行为不变。sidecar 控制帧中明确区分 `weight_materialization` 和 `segment_materialized`，commit ACK 必须证明只物化本段，不是全模。Scheduler 只通过显式 `begin_qwen3_pipeline_sidecar()` 进入该模式；TCP 断线会主动 abort sidecar 并回收。
- sidecar/session/loopback/transaction 专项和 QW3 扩大回归共 `488 passed`；固定 4 worker 的完整 unit 通道为 `2201 passed / 6 skipped`（约 86 秒），`py_compile`/`git diff --check` 通过。本票的真实双机执行、CUDA 峰值和结果等价仍未验收；这些不能被合成 session 测试代替。

### 8.19 `PT-PIPE-QW3.7` 单机多 sidecar hidden handoff 与 KV 执行（开发门 Completed，2026-08-14）

- 新增 `Qwen3PipelineMultiSidecar`，可从 canonical `node_local_sidecar` 合同创建 2/3 个 sidecar session，按 `prepare -> commit -> prefill -> decode -> release` 串行推进；任何阶段异常都会对全部 session 执行 abort。
- hidden/KV 不进入 JSONL 控制帧，sidecar 只读写 controller-owned 本机 artifact；控制帧绑定 artifact root、输入/输出 SHA-256 与字节数、chain/segment、shape、dtype/device、sequence length 和 generation。输出证据拒绝整模物化，且摘要不符、工件越界、hidden shape 或 KV 长度变化均 fail-closed。
- 增加取消和重启恢复入口，按 chain token 回收遗留 artifact；release/abort 返回清理证据。`from_contract()` 将 QW3.6 canonical 合同绑定到本机多 sidecar 编排，未改变默认 metadata-only 或跨节点生产准入。
- 多 sidecar/runtime worker 专项回归 `10 passed`；QW3 扩大回归 `495 passed`；固定 4 worker 的完整 unit 通道为 `2208 passed / 6 skipped`（约 105 秒），`py_compile`/`git diff --check` 通过。
- 本票仍未验收真实 Qwen3-4B CUDA 结果等价、异构设备转换、跨机 hidden tensor/KV 传输、网络重连或生产准入；本机 artifact 测试不能替代这些门。

### 8.20 `PT-PIPE-QW3.8` Scheduler 本地链入口与 CPU parity gate（开发门 Completed，2026-08-14）

- 新增主节点专用、显式 opt-in 的 Scheduler 本地链入口和单体/scheduler-svc 等价 API，覆盖 begin、prefill、decode、parity、release、cancel 与只读状态；默认聊天、metadata-only 和既有生产流水线均未接入该入口，返回固定 `production_admitted=false`。
- chain/config/plan/generation/phase/segment count/cleanup/parity 元数据投影到用户主节点 SQLite；不保存绝对路径、tensor、logits 或工件正文。启动/状态查询会把无内存会话但仍处活动阶段的记录收敛为 `recovered_aborted`，并按 chain token 只清理本链遗留工件；同合同重放和陈旧 generation 均被 fencing。
- CPU parity gate 对 2/3 段 prefill/decode 的最终 logits 做有界容差比较，同时复核逐段 artifact 字节数/SHA-256、`full_model_materialized=false`、KV phase/generation/sequence length 和 hidden handoff shape。失败会取消全部 sidecar、清理工件并返回结构化拒绝，明确 `full_model_fallback=false`。
- 合成 Qwen3-like CPU 小模型使用同一组权重完成整链参考与 2/3 段执行对照；另覆盖数值不一致、工件篡改、代际不符、SQLite 白名单、重启恢复、重复提交和主从权限。QW3.8 新增专项 `15 passed`，QW3 专项 `102 passed`；固定 4 worker 的完整 unit 通道为 `2223 passed / 6 skipped`（约 112 秒）。
- 全量复核同时发现并修复 QW3.7 的独立 Worker 导入缺口：运行时 segment/KV 合同移入 `src/qwen3_pipeline_contract.py`，打包/独立进程不再依赖仓库根或 `scripts.model_tools` 可导入。真实 Qwen3-4B CUDA parity、异构转换、跨机 hidden/KV 和生产准入仍未开放。

### 8.21 `PT-PIPE-QW3.9` 网络工件传输合同与 loopback 故障矩阵（开发门 Completed，2026-08-14）

- 新增 Qwen3 专用 HMAC 工件票据，完整绑定 authenticated peer、chain、generation、phase、from/to segment、字节数、SHA-256、过期时间与 nonce；接收计划、状态和 receipt 均为严格 metadata-only，不返回 tensor、KV、本机绝对路径或票据。票据仅服务一个 transfer session，不与 SD 图像 CAS/SQLite 混用。
- 新增有界接收存储和 loopback 客户端：最多 4 MiB 顺序 PATCH、逐块落盘并 `fsync`、`Upload-Offset` 背压、`.part` 断点恢复、最终全文件 SHA-256/字节复核和同卷 `os.replace` 原子提交。客户端强制 loopback、禁系统代理和重定向，控制响应上限 64 KiB；controller 不聚合完整 artifact 字节。
- 内部 FastAPI adapter 只读取认证传输层注入的 `request.scope.qlh_authenticated_peer_id`，不接受 caller header 声称节点身份。精确重复块幂等；跨 peer/跨 session 被拒绝且不得破坏合法 staging；错序、变化重放、越界和摘要不符立即失败并清理。连接中断保留有界 `.part` 供续传，取消或 TTL 到期定向回收。
- 2/3 段、prefill/decode 的实际 loopback GET/PATCH/commit 矩阵，以及断线后 ACK 丢失、截断提交、重放、越权、错序、摘要不符、取消、过期、超限和路径逃逸均已覆盖。新增专项 `14 passed`，QW3 专项 `107 passed`；固定 4 worker 的完整 unit 通道为 `2237 passed / 6 skipped`（约 129 秒）。
- 本票只落下独立内部 router/runtime 与传输 fixture，尚未在生产 `api_server` 注册，也未由 Scheduler 向真实远端 peer 签发计划；因此不构成真实双机 hidden/KV 传输、CUDA parity 或生产准入。

### 8.22 `PT-PIPE-QW3.10` Scheduler 认证网络接线与模拟多节点链（开发门 Completed，2026-08-14）

- 新增 `qwen3_pipeline_peer_auth.py` 请求级 HMAC proof：proof 绑定 HTTP method/path、Bearer 摘要、时间窗口和 nonce，并只把认证层注入的 peer identity 交给 transfer/data-plane；新增 `qwen3_pipeline_control.py`，控制请求体以规范 JSON + SHA-256 绑定，目标端不得从普通 header 或未提交 topology 取得权限。
- `Qwen3NetworkTransferCoordinator` 将活动 canonical contract、generation、prefill/decode 阶段、相邻 source/target topology 与 transfer session 互相 fencing；`Qwen3PipelineMultiSidecar` 现在显式产生 path-free `local`/`network` artifact reference，目标端摘要复核、原子提交后才向下一段暴露内部路径，控制元数据不携带路径、票据或 tensor。
- Scheduler 以显式 `configure_qwen3_artifact_transfer(..., network_coordinator=...)` 接入 runtime、peer verifier 和可选 handoff transport；`scheduler_svc_http` 仅在显式配置时注册内部 control/data router，`production_admitted` 仍为 `false`，默认 API/聊天路径不进入该支线。
- 新增 `tests/helpers/qwen3_network_node.py` 同机目标进程，2/3 节点 prefill/decode、断线续传、目标失败、全链取消、阶段未完成拒绝、body 重绑定、重启后孤儿回收/陈旧合同拒绝均通过。专项网络 `14 passed`，QW3 支线 `121 passed`，Python 全量 `2252 passed / 8 skipped`；`py_compile` 与 `git diff --check` 通过。
- 进程 helper 用固定 allow-list 模拟已注册 peer；真实 TCP 注册生命周期、远端 sidecar 在目标节点执行、跨机吞吐、CUDA 数值等价、异构设备转换和生产路由仍未关闭，不能宣称真实双机或生产准入。

### 8.23 下一票：`PT-PIPE-QW3.11` TCP peer 生命周期与目标端 sidecar 执行边界

1. 将 QW3.10 control client 的 peer allow-list 替换为现有 TCP HMAC 注册/断开事件投影，覆盖注册、撤销、重连和源节点身份变化；控制请求、artifact ticket 与 Scheduler transaction 必须共享同一 live epoch。
2. 把 network artifact reference 的消费端下沉到目标节点 sidecar：源端只提交 path-free reference，目标端本地解析并执行下一段，返回 hidden/KV/artifact 的摘要和形状合同；目标进程不得把本地绝对路径回传控制面。
3. 保持 2/3 节点 CPU 合成链和全链清理回归；真实双机网络、CUDA parity、异构转换、吞吐/时延和生产准入继续作为独立外部门，不下载新模型。

### 8.24 `PT-PIPE-QW3.11` TCP epoch 与目标侧消费边界（开发门 Completed，2026-08-14）

- `src/tcp_comm.py` 为每次成功的 HMAC 注册分配单调 `registration_epoch`；服务端暴露当前已确认 peer 的 epoch，客户端在注册成功和重连后更新本地 epoch。旧连接断开或身份变化不会继续复用旧 epoch。
- `src/qwen3_pipeline_peer_auth.py` 升级 proof schema，proof 绑定 `peer_epoch`；Scheduler 在显式配置 Qwen3 data plane 时同时投影 peer 身份与 epoch。测试 helper 改为可注册/撤销/重连的 TCP 注册事件投影，不再把静态 allow-list 当作生命周期事实。
- transfer ticket、descriptor、状态/receipt 和 network coordinator 授权均绑定 epoch。peer 重连后，旧 ticket 即使仍在 TTL 内也会 fail-closed；控制请求、artifact ticket 和活动 network contract 使用同一 live epoch。
- `POST /internal/v1/qwen3/network-control/consume` 将已提交 artifact 的解析和执行留在目标进程。目标 executor 才能看到目标本地路径，返回值仅含 artifact SHA/bytes、hidden handoff、KV shape/phase 摘要，不返回路径、ticket、tensor 或完整模型；`Qwen3NetworkHandoffTransport.consume_target()` 提供统一调用边界。
- 回归新增 epoch 变更/陈旧 proof、目标消费 path-free 响应和注册 projection 覆盖；QW3 网络/传输专项当前 `30 passed`。本票仍只通过 CPU/合成工件开发门，真实双机网络、CUDA parity、异构转换、吞吐/时延和生产路由继续后置。

### 8.25 下一票：`PT-PIPE-QW3.12` 目标侧真实 sidecar 串联与消费后工件生命周期

1. 将 `consume` 接到目标 sidecar session 的真实 filtered assignment/hidden/KV 执行，要求输入 artifact 只在目标进程内解析，并把下一跳输出重新登记为新的 path-free network reference。
2. 增加消费中断、重复 consume、目标重启、epoch 变化和输入/输出摘要不一致的事务回收矩阵；消费失败必须回收当前 transfer 和目标 sidecar，不得回退整模。

### 8.26 `PT-PIPE-QW3.12` 目标侧 sidecar 消费闭环（开发门 Completed，2026-08-14）

- `Qwen3NetworkSidecarExecutor` 将目标节点的 sidecar session 接到 network `consume`：filtered assignment、hidden handoff 和 KV artifact 只在目标进程本地解析，decode 复用目标段 prefill KV；控制面不接收本地路径、tensor 或完整模型。
- coordinator 新增 `receiving -> committed -> consuming -> consumed` 状态机。相同消费合同重试返回缓存结果，合同变化和并发重复消费 fail-closed；目标 executor 异常、取消、sidecar 清理失败和输入摘要不一致都会回收 transfer/输出工件，禁止 full-model fallback。
- 目标 executor 产出的文件由 coordinator 在目标 artifact root 内复核字节数/SHA 并登记新的 path-free `output_reference`，绑定下一段 source/target、chain、generation、phase 和 segment boundary；目标重启时清理遗留 `qwen3-consume-*` 工件，peer epoch 前进时回收旧 transfer 与已登记输出。
- `Qwen3NetworkHandoffTransport.transfer_reference/transfer_and_consume` 提供不解析目标路径的上传与消费 API；旧 `transfer/resolve` 保留给已有同机模拟链，真实远端下一跳数据搬运、双机 CUDA parity、异构转换和生产路由仍后置。
- 新增目标执行适配、重复消费、失败回收、epoch fencing、重启工件回收、path-free transfer-and-consume 及 sidecar executor 专项；QW3 专项当前 `144 passed`，真实模型/双机/CUDA 验收未开启。

### 8.27 下一票：`PT-PIPE-QW3.13` path-free 输出 reference 的远端下一跳传输

1. 基于 `output_reference` 增加源节点目标端有界输出读取/下一目标上传协议，保持 source/target 两端都不交换本地路径，覆盖 2/3 节点 prefill/decode 和断线续传。
2. 把 target sidecar session 的持久 generation/KV 账本接入下一跳重试，补目标重启后的 reference 失效、输出摘要复核和逐节点 release；真实双机/CUDA parity 与生产准入继续独立后置。
3. 在不改变主运行时 `transformers==4.47.x` 和不引入硬件前置的前提下，先完成 2/3 节点 CPU 合成链，再把真实双机、CUDA parity、异构 dtype/device 转换交给独立验收票。

### 8.28 `PT-PIPE-QW3.13` path-free 输出下一跳数据面（开发门 Completed，2026-08-14）

- `Qwen3NetworkTransferCoordinator.read_output_chunk` 以 `output_reference`、认证 peer 和有界 offset/limit 读取源端登记输出；每个 chunk 都重新核对文件 bytes/SHA，路径不存在、摘要变化或 peer 越界会立即使 reference 失效并定向清理。
- artifact transfer client 新增 `upload_chunks`，统一复用 GET offset、顺序 PATCH、断线续传和 POST commit；本地路径上传也改走该状态机。目标端新增二进制 output route，仅返回 bytes、offset/total、SHA 和 EOF 响应头，不返回路径或完整控制 JSON。
- `Qwen3NetworkHandoffTransport.transfer_registered_output` 将源端 chunk reader 接到下一目标的既有 `.part`/receipt 管线，提交前后复核 chain、generation、phase、segment boundary、bytes/SHA；连接中断保留 target staging 与 source reference 供重试，协议/摘要/拓扑错误才回收。
- CPU 合成覆盖 2/3 节点 path-free 下一跳、chunk provider 元数据绑定、输出摘要篡改、offset/EOF、逐条 release；传输/网络专项当前 `36 passed`。尚未接入真实双机、CUDA parity、持久 generation/KV ledger 和生产路由，不能据此宣称远端模型等价。

### 8.29 下一票：`PT-PIPE-QW3.14` 持久账本与多阶段重试收口

1. 将 generation/KV ledger 与 `transfer_registered_output` 的计划、确认、重试 offset 和 target restart projection 持久化绑定；target 重启后只接受仍在活动合同且摘要可复核的 reference。
2. 增加 source output reference 的显式逐节点 release/lease、重复提交幂等、跨阶段 prefill/decode 和 2/3 节点完整重试矩阵；连接失败不得丢失可续传状态，过期/撤销必须清理两端。
3. 继续保持主运行时 `transformers==4.47.x`、`full_model_materialized=false` 和无硬件开发门；真实双机/CUDA parity、吞吐/时延与生产准入另开验收票。

### 8.30 `PT-PIPE-QW3.14` 持久账本与多阶段重试（开发门 Completed，2026-08-14）

- 新增用户主节点 SQLite `qwen3_network_ledger_v1` 投影，保存 active contract、generation、phase、transfer descriptor/状态、确认 offset、KV shape/phase 摘要和 output lease/next transfer 关系；不保存路径、ticket、tensor 或完整模型。
- Scheduler 配置 network coordinator 时自动挂接 ledger load/save。coordinator 在 begin/commit/consume/progress/lease/output-progress/commit-output/release 等边界同步投影；同一 lease/commit 可幂等重放，已确认 offset 禁止回退。
- 目标/源节点重启加载账本后，旧 transfer/output 记录 fail-closed 标为 `invalidated`，旧消费输出不复活；仍可在相同 canonical contract/generation 下重新 activate，陈旧 generation 继续拒绝。prefill 消费完成后允许 phase finish 并进入 decode，KV contract 保留 generation/phase 摘要。
- `transfer_registered_output` 在同一 transport 内复用 pending plan/ticket，断线后的再次调用从目标已确认 offset 继续；连接失败保留 lease/staging，摘要、协议、拓扑和过期错误同时回收两端。显式 `release_registered_output` 完成源 output 逐节点释放。
- 账本/网络专项 `44 passed`，QW3 全线专项 `153 passed`。真实双机、CUDA parity、真实模型、生产路由和生产准入仍未开启。

### 8.31 `PT-PIPE-QW3.15` 实施计划（已执行，2026-08-14）

1. 在现有独立进程 helper 上接通 `consume -> output_reference -> transfer_registered_output -> consume` 自动串联，覆盖 2/3 节点 prefill/decode、KV 代际和 source output release，不再只由测试逐步调用底层 API。
2. 增加 target restart、peer epoch 变化、ticket 过期、重复提交和中途撤销的端到端矩阵；对不可恢复错误输出可审计的 ledger terminal state，禁止残留 lease/staging。
3. 继续保持主运行时 `transformers==4.47.x`、`full_model_materialized=false` 和无硬件开发门；真实双机/CUDA parity、吞吐/时延和生产准入另开验收票。

### 8.32 `PT-PIPE-QW3.15` 多进程端到端 sidecar 串联（开发门 Completed，2026-08-14）

- `Qwen3NetworkHandoffTransport.execute_target_chain` 已把 `consume -> output_reference -> transfer_registered_output -> next consume -> source release` 收口为一次受 canonical contract、generation、phase、segment topology 和 peer authorization 约束的调用，覆盖 2/3 节点 prefill/decode 与 KV generation；失败按 connection retry 或 fail-closed cleanup 分类。
- `tests/helpers/qwen3_network_node.py` 新增用户主节点 ledger 的 JSON 投影测试适配器和目标进程 synthetic executor。跨进程测试只让目标 executor 读取本地 artifact，返回控制面的是摘要、bytes、shape、phase 等 metadata；不交换路径、ticket 或 tensor。
- 目标节点使用同一 ledger 重启后，旧 transfer/output 记录被标记为 `invalidated`，output reference 不可重放；孤儿消费文件、`.part` staging 和旧 lease 被清理。当前已覆盖 happy path 与 target restart，TTL、peer epoch、重复提交、中途撤销的完整矩阵仍待下一票。
- `.venv-test` 验证：网络专项 `28 passed`，ledger/transfer/sidecar/state 合并 `53 passed`，QW3 全线 `156 passed`；此前 `pytest-timeout` 错误来自主环境调用嵌套 pytest，`scripts/run_simulation.py` 现优先选择仓库 `.venv-test`。

### 8.33 下一票：`PT-PIPE-QW3.16` 故障矩阵与真实 sidecar CPU smoke

1. 在独立进程中补齐 peer epoch 变化、ticket TTL、重复 submit/commit、中途 revoke、断线重试和 ledger terminal state，逐项断言 output lease、transfer record 与 `.part` staging 无残留。
2. 将 synthetic target executor 替换为已经存在的隔离 sidecar executor，先以无硬件依赖的 Qwen3 assignment CPU smoke 验证真实 hidden/KV 输入输出，再决定是否进入真实 CUDA/双机验收。
3. 保持主运行时 `transformers==4.47.x`、sidecar 隔离、`full_model_materialized=false` 和禁止 full-model fallback；生产 API 路由、双机/CUDA parity、吞吐时延和生产准入继续后置。

### 8.34 `PT-PIPE-QW3.16` 故障矩阵与真实 sidecar CPU smoke（开发门 Completed，2026-08-14）

- `Qwen3ArtifactReceiver.session_status` 与 `Qwen3NetworkTransferCoordinator.cleanup_expired` 已接通。过期 receiver session 会被 coordinator 回收并写入 ledger `expired`；peer registration epoch 前进仍由 `_fence_peer_locked` 回收 transfer，并以 `invalidated` 终态保存。未配置 network coordinator 的普通 transfer runtime 继续保留原有 TTL 行为。
- `/internal/v1/qwen3/artifact-transfer/status` 和受保护数据路由优先调用 coordinator reconciliation，因此轮询/访问即可完成过期清理；`.part`、committed artifact、sidecar output 和 output lease 均按终态清理。ledger 恢复时允许 `expired/failed` 作为已有终态，不会被误写成新的活动 transfer。
- 独立进程新增 TTL 过期、注册 epoch 变化、旧 transfer 不可复活和终态 ledger 断言；控制面不返回路径、ticket 或 tensor。网络专项 `30 passed`，QW3 全线 `158 passed`。
- 真实隔离 sidecar CPU assignment smoke 已用 `models/qwen3-4b` `[0,1]+embedding` 完成：13 tensors、`979,779,072` source bytes，`transformers 4.57.6`、`torch 2.13.0+cu126`、`accelerate 1.14.0`，prefill/decode KV 和 `full_model_materialized=false` 通过。该 worker 的 forward 仍是受控 synthetic forward，未宣称真实多节点结果等价。

### 8.35 下一票：`PT-PIPE-QW3.17` 真实 network sidecar 进程链

1. 给独立进程 helper 增加真实 `Qwen3NetworkSidecarExecutor` 接线，用真实 torch artifact 完成 2 节点 CPU prefill/decode；随后扩展 3 节点 output reference、KV generation 和源端 release。
2. 加入 PATCH 后断线、重复 status/commit、revoke 中断和 `.part` offset/ledger progress 对照，确认重试只续传确认 offset，不重复物化或保留旧 lease。
3. 保持 sidecar 独立环境、主运行时 `transformers==4.47.x`、`full_model_materialized=false` 和禁止整模回退；真实 CUDA/双机 parity、吞吐时延和生产准入继续后置。

### 8.36 `PT-PIPE-QW3.17` 真实 network sidecar 进程链（开发门 Completed，2026-08-14）

- 独立进程 helper（`tests/helpers/qwen3_network_node.py`）新增真实 executor 接线：`--sidecar-model-path/--sidecar-python/--sidecar-layer-range/--sidecar-has-embedding/--sidecar-has-lm-head` 时进程内构造隔离 `Qwen3PipelineSidecarSession`（真实 `.venv-qwen3-sidecar`）并接 `Qwen3NetworkSidecarExecutor` 到 consume 边界。
- **修复 `Qwen3PipelineSidecarSession._start` 重复启动 bug**：每次 `_exchange` 都 Popen 新 worker，导致 prepare 的 phase 在 commit 时必然丢失（"commit does not match a prepared assignment"）。加存活 worker 守卫后 prepare→commit→release 单 worker 全生命周期通过。
- **错误透传改进**：`consume_transfer` 的 sidecar 失败原因（错误码/消息，无路径/tensor）现随 `qwen3_network_execution_failed` 透传，远端不再只剩笼统消息。
- 新测试 `tests/test_qwen3_network_sidecar_chain.py`（真实工件门，缺失 `models/qwen3-4b` 或 sidecar venv 整文件 skip）：
  - **2 节点真实链**（node-b `[0,1]+embedding`）：真实 torch artifact 的 prefill→decode、KV 契约（prefill 4 + 新 token 1 = 5）、`full_model_materialized=false`、无段间 release（末段输出保留）；
  - **重复 consume 幂等**（同合同同参数返回缓存结果，输出 reference 一致）；
  - **revoke 中断清理**（取消后无 `qwen3-consume-*` 残留，已撤销 transfer 再 consume fail-closed）；
  - **3 节点链**（node-b `[0,1]+embedding` → node-c `[1,2]` 级联 output reference 与 KV generation 交接）：**RAM 腾出后实测通过（2026-08-14 晚，可用 9.7GB）**，`prefill/decode` 各 2 段执行、段 1 输出转发前 release 一次、`full_model_materialized=false`；decode 依赖 prefill KV 的跨阶段保留由 executor KV 副本承载（见下）。
- **修复 `Qwen3NetworkSidecarExecutor` KV 生命周期**：prefill 段间转发完成后传输层 release 会删除 output 文件（传输副本），而 decode 仍需 prefill 的 `past_key_values`——executor 现为 KV 单独保留副本（`qwen3-kv-*.pt`，cleanup 时一并清理），decode 不再丢失 KV 载体（3 节点链实测暴露并修复）。
- 回归：sidecar/network/真实链专项当前 `39 passed / 1 skipped`（默认 `-n 4` 由跨 worker 文件锁串行真实模型测试）；QW3 全线 `159 passed / 3 skipped`。QW3.17 全程保持 sidecar 独立环境、`transformers==4.47.x` 主运行时、禁整模回退；真实 CUDA/双机 parity、吞吐时延与生产准入继续后置。

### 8.37 `PT-PIPE-QW3.18` PATCH 断线续传矩阵与真实链收口（开发门 Completed，2026-08-14）

- `Qwen3ArtifactTransferClient.upload` 现在接受 `progress_callback`；普通 `transfer` 与 `transfer_reference` 将每次 receiver 确认的 offset 写入目标 coordinator ledger。进度更新保持单调；已提交 transfer 在重复 status/commit 重试时允许终值 offset 重放，不会把 `committed` 回退成 `receiving`。
- 真实 helper 测试模拟“目标已持久化 PATCH、客户端丢失 ACK”：150,000 字节工件通过 GET offset 续传，最终文件与源 SHA/字节一致，ledger `received_bytes=size_bytes/status=committed`；同一 `commit_reference` 连续调用两次返回完全相同引用。既有 revoke/TTL/epoch 终态和 `.part` 清理继续通过。
- 回归：QW3 全线 `161 passed / 3 skipped`（164 项，`.venv-test` 默认 `-n 4`）；sidecar/network/真实 helper 合并专项 `40 passed / 2 skipped`。真实 helper 测试继续使用跨 xdist worker 文件锁，避免低 RAM 并发假失败；全程保持 sidecar 独立环境、主运行时 `transformers==4.47.x`、`full_model_materialized=false`，不宣称真实双机/CUDA parity 或生产准入。

### 8.38 `PT-PIPE-MM1.1/.2` 多模态合同与 transfer binding（开发门 Completed，2026-08-14）

- 新增 `src/qwen3_multimodal_contract.py`：从本地 Qwen3-VL/Qwen3.5 `config.json` 提取 path-free profile，不读取权重；固定 text/vision/processor/component manifest 字段、格式/修订/SHA/大小约束和 `full_model_materialized=false`。
- 视觉到文本 handoff 固定为 `[batch,tokens,hidden]`、dtype/device、image/video 元数据和已提交 artifact 摘要；拒绝路径、ticket、prompt、tensor 内容等敏感字段，拒绝视觉 hidden 与 manifest 不一致。
- 容量预算按唯一 `artifact_id` 去重，分别报告视觉/文本权重、激活、handoff 和安全余量；实际本地 profile 已核对 Qwen3-VL-4B（文本 36 层/hidden 2560）与 Qwen3.5-2B（文本 24 层/hidden 2048）。
- 新增 transfer binding：仅在 QW3 receiver reference 已 `committed` 后绑定 artifact/SHA/大小/节点/代际/phase，结果仍为 path-free ledger 摘要；断线重试、取消、lease/revoke、清理继续复用 QW3.18 状态机。
- MM1.1/.2 定向 `6 passed`；QW3+MM1 联合回归 `169 passed / 1 skipped`。未宣称真实视觉模型 forward、CUDA/双机 parity、吞吐或生产准入。

### 8.39 `PT-PIPE-MM1.3` 合成视觉组件生命周期与 QW3 consume 接线（开发门 Completed，2026-08-14）

- 新增 `src/qwen3_multimodal_runtime.py` 的 `Qwen3MultimodalSyntheticExecutor`，作为目标侧 consume callback：接收已提交 QW3 reference 后构造并校验 MM1 visual→text handoff/binding，生成小型确定性输出，不加载视觉权重。
- `Qwen3NetworkTransferCoordinator.consume_transfer` 可携带 path-free `mm1_binding_sha256` 摘要；执行失败、取消、TTL 和 peer epoch 继续复用原有 coordinator ledger 与 executor cleanup，不新建第二套状态机。
- 定向 MM1 runtime `3 passed`；QW3/MM1 联合回归 `172 passed / 1 skipped`。覆盖成功 consume、binding 失败后的 terminal `failed`/输出清理、TTL `expired` 和 peer epoch `invalidated`；真实视觉 forward、跨进程 sidecar、CUDA/双机 parity、吞吐时延和生产准入继续后置。

### 8.40 `PT-PIPE-MM1.4` 真实隔离 sidecar 输出适配（开发门 Completed，2026-08-15）

- 新增 `Qwen3MultimodalSidecarAdapter`，包裹现有 `Qwen3NetworkSidecarExecutor`：receiver reference 提交后生成 MM1 handoff/binding，把 `mm1_binding_sha256` 投影到 path-free consume 结果，并将 cleanup 原样转发给真实 sidecar executor。
- adapter 对 sidecar 主动返回的 hidden metadata 校验 batch、text hidden size、dtype/device；旧 sidecar 未返回 hidden 字段时只保留 manifest 投影，不伪造真实视觉输出。测试 helper 新增受控 `--mm1-manifest/--mm1-visual-tokens` 参数。
- 真实隔离 `.venv-qwen3-sidecar` CPU prefill/decode smoke 已通过，使用真实 Qwen3-4B torch artifact 与 Qwen3-VL config-derived MM1 manifest；这验证的是跨进程 metadata 接线，不是视觉塔 forward 或图像语义。MM1 runtime `4 passed`、真实 MM1 sidecar `1 passed`、QW3/MM1 联合 `174 passed / 1 skipped`。

### 8.41 计划记录：`PT-PIPE-MM1.5` processor/视觉 worker 离线预检合同

1. 为 Qwen3-VL/Qwen3.5 定义隔离 visual worker 的 request/response schema，固定 processor revision、image/video token、像素/帧预算、组件 assignment 与 `trust_remote_code=false`/offline 约束。
2. 先做不加载视觉权重的 config/processor metadata preflight，覆盖 Qwen3-VL 与 Qwen3.5、缺组件/越界 media/运行时不匹配 fail-closed；图像字节和本地路径不得进入控制面。
3. 真实视觉塔 forward、图像/视频语义质量、CUDA/双机 parity、峰值标定和生产准入继续单独后置。

### 8.42 `PT-PIPE-MM1.5` processor/视觉 worker 离线预检合同（开发门 Completed，2026-08-15）

- 新增 `src/qwen3_multimodal_preflight.py`：对本地 `config.json`、`preprocessor_config.json`、`video_preprocessor_config.json` 和 `tokenizer_config.json` 做有界 JSON 读取、SHA-256 摘要和模型族/视觉几何/tokenizer 交叉校验；不导入 Transformers，不读取 safetensors 权重，不处理图像或视频字节，返回结果不含本地路径。
- 固定 path-free visual worker request/response schema：处理器 revision、`Qwen3VLProcessor`/`Qwen2Tokenizer`、image/video token、patch/merge/temporal 参数、图像/视频像素与帧预算、节点和 manifest 摘要；安全字段强制 `offline=true`、`local_files_only=true`、`network_disabled=true`、`trust_remote_code=false`。
- 组件分配按最小权限收口：`transformers_sidecar` 仅允许 `processor + vision_weights`，`llama_cpp_mtmd` 仅允许 `processor + mmproj`，明确拒绝 `text_weights`/`mtp` 进入视觉 worker。`full_model_materialized` 与 `weight_materialized` 固定为 `false`。
- 两个真实本地目录 Qwen3-VL-4B 与 Qwen3.5-2B 均通过 image/video metadata preflight；新增 MTMD 最小分配、安全策略、越界 media、摘要篡改和敏感字段拒绝矩阵。MM1.5 定向 `15 passed`；QW3/MM1 联合回归 `190 passed`（`.venv-test`，4 workers，78.67s）。主 Python 不含 pytest，不能作为测试环境结论。
- 本票只证明 processor metadata 与视觉 worker 启动条件可审计，不宣称 AutoProcessor 实例化、视觉塔 forward、图像/视频语义、CUDA/双机 parity、峰值/吞吐或生产准入。

### 8.43 计划记录：`PT-PIPE-MM1.6` 隔离 AutoProcessor 离线构造 smoke

1. 在 sidecar 独立 venv 中固定 Transformers 版本与 `local_files_only`，仅用已通过 MM1.5 的 metadata 构造 `AutoProcessor`，把实际类名、token/geometry 和清理状态投影回 path-free response。
2. 覆盖缺少 processor 文件、版本/类名漂移、远程代码开关、异常退出和重复 cleanup；不加载 vision/text weights，不接图像语义和网络传输。
3. 只有该 smoke 与后续人工审计同时通过，才讨论真实视觉张量 handoff；CUDA、双机 parity、长视频和生产路由继续后置。

### 8.44 `PT-PIPE-MM1.6` 隔离 AutoProcessor 离线构造 smoke（开发门 Completed，2026-08-15）

- 新增 `scripts/model_tools/qwen3_multimodal_processor_probe.py` 与独立 `qwen3_multimodal_processor_probe_worker.py`。controller 先复核 MM1.5 manifest/request，worker 在 `.venv-qwen3-sidecar` 内强制 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、清除代理，并只调用 `AutoProcessor.from_pretrained(local_files_only=true, trust_remote_code=false)`。
- Qwen3-VL-4B 与 Qwen3.5-2B 真实构造均通过：Transformers `4.57.6`、`Qwen3VLProcessor`、`Qwen2VLImageProcessorFast`、`Qwen3VLVideoProcessor`、`Qwen2TokenizerFast`，image/video token 与 patch=16、temporal=2、merge=2 均与 MM1.5 request 一致。未调用 processor media forward，未读取权重分片。
- pipeline sidecar 依赖补齐并锁定 `Pillow>=10,<13`、`torchvision>=0.28,<0.29`；setup `_ready(pipeline=true)` 现在会检查 Torch/torchvision/Pillow，避免缺视觉依赖时误报环境 ready。主运行时依赖不变。
- 响应合同额外固定 processor 实例化状态、runtime 隔离、offline 参数、实际类名、cleanup 完成和 `weight_materialized=false/full_model_materialized=false`；异常、缺文件、旧 Transformers、类漂移、非法网络协议和重复运行均 fail-closed。MM1.6 定向 `8 passed`；MM1.5+MM1.6 定向 `23 passed`；串行 QW3/MM1 全量 `198 passed in 111.12s`。4 worker 并发时既有真实 sidecar 资源门可能出现 `<3GB/<5GB` skip，不是代码失败。
- 本票仍不宣称图像/视频处理输出、视觉张量 handoff、语义质量、CUDA/双机 parity、峰值/吞吐或生产准入。

### 8.45 下一票：`PT-PIPE-MM1.7` CPU 合成媒体预处理与张量摘要合同

1. 使用有限尺寸的内存 PIL 图像与短视频帧序列调用已构造的 processor，限制像素/帧/输出字节，并只投影 shape、dtype、grid 和 token 数摘要。
2. 覆盖超限、空媒体、帧数/geometry 不符、异常退出和清理；控制面禁止 pixel_values、原始媒体和路径，仍不加载视觉塔或文本权重。
3. 真实图像语义、跨节点 hidden handoff、CUDA/双机 parity、长视频和生产路由继续后置。

### 8.46 `PT-PIPE-MM1.7` CPU 合成媒体预处理与张量摘要合同（开发门 Completed，2026-08-15）

- `qwen3_multimodal_processor_probe_worker` 在 MM1.6 的 processor 构造后新增媒体预处理阶段：受限合成媒体（内存 numpy 图像/帧，边长 8..1024、帧数 1..32，契约层拒绝超限/空/过小）调用 `image_processor`/`video_processor` 预处理，只投影 `pixel_values_shape`、dtype、patch 网格 token 数估计与输出字节估计；pixel_values、原始媒体、路径与 prompt 一律不进响应（`weight_materialized=false/full_model_materialized=false`）。
- 真实 sidecar（`.venv-qwen3-sidecar`，transformers 4.57.6）双模型实测：Qwen3-VL-4B 与 Qwen3.5-2B 的 image 预处理均产出张量摘要（shape 为网格化 [H,W] 形态、dtype `torch.float32`、token 估计按 patch=16 网格）；Qwen3.5-2B 的 video 预处理对合成帧不产张量（摘要空，非空时校验）。
- 媒体异常统一 `processor_contract_rejected`（fail-closed）；`media_smoke` 经 controller 参数透传（缺省合成 32x32/2 帧）。MM1.7 定向 `11 passed`（含真实双模型）；未加载视觉塔或文本权重，真实图像语义、跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续后置。

### 8.47 下一票：`PT-PIPE-MM1.8` 媒体张量跨边界投影与视觉组件占位

1. 将 MM1.7 的媒体摘要接入视觉组件边界：投影为 path-free 的媒体张量参考（shape/dtype/grid/token 契约），供后续视觉塔执行器的输入占位与容量预算；覆盖摘要与权重/媒体路径解耦的回归。
2. 保持不加载视觉塔与文本权重、CPU 合成、`weight_materialized=false`；真实图像语义、跨节点 hidden handoff、CUDA/双机 parity 与生产路由继续独立后置。

### 8.48 `PT-PIPE-MM1.8` 媒体张量跨边界投影与视觉组件占位（开发门 Completed，2026-08-15）

- `build_mm1_media_tensor_reference` 把 MM1.7 媒体摘要投影为 **path-free 媒体张量参考**（`qwen3_visual_media_tensor_placeholder`）：image/video 的 shape/dtype/patch 网格 token 估计 + `capacity.total_media_tokens`/`output_bytes_estimate` 容量预算（供视觉塔执行器输入占位与容量规划）；`weight_materialized=false/full_model_materialized=false`，pixel_values/原始媒体/路径与 prompt 一律不进入。
- `validate_mm1_media_tensor_reference` 只读校验：身份（model_id/component_ids）、空媒体条目、digest（篡改 fail-closed）。worker 响应新增 `media_tensor_reference`（与 `media_summary` 解耦并列）。
- 定向 `14 passed`（MM1.7 全量 + MM1.8 投影/拒绝/真实双模型）；未加载视觉塔或文本权重，真实图像语义、跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续后置。

### 8.49 下一票：`PT-PIPE-MM1.9` 视觉塔执行器输入占位与容量预算接线

1. 把 `media_tensor_reference` 接到视觉塔执行器边界：以张量参考为输入占位构造 path-free 的视觉输入契约（grid/token/字节预算与节点容量比对），覆盖预算不足 fail-closed 与参考解耦回归。
2. 保持不加载视觉塔与文本权重、CPU 合成、`weight_materialized=false`；真实图像语义、跨节点 hidden handoff、CUDA/双机 parity 与生产路由继续独立后置。

### 8.50 `PT-PIPE-MM1.9` 视觉塔执行器输入占位与容量预算接线（开发门 Completed，2026-08-15）

- `build_mm1_visual_input_contract` 以媒体张量参考为输入占位构造 **path-free 视觉输入契约**（`qwen3_visual_input_placeholder`）：`input.total_media_tokens/total_bytes_estimate/grid`（image/video shape）+ `capacity`（节点容量、required = 字节估计 × margin、**admitted**）；required > node_capacity 时 **admitted=false（fail-closed，视觉塔不执行）**。
- `validate_mm1_visual_input_contract`：身份（model_id/media_reference_sha256）、admitted 与容量一致性（不足却标 True → 拒绝）、digest 校验；worker 响应新增 `visual_input_contract`（`node_capacity_bytes` 经 controller 透传，缺省 None）。
- 定向 `17 passed`（MM1.7/1.8 全量 + MM1.9 预算足/不足/篡改/真实模型接线）；未加载视觉塔或文本权重，真实图像语义、跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续后置。

### 8.51 下一票：`PT-PIPE-MM1.10` 视觉塔组件容量规划与多模态资源账本

1. 把 MM1.9 的视觉输入契约接入节点容量账本：视觉塔权重字节 + 媒体输入预算 + 文本段预算统一入账，覆盖组合超限 fail-closed 与账本幂等（重复入账不重复计费）。
2. 保持不加载视觉塔与文本权重、CPU 合成、`weight_materialized=false`；真实图像语义、跨节点 hidden handoff、CUDA/双机 parity 与生产路由继续独立后置。

### 8.52 `PT-PIPE-MM1.10` 视觉塔组件容量规划与多模态资源账本（开发门 Completed，2026-08-15）

- `mm1_ledger_commit`：多模态资源账本（`qwen3_multimodal_resource_ledger`）——视觉塔权重 + 媒体输入 + 文本段三类条目统一入账，`total_bytes/remaining_bytes/admitted`；**组合超限 fail-closed**（total > node_capacity → admitted=false，视觉塔不执行）；**幂等**：同 ledger_id 下相同 entry_id 重复入账按 entry_id 去重（不重复计费）。
- `mm1_ledger_release`：按 entry_id 释放（total 减少；已释放条目为 no-op，幂等）；`validate_mm1_resource_ledger`：条目去重、kind 白名单、total 一致性、admitted 与容量一致性、digest（篡改拒绝）。账本只记录字节与类别摘要，不携带路径/权重/像素（`weight_materialized=false`）。
- 定向 `20 passed`（MM1.7-1.9 全量 + MM1.10 入账/幂等/超限/篡改/释放）；未加载视觉塔或文本权重，真实图像语义、跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续后置。

### 8.53 下一票：`PT-PIPE-MM1.11` 视觉塔组件放置与纯文本请求守卫

1. 基于资源账本做视觉塔组件放置决策（节点容量账本 → 视觉塔可放置性），并保证**纯文本请求不错误加载视觉塔**（视觉组件按需激活，文本段独立可执行）。
2. 保持不加载视觉塔与文本权重、CPU 合成、`weight_materialized=false`；真实图像语义、跨节点 hidden handoff、CUDA/双机 parity 与生产路由继续独立后置。

### 8.54 `PT-PIPE-MM1.11` 视觉塔组件放置与纯文本请求守卫（开发门 Completed，2026-08-15）

- `mm1_vision_tower_placement`：基于资源账本做视觉塔放置决策（`qwen3_vision_tower_placement`）——**纯文本请求守卫**：无媒体时 `vision_tower_active=false`（视觉塔不激活、不加载，文本段独立可执行，`reason=text_only_request_guard`）；有媒体时账本剩余容量 ≥ 视觉塔权重字节 → `active=true`（`capacity_admitted`），否则 **fail-closed**（`vision_tower_capacity_insufficient`）。
- `validate_mm1_vision_tower_placement`：身份、**纯文本守卫不可绕过**（无媒体却 active=true → 拒绝）、容量一致性、digest；决策对象只含布尔/字节/理由摘要（无路径/权重/像素，`weight_materialized=false`）。
- 定向 `23 passed`（MM1.7-1.10 全量 + MM1.11 纯文本守卫/容量足/容量不足/守卫绕过篡改）；未加载视觉塔或文本权重，真实图像语义、跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续后置。

### 8.55 下一票：`PT-PIPE-MM1.12` 视觉塔执行器骨架与 text-only 会话回归

1. 把 MM1.11 的放置决策接到视觉塔执行器骨架：media 请求激活视觉占位执行路径，text-only 会话全程不触碰视觉塔（含既有文本链路回归）；覆盖放置决策与执行路径一致性。
2. 保持不加载视觉塔与文本权重、CPU 合成、`weight_materialized=false`；真实图像语义、跨节点 hidden handoff、CUDA/双机 parity 与生产路由继续独立后置。

### 8.56 `PT-PIPE-MM1.12` 视觉塔执行器骨架与 text-only 会话回归（开发门 Completed，2026-08-15）

- `run_mm1_visual_tower_skeleton`（`src/qwen3_multimodal_runtime.py`）：按放置决策选执行路径——**text-only 会话** `visual_path="skipped"`（全程不触碰视觉塔，携带媒体参考即拒绝）；**media + active** `visual_path="placeholder_ready"`（占位执行，绑定 media_reference_sha256 与 total_media_tokens）；**media + inactive** fail-closed（`qwen3_mm1_vision_tower_inactive`）；放置决策与 text-only 标志矛盾拒绝（`qwen3_mm1_placement_inconsistent`）。
- 骨架不加载视觉塔或文本权重（`weight_materialized=false`）；text-only 回归 = 骨架 + 放置守卫双保险，既有文本链路不触碰视觉塔。
- 定向 `27 passed`（MM1.7-1.11 全量 + MM1.12 四路径：skipped/占位/未激活 fail-closed/矛盾拒绝）；真实图像语义、跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续后置。

### 8.57 下一票：`PT-PIPE-MM1.13` 视觉塔占位执行路径与媒体参考消费契约

1. 把 MM1.12 的 `placeholder_ready` 路径接到媒体参考消费边界：以占位执行产出 path-free 的视觉特征摘要（特征 shape/dtype/token 数），并绑定回文本段 hidden handoff 契约；覆盖占位执行与文本段消费一致性。
2. 保持不加载视觉塔与文本权重、CPU 合成、`weight_materialized=false`；真实图像语义、跨节点 hidden handoff、CUDA/双机 parity 与生产路由继续独立后置。

### 8.58 `PT-PIPE-MM1.13` 视觉塔占位执行路径与媒体参考消费契约（开发门 Completed，2026-08-15）

- `run_mm1_visual_placeholder_execution`：`placeholder_ready` 骨架 + 媒体参考 → **path-free 视觉特征摘要**（`qwen3_visual_feature_placeholder`）——特征 shape `[1, media_tokens, vision.output_hidden_size]`（合成特征 `synthetic=true`，不加载视觉塔）；骨架非 ready（skipped）即 fail-closed（`qwen3_mm1_skeleton_not_ready`）。
- `bind_mm1_visual_feature_handoff`：特征绑定回文本段 hidden handoff（`visual_to_text` 边界，`build_mm1_handoff_contract`），**消费一致性**：handoff tensor token 数 == 特征 token 数；绑定证据 = artifact.sha256 == media_reference_sha256（path-free 消费）。
- 定向 `30 passed`（MM1.7-1.12 全量 + MM1.13 特征投影/非 ready fail-closed/绑定一致性）；未加载视觉塔或文本权重，真实图像语义、跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续后置。

### 8.59 下一票：`PT-PIPE-MM1.14` 视觉占位链端到端合成回归（CPU）

1. 串联 MM1.7→MM1.13 全链：合成媒体 → 摘要 → 张量参考 → 容量账本 → 放置决策 → 骨架 → 占位执行 → visual_to_text handoff，作为一条 CPU 合成端到端回归（全程不加载视觉塔/文本权重）。
2. 保持 `weight_materialized=false`；真实图像语义、跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续独立后置。

### 8.60 `PT-PIPE-MM1.14` 视觉占位链端到端合成回归（开发门 Completed，2026-08-15）——MM1 合成链收口

- `run_mm1_synthetic_visual_chain`（runtime 编排）：串联 **MM1.7→MM1.13 全链**——合成媒体摘要 → 媒体张量参考 → 资源账本（视觉塔/媒体/文本三段）→ 放置决策 → 执行器骨架 → 占位执行（视觉特征摘要）→ `visual_to_text` handoff；**消费一致性**（handoff token == 特征 token == 媒体 token）。text_only 路径全链跳过视觉塔（骨架 skipped，零媒体参考）。
- fail-closed：容量不足 → 账本/放置/骨架任一环节拒绝（视觉塔不执行）；媒体请求缺 media_smoke 拒绝；全程 `weight_materialized=false/full_model_materialized=false`。
- 定向 `33 passed`（MM1.7-1.13 全量 + MM1.14 端到端 media/容量不足/text_only 三路径）；**MM1 合成链（不依赖真机的部分）至此收口**：真实图像语义、跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续独立后置。

### 8.61 下一票：`PT-PIPE-MM1.15` 视觉塔真实权重接入与合成/真实对照

1. 在隔离 sidecar 内加载视觉塔真实权重（mmproj 或视觉塔 safetensors），与 MM1.14 合成占位对照（真实特征 shape/dtype/token vs 合成投影），覆盖加载门、资源准入与对照一致性。
2. 保持文本段零加载、CPU 合成先行、`weight_materialized` 如实登记；跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续独立后置。

### 8.62 `PT-PIPE-MM1.15` 视觉塔真实权重接入与合成/真实对照（开发门 Completed，2026-08-15）

- `qwen3_multimodal_vision_tower_probe`（controller + 隔离 worker）：sidecar 内**只加载视觉塔真实权重**（`Qwen3VLVisionModel`，从 safetensors 分片 filter `model.visual.` 前缀，315 键 strict 加载）——**文本权重零加载**（`text_weights_loaded=false`）；合成图经真实 `Qwen2VLImageProcessorFast` 预处理 → 视觉塔 forward → **真实特征** `[1, 64, 2560]`（float32，merger 投影到文本 hidden 2560）。
- **合成/真实对照**：`tokens_match=true`（64 tokens 与 MM1.14 合成占位一致）、`hidden_matches_text_config=true`（特征维度 == 文本段 hidden，visual_to_text 边界正确）；视觉塔权重已加载如实登记（`weight_materialized=true`——与合成链 `false` 区分），`full_model_materialized=false`。
- 资源门：可用 RAM < 4GB skip；加载后释放（del/gc）。定向 `34 passed`（MM1.7-1.14 全量 + MM1.15 真实加载/对照）；跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续独立后置。

### 8.63 下一票：`PT-PIPE-MM1.16` 视觉特征接入文本段合成解码（CPU 混合链）

1. 把 MM1.15 的真实视觉特征接入文本段合成解码：视觉特征 → 文本段 hidden 占位 → 合成 decode（含 visual token 位置对齐），覆盖特征与文本段输入一致性；真实图像语义仍不宣称。
2. 保持文本段权重零加载、CPU 合成、`weight_materialized` 如实登记；跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续独立后置。

### 8.64 `PT-PIPE-MM1.16` 视觉特征接入文本段合成解码（开发门 Completed，2026-08-15）

- `run_mm1_synthetic_text_decode`：MM1.15 真实视觉特征接入文本段合成解码——**序列布局** = 视觉 token 区（`[0:visual_tokens]`）+ 文本 token 区（`[visual_tokens: +prompt_tokens]`），**visual token 位置与特征 token 数对齐**；合成 decode（文本段零权重，`text_weights_loaded=false`，不宣称真实语义）。
- fail-closed：特征 hidden 与文本段 hidden 不符（`qwen3_mm1_hidden_mismatch`）、视觉+文本超序列预算（`qwen3_mm1_sequence_overflow`）、维度越界；`weight_materialized=false`（文本段），视觉塔 `weight_materialized` 由特征如实携带。
- 定向 `36 passed`（MM1.7-1.15 全量 + MM1.16 对齐/不匹配/超预算）；真实图像语义仍不宣称，跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续独立后置。

### 8.65 下一票：`PT-PIPE-MM1.17` CPU 混合链端到端合成回归（视觉塔真实 + 文本合成）

1. 串联 MM1.15+MM1.16 与 MM1.14 合成链：真实视觉特征 → 文本段合成解码，作为 CPU 混合端到端回归（视觉塔真实权重、文本段零权重、视觉/文本 token 对齐）。
2. 保持 `weight_materialized` 如实登记（视觉塔 true/文本 false）、CPU 合成；真实图像语义、跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续独立后置。

### 8.66 `PT-PIPE-MM1.17` CPU 混合链端到端合成回归（开发门 Completed，2026-08-15）——MM1 视觉塔真实阶段收口

- `run_mm1_synthetic_hybrid_chain`：**CPU 混合链端到端**——合成媒体链（摘要→张量参考→账本→放置→骨架，MM1.14 路径）→ **真实视觉特征**（MM1.15 产出，替代占位执行）→ 文本段合成解码（MM1.16，视觉区 `[0:64]` + 文本区 `[64:68]` token 对齐）；容量不足 fail-closed。
- **权重如实登记**：`vision_tower_weight_materialized=true`（真实特征携带）、`text_weight_materialized=false`（文本零加载）、`full_model_materialized=false`。
- 定向 `39 passed`（MM1.7-1.16 全量 + MM1.17 混合链/容量不足/**真实串联端到端**）；真实图像语义仍不宣称，跨节点 hidden handoff、CUDA/双机 parity、长视频与生产路由继续独立后置。

### 8.67 下一票：`PT-PIPE-MM1.18` 真实视觉语义 smoke 与跨节点 hidden handoff 前置

1. 以固定测试图走真实视觉塔 + 真实文本段前向（首次加载文本权重），产出视觉语义描述并对照人工/合成基线；随后将视觉特征接入跨节点 hidden handoff 契约（text_chain 边界）。
2. 首次引入文本段权重加载与真实语义，需独立资源门与登记；CUDA/双机 parity、长视频与生产路由继续独立后置。

### 8.68 `PT-PIPE-MM1.18` 真实视觉语义 smoke 与跨节点 hidden handoff 前置（研究工具完成，生产能力撤回，2026-08-15）

- `qwen3_multimodal_vision_text_smoke`（controller + 隔离 worker）：**首次文本权重加载**——4-bit（BitsAndBytes）加载完整 `Qwen3VLForConditionalGeneration`（CPU，RAM 门 ≥10 GiB，显式 opt-in）；固定测试图的研究性语义 smoke 结果如实标记 `full_model_materialized=true`。
- 原先把完整模型 smoke 写成 `full_model_materialized=false` 的 handoff/完成结论已撤回；该入口不签发真实跨节点 hidden handoff，也不进入生产路由。
- 定向回归保留；真实图像语义仅是固定图研究证据，CUDA/双机 parity、长视频与生产路由继续独立后置。

### 8.69 `PT-PIPE-MM1.19` 真实语义扩展与生产路由评估（开发门完成，实机数据待资源恢复，2026-08-15）

1. controller/隔离 worker 支持 1-4 张有界本地图；同一模型只加载一次后顺序处理，逐图登记 `image_sha256`、关键词命中、语义门和 `latency_ms`，支持每图独立关键词基线（sd-001 与 90s 风格不再共用错误基线）。
2. 新增 path-free CPU 资源观测：进程 RSS before/after/peak/delta、可用 RAM、CUDA available/allocated/reserved peak；10 GiB 资源门在加载前 fail-closed。
3. 新增生产路由评估：原生 sidecar 必须通过无完整模型物化、多图语义、资源观测、CUDA parity、分布式 hidden handoff、长视频等门；当前研究 smoke 明确推荐条件式 `external_api`，不构成生产准入。
4. 当前开发机真实双图请求已在模型加载前返回 `resource_rejected/insufficient_ram`（可用 RAM 约 6.5 GiB），所以真实多图时延/显存/内存表待资源恢复后补登；CUDA/双机 parity、长视频和跨节点实测继续后置。

### 8.70 `PT-PIPE-MM1` 复核修正（2026-08-15）

- 复核发现 MM1.15 原实现使用 `safetensors.torch.load_file()` 读取完整分片后再过滤视觉键，会在 4.9GB 分片上制造不必要的 RAM 峰值；现改为读取 index/header 后用 `safe_open.get_tensor()` 逐张量复制，仅保留视觉塔权重，并校验 shard 路径、键集合、形状和 Qwen3-VL model type。
- 原 MM1.18 把完整 `Qwen3VLForConditionalGeneration` 4-bit 加载标记为 `full_model_materialized=false`，与分布式无峰值目标矛盾。该完成结论撤回：默认请求现在在加载前 `resource_rejected/full_model_materialization_disabled`；研究性完整模型 smoke 必须显式传 `allow_full_model_materialization=true`，成功响应真实标记 `full_model_materialized=true`，不进入生产路由或 hidden handoff。
- MM1.8-11 的 media reference、容量合同、资源账本和 placement 校验已收紧为严格类型、容量/摘要一致、冲突 entry 拒绝和 text-only 守卫；新增 `tests/test_qwen3_multimodal_review.py` 回归。
- 当前可宣称范围：MM1.7-1.17 的合成/视觉塔 CPU 骨架和真实视觉塔前向；MM1.18/1.19 仅为显式 opt-in 研究工具和资源观测，完整模型物化、真实跨节点 handoff、CUDA parity 和生产准入仍未完成。下一票为 `PT-PIPE-MM1.20`，先做分段文本执行器前置，不得绕过完整模型 opt-in 门。

### 8.71 `PT-PIPE-MM1.20` 分段文本执行器前置与无峰值 handoff 门（开发门 Completed，2026-08-15）

1. 新增 `build_mm1_staged_text_contract` / `validate_mm1_staged_text_contract`：把 MM1.15 视觉特征摘要与完整 2/3 段文本拓扑绑定；校验模型/manifest/assignment SHA、连续层范围、首段节点、单段权重与激活预算、节点容量以及 contract SHA。控制合同只含 artifact/shape/dtype/token/chain 摘要，不含路径、像素或 prompt 内容。
2. 首段输入显式登记 `visual_span + text_span`、总序列长度与 sequence budget，并以 `[batch, visual_tokens + prompt_tokens, hidden]` 约束下一段 hidden handoff；任一段激活预算低于合并输入、容量不足、序列溢出、dtype/device 不匹配或完整模型物化均 fail-closed。
3. 新增 `Qwen3MultimodalStagedTextFixture` 和 `execute_mm1_staged_text_contract`：CPU fixture 覆盖首段边界、下一段请求、成功释放、执行异常释放、cleanup 不完整拒绝和合同篡改拒绝。证据明确为 `evidence_kind=cpu_fixture`，`text_weights_loaded=false`、`segment_materialized=false`、`full_model_materialized=false`；仅 `fixture_segment_materialized` 描述测试生命周期，不宣称真实权重已加载。
4. MM1.20 专项 `7 passed`；MM1 合同/预检/runtime/processor/MM1.19-20 联合 `80 passed / 1 skipped`。真实首段权重加载与 forward、本地 artifact 数据面、CUDA/双机 parity、跨节点网络和生产准入仍未完成。

### 8.72 `PT-PIPE-MM1.21` 首段 sidecar 本地 artifact 适配（开发门 Completed，2026-08-15）

1. 新增独立的 `qwen3_mm1_first_segment_artifact` 绑定合同：把 MM1.20 staged contract、首段 assignment、合并 hidden 的 shape/dtype、实际序列化工件大小/SHA 和下一段目标绑定；媒体摘要不再冒充 `.pt` 文件摘要。合同及返回结果均拒绝路径、文件字段、像素和 prompt 内容。
2. 新增 `Qwen3MultimodalFirstSegmentSidecarAdapter`，复用现有 `Qwen3PipelineSidecarSession` 执行 prepare → commit → prefill → release；输入/输出路径只在受控本地 data plane 使用，scope fence、输入提交证据、输出回读证据、hidden handoff 和 sidecar assignment 均逐项校验。成功后仅以本机 `artifact_id` 保留输出供下游消费，控制结果只含大小、SHA 和 handoff 摘要。
3. 取消、prefill 失败、输出 SHA 不符、release/abort 未证明 `cleanup_complete=true` 时均 fail-closed：删除本票创建的输出并 abort；成功 release 后 `segment_materialized=false`，下游提交后可通过本地清理接口删除保留输出。输入工件仍由上游 controller 持有，不越权删除。
4. MM1.21 CPU fake-runner 专项 `9 passed`；排除 `real_model` 的 MM1 合同/预检/runtime/processor/MM1.19-21 与通用 sidecar 稳定回归 `91 passed`。包含真实模型门的最近一次运行 `94 passed / 2 skipped`：MM1.17 因当时可用 RAM `<4GB` 跳过，MM1.18 固定为显式 opt-in。上述结果证明合同、文件证据和生命周期接线，不证明真实 Qwen3-VL 文本权重、真实 forward、CUDA、跨节点或生产准入。

### 8.73 `PT-PIPE-MM1.22` 本机 2/3 段 sidecar artifact 串联（开发门 Completed，2026-08-15）

1. 新增 `Qwen3MultimodalMultiSidecarAdapter`，把 MM1.20 staged contract、MM1.21 首段输入 artifact binding 与现有 `Qwen3PipelineMultiSidecar` 严格绑定；首段输出在同一条本地链内直接作为下一段输入，避免重复执行首段。每个 session 的 model/node/layer/embedding/LM-head/generation/assignment、边界 dtype/device 和 hidden size 均在执行前或下一段启动前校验。
2. 每段输出文件实际大小/SHA、local handoff reference、hidden shape/dtype/device、KV contract、generation 和最终段输出摘要均回读核对；控制结果只含 segment/handoff/KV/最终 artifact 摘要，不返回路径。通用 multi-sidecar 新增可选 hidden-size 门，损坏 handoff 会在下一段启动前终止。
3. CPU fake runner 覆盖 2 段/3 段成功串联、最终 artifact 保留、逐段证据、中段失败、handoff shape 篡改、取消、重启残留清理和 path-free 返回。专项 `7 passed`，通用 multi-sidecar `5 passed`，MM1/sidecar/network 联合 `138 passed / 1 skipped`。真实权重/forward、视觉语义、CUDA/双机、网络传输和生产路由继续后置。

### 8.74 `PT-PIPE-MM1.23` 2/3 段 prefill → decode/KV 生命周期（开发门 Completed，2026-08-15）

1. 新增 path-free `qwen3_mm1_decode_input_artifact` 合同，绑定 decode 输入工件、prefill/decode generation、batch/token 数、prefill 完整序列长度与 decode 增量长度；decode 输入路径只在本地 data plane 解析，控制面只返回证据摘要。
2. `Qwen3MultimodalMultiSidecarAdapter` 新增 prefill → decode 原子流程：不提前 release，逐段复用并校验 prefill KV，decode 使用完整 KV sequence length，同时用 `input_sequence_length` 正确表达本轮新增 token 的 hidden handoff；最终只保留 path-free final decode artifact/KV 摘要。
3. 通用 `Qwen3PipelineMultiSidecar.decode` 增加可选 `input_sequence_length`，兼容旧调用并修正 decode hidden handoff 与 KV 总长度混用问题。取消、KV/输入/合同篡改、generation 不一致、异常 abort、重启和最终 cleanup 均 fail-closed。
4. MM1.23 专项 `13 passed`；MM1/sidecar/network 联合 `144 passed / 1 skipped`。真实 Qwen3-VL forward、真实 logits 质量、CUDA/跨节点与生产准入继续后置。

### 8.75 `PT-PIPE-MM1.24` decode 结果消费与 path-free 质量摘要（开发门 Completed，2026-08-15）

1. 新增 `Qwen3MultimodalDecodeArtifactConsumer`，只允许读取受控 artifact root 内、状态为 committed 且类型为 `final_decode_output` 的本地工件；消费前复核实际大小/SHA，默认以 `torch.load(..., map_location="cpu", weights_only=true)` 读取 sidecar 原生 `.pt`，控制结果不返回路径或张量。
2. 消费门严格要求 payload 只有 `logits/hidden_states/past_key_values`，且恰有一个输出张量；校验 rank-3、batch、decode token 数、dtype、CPU 映射和非有限值，并把 KV 层数/完整 sequence length 与 decode generation 合同交叉核对。返回仅含 artifact 证据、token/shape/dtype/device/KV 质量摘要和 `full_model_materialized=false`。
3. 同一进程默认拒绝重复消费；显式 `reset()` 只清除重放账本，不删除工件，最终工件仍由 MM1 adapter 的统一 cleanup/restart recovery 管理。缺失、越界、SHA 篡改、shape/dtype/KV 不符、重复消费和清理所有权均 fail-closed。
4. MM1.24 新增 `9` 个 CPU 小工件回归，MM1.22-24 专项共 `22 passed`；MM1/sidecar/network 联合 `153 passed / 1 skipped`。这证明本地 decode 工件证据和摘要消费，不证明真实 Qwen3-VL logits/文本质量、CUDA、跨节点或生产准入。

### 8.76 `PT-PIPE-MM1.25` 有界多步 decode 与逐段 KV 滚动（开发门 Completed，2026-08-15）

1. `Qwen3PipelineMultiSidecar` 已允许 committed prefill 后连续 decode；每一步 generation 自动递增，完整 KV sequence 必须按 `input_sequence_length` 单调扩展。所有 segment 的新输出均成功后才切换 KV 引用并删除上一代，任一段失败、长度跳变或取消都会 abort 全链并删除各代本地工件。snapshot 记录有界 decode 水位和 trace，不汇总模型或跨段 KV。
2. `Qwen3NetworkSidecarExecutor` 同步改为 prefill/decode 均制作 node-local KV 副本，新一代复制与 SHA 复核完成后才替换旧代；每个 chain/segment 记录 generation/sequence 水位，陈旧或不增长的远端 decode 在执行前拒绝，复制失败恢复旧引用。cleanup 同时清除 KV 文件和水位。
3. `Qwen3MultimodalMultiSidecarAdapter.execute_bounded_decode` 支持 batch=1、每步单 token 的有界 greedy 循环（`max_new_tokens<=4096`、最多 64 个 EOS ID）。末段 `.pt` 只在本地 data plane 以 weights-only CPU 方式校验并 `argmax`，下一 ID 仅写入临时 `input_ids` 工件；EOS、上限和显式取消均可停止，临时输入在消费后立即删除并纳入 restart cleanup。控制结果不含 token ID、logits tensor 或路径。
4. 为避免长生成撑大控制帧，逐步证据使用增量 trace SHA，控制面只返回步数、首末 generation/sequence、停止原因和最终 logits/KV/SHA 摘要，结果大小不随生成步数线性增长。KV 层数门同时收紧为必须等于末段 layer range。
5. MM1.25 新增 `6` 个 MM1 用例和 `2` 个通用 multi-sidecar 用例；MM1.22-25 专项 `28 passed`，sidecar/multi-sidecar/network 定向 `44 passed`，MM1/sidecar/network 最近联合 `161 passed / 1 skipped`。唯一 skip 为 MM1.18 完整模型显式 opt-in；真实 helper/本地调度器另组 `6 passed / 6 skipped`，跳过项仍受隔离运行时/外部条件门限制。上述结果不证明真实 logits/文本质量、CUDA、双机或生产准入。

### 8.77 `PT-PIPE-MM1.26` 本地生成 token 账本与隔离 tokenizer 最终解码（开发门 Completed，2026-08-15）

1. `execute_bounded_decode` 每步选中 ID 只追加到用户主节点拥有的本地 JSON ledger；每条记录绑定 text chain、generation、sequence、输出 artifact SHA 和停止原因。账本采用固定 schema/canonical digest/file SHA，限制最多 `4096` token、`256 KiB`，token ID 受整数边界约束；控制面只返回账本摘要，不返回 token ID 或路径。
2. ledger 通过临时文件原子提交并归 adapter 所有；成功、失败、取消、TTL 与重启 recovery 都清除 `.pt` 与 `.json` 临时工件，不留下可重放残留。
3. 新增隔离 controller/worker：强制专用 Python、离线环境、`local_files_only=true`、`trust_remote_code=false`，严格复核 ledger 身份、水位、内部摘要与每条 artifact SHA；仅用本地 tokenizer 解码，跳过 special token，执行严格 UTF-8 与 `64 KiB` 文本上限，输出文本及 SHA/字节证据，不回显 prompt、token ID、路径或权重状态。
4. MM1.26 新增 `7` 项回归（6 个 worker/controller 合同、1 个 adapter 集成）；专项 `35 passed`，加入 pipeline/sidecar/network 后 `79 passed`，随后 MM1.27 sampling 回归使最新联合结果为 `176 passed / 3 skipped`。跳过项为可用 RAM 不足的 MM1.15、MM1.17 以及 MM1.18 完整模型显式 opt-in。结果不证明真实模型语义质量、CUDA/双机、跨节点或生产准入。

### 8.78 `PT-PIPE-MM1.27` 本地 sampling 合同与可复现生成（开发门 Completed，2026-08-15）

1. `execute_bounded_decode` 新增严格的 temperature `0..2`、top-k `0..4096`、top-p `(0,1]` 和 seed `0..2^63-1` 合同；temperature=0 保持旧 greedy，显式正温度使用 CPU 本地 `torch.Generator` 的 multinomial sampling，固定 seed 不触碰全局 RNG 状态。
2. logits/probabilities 全程留在本地 data plane，top-k/top-p 过滤和非有限值检查在选 token 前完成；ledger 增加 sampler canonical digest、逐步 draw evidence，并由 worker 复核 sampler、record、artifact SHA 的一致性。控制面只返回采样配置摘要与 draw 数，不返回 RNG state、tensor 或 token ID。
3. 新增参数边界、同 seed、top-p 边界、ledger 证据和 adapter 集成回归；MM1/sidecar/network 联合 `176 passed / 3 skipped`。结果不证明真实模型采样质量、CUDA/双机、跨节点调度或生产路由。

### 8.79 `PT-PIPE-MM1.28` sampling 质量摘要与受控重放边界（开发门 Completed，2026-08-15）

1. 每步 sampling 在本地计算候选集大小、熵、置信度和 top-p 截断水位，只输出聚合的 min/max 摘要与滚动 SHA；摘要绑定 sampler digest、step artifact SHA 和 token 数，不回传 logits、概率或 token ID。
2. ledger 扩展 `quality_summary`，由 controller/worker 校验字段、水位、sampler 关联、质量范围和 metadata SHA；新增只读 `qwen3_token_ledger_replay` operation，仅验证 ledger/sampler/quality 证据，不加载 tokenizer、模型或完整 RNG 状态。
3. replay 支持期望 sampler/quality digest，摘要篡改、链身份不符、过期/缺失 ledger 和旧版 ledger 均 fail-closed；MM1.28 新增 replay/quality 集成回归，定向 `46 passed`，联合回归为 `177 passed / 3 skipped`。结果不证明真实模型采样质量、CUDA/双机、跨节点调度或生产路由。

### 8.80 `PT-PIPE-MM1.29` sampling 策略快照与路由接线（开发门 Completed，2026-08-15）

1. 新增 sampling policy snapshot：固定策略版本、policy ID、issued/expiry、replay 权限和 `mm1-bounded-decode` route scope；快照 canonical SHA 与 sampling digest 绑定。无快照的旧客户端保持 greedy，任何隐式正温度 sampling 在 `prepare` 前拒绝。
2. adapter 将策略摘要投影到 path-free 结果并写入新 ledger；worker/replay 校验策略身份、过期窗口、replay 权限、期望 policy SHA 和 sampling 一致性，不把完整快照、路径、prompt、token ID 或 RNG 状态放入控制结果。
3. 新增策略快照、隐式 sampling、过期策略、policy/ledger/replay 集成回归；定向 `48 passed`，最新 MM1/sidecar/network 联合 `181 passed / 1 skipped`（唯一 skip 为 MM1.18 完整模型显式 opt-in；RAM 门本次满足）。真实模型采样质量、CUDA/双机、跨节点路由和生产准入仍未验。

### 8.81 历史排期：`L4-GEMMA4 G4.3.2B`（已完成，2026-08-16 复核）

1. `3a3ee84`、`b41788c` 已完成官方 Ollama `gemma4:12b` GGUF/mmproj 身份下的 CPU-only 原生图片 smoke；本条不再作为下一票。
2. 该结论只属于已冻结 GGUF/mmproj 路径，不代表 23.9GB PyTorch 工件、分布式 pipeline、音频、长上下文或生产路由已经验收。

### 8.82 `KIP-11A` Gemma 4 Unified 元数据、分配与隔离物化合同（开发门 Completed，2026-08-16）

1. `pipeline_model_descriptor` 已登记官方 `gemma4_unified` 嵌套布局；descriptor 独立统计 text layer、embedding/norm/head 与 `embed_vision`/`embed_audio` 多模态字节，运行时能力继续显式关闭。
2. assignment manifest 只允许分配 `model.language_model.*` 和 `lm_head.*`；未规划视觉/音频组件时容量与物化均 fail-closed，tied head 的源权重按架构布局解析，不再硬编码 Qwen2 路径。
3. 新增隔离 `gemma4_pipeline_adapter`：要求 filtered safetensors index、`local_files_only=true`、`trust_remote_code=false`，先验证完整 assignment/key 集合再读取 tensor；只创建 meta 骨架和目标层段，清除未分配首尾与两种模态模块，指标固定声明 `full_model_materialized=false`、`multimodal_materialized=false`。
4. 主运行时在读取本地 config 后即拒绝 `gemma`/`gemma4_unified`，不会落入旧 Transformers 的未知架构错误，也不会复用 Qwen2 前向。定向回归 `125 passed / 1 skipped`（含官方默认 tied output），Scheduler/API/模型同步扩展回归 `348 passed`。
5. **已转入 KIP-11B**：见下一节；在真实工件、CUDA/双机和生产路由验收完成前，`pipeline_runtime_supported` 必须保持 `false`。

### 8.83 `KIP-11B` Gemma 4 隔离分段执行与本地 Scheduler 链（开发门 Completed，2026-08-16）

1. 新增 `packaging/requirements-gemma4-pipeline-sidecar.txt` 与 `scripts/setup_gemma4_pipeline_env.py`；`.venv-gemma4-pipeline` 独立安装 Transformers `5.10.1`、Torch、Accelerate、Safetensors，脚本拒绝复用 `.venv-gemma4-native`，支持显式 PyTorch wheel index、wheelhouse 和用户代理。
2. `Gemma4PipelineAdapter` 保持官方绝对 layer index，按 `full_attention/sliding_attention` 构造官方 mask；处理跨段 prefill 的位置/掩码、连续 decode、native `Cache` 和 shared-KV source/consumer，纯 shared 段使用只读 mask cache geometry，避免把上游 cache 重复计入。
3. 新增 path-free `gemma4_pipeline_contract`、sidecar controller/worker、2/3 段 `Gemma4PipelineMultiSidecar` 与 Scheduler 本地显式 opt-in 接线；prepare/commit/execute/release/abort、assignment SHA、资源门、工件证据和 cleanup 均 fail-closed。控制结果固定 `full_model_materialized=false`、`multimodal_materialized=false`，不提供网络执行器或生产路由。
4. 验证：KIP-11B 合同/adapter/sidecar/env 专项 `18 passed`；官方 Transformers 5.10.1 CPU tiny smoke（整模 vs 两段 prefill/decode）`prefill_max_abs_error=0`、`decode_max_abs_error=0`；主 `.venv-test` 官方 smoke 因旧 Transformers 版本跳过，未把测试依赖写入 sidecar。
5. **KIP-12 已完成**：`gemma4-native` 受管 binding 已冻结路径、版本、revision、patch SHA、ABI 摘要和 package/frozen marker；缺失或篡改均 fail-closed。当前下一票转 KIP-13；KIP-11B 真实 23.9GB Safetensors header/assignment、CUDA、双机 handoff、断线恢复、真实语义和生产 allow-list 继续后置。
