# 旧仓库功能面盘点 — 重写取舍依据

> 盘点对象：`G:\C\PYT\qlh\src\`，**232 个 .py 文件 / 129,337 行非空行**（含空行 142,827）。
> 只读盘点，未修改仓库任何文件。
> 对照基准：`DESIGN.md` / `ARCHITECTURE.md` / `PROGRESS.md` / 仓库根（已有 1,276 行非空原型）。
> 新定位：**只保留「llama 工件裁切」+「层段接力」**，首期 1000 行左右。

---

## §1 功能域总表

行数口径 = 非空行。判定：**保留 / 部分保留 / 抛弃**。

| # | 功能域 | 文件 | 行数 | 占比 | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| F01 | 层段接力核心（relay_*） | 12 | 4,491 | 3.5% | **部分保留**（仅传输+契约思想） |
| F02 | keep-head 上游通道 | 3 | 1,034 | 0.8% | **保留**（唯一 hidden 通道） |
| F03 | 工件裁切 | 5 | 2,943 | 2.3% | **部分保留**（约 200 行是真裁切） |
| F04 | 引擎适配层（llama_engine 等） | 12 | 7,066 | 5.5% | **抛弃** |
| F05 | 模型供给（下载/搜索/清单/意图） | 27 | 14,157 | 11.0% | **抛弃** |
| F06 | 调度控制面（scheduler 全家） | 28 | 29,752 | 23.0% | **抛弃** |
| F07 | qlh_core 抽象层 | 47 | 8,108 | 6.3% | **抛弃** |
| F08 | 网络传输（tcp_comm 等） | 6 | 4,976 | 3.8% | **部分保留**（帧编解码 ~280 行） |
| F09 | 认证与引导 | 6 | 1,444 | 1.1% | **抛弃** |
| F10 | API server（FastAPI 全家） | 26 | 16,018 | 12.4% | **抛弃** |
| F11 | TUI（Textual） | 6 | 4,789 | 3.7% | **抛弃**（新原型自写最小 CLI/TUI） |
| F12 | 多模态 / 跨框架 / PyTorch | 37 | 27,825 | 21.5% | **抛弃** |
| F13 | 持久化与日志（SQLite/journal） | 9 | 3,449 | 2.7% | **抛弃** |
| F14 | 启动 / 配置 / 发布 | 8 | 3,285 | 2.5% | **抛弃** |
| | **合计** | **232** | **129,337** | 100% | |

### §1.1 六个「看起来核心」模块的逐个裁定

| 模块 | 行数 | 层段接力真的需要它吗？ | 裁定 |
| --- | ---: | --- | --- |
| `scheduler_pipeline.py` | 8,691 | **不需要**。它的主体是 A3 `stage_offer` 的**供给编排**（准备→提交→就绪→清理），共 637 行 `_push_layer_config_to_clients_locked` + 948 行 `_handle_layer_config_locked` + 399 行 `_handle_layer_forward_locked`。真正的接力循环 `qlh_core/execution/linear.py` 只有 **62 行**。 | 抛弃 |
| `scheduler.py` | 4,697 | **不需要**。职责是把 4 个 mixin 拼起来（`scheduler.py:505-508`）：`Sidecar + TaskWorker + Cluster + Pipeline`。接力只要「一个 token 依次经过各节点」。 | 抛弃 |
| `scheduler_cluster.py` | 2,268 | **不需要**。节点成员管理 + 容量规划 + reshard 恢复。新原型段落在 TOML 里静态声明（原型 `topology.py` 已这么做）。 | 抛弃 |
| `api_server.py` | 4,529 | **不需要**。FastAPI 门面 + 前端静态资源 + OpenAI 兼容层。用户明确不做 OpenAI SSE。且 SSE 已迁走：`api/routes_chat.py:1223` 是唯一的 `text/event-stream`。 | 抛弃 |
| `tcp_comm.py` | 2,834 | **部分需要**。只要帧编解码（`pack_data`/`unpack_header`/`recv_exact`/`build_message`/`parse_message`，约 280 行）。`TCPServer` 955 行 + `TCPClient` 1,017 行带心跳/重连/epoch/注册重放防护，**不要**。 | 部分保留 |
| `model_supply*.py` | 5,981 | **不需要**。真裁切外壳只有 `_run_cut`（48 行，`model_supply_assets.py:1242-1289`，起子进程调 `scripts/cut_layers.py`）+ `build_gguf_segment` 的入参校验（约 130 行）。其余是段复用/generation/哈希缓存/断点续传。 | 抛弃壳，保留 `scripts/cut_layers.py` |
| `tui_textual.py` | 3,152 | **不需要**。Textual App + SplashScreen + ChatPane。新原型要 TUI 就在 Rust 侧重写。 | 抛弃 |
| `app_bootstrap.py` | 1,567 | **不需要**。安装包启动器：Tailscale 检测、pywebview、console 观感、端口占用清理。 | 抛弃 |

---

## §2 保留项详述

### §2.1 keep-head 上游通道（最关键的保留项）

**为什么必须保留**：llama.cpp 的 `llama_get_embeddings_ith` 返回的是 `output_norm(H)`（证据
`scripts/relay_diag_head_norm.py`：`rel_err=0.0018` / `cos=0.999998`），**比接力需要的 hidden 多一次归一化**。
用它当上游，下游首步即分叉。项目已有一个 llama.cpp 补丁
（`scripts/model_tools/patches/llama-cpp-layer-forward-api.patch`，248 行）导出
`llama_set_embeddings_layer_inp` / `llama_get_embeddings_layer_inp`，shim 建立在它之上。

**这是新原型的硬依赖，不是可选件**：`engine.py:19-24` 目前直接
`sys.path.insert(0, "G:/C/PYT/qlh/src")` 再 `from llama_keep_head import KeepHeadUpstream`。
原型 P1 的「逐位相同」结论（`PROGRESS.md` §1，`max abs diff = 0.0`）就建立在这个通道上。

**最少需要多少行**：
- C shim：`qlh_keep_head.c` 620 行非空 → GPU 版已改为 `H:\slimarc-proto\shim\qlh_keep_head_gpu.c`（741 行，81 行差异）。**C 代码不计入 Python 1000 行预算**，但必须算作移植工作量。
- Python 绑定：`src/llama_keep_head.py` 871 行里，接力实际用到的是 `forward_tokens_to_hidden` /
  `forward_hidden_to_hidden` / `forward_hidden_to_token` / `reset` / `close` 五个方法 + `_add_dll_dirs`
  的 DLL 搜索路径修复。**可精简到 250–300 行**（去掉 subprocess 隔离 worker、
  多序列 argmax、`materialization_observation`、`skip_hidden_logits` 探测缓存）。
- ⚠️ **不要去掉** `_add_dll_dirs`：shim 依赖 `libgcc_s_seh-1.dll` 等同目录 DLL，缺了直接加载失败。

### §2.2 工件裁切：真身在 `scripts/`，不在 `src/`

**重要事实**：`src/model_supply_assets.py` 的 2,353 行里，GGUF 裁切相关只有约 **200 行**
（`_gguf_table` 10 / `_gguf_layers` 12 / `_layer_sizes` 20 / `_run_cut` 48 / `build_gguf_segment` 的校验段 ~110）。
真裁切在 **`scripts/cut_layers.py`（767 行 / 690 非空）**，由 `_run_cut` 以子进程方式调用
（`model_supply_assets.py:1242-1289`）。

**必须保留的裁切语义**（`scripts/cut_layers.py:530-566` 的 `_plan_tensors`）：
1. `--keep-head N`：保留 `blk.0..N-1`，**不重命名**（头段）。
2. `--k K --end E`：丢区间外，把 `blk.K..` 重编号为 `blk.0..`（中段/尾段）。
3. 非 `blk.*` 张量原样保留；非尾段可省略 `output.weight`。
4. **切点必须是 `<arch>.full_attention_interval` 的整数倍**（Qwen3.5 MoE 为 4），
   否则 hybrid 层类型错位、llama.cpp 报 `missing tensor 'blk.x...'` 无法加载。
   这是已验证的硬约束（`DESIGN.md:172-181`）。

**最少需要多少行**：**150–200 行**。可以砍掉：`_validate_layer_types` 的 HF config 交叉校验（137 行）、
`_validate_model_preflight_metadata`（19 行）、`_per_layer_array_overrides` 的 per-layer KV（46 行，
只为多序列）、`_manifest` 的完整 340 行清单。核心只需：读 GGUF 头 → 选张量 → 改名 → 写 GGUF + 一个 6 字段 manifest。

### §2.3 接力传输：**只能保留协议形状，不能保留实现**

`relay_transport.py`（1,247 行）里值得抄的只有：帧结构 `MAGIC "QLHR" + version + kind + seq + dtype + payload`
（`relay_transport.py:37-70, 382-470`），以及「raw float32 直传、不做 base64/JSON」这个决定。
**实现必须重写**，理由见 §4.1。

---

## §3 抛弃项详述

| 域 | 行数 | 替换方案 / 为什么不需要 |
| --- | ---: | --- |
| F04 引擎适配层 | 7,066 | `llama_engine.py` 2,694 行覆盖 binding/server/rpc 三种形态 + 多模态 + 校准。新原型每段一个 `KeepHeadUpstream` 就够（原型 `engine.py` 仅 83 行）。`llama_rpc_*` 是 llama.cpp RPC 后端，与「按层切」是两套正交机制，不需要。 |
| F05 模型供给 | 14,157 | 新原型假设「GGUF 已在本地」。下载/搜索/意图/HF 同步/SQLite 注册表全部不需要。`model_intent*.py` 是「用户选模型 → 主节点分发」的编排，静态拓扑下无意义。 |
| F06 调度控制面 | 29,752 | 静态 TOML 拓扑替换动态调度（原型 `topology.py` 200 行已实现并带校验）。容量求解 `qlh_core/planning/capacity.py` 1,955 行在「人工定切点」场景下无价值。 |
| F07 qlh_core | 8,108 | 纯度很高的六边形架构抽象：contracts / adapters / bootstrap / control / execution / planning 六层端口。对 1000 行原型是纯负担。**唯一值得读的是 `execution/linear.py`（62 行）**，它把接力循环表达得最清楚。 |
| F09 认证引导 | 1,444 | bootstrap 凭证的 RSA 临时公钥 + nonce 台账 + AES-GCM 本地密钥库。新原型跑 Tailscale/内网，无此需求。 |
| F10 API server | 16,018 | 无 HTTP 层。Rust 控制面直接用 TCP/UDS 与 worker 通信。 |
| F11 TUI | 4,789 | Rust 侧重写。`tui_shared.py` 的 `API_PATHS` 常量表随 API 一起消失。 |
| F12 多模态/跨框架 | 27,825 | 用户明确不接跨框架、不引入 PyTorch。这是整仓最大单块（21.5%）。 |
| F13 持久化/journal | 3,449 | 用户明确不要 SQLite。`local_store.py` / `recovery_journal.py` / `journal_replication*.py` 全部去掉。 |
| F14 启动配置 | 3,285 | `app_bootstrap.py` 1,567 行是安装包启动器。`config.py` 975 行读 **94 个 QLH_* 变量**（全 src 共 **180 个**），原型只需 9 个（`config.py:70`）。 |

---

## §4 过度工程化清单

### §4.1 ⚠️ 最大发现：`relay_transport` 强制 loopback-only，与「跨机传 hidden」定位直接冲突

`relay_transport.py:535-536`：
```python
if not is_loopback_host(host):
    raise RelayProtocolError("non_loopback_endpoint_rejected")
```
`config.py:731` 与 `scheduler_pipeline.py:6710` 都写明：**跨机必须走本地 SSH 隧道**。
也就是说，旧仓库的「层段接力」在生产形态下**根本不做跨机直连**——它靠外部 SSH 隧道把远端
映射回 `127.0.0.1`。`relay_contract.py:41` 的 `XFRAME_NETWORK_MODES = {"loopback", "ssh_tunnel"}`
是同一个约束的第二次表达。

**含义**：新原型「一个 token 依次经过各节点」要真跨机，这条传输层**必须重写**，不能复用。
同时它也解释了为什么旧仓库接力很慢（`DESIGN.md:92`：链路 + 固定开销 297 ms，占 49.7%）——
SSH 隧道 + 每段 2 次往返（`relay_cut_objective.py` 的 `STAGE_ROUND_TRIPS = 2`）。

### §4.2 状态机 / 租约 / 事务 / 两阶段提交 / 代际

| 痕迹 | 位置 |
| --- | --- |
| 两阶段提交（prepare→commit→ready / abort→release） | `scheduler_pipeline.py:1698 start_pipeline_model_transaction`、`:3163 _abort_pipeline_load_transaction`、`:3361 _commit_pipeline_load_transaction`、`:3594 _complete_pipeline_load_transaction`（275 行） |
| 代际 generation | `scheduler_pipeline.py:601,620,754,780,790`；`scheduler_pipeline.py:3019 _assignment_connection_generation` |
| 分片租约 + 续期 | `llama_rpc_contract.py:12 class RpcShardLease`（含 `lease_expires_at`）、`:34 RpcShardLeaseBook` |
| 阶段租约（默认 300 s，实测 63 s 就 `lease_expired`） | `config.py:630-662 pipeline_stage_lease_seconds` |
| 请求生命周期相位 + 互斥终态 | `qlh_core/contracts/outcome.py:1-27`：`admitted/started/completed/fallback/refused/cancelled/failed` 七相位，优先级 `cancelled > refused > failed > fallback_completed > completed > incomplete` |
| 注册 epoch + 重放防护 | `tcp_comm.py:140 RegistrationReplayGuard`、`:1303 _next_registration_epoch`、`:1140 registration_epoch`（全文 30+ 处引用） |

### §4.3 重试 / 退避 / 心跳 / 故障恢复

| 痕迹 | 位置 |
| --- | --- |
| 层配置重试监控线程 + 5 s 退避 | `scheduler_pipeline.py:3912 _start_layer_config_retry_monitor`、`:3924 _layer_config_retry_loop`、`:3931 _retry_pending_layer_configs`；退避常量 `:794,2491,3061` 均为 `time.monotonic() + 5.0` |
| 恢复状态装载 / 恢复栅栏 / 恢复模型重建 | `scheduler_pipeline.py:1021 _load_pipeline_recovery_state`、`:1251 _restore_pipeline_model_for_recovery`（182 行）、`:1433 _clear_pipeline_recovery_fence` |
| 拓扑重切（reshard）+ 旧拓扑隔离 | `pipeline_reshard.py` 975 行，模块 docstring 自称 "fail-closed recovery planning" |
| 心跳 / 重连 / 漏跳上限 | `tcp_comm.py:285-286 HEARTBEAT/HEARTBEAT_ACK`、`:1166 MAX_HEARTBEAT_MISSED`、`:2032` 超时判定 |
| 集群恢复决策 | `cluster_recovery.py` 156 行（"Fail-closed recovery decisions"） |
| 断点续传下载 | `model_supply_assets.py:2084 _read_resume_journal`、`:2130 class _ResumeJournal`、`:2253 _copy_partial_prefix`（约 300 行） |

### §4.4 多套并行的抽象层（provider / adapter / worker 三层套娃）

- **provider 3 份**：`execution_provider.py` 1,135（`ExecutionProvider` Protocol + `ProviderRegistry` + `LocalFullModelProvider` + `CallbackExecutionProvider` + `InProcessWorkerProvider` + `DeterministicFakeProvider`）、`external_provider.py` 684、`provider_soak.py` 92。
- **worker 12 份 / 9,094 行**：`scheduler_task_worker.py` 2,730、`task_worker_adapter.py` 1,968、`task_worker_protocol.py` 1,661、`worker_prepare.py` 674、`worker_prepare_coordinator.py` 424、`worker_assignment_state.py` 384、`task_worker_io.py` 289、`task_worker_chunks.py` 247、`worker_weight_warmup.py` 213、`worker_prepare_contract.py` 197、`worker_runtime_registry.py` 190、`task_worker_capabilities.py` 117 —— 其中很大一部分只为回答「谁有权关闭一个 native 运行时」（`worker_runtime_registry.py` docstring 原话）。
- **contract 10 份 / 3,175 行**：`pipeline_node_contract.py` 887、`qwen3_multimodal_contract.py` 557、`relay_contract.py` 457、`gemma4_pipeline_contract.py` 268、`gguf_metadata_contract.py` 223、`worker_prepare_contract.py` 197、`model_runtime_contracts.py` 194、`llama_rpc_contract.py` 155、`qwen3_pipeline_contract.py` 94、`release_contract.py` 143。

### §4.5 同一功能的多份实现

- **scheduler 8 个文件 / 21,016 行**（占全仓 16%）：`scheduler_pipeline` 8,691 + `scheduler` 4,697 + `scheduler_task_worker` 2,730 + `scheduler_cluster` 2,268 + `scheduler_sidecars` 1,430 + `scheduler_svc_http` 771 + `scheduler_layer_plan` 221 + `scheduler_types` 208。
- **GGUF 头解析 2 份**：`gguf_metadata_contract.py:51 read_gguf_header` 与 `relay_segment_info.py:215 _gguf_header`；`scripts/cut_layers.py:112` 又直接 `import gguf`（第三方库）。三套并存。
- **传输 3 套**：`tcp_comm.py`（生产 TCP）、`wss_loopback.py` 467（WSS 开发门）、`transport_runtime.py` 241（v2 信封）+ `cluster_transport.py` 692（"contains no socket implementation"的纯契约）。
- **段角色 2 套**：`relay_contract.py` 的 `RelayTrimPlan/RelayHandoff` 与 `pipeline_node_contract.py` 的 `PipelineNode/PipelineLayout`。

### §4.6 为兼容历史行为而保留的分支

- 全 `src/` 有 **439 处**「兼容 / legacy / 向后兼容 / 旧行为」标记，`scheduler_pipeline.py` 单文件 115 处。
- `pipeline_capacity.py` 21 行整个文件只是转出口（"Private helpers remain importable during the staged migration"）。
- `model_supply_assets.py:1378-1381`：为「旧 receipt 永久阻塞重建」专门开一个 `_gguf_metadata_contract: 1` 的**第二 generation 目录**。
- `relay_a1_legacy.py` 109 行：A1 relay **已从产品入口剔除**（`config.py:741-748`，`QLH_RELAY_PROBE_ONLY` 默认 1），文件保留只为「历史证据」。
- `request_deadline.py` / `request_outcome.py` 各 2 行，纯转口 `qlh_core.contracts.*`。

### §4.7 数据库 / 持久化

SQLite 出现在 4 处：`local_store.py`、`recovery_journal.py`、`journal_replication.py`、`bootstrap_nonce_ledger.py`；
`qlh_core/adapters/` 下还有 6 个 SQLite 仓储（`catalog_repository` / `model_control_repository` / `request_repository` / `sqlite_journal` / `publication_cleanup` / `local_store`）。
用户明确不要 —— **全部抛弃，零替换**。

---

## §5 新原型 1000 行预算建议

最小可行面 = **裁切一次（离线）+ 段进程常驻 + 一 token 一帧串行**。无 HTTP、无 DB、无状态机、无租约。

| 模块 | 内容 | 预算 | 备注 |
| --- | --- | ---: | --- |
| `config.py` | 9 个环境变量 + 拓扑从 TOML 读 | 60 | 原型已有 70 行，可再压 |
| `protocol.py` | 6 帧（hello/forward/result/error/reset/bye），`[len][type][meta][blob]` | 70 | 原型已有 66 行，形状正确 |
| `cut.py` | GGUF 头解析 + 张量选择 + 重命名 + 写出 + 6 字段 manifest | 180 | 从 `scripts/cut_layers.py` 690 行砍到 180；**保留 4 倍对齐校验** |
| `engine.py` | keep-head 绑定：`tokens→hidden` / `hidden→hidden` / `hidden→token` / reset | 260 | 从 `llama_keep_head.py` 871 行砍到 260；删 subprocess worker、多序列、探测缓存 |
| `stage_worker.py` | 单段进程：监听 + 循环 forward + KV 位置推进 | 120 | 原型已有 112 行 |
| `runner.py` | 逐 token 两段串行 + 贪心 + detokenize | 130 | 原型 `runner.py` 79 + `chain_client.py` 144 |
| `cli.py` | 子命令：`cut` / `serve` / `run` | 60 | |
| `selftest.py` | hidden 逐位对拍（整模 `layer_inp` vs 裁层件） | 120 | **不能砍**，这是唯一的正确性防线 |
| **合计** | | **1,000** | |

**预算外但必须计入工作量**：
1. **C shim**（`qlh_keep_head_gpu.c` 741 行）+ `llama-cpp-layer-forward-api.patch` 248 行 —— 这是既有资产，直接沿用；CUDA 版构建脚本 `build_cuda_shim.ps1` 已在原型目录。
2. **不预留 Rust 控制面 / TUI 行数** —— 那是另一个语言栈，不占这 1000 行 Python 预算。

**与现有 1,276 行原型的差额处理**：现有原型多出的 276 行主要来自 `topology.py`（200 行，支持 N 段 + TOML 校验）、`test_topology.py`（291 行测试）、`bench.py`/`breakdown.py`/`probe_hops.py`/`smoke.py`（共 302 行工具）。建议：**拓扑校验砍到 60 行、固定两段**；工具脚本移出核心包，不进交付行数。

---

## §6 与预期不符的五点（供决策）

1. **`relay_transport.py` 不能复用**。它硬性拒绝非 loopback 端点（`relay_transport.py:535`），跨机靠外部 SSH 隧道。新原型要真跨机，传输层必须重写——但只抄它的 6 帧形状，不抄 1,247 行实现。
2. **keep-head 是硬依赖，不是可选加速件**。裸 llama.cpp 拿不到 `output_norm` 之前的 hidden，必须走补丁 + shim。原型 `engine.py:19` 已经在 `sys.path` 注入旧仓库——这个耦合必须在合并时显式化（把 `llama_keep_head.py` 精简后收进仓库根，C shim 进 `shim/`）。
3. **裁切真身不在 `src/`**。如果只盘 `src/`，会得出「裁切功能只有 200 行」的错误结论——实际 690 行在 `scripts/cut_layers.py`，且是全仓唯一被原型依赖的旧代码（`DESIGN.md:174,196`）。
4. **`scheduler_pipeline.py` 的重点不是接力**。接力循环本体在 `qlh_core/execution/linear.py`，只有 62 行；8,691 行的主体是 A3 供给编排（准备/提交/就绪/清理/重试/恢复）。真正纯接力路径（A1）已被降级为探针（`config.py:741-748`）。这解释了「旧仓库功能多」的具体形态：**不是功能多，是把一条 62 行的循环包在 29,752 行控制面里**。
5. **`tcp_comm.py` 的价值被高估**。2,834 行里可直接迁移的只有约 280 行帧编解码；余下 2,000+ 行是心跳/重连/epoch/注册重放，全部属于「首期不做」。
