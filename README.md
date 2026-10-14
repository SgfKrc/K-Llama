# K-Llama

K-Llama（Llama for Koakuma）是面向异构边缘设备的分布式推理核心：主线是 GGUF/llama.cpp 轻量引擎，主仓同时拥有 PyTorch 分层分布式引擎与**层流水线**（含跨框架逐层接力），交互入口是跨平台 Textual TUI。

> **独立项目 · 非官方**：K-Llama 是独立的学生创新项目（北京交通大学 2026 大创），与 llama.cpp 项目没有隶属、赞助或背书关系，也不代表其官方立场。项目基于 llama.cpp 构建（描述性引用），不主张对 "llama.cpp"、"llama" 或任何上游名称的所有权或商标权；上游组件保留其自身许可证与版本锁定。

> 状态：**现行**
>
> 更新日期：2026-10-14
>
> 本 README 只描述主仓当前边界和可复现入口。实验记录、历史实现和外置子项目不等同于主仓生产能力。
>
> English: [docs/README.en.md](docs/README.en.md)

## 主仓做什么

- 在 Windows/Linux PC、无 CUDA 设备和 Android 节点上运行或协调 GGUF 推理。
- 让单机装不下的模型由多个节点按层段合同共同承载；每个节点只持有实际分配的模型部分。
- 让 Edge 节点默认使用 1B 以内模型完成本地推理，同时保留作为大模型 RPC worker 的能力。
- 用同一套层段合同管理三类节点（本机、远端 RPC、跨框架），并按设备画像与能力选择引擎。
- 通过设备画像、容量计划、模型身份、层段合同、租约和 epoch fencing 控制分布式准入与故障恢复。
- 用 Textual TUI 访问聊天、状态、模型资产、节点、分布式布局、队列、设备、日志和设置；只读单命令走标准库薄层。

引擎分**双档**，两档都在主仓：

| 档 | 引擎 | 能力 | 依赖 |
| --- | --- | --- | --- |
| **L 档** | llama.cpp / GGUF | 单机推理、RPC 部分驻留、TUI 对话、优化三件套 | 无 torch（Edge 默认） |
| **D 档** | PyTorch / Safetensors | 层拆分与张量放置、层间流水线、多节点层段承载、跨框架接力上游侧，以及同卡对照实验 | torch |

L 档内部按能力再分两条路径，由 `src/inference_path_selector.py` 的 `select_inference_path` 裁决、`src/config.py` 的 `INFER_PATH` 开关控制（默认 `auto`）：检测到 CUDA 且本机有 `llama-server` 二进制时走 `llama_server` 子进程路径，否则走 QLH 引擎内的 llama.cpp 绑定路径；层段接力强制绑定，显式请求 `llama_server` 而二进制缺失时 fail-closed。两条路径对外都报 `engine_type="llama_cpp"`（`src/llama_server_engine.py`），差异经 `inference_path` / `capability()` / `supports()` 暴露。

D 档不进入 Edge 默认依赖，但层拆分、层流水线、跨框架接力由 PyTorch 系实现（`src/model_module.py`、`src/tcp_comm.py`、`src/qwen3_pipeline_*.py`）。同卡单序列实测 llama.cpp 更快（约 4.3×，口径见[跨框架层接力项目报告](docs/跨框架层接力-项目报告.md)），默认生产路径是 L 档。

## 架构总览

K-Llama 是**一个进程里的两层**：面向人的控制面，以及面向机器与协议的引擎层。两层之间只有一条边界 —— 层段合同 `(layer_range, engine, location)`。

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 控制面（面向人）                                                            │
│ Textual TUI · 只读单命令 · HTTP API(/api/cluster/*)                         │
│ 设备画像 · 容量计划 · 层段合同 · 租约 / epoch fencing · 准入                 │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ 层段合同 (layer_range, engine, location)
┌──────────────────────────────┴─────────────────────────────────────────────┐
│ 引擎层（面向机器）                                                          │
│ L 档 llama.cpp / GGUF               D 档 PyTorch / Safetensors             │
│ ├ 单机推理（Edge 默认）             ├ 层拆分与张量放置                      │
│ ├ ggml RPC worker（借算力）         ├ 层间流水线（qwen3_pipeline_*）         │
│ └ 层段前向（keep-head shim）        └ 跨框架接力上游（model_module）         │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ 层段通道：v3 stage（stage_offer_v3 / layer_forward）
┌──────────────────────────────┴─────────────────────────────────────────────┐
│ 节点与传输（跨机、异构）                                                    │
│ 本机 loopback · SSH 隧道 · Surface(x86_64, Windows) · y700(ARM64, Termux)   │
│ hidden 压缩：f32 / f16 / bf16 / int8_block128 · 弱网带宽与额外延迟口径       │
└────────────────────────────────────────────────────────────────────────────┘
```

一条已实测通过的四段链路：

```
prompt → torch(0..7) →hidden→ Surface(8..15) →hidden→ y700(16..19) →hidden→ llama(20..23+head) → token
           本机 CUDA              x86_64 Windows         ARM64 Android           本机 llama.cpp
```

判据是**与同精度整模逐 token argmax 一致**；分叉即标 FAIL，不使用近似判据。逐段记录见[跨框架接力当前有效基线](docs/跨框架接力-当前有效基线与后续优化计划-2026-09-21.md)。

两层的验证强度不同，改动落点决定验证成本：

| 判据 | 控制面 | 引擎层 |
| --- | --- | --- |
| 主要使用者 | 人（终端用户 / 运维） | 其他软件（TUI、API 编排、跨节点对端） |
| 失败后果 | 体验退化，可重试 | 数值错误，可能**静默**错算 |
| 接口稳定性 | 可演进（页面 / 命令可改） | 强契约（层段合同、协议版本、记录 schema） |
| 可否单独替换 | 可以（换前端不动引擎） | 换引擎即换数值语义，须重跑逐 token 对照 |

因此：改控制面跑 UI 与合同测试；改引擎层必须逐 token 对照 + 记录 schema 校验 + 矩阵化实测。

## 层流水线与跨框架逐层接力

### 统一节点抽象

层流水线把"谁持有哪些层、用什么引擎、在哪里、容量多大"统一成一条抽象：`(layer_range, engine, location)`。三种节点同构，共享同一套合同与校验：

| `kind` | 含义 | 引擎 | 通信 |
| --- | --- | --- | --- |
| `local` | 本机进程内的层段 | llama.cpp / pytorch | 进程内 |
| `remote_rpc` | 借来的算力（Android/PC 上的 `ggml-rpc-server`） | llama.cpp | 网络（ggml RPC） |
| `cross_framework` | 跨引擎的层段接力（torch 上游 + llama.cpp 下游） | 两种 | 进程内或 stdio |

实现与端点：

- `src/pipeline_node_contract.py`：`PipelineNode` 合同、既有产物映射、布局 fail-closed 校验；
- `src/pipeline_capacity.py` 容量求解，`src/pipeline_assignment_manifest.py` 分配 manifest；
- `src/pipeline_reshard.py`：容量重解 + 工件就绪门 + epoch 原子提交；
- `GET /api/cluster/layers`、`GET /api/cluster/pipeline-capacity`、`GET /api/cluster/pipeline-reshard`。

### 跨框架逐层接力（D→L）

上游 PyTorch 层段算到第 N 层，把 hidden states 交给下游 llama.cpp 裁层模型继续算完。注入点是 llama.cpp 的 `llama_batch.embd` 字段，PyPI 版 `llama-cpp-python` 即可完成，无需 fork 或重新编译。

它解决两件事：让单机装不下的模型被多台设备承载（层流水线共用同一套层切分与 hidden 传递机制），以及让引擎能力不同的节点（`local` / `remote_rpc` / `cross_framework`）共处一个层流水线；切点分配、混合精度、算子替换、批量交叠等实验也建立在层间可传 hidden 之上。

**当前结论**（判据、速度、容量分开记录，详细数据与实验方法见[当前有效基线](docs/跨框架接力-当前有效基线与后续优化计划-2026-09-21.md)）：

| 项 | 口径 |
| --- | --- |
| 正确性 | 主仓双引擎 D→L 矩阵 27/27 逐 token 一致（qwen2.5-0.5B K=4/8/12/16/20、qwen3.5-2B K=8/12/16/20；负载 prefill 32/128/512、decode 32/64/256；batch 2/4；混合精度 fp16·f32·NF4 × Q4_K_M） |
| 样本 | qwen2.5-0.5B、12+12 层、gen=32：上游 `model_module.forward_layers` + 下游 `llama_engine.forward_layers_from_hidden`，**47.501 ms/步**，与纯 llama.cpp 整模逐 token 一致 |
| 容量 | 两段切分收益 qwen2.5-0.5B **1.568×** / qwen3-2b **1.457×**（严格口径：`C` 只计不属于任何 `*.layers.<i>.*` 的张量，实算见 `src/relay_capacity_budget.py`；把 `model.visual.*` 视觉塔也算可分层时为 1.613×）；同一受控 3.0 GB CUDA 预算下整模被拒、12 层上游通过（该预算是可复现实验约束，非物理 OOM） |
| 上游 compile | 上游 PyTorch 全精度 + 下游 GGUF 量化是刻意的"不完整量化"策略，必须与同精度下游整模对拍 |
| L→L 通道 | pip 绑定的 `llama_get_embeddings_ith` 返回 `output_norm(H)`（实测 cos 0.999998），不能作层接力上游；补丁版 keep-head 通道已打通，`--path l2l_keep_head` / `d2l2l_keep_head` 实测 32/32（含「1 torch 上游 + 2 llama 下游」三段） |
| 切点求解 | `scripts/relay_cut_plan.py` + `src/relay_cut_objective.py`：从实测拟合段画像（固定开销 + 每层耗时）再求解，输出 `capacity_feasible` / `latency_estimate` / `risk_penalty`；n 段、含 Qwen3.5 的 4 层倍数硬约束。2 段闭环在 qwen2.5（r² 0.96/0.99）与 qwen3.5（0.79/0.96）通过 |
| 生产定位 | 正确性已满足 Relay 合同准入；速度只影响默认路由倾向。跨机长时（LL/DL 三段 `gen=256` 逐 token 一致）与多段切点组合（G2/G3 32/32）已于 2026-09-26 通过；资产自动分发与故障注入验收未完成 |

**关键量化结论**：两侧计算占 99.3%（通信 + 同步 + 批量仅 0.34%）；单项最大收益是消除上游对 20 层的空转（10.7×）；层流水线上游只载 `embed_tokens + L0-3`（1.47 GB vs 4.55 GB，3.1×）。已否证的假设：进程边界（仅 8%）、复用 `llama_batch`（0.09%）、上游层朴素截断（数值错）、把 `--override-tensor` 当提速（实为容量旋钮）。瓶颈在两侧计算（切点、kernel、批量），不在传输层。

**P0 切点扫描（2026-09-18 实测，上游跑 GPU）**：最优 N=20 为 129.1 ms/步，比不接力的 186.8 ms/步快 1.45×；上游 GPU 约 2.5–4.3 ms/层，下游 CPU llama.cpp 约 5.8–8.6 ms/层。上游跑 CPU 时相反：不接力最快，总时间对切点几乎不敏感。两条通用约束：正确性不随切点变化；切点必须是 `full_attention_interval`（Qwen3.5 = 4）的整数倍，否则裁层 GGUF 层类型错位而无法加载。

**与"全 llama.cpp + CUDA"的对照（2026-09-18 实测）**：llama.cpp + CUDA 整模（build-cuda，`-ngl 24`，f16）为 26.6 ms/token，跨框架接力最优为 129.1 ms/token，慢 4.85×。集群里有 CUDA 节点时，最佳实践是把它当作 llama.cpp 的 CUDA worker（RPC / 分片），而不是接力上游。P2 交叠（上游 GPU torch 与下游 CPU llama.cpp 交错推进）把 serial 101.11 降到 78.31 ms/token（1.291×）。以上均为同机数据。

报告：`local_docs/evidence/relay-xframe/CORE-RELAY-XFRAME-02-*.json`；票号口径见[验收清单](docs/验收清单与资源限制登记.md)。

### torch.compile 与层循环开关

分段前向要吃到 `torch.compile` 收益，靠 `src/config.py` 的两个开关配合（可用环境变量覆盖）：

| 开关 | 默认 | 作用 |
| --- | --- | --- |
| `USE_COMPILE` | `True` | 启用编译。编译不可用时告警并回退 eager，不影响启动；Windows 未装 [`triton-windows`](requirements-compile.txt) 时走此路径，装了即可用（2026-09-19 起原生实测编译成功） |
| `USE_MONOLITHIC_FORWARD` | `False` | 打开后额外编译层循环（`_LayerLoop`），供 `forward_layers()` 使用 |

只编译层循环的原因：`Qwen2Model.forward()` 的返回值要过 `self.norm`，而分段前向在 `has_lm_head=False` 时必须返回未过 norm 的 raw hidden，所以只能包住层循环，前后置仍由 `forward_layers()` 负责。实测收益（`USE_MONOLITHIC_FORWARD=True`）：Qwen2.5-0.5B 12 层 1.674×、Qwen3.5-2B 24 层 1.269×（两组模型/层数/prefill 口径不同，加速比不可直接比较）。

三条边界：compile 与 eager 非逐位一致（hidden 差异为 f16 的 1 ULP，来源是 attention 实现通路，不能据此判定 compile 更差）；有"逐 token 一致"验收判据的场景不得开启 compile；Windows 需要 `PYTHONUTF8=1` 与 `triton-windows`，两者缺失只会拿不到收益，不会崩。报告：`local_docs/evidence/relay-xframe/CORE-RELAY-XFRAME-02-a4-layer-loop-2026-09-18.json`、`…-b14-hybrid-layer-loop-2026-09-18.json`、`…-compile-numerics-2026-09-18.json`。

### 顶层透明性

TUI 与 API 顶层只需知道聚合资源（GPU/CPU/内存）和是否分布式，引擎选择按资源 + 能力 + 目标决定。见[层流水线的节点类型与顶层透明性](docs/archive/relay/层流水线节点类型与顶层透明性-可行性确认-2026-09-17.md)。

## 当前状态

| 能力 | 当前口径 |
| --- | --- |
| Textual TUI | 统一 `qlh` 入口；聊天、9 个功能屏和 1 个调试兜底屏共用一个进程，可在本机按需启动后端；模型下载/搜索/预检/登记、集群配置、节点管理、日志筛选/统计/导出、设备配置、用户设置等写操作经确认框闸门 |
| Edge <=1B 单机 | 已有模型画像、GGUF/llama.cpp 路径和边缘预检；默认不加载 torch |
| 同机双进程 RPC | 已有 llama host + `ggml-rpc-server` 模拟及合同测试；不等同于跨机生产准入 |
| PC RPC | 有设备评分、自动层数规划、租约/断线回退和资产同步合同；吞吐收益未宣称，容量收益与 D→L 分开验收 |
| 层段合同与自动重分片 | 合同、布局 fail-closed 校验、容量重解与 epoch 原子提交的开发门已完成；真实 PC/Android 故障注入、长时和性能验收待做 |
| 跨框架逐层接力（D→L） | 正确性已准入（27/27 逐 token 一致）；跨机长时（LL/DL 三段 `gen=256`，256/256）与多段切点组合（32/32）已通过；资产自动分发与故障注入验收待补。默认路由优先可直接整模运行的 L/RPC 路径 |
| PyTorch D 档 | 层拆分/层间流水线/多节点层段承载的实现方，兼作对照实验；不进入 Edge 默认依赖 |
| Relay R | L→L、D→L、f32/采样矩阵和 SSH 跨机证据已完成正确性验证；容量收益已量化；默认路由不替代可直接整模的 L/RPC |
| Android | `qlh-android` P0 交叉编译/JNI 已完成；P1 的设备运行、RPC worker、断线、热/电和安全证据未完成 |
| 运行环境 | 版本窗口以 [`requirements.txt`](requirements.txt) 为准（`transformers>=5.17.0,<5.18.0`、`llama-cpp-python==0.3.35`）；全部 PyTorch 侧车统一到 5.17.0；`.venv-test` 含 `triton-windows`，Windows 原生 `torch.compile` 可用 |
| 模型资产 | 模型文件不入 Git（本机 `models/` 是 Junction，指向仓库外目录）；主仓资产清单支持 Qwen2.5-0.5B、Qwen3-0.6B、Qwen3.5-2B、Qwen3VL-4B、MiniCPM4-0.5B、DistilQwen2.5-DS3-0324-7B 等登记模型；Qwen3.6-35B-A3B 及其社区蒸馏/派生版（`qwen35moe` 与 `qwen3next` 两种架构头）已在本机 llama.cpp `b10964` 实测可加载 |
| 选模与供给意图 | 统一入口 `POST/GET/DELETE /api/models/intent`（含 `/catalog`、`/current`、`/connect`）：用户只提交模型 ID、engine、量化与自动策略，master 沿既有供给与 native prepare 事务完成规划、传输、校验、装载与提交；Web 模型页、Textual TUI 与 Android 选模面板共用同一目录与 operation 进度（`src/model_intent*.py`、`src/api/routes_model_intent.py`） |

没有标注真实设备、跨机或生产验收的实验，只作为开发证据或 PoC 使用。

## 主仓边界

| 保留在主仓 | 外置或不再回流 |
| --- | --- |
| llama.cpp/GGUF 引擎适配、RPC/层段合同、调度与故障恢复 | Android UI/JNI 工程：`qlh-android` |
| FastAPI 控制面、模型/节点/能力合同 | 产品壳和前端：`qlh-shell` |
| 跨平台 Textual TUI、只读命令薄层、Edge 入口和质量门 | 发布/安装器：`qlh-release` |
| 单机、同机双进程和 PC/Android 主线实验接口 | Toolbox：`qlh-toolbox` |
| PyTorch D 档引擎（层拆分、层流水线）与跨框架接力实现 | 生图、Web 产品界面、邮件、运营工作台 |
| 模型注册、下载校验、设备画像和分布式观测 | Koakumix harness 的定制实验、生图和侧车能力 |

主项目不保留生图运行时和生图资产；生图唯一归属 Koakumix。多模态不作为固定主线依赖，由模型舰队按设备能力选择文本或视觉模型。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `src/` | K-Llama 主代码：控制面、引擎、层段/层流水线合同、TUI（分组见下） |
| `tests/` | pytest 套件（TUI、RPC/层段、调度、合同、文档门） |
| `scripts/` | 验证、实验、环境与文档工具（`edge_preflight.py`、`android_validation.py`、`llama_rpc_*.py`、`relay_*.py`、`run_doc_checks.py` 等） |
| `docs/` | 现行文档；历史与已迁移内容在 `docs/archive/` |
| `schemas/` | 跨进程/跨仓 JSON Schema 合同（artifact-manifest、cluster-profile、experiment-record 等） |
| `fixtures/` | 测试固件（含离线聊天事件回放，供 `qlh chat --fixture` 使用） |
| `local_docs/` | 本地实验与验收原始记录；不作为公开源码接口 |
| `runtime/` | 运行期日志与 llama.cpp 运行时目录 |
| `qlh.py` / `qlh_edge.py` | 交互 TUI/CLI 入口与 Edge 入口 |
| `qlh.bat` / `qlh.sh` / `K-Llama.bat` / `K-Llama.sh` / `bjtu.*` / `koakuma.*` | 启动器；`K-Llama` 是推荐别名，`bjtu`、`koakuma` 是兼容别名，统一入口脚本是 `qlh.py` |
| `start_tui.*` / `start_backend.bat` / `setup_all_envs.*` | 一键启动与多环境安装脚本 |
| `requirements*.txt` / `pytest.ini` / `pyrightconfig.json` | 依赖清单与工具配置 |
| `models/`、`chat_history/`、`dist/`、`build/`、`test-results/`、`logs/` | 本地产物或归档区，不入 Git |

**子模块（4 个）** —— 由 `.gitmodules` 登记，`git submodule update --init` 拉取：

| 子模块路径 | 远端 |
| --- | --- |
| `android/` | `qlh-android` |
| `frontend_cybergothic/` | `qlh-shell` |
| `packaging/` | `qlh-release` |
| `harness_workbench/` | `Koakumix` |

**关联仓库（开发/实验工具，不是子模块）** —— 需另行 clone 到本地，已被 `.gitignore` 覆盖、不进主仓：

| 本地目录 | 远端 | 说明 |
| --- | --- | --- |
| `tools/docagent/` | `qlh-docagent` | 文档维护扫描器 |
| `tools/toolbox/` | `qlh-toolbox` | 工具集 |
| `tools/reasonix-codex-bridge/` | `reasonix-codex-bridge` | Reasonix ↔ Codex 受控桥（MCP + ACP） |
| `tools/dsh-codex-bridge/` | `dsh-codex-bridge` | 孪生项目（DSH 侧，vendor 了桥接器运行时） |
| `packages/spawnledger/` | `spawnledger` | 进程归属票据工具 |

`src/` 按职责分组（逐模块接口见[模块接口说明](docs/模块接口说明.md)、[核心技术原理](docs/核心技术原理.md)）：

| 分组 | 代表模块 |
| --- | --- |
| 控制面与入口 | `api_server.py`、`api_errors.py`、`config.py`、`bootstrap.py`、`model_api_access.py`、`review.py`、`local_store.py` |
| 调度与集群 | `scheduler.py`、`scheduler_svc_http.py`、`cluster_join.py`、`cluster_transport.py`、`edge_cluster.py`、`node_config.py`、`node_runtime.py`、`transport_runtime.py`、`transport_port.py`、`network_address.py`、`network_path.py`、`proxy_config.py`、`wss_loopback.py` |
| 引擎 | `llama_engine.py`（L 档）、`model_module.py`（D 档层拆分）、`island_engine.py`、`koakuma_engine.py`、`tcp_comm.py`、`inference_client.py`、`inference_svc_main.py`、`inference_service/`、`paged_kv_cache.py`、`external_provider.py`、`speculative*.py` |
| RPC 与层接力 | `llama_rpc_contract.py`、`llama_rpc_device.py`、`llama_rpc_planner.py`、`relay_contract.py`、`relay_planner.py`、`relay_transport.py` |
| 层段/层流水线合同 | `pipeline_node_contract.py`、`pipeline_capacity.py`、`pipeline_assignment_manifest.py`、`pipeline_model_descriptor.py`、`pipeline_reshard.py`、`cache_unit_layout.py` |
| 跨框架/多模态流水线 | `qwen3_pipeline_*.py`、`qwen3_multimodal_*.py`、`gemma4_pipeline_*.py`、`multimodal.py` |
| 模型与资产 | `model_config.py`、`model_host.py`、`model_sync.py`、`model_downloader.py`、`model_download_jobs.py`、`model_search.py`、`model_registry_validation.py`、`model_runtime_contracts.py`、`local_model_assets.py` |
| 任务图与工作流 | `task_graph*.py`、`task_journal.py`、`task_provider.py`、`task_worker_*.py`、`graph_orchestrator.py` |
| TUI 与交互 | `tui_textual.py`、`tui_api.py`、`tui_shared.py`、`tui_sse.py`、`tui_backend.py`、`tui_commands.py` |
| 设备与压测 | `device_profiler.py`、`provider_soak.py`；RAG 已外置至 `harness_workbench`/Koakumix |

## 快速开始

### 1. 获取代码和子模块

```bash
git clone https://github.com/SgfKrc/LEDS_BJTU.git
cd LEDS_BJTU
git submodule update --init --recursive     # 只拉 4 个子模块
```

子模块远端见 `.gitmodules`。关联仓库（开发/实验工具）按需另 clone 到 `tools/`、`packages/` 下：

- 子模块：`qlh-android` · `qlh-shell` · `qlh-release` · `Koakumix`
- 关联仓库：`qlh-docagent` · `qlh-toolbox` · `reasonix-codex-bridge` · `dsh-codex-bridge` · `spawnledger`

### 2. 选择运行环境

主环境带 torch，适合 D 档层流水线、跨框架接力和完整 API：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Edge 环境只装 GGUF/llama.cpp 与控制面依赖，不装 torch、Transformers 或 bitsandbytes：

```powershell
python -m venv .venv-edge
.\.venv-edge\Scripts\python.exe -m pip install -r requirements-edge.txt
.\.venv-edge\Scripts\python.exe scripts/edge_preflight.py --python .venv-edge\Scripts\python.exe --json
python scripts/llama_dependency_contract.py --json
```

Gemma 4 MTMD 档位独立：`.venv-gemma4-native` 使用冻结的 `llama-cpp-python==0.3.28` 绑定，不复用普通 CPU wheel，其 ABI 标记与 lock 单独校验。

Ubuntu/macOS 将 `Scripts\python.exe` 换成 `bin/python`。交互 TUI 依赖 Textual，只读命令、协议层与 CI 检查不需要它；`uvicorn/FastAPI` 只在本机自动启动后端时需要，远程 TUI 不会替远端启动本机后端。只装交互 TUI 依赖：

```powershell
python -m pip install -r requirements-tui.txt
```

### 3. 启动 TUI

```bash
python qlh.py chat
python qlh.py chat --route distributed_preferred --thinking
python qlh.py chat --fixture fixtures/chat_interactive_fixture.sse
python qlh.py status
python qlh.py models
```

`qlh chat` 在本机后端未运行时于当前进程的 daemon 线程中启动后端，并在启动屏显示探活阶段；后端日志写入既有日志文件与日志页面。`status`、`models` 等单命令不自动启动后端，`--fixture` 是不联网的聊天事件回放路径。写操作从外壳发起：模型屏 `L` 选模（自动供给并准备）/ `U` 卸载 / `C` 取消选模，队列屏 `P` 暂停-恢复 / `S` 策略 / `C` 清空排队，聊天屏可用 `/model`、`/queue`、`/new`、`/resume`、`/rename`、`/sessions`、`/delete-session`、`/reset`；破坏性与长耗时操作先弹确认框。模型控制接口按 loopback 默认放行，远程控制主节点需主节点配置 `QLH_MODEL_API_TRUSTED_CIDRS`。

Windows 可直接用 `qlh.bat`（或 `K-Llama.bat`），Linux/macOS 用 `qlh.sh`（或 `K-Llama.sh`）；三者转发到同一入口脚本 `qlh.py`。页面清单与命令语义见 [TUI 使用指南](docs/TUI使用指南.md) 与 [TUI 指令集](docs/TUI指令集.md)。

## 模型与分布式

模型工件、下载缓存和大型 GGUF 文件不进入 Git。模型清单和能力画像由主仓 API/TUI 管理，模型必须通过格式、摘要、架构、模板、thinking、设备预算和来源校验后才能进入可用列表。

当前推荐验证顺序：

1. 单机加载一个 <=1B GGUF，验证模板、thinking 开关和流式输出。
2. 同机启动 host 与 `ggml-rpc-server`，验证部分驻留、容量合并、输出对拍和 worker 断开。
3. 在 PC 节点完成真实 RPC、资产同步、租约和故障恢复验收。
4. 涉及跨框架或层间流水线时，先在本机复核 D→L 接力的数值一致性与性能边界。
5. 再进入 ARM64/Android worker 验收。

不能用完整模型复制、任务图整请求并行或旧 PyTorch 双机结果宣称"模型已分片"。节点故障时的小模型绕行是调度策略，不是独立产品形态。模型获取与设备画像分档见[项目速览](docs/项目速览-QLH-at-a-Glance.md)。

## Android 验证

Android 工程位于子模块 `android/`，开发机没有 Android 真机时使用分层证据：

```powershell
# JVM 协议/状态机/能力合同测试
python scripts/android_validation.py

# 同时构建 fullDebug APK
python scripts/android_validation.py --assemble

# 连接模拟器或 adb 真机后，安装并启动控制面
python scripts/android_validation.py --assemble --install --launch --serial emulator-5554

# 输出机器可读证据
python scripts/android_validation.py --assemble --json
```

x86_64 模拟器可验证 APK、UI、权限、网络和生命周期，不能证明 `arm64-v8a` JNI RPC worker；ARM64 AVD/QEMU 只能补 ARM 兼容性，不能替代真实手机的热/电、后台回收、弱网和长时测试。Android P1 的完整判据和 AVD/QEMU/远程 adb 方案见 [Android 验证替代路径](android/Android验证替代路径-2026-09-18.md)。

## 测试

测试环境独立于主环境：

```powershell
python -m venv .venv-test
.\.venv-test\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv-test\Scripts\python.exe -m pytest -q
```

高风险主线的定向检查：

```powershell
.\.venv-test\Scripts\python.exe -m pytest -q tests/test_tui_textual.py tests/test_tui_write_ops.py tests/test_tui_shared.py tests/test_tui_sse.py
.\.venv-test\Scripts\python.exe -m pytest -q tests/test_llama_rpc_planner.py tests/test_llama_rpc_device.py
.\.venv-test\Scripts\python.exe -m pytest -q tests/test_pipeline_node_contract.py tests/test_pipeline_reshard.py tests/test_pipeline_capacity.py
```

串行全量基线：**6055 passed / 33 skipped / 0 failed**（`-n 0`，2026-10-14，HEAD `b84522f3`）。xdist 并发下偶有 flaky，判定以串行为准。

真实硬件、跨机网络、Android ARM64、性能和长时 soak 必须另存原始命令、环境、模型摘要、拓扑、输出和失败边界，测试绿灯不替代这些证据。测试通道划分与标记语义见[测试通道运行说明](docs/测试通道运行说明.md)，判据与对照组见[测试与评判标准](docs/测试与评判标准.md)。

**文档检查**（纯静态、只需标准库）：`python scripts/run_doc_checks.py` 跑两条检查 —— 相对链接死链、README 双语结构同步。同一套检查在 CI（[`.github/workflows/checks.yml`](.github/workflows/checks.yml)）与本地 pre-push 钩子（[`.githooks/`](.githooks/README.md)，用 `git config core.hooksPath .githooks` 启用）各跑一遍 —— 一处定义、两处复用。

## 文档入口

- [整体架构](docs/整体架构.md) · [模块接口说明](docs/模块接口说明.md) · [核心技术原理](docs/核心技术原理.md) · [分布式资源调度系统](docs/分布式资源调度系统.md)
- [跨框架接力-当前有效基线与后续优化计划](docs/跨框架接力-当前有效基线与后续优化计划-2026-09-21.md)（接力方向的索引文档）· [项目报告](docs/跨框架层接力-项目报告.md) · [容量收益实测](docs/跨框架层接力-容量收益实测-2026-09-21.md)
- [TUI 使用指南](docs/TUI使用指南.md) · [TUI 指令集](docs/TUI指令集.md)
- [测试与评判标准](docs/测试与评判标准.md) · [测试通道运行说明](docs/测试通道运行说明.md) · [发行物门](docs/发行物门-RELEASE-ARTIFACTS.md)
- [主线开发计划：分布式推理与边缘优化](docs/主线开发计划-分布式推理与边缘优化-2026-09-14.md) · [主节点动态选举与分布式管理 P4.5 立项](docs/主节点动态选举与分布式管理-P4.5立项-2026-09-21.md)
- [分布式推理并行与跨框架路线调研汇总](docs/分布式推理并行与跨框架路线调研汇总-2026-09-15.md) · [KTransformers 优化迁移调研与算法数据层优化方向](docs/KTransformers优化迁移调研与算法数据层优化方向-2026-09-23.md)
- [遗留清单（当前台账）](docs/遗留清单-2026-10-07.md) · [已知问题记录](docs/已知问题记录.md) · [验收清单与资源限制登记](docs/验收清单与资源限制登记.md) · [未完成工作备忘（历史登记）](docs/未完成工作备忘-2026-09-23.md) · [DIST 后续核心链路缺陷审计](docs/DIST后续核心链路缺陷审计-2026-10-07.md)
- [文档状态与清理清单](docs/文档状态与清理清单.md) · [归档索引](docs/archive/README.md)
- [Android 验证替代路径](android/Android验证替代路径-2026-09-18.md)

历史计划和已迁移能力位于 `docs/archive/`；本地实验产物和验收原始记录位于 `local_docs/`，不作为公开源码接口。

## 许可证

主仓使用 [MIT License](LICENSE)。各子模块和上游 `llama.cpp` 保持其自身许可证与版本锁定。
