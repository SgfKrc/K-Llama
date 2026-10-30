# SLIM-ARC 层段推理原型 — 设计

> 状态：**设计定稿，P0 待启动**
> 更新日期：2026-10-28
>
> 适用范围：以 llama.cpp 的 MoE 专家 offload 做层段拆分的**另起原型**。代码目录最终
> 落在 K-Llama 仓库的 `slimarc/`，作为新分支提交；开发期在 `H:\slimarc-proto`。
> 不含 OpenAI SSE 接口、PyTorch、SQLite、Android/Kotlin。NInfer 为并行评估支线。

---

## 1. 关键实测（2026-10-28，本机）

全部数据来自 `build/moe-cache-20261026/b11889/llama-bench.exe`，
模型 `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`（20.81 GiB，40 层，256 专家，激活 8，
hidden 2048）。原始输出 `H:\slimarc-proto\exp\`，汇总 `summary.json`。

### 1.1 决定性结果：只把专家放 GPU

| 方案 | 配置 | pp512 | tg |
| --- | --- | ---: | ---: |
| 纯 CPU | `-ngl 0 -t 8` | 91.5 | 8.23 |
| **专家上 GPU（11 层）** | `-ngl 99 -ncmoe 29` | 263.2 | **30.45 ± 0.27** |
| 整层上 GPU（11 层） | `-ngl 11 -ncmoe 29` | 201.2 | 10.11 |

**只把专家放 GPU 比整层上 GPU 快 3.0 倍。**

原因是 MoE 的权重分布：专家占 88%（19.67 GB），但每 token 只用 8/256。
把「体积最大、访问最稀疏」的部分放进最快的设备，收益最大；而 attention/norm
每层必读、体积小（4.9% + 1.3%），留 CPU 影响有限。

### 1.2 显存硬边界：11 层可装下，12 层溢出

| 专家层数 | `-ncmoe` | tg | 说明 |
| ---: | ---: | ---: | --- |
| 9 | 31 | 27.86 ± 0.62 | |
| 10 | 30 | 29.43 ± 0.44 | |
| **11** | **29** | **30.45 ± 0.27** | 最优 |
| 12 | 28 | 15.29 ± 0.20 | **断崖，溢出到系统内存** |

11 层 × 256 专家 × 1.72 MiB ≈ **4.84 GiB**，加非专家 0.4 GiB 与 KV/workspace
≈ 5.7 GiB < 8 GiB，与实测边界吻合。

**Windows 特有风险**：超显存不报错，WDDM 静默把系统内存当显存用，只表现为掉速
（12 层掉到 15.29）。切分点必须留余量，不能贴边。

### 1.3 次要结论

- **线程数已非瓶颈**：在最优切分下 t=4/6/8/14 得 tg 26.68/27.29/28.18/27.83，
  差异 <6%。CPU 段不再主导。
- **FlashAttention 无差异**：`-fa on` 29.47 与 `-fa off` 29.65 在噪声内。
- **`-ncmoe N` 语义**：前 N 层专家留 CPU，其余 `40−N` 层专家上 GPU。

---

## 2. 目标与非目标

### 2.1 本期目标

| 项 | 内容 |
| --- | --- |
| 主验收 | 同机两进程拆分后，decode 吞吐**保住 29.74 t/s**（单进程基线 30.45，允许少量损耗） |
| 正确性 | 逐 token argmax 与整模推理一致 |
| 结构 | ≈1,400 行 Python、6 种消息、≤14 个环境变量、无数据库、无状态机 |

### 2.2 明确不做

OpenAI SSE 接口、PyTorch、SQLite/任何 DB、Android/Kotlin、4 段以上切分、
租约协议、供给状态机、配置热加载、插件体系。

---

## 3. 架构

```
┌─────────────────────────────────────────────────────┐
│ sa-run     一次请求：tokenize → 逐 token 两段串行 → detokenize │
│            ≈220 行，纯 asyncio，无状态                     │
└─────────────────────────────────────────────────────┘
              │ hidden (BF16, 4 KiB)      ▲ logits/token
              ▼                           │
┌──────────────────────────┐  ┌──────────────────────────┐
│ GPU 进程                  │  │ CPU 进程                  │
│ llama.cpp + CUDA          │  │ llama.cpp                 │
│ 层 [0, 11) 的专家在显存     │  │ 层 [11, 40)               │
│ `-ngl 99 -ncmoe 29`       │  │ SLIM-ARC 可选             │
└──────────────────────────┘  └──────────────────────────┘
```

### 3.1 为什么两段

现有仓库是 4 段，每 token 3 次边界往返；实测链路 + 固定开销约 297 ms，占 49.7%。
本期先做 **2 段（1 次边界）**，把「拆分是否丢性能」与「多段能否配平」分成两件事。

### 3.2 为什么 GPU 段不用 NInfer

NInfer 上游 `cmake/NinferPlatform.cmake` 锁死 `sm_120a`，本机是 sm_89。
社区两个 fork 各有缺失：`CraneBW/ninfer-ternary-bonsai-ada` 有 sm_89 但无 Host
offload 枚举、且是不同代码代际；`headpiece747/ninfer-5090-windows` 有 Windows 与
offload 但锁死 120a。

**层段架构大幅降低了 NInfer 的门槛**（11 层专家 4.84 GiB 全驻显存，不需要 offload、
不需要 pinned host bank、不需要每 token 搬专家），因此 NInfer 仍作为并行支线评估，
但**不阻塞主线**。主线用 llama.cpp CUDA，与 CPU 段同源、边界表示零转换。

---

## 4. 组件预算

| 文件 | 职责 | 预算 |
| --- | --- | ---: |
| `config.py` | 环境变量读取与校验 | 80 |
| `protocol.py` | 6 种消息帧编解码 | 120 |
| `engine.py` | llama.cpp 封装（两段共用） | 300 |
| `gpu_worker.py` | GPU 段进程 | 200 |
| `cpu_worker.py` | CPU 段进程 + SLIM-ARC 开关 | 220 |
| `runner.py` | 两段串行、采样循环 | 220 |
| `cli.py` | 启动入口 | 80 |
| **合计** | | **≈1,220** |

### 4.1 协议：6 种消息

```
hello    worker → runner：层区间、容量、就绪
forward  runner → worker：hidden + positions + seq_id
result   worker → runner：hidden（末段回 logits/token）
error    worker → runner：错误码
reset    清 KV（新会话）
bye      下线
```

不做：offer/accept 两段式、lease 续期、ACK、分片协商、心跳。
现有仓库每段 2 次往返（`relay_cut_objective.py` 的 `STAGE_ROUND_TRIPS = 2`），
本设计压到 **1 次**。

### 4.2 环境变量（14 个）

```
SA_MODEL_FULL    整模 GGUF 路径（两段共用同一文件，各取自己的层）
SA_GPU_LAYERS    GPU 段层数（默认 11，实测最优点）
SA_CTX           上下文长度（默认 8192）
SA_N_THREADS     CPU 段线程数（默认 8）
SA_GPU_ADDR      GPU worker 地址（默认 127.0.0.1:9101）
SA_CPU_ADDR      CPU worker 地址（默认 127.0.0.1:9102）
SA_MAX_TOKENS    单次生成上限（默认 256）
SA_KV_TYPE_K     KV K 量化（默认 q4_0）
SA_KV_TYPE_V     KV V 量化（默认 q4_0）
SA_FA            FlashAttention（默认 auto）
SA_SLIMARC       启用 SLIM-ARC（默认 0）
SA_NO_EXPERT_PREFETCH  SLIM-ARC 专家预取开关（默认 1）
SA_SSD_BUDGET_GB 每小时写入预算上限（超过则停止换入）
SA_LOG_LEVEL     日志级别
```

**不做**：配置热加载、版本化、快照、审计。

---

## 5. 阶段

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| **P0** | 单进程跑通 `-ncmoe 29`，建立 30.45 t/s 基线复现 | 三次重复，方差 ≤5% |
| **P1** | 两进程拆分（GPU + CPU，localhost） | 逐 token argmax 与 P0 一致 |
| **P2** | 吞吐回归：拆分后 ≥29.74 t/s | 允许 ≤2.3% 损耗 |
| **P3** | SLIM-ARC 可选开关 + A/B | 明确开/关哪个更快 |

### 5.1 P1 技术验证结果（2026-10-28，已完成）

**结论：可行，且不需要新写隐藏态通道。**

#### 切分点受硬约束：必须是 4 的倍数

`scripts/cut_layers.py` 对 `qwen35moe` 正确识别并给出确定性拒绝：

```
FAIL: --keep-head N=11 不是 full_attention_interval(4) 的整数倍
      —— hybrid 架构会层类型错位、无法加载
```

`full_attention_interval=4`，因此可选切点为 4/8/12/16...

#### 8 层是可行窗口内的最优

| 专家上 GPU 层数 | `-ncmoe` | tg256 | 说明 |
| ---: | ---: | ---: | --- |
| 8 | 32 | **28.47 ± 0.34** | 可行 |
| 11 | 29 | 30.45 ± 0.27 | 最优，但违反 4 倍数 |
| 12 | 28 | 16.03 ± 0.10 | 显存溢出 |

**采用 8 层**：28.47 t/s 仍显著高于纯 CPU 的 8.23 与目标基线的 20。

#### 段工件已生成并验证独立可跑

```
F:\qlh-models\stage-split\gpu-0-8.gguf    5.03 GB  149 张量  (--keep-head 8)
F:\qlh-models\stage-split\cpu-8-40.gguf  16.81 GB  587 张量  (--k 8)
```

两段独立跑通：

| 段 | 配置 | pp64 | tg16 |
| --- | --- | ---: | ---: |
| GPU（8 层） | `-ngl 99 -t 8` | 1881.7 | 159.3 |
| CPU（32 层） | `-ngl 0 -t 8` | 6.9 | 6.6 |

（单段数字不代表串联性能，仅证明工件可加载、可前向。）

#### hidden 通道已存在，不必新写

`include/llama.h:1058` 只有**导出**接口 `llama_set_embeddings_layer_inp(ctx, lid, true)`，
无注入接口。但 QLH 已通过 keep_head shim 实现完整通道：

| 函数 | 语义 |
| --- | --- |
| `qlh_kh_forward_embd_tokens` | hidden 进、hidden 出（中段）★ 本次用 |
| `qlh_kh_forward_embd_token` | hidden 进、token 出（末段） |
| `qlh_kh_set_skip_hidden_logits` | 中间段跳过 `output.weight` |

Python 侧封装：`src/llama_keep_head.py` 的 `forward_hidden_to_hidden`（中段）、
`forward_hidden_to_token`（末段）。

**因此 P1 的实现面是「复用 shim + 写两进程调度」，不是「改造 llama.cpp」。**

#### 一个必须记录的权衡

整模 `-ncmoe 32` 只把 8 层的**专家**放 GPU（约 3.5 GiB），而切分方案的 GPU 段
含 8 层**全部张量**（5.03 GiB）。**切分方案显存占用更高**，且多一次 hidden 传递。

因此 P2 的吞吐验收（≥29.74 t/s）需要重新对照：
- 整模 `-ncmoe 32` 实测 26.92–28.47 t/s
- 拆分的价值不在单机提速，而在**能把 CPU 段挪到另一台设备**

单机两段拆分若反而慢于整模，属预期，应记录为「分布式入场成本」，而非失败。

---

## 6. 验收口径

分开计时，不合并：

- 冷加载耗时
- 首次 token 延迟（TTFT）
- decode tok/s（主指标）
- 段间单次往返耗时
- 峰值显存 / 峰值内存

**基线**：单进程 `-ngl 99 -ncmoe 29`，tg256 = 30.45 ± 0.27（-r 5，2026-10-28）。

---

## 7. 已知风险

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 显存超界静默降速 | 掉到 15 t/s | 切分点留余量，不贴 11 层边界 |
| llama.cpp 无层段 API | P1 阻塞 | 先做技术验证，退路是裁层工件 |
| 两进程各持一份模型 | 内存翻倍 | 用 mmap 共享同一文件，验证 RSS |
| SSD 磨损 | 低（40–199 年达 TBW）| 但需 `SA_SSD_BUDGET_GB` 刹车 |
| SLIM-ARC 在此场景可能负收益 | 无效开销 | 默认关闭，A/B 后再决定 |
