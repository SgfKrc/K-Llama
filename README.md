# K-Llama v2

**中文** | [English](docs/README.en.md)

> 状态：两段跨机链路可用；DFlash 并行草稿、SLIM-ARC prefetch、TUI、OpenAI 兼容接口、
> 采样与显存校验均已接入
> 更新日期：2026-10-31

面向**非整模加载**的定制 llama.cpp 引擎。整模文件留在原地，每个节点只把属于自己那几层
读进内存，段间只传 hidden state（`n_embd` 个 float32，8 KB）。

Qwen3.6-35B-A3B（MoE，40 层，14.8 GiB）在两台机器上分成 32 + 8 层：本机背 32 层、
注意力与 dense 全上卡，远端背 8 层纯 CPU。解码阶段 **20.9 tok/s**（DFlash k=5），
同一个模型在本机单卡整模只有 10.5 tok/s。

**机制怎么成立的**：每个节点都持有同一份整模，加载时只创建、只读取属于自己那几层的张量——
三层机制是层数覆盖（`LLAMA_LAYER_RANGE`）、名字重映射（本地 `blk.i` → 源 `blk.lo+i`）、
全局张量按需 lazy。接力向量取 `output_norm` 之前的值，段间只传 8 KB hidden。
全文见 **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**。

---

## 与 SLIM-ARC 的关系

本项目的层段接力是自研的，与 [SLIM-ARC](https://github.com/Nexa-Language/SLIM-ARC)
不是同一层的东西。

SLIM-ARC 是中山大学参加 2026 年全国大学生计算机系统能力大赛操作系统设计赛 Proj59 的作品，
定位是**内存受限端侧系统上的 MoE 推理运行时**：mmap 按需加载、阶段感知页建议、
MoE 专家预测与回收、压力感知驻留、统一 I/O 预算与 KV cache 管理。它解决的是
"模型体积远大于物理内存"——代表结果是把 Qwen3-Next-80B-A3B 塞进 8 GiB cgroup，
decode 0.08 → 0.43 tok/s。

同一个问题的两条路径：

| | 本项目 | SLIM-ARC |
| --- | --- | --- |
| 做法 | **横向扩**：把层分给多台机器 | **纵向压**：单机按页换入换出 |
| 前提 | 每个节点装得下自己那段 | 单机装不下整模 |
| 交叉代价 | 8 KB hidden 过网络 | 缺页读盘 |
| 形态 | 多进程 + TCP | 单进程 + 页级调度 |

借用的只有它的 **prefetch 模块**——按层边界预读，把随机缺页变成顺序读。已编译进
`llama.dll` 并做了 win32 移植；它的专家驻留、KV 驱逐、统一 I/O 调度器、on-demand
加载器都没有接。触发条件见 [`docs/SLIM-ARC-接入计划.md`](docs/SLIM-ARC-接入计划.md)，
接入形态见 [`docs/PATCHES.md`](docs/PATCHES.md)。

---

## 这套东西的价值在哪

拆分的用处在单机装不下的时候：那时两台机器的资源是相加的，而不是共享一块显存。

但**拆分本身是负收益**——同一块显存分摊给两段，比单进程慢。跨机之后快出来的部分是
投机解码挣的，与拆分无关：

| 环节 | 手段 | 解码阶段吞吐 |
| --- | --- | ---: |
| 无投机 | 每步 1 token、一次绕链 | 12.2 |
| MTP 递归 | k=2，草稿在末段的 NextN 头 | 1.13× |
| **DFlash 并行块** | k=5，一次出 6 个候选、一轮验证 | **20.9** |

口径：32:40 链路、`def quicksort(arr):`、64-token 生成、只计解码阶段。
写满 128 token 时接受率从 96% 掉到 70%，吞吐 13.4——见 [`docs/SPECULATION.md`](docs/SPECULATION.md)。

---

## 适用范围

- **架构**：Qwen 3.5 及以上（`qwen35` / `qwen35moe`）。别的架构要支持，得先补 tensor 命名、
  RoPE 变体、MTP 位置这几处差异。
- **量化**：8 bit 以下（Q4_K 这类）。这不只是限制，还是**段间传 float32 就够**的前提：
  Q4_K 的相对误差约 1e-2，比 float32 传输误差大得多，传输误差落在量化噪声底下。
- **段数 = 节点数**。同一台机器拆两段是设计错误：每多一个进程就多一份全局张量读取、
  一块 compute buffer、一轮页竞争。

### 传输 dtype 保持 float32

试过压到 4 KB，两个方案都不行：

| 传输 dtype | 单跳 max\|diff\| | 贪心 token 一致 |
| --- | ---: | ---: |
| float32 | 0 | 5/5（基线） |
| float16 | 3.9e-3 | **4/5**，第 22 步分叉 |
| int16 定点 | 2.2e-4 | **4/5**，第 16 步分叉 |

int16 的量化步长比 float16 细 20 倍，照样翻转——分叉点都落在输出接近平局的位置，
1e-4 级的扰动就够。判据取**全部 prompt 的贪心 token 逐位一致**，"大多数一致"不够。

---

## 快速开始

```powershell
koakuma                                    # 一条命令进 TUI
koakuma --topology topology-cross.toml     # 跨机两段

# 直接跑链路（--spec-k 默认 -1 = 按负载自动选）
python chain_client.py --topology topology-cross.toml --prompt-text "中国的首都是" --chat --gen 32

# 加 DFlash 并行草稿（k=5 是实测拐点；需要 --greedy）
python chain_client.py --topology topology-cross.toml --greedy --dflash <dflash.gguf> \
       --dflash-block 6 --prompt-text "def quicksort(arr):" --gen 64
```

依赖：`pip install numpy psutil gguf textual`（`gguf` 只有裁切与溯源校验要，`textual` 只有 TUI 要）。

手动三步：编译 shim（`shim\build_shim_msvc.ps1`）→ 每台放同一份整模 →
拓扑里的 `layer_range` 决定各段拿哪几层。远端 worker 由拓扑的 `start_cmd` 拉起，
完整流程见 [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)。

---

## 结构

```
protocol.py           帧格式与消息编解码
wire.py               段间 hidden 的传输 dtype（默认 float32）
topology.py           拓扑解析 + fail-closed 校验
keep_head.py          同进程的 shim 封装
keep_head_worker.py   子进程封装（n_gpu_layers=0 必须新进程）
engine.py             段的加载与前向
stage_worker.py       单段进程（TCP 服务端）
chain_client.py       拉起 worker、驱动解码、两个草稿源、内存与显存预算
stream.py             流式生成与 UTF-8 重组
server.py             /v1/chat/completions、/v1/models
proto_responses.py    /v1/responses 的纯翻译层
koakuma.py            TUI 与启动画面
cut.py                离线裁段（现在只用于比对）
checks/               20 份验收脚本
```

不写行数：这类数字每次重构都会过期，而职责不会。

---

## 四条必须知道的约束

**① 段必须精确拼满源模型。** 缺层不会报错，只会输出通顺但错的中文。`topology.load()`
每次校验区间连续、无重叠、宽度与声明一致。

**② 显存超额是静默的。** WDDM 允许分配超过专用显存，用系统内存兜住，llama.cpp 照旧打印
`offloaded N/N`。现在启动前预算、加载后实测核对（`chain_client.py` 的 `VRAM_PER_BYTE`）。

**③ 不要绑核。** worker 有 5+ 活动线程，绑到同一物理核的两个超线程上会互相抢占。

**④ 切点和末段层数必须是 `full_attention_interval`（=4）的倍数。** qwen35moe 用本地层号
推算层类型，而本地层号被映射回源层号，只有 `lo` 是 4 的倍数时两者模 4 同余、类型才对得上。
`32:38` 这种 6 层划分会在解析阶段被拒绝。

---

## 实测数字

Qwen3.6-35B-A3B（MoE，40 层，14.79 GiB 权重），单序列，greedy：

| 配置 | 吞吐 | 口径 |
| --- | ---: | --- |
| 本机单卡整模 `-ngl 6` | 10.5 tok/s | 端到端 |
| 跨机两段 32:40，无投机 | 12.2 tok/s | 解码阶段、128-token 生成 |
| **跨机两段 + DFlash k=5** | **20.9 tok/s** | 解码阶段、64-token 生成 |
| 跨机两段 + DFlash k=5 | 13.4 tok/s | 解码阶段、128-token 生成 |

跨机比单机快的那部分来自投机；拆分本身仍是负收益。段内耗时随 `n_past` 呈 U 型，
量法见 [`docs/MEASUREMENT-PITFALLS.md`](docs/MEASUREMENT-PITFALLS.md)。

---

## 未做（按本机可验证性排序）

**1. 非贪心下的投机。** 验证器只暴露 argmax；采样需要 rejection sampling，要 shim 回传
每个位置的 logits。链路是量化权重 + 跨机分段，"两个分布一致"这个前提比单机更弱。

**2. SLIM-ARC 的 on-demand 加载器。** 现在只接 prefetch。要验证它得先造出"段装不下"
的场景——本机整模在页缓存里，`issued` 恒为 0。要么等弱设备，要么人为把可用内存压下去。
它是降级档的优化，不是当前瓶颈。

**3. 多序列 token 共享。** 需要 `kv_unified`，且每段要维护多序列位置。当前 server 用一把
全局锁串行处理请求，单用户场景下并发需求弱。真要做，先想清楚 hybrid 模型那 30 层 linear 的
recurrent state 怎么按序列隔离。

**已完成**：裁段工具退场、两个草稿源接入、DFlash 的 k 扫描、SLIM-ARC prefetch 与 win32 移植、
把投机接进服务路径、TUI 的采样开关。

服务路径那次：`stream.py` 原本自己写了一遍朴素解码循环，导致 server 与 TUI 拿不到 DFlash；
现在它只保留 UTF-8 重组、计时与停止条件，解码交给 `chain_client` 的
`forward` / `speculate` / `DFlashDriver`。server 与 TUI 都新增了投机参数，只对
`temperature=0` 生效——实测同一 greedy 请求开启与关闭 DFlash，输出逐字相同。
细节见 [`docs/PROGRESS.md`](docs/PROGRESS.md)。

---

## 文档

| 文件 | 讲什么 |
| --- | --- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | **核心机制**：凭什么非整模加载、硬约束、判据、与 RPC 的分界 |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | 每台放什么、换分段怎么做 |
| [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md) | 全部实测数据 |
| [`docs/MEASUREMENT-PITFALLS.md`](docs/MEASUREMENT-PITFALLS.md) | 段内计时为什么骗人 |
| [`docs/MEMORY-LIMITS.md`](docs/MEMORY-LIMITS.md) | 内存与显存的账怎么算 |
| [`docs/SPECULATION.md`](docs/SPECULATION.md) | 投机解码的判定、成本模型与实测 |
| [`docs/PATCHES.md`](docs/PATCHES.md) | 三代补丁的差异、补丁全文、与原生 RPC 的分界 |
| [`docs/SLIM-ARC-接入计划.md`](docs/SLIM-ARC-接入计划.md) | 缺页优化在什么条件下才启用 |
| [`docs/LILICORR-轻量头.md`](docs/LILICORR-轻量头.md) | DFlash 的块内一致性重排，以及训练成本 |
| [`docs/ANDROID-SURFACE-段接入评估.md`](docs/ANDROID-SURFACE-段接入评估.md) | 弱设备接入的条件与安卓侧硬约束 |
| [`docs/SAMPLING.md`](docs/SAMPLING.md) · [`docs/TUI.md`](docs/TUI.md) | 采样映射 · TUI 设计 |
| [`docs/ISSUE-recurrent-snapshot-rollback.md`](docs/ISSUE-recurrent-snapshot-rollback.md) | hybrid 模型跨 ubatch 回滚缺陷与修复 |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | 当前能力快照与已做的优化 |

验收：`python checks/verify_*.py`，`python -m unittest test_topology`。
