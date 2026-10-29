# slimarc 架构（定稿）

> 状态：**设计定稿**
> 更新日期：2026-10-28
>
> 单机性能数据见 `MEASUREMENTS.md`，旧仓库盘点见 `INVENTORY.md` / `BASELINE.md`。

---

## 1. 核心思想

**每个节点加载自己那份层段工件，而 llama.cpp 把每份工件当作完整模型对待。**

由此得到两条互不干扰的优化路径：

```
路径 A（段内）：每段独立 offload
  GPU 节点：非专家上卡，专家按显存余量分配
  CPU 节点：全留内存，或交给 SLIM-ARC 做页级优化

路径 B（段间）：按层切分，跨机传 hidden
  一台机器装不下时，把层分给多台
```

**两条路径正交。** 段内怎么 offload 不影响段间怎么切，反之亦然。这是本设计的全部复杂度来源，
也是它唯一的抽象。

### 1.1 为什么不"整模 + 分布式 offload"

llama.cpp 的 `-ncmoe` 只能在**同一进程**内把张量分配到显存/内存。跨机没有对应机制
（其 RPC 后端是另一套东西，且要求同构构建）。

所以跨机只能按层切。一旦按层切，每段自然就是独立的 llama.cpp 实例，offload 自动可用。

### 1.2 装得下时不该拆

在 35B-A3B + 31.6 GB RAM 这个组合下：

| 方案 | tg128 |
| --- | ---: |
| 单机单进程 `-ncmoe 32` | **28.02** |
| 单机两进程 | ~17 |
| 三机三段（预期） | ~17.5 |

**拆分在本机是倒退。** 它的价值只在"单机装不下"时体现（如未来 80B-A3B 45 GB）。
这一点必须写进设计，避免后人误以为拆分能提速。

---

## 2. 设备与切分

| 节点 | 硬件 | 层段 | 引擎 | offload 配置 |
| --- | --- | --- | --- | --- |
| master | 14C/20T / 31.6 GB / RTX 4060 8GB | `[0,8)` | llama.cpp CUDA | `-ngl 99 -ncmoe 0`（全上卡） |
| master | 同上 | `[8,24)` | llama.cpp CUDA | `-ngl 99 -ncmoe 13`（前 13 层专家留 CPU） |
| ROG | 6C/12T / 15.15 GB / RTX 5060 | `[24,40)` | llama.cpp **纯 CPU** | `-ngl 0`，SLIM-ARC 可选 |

**ROG 的显卡不用**（Blackwell sm_120，现有构建是 sm_89，需重编；纯 CPU 足够验证机制）。

切点约束：`full_attention_interval=4`，所有切点必须是 4 的倍数。

---

## 3. 三段工件

| 段 | 文件 | 大小 | 张量 | 含全局张量 |
| --- | --- | ---: | ---: | --- |
| `[0,8)` | `stage-0-8.gguf` | 4.53 GB | 148 | `token_embd` ✓ |
| `[8,24)` | `stage-8-24.gguf` | 8.36 GB | 294 | 无 |
| `[24,40)` | `stage-24-40.gguf` | 8.96 GB | 295 | `output.weight` ✓ |

合计 21.85 GB，整模 20.81 GB，**多出 1.04 GB**——中段和末段各自带的重复全局张量。
这是拆分的固定代价。

---

## 4. SLIM-ARC 的接入位置

SLIM-ARC 是 llama.cpp 的 patch set，**编译期接入**，运行时由环境变量控制。

### 4.1 激活门槛

`slim-arc-runtime.h:14`：

```cpp
use_mmap_buffer && mapping_count > 0 && size_data > (6ULL << 30) && enabled
```

`size_data` 是**模型文件大小**。对照三段：

| 段 | 大小 | 过 6 GiB 门槛 |
| --- | ---: | --- |
| `stage-0-8` | 4.53 GB | ❌ 不激活 |
| `stage-8-24` | 8.36 GB | ✅ 激活 |
| `stage-24-40` | 8.96 GB | ✅ 激活 |

### 4.2 预期行为

用户要求「装得下就让它自己退化回普通 llama.cpp」。按门槛设计，退化由两件事保证：

1. **文件小于 6 GiB 直接跳过**（首段）
2. **内存充足时不产生 page fault**，prefetch 无事可做（中段、末段在 15–31 GB 内存下）

因此明天的 A/B 预期是**持平**。如果实测有明显差异（正或负），说明对退化条件的理解
需要修正——这本身就是有价值的结论。

### 4.3 接入代价

`apply-slim-arc.py`（734 行，19 处 `replace_required` 锚点）做**文本级源码改写**，
不是 `git apply`。这意味着：

- 构建必须 fail-fast（锚点找不到就报错，不能静默跳过）
- 构建脚本要记录 patch 版本与锚点校验结果
- `slim-arc-integration.patch` 是**废弃的**（引用了不存在的符号），不要用

---

## 5. 控制面：Rust

### 5.1 分工

```
Rust 控制面
  ├─ 读 topology.toml
  ├─ 按 host 决定 spawn 本地 worker 还是连接远端
  ├─ 驱动链式调用（token → 段0 → 段1 → 段2 → token）
  ├─ 统计每段耗时、吞吐
  └─ TUI 显示

llama.cpp worker（独立进程，C++）
  ├─ 加载一份段工件
  ├─ 段内 offload（-ngl / -ncmoe）
  └─ SLIM-ARC（如启用）
```

**llama.cpp 不嵌入 Rust 进程。** 理由：

- 崩溃隔离：worker 段错误时控制面能报清楚是哪段、哪个工件
- 复用现有 shim（`qlh_keep_head.c`，620 行 C），不必重写 FFI 绑定
- SLIM-ARC 的 patch 在 llama.cpp 侧，与调用方语言无关

### 5.2 与 Python 版的关系

**协议字节级兼容**（6 种消息，长度前缀帧）。好处：

- Rust 版可逐步替换：先换控制面，worker 仍是 Python
- Python 版（872 行，已验证 25 项测试通过、数值逐位一致）作为参考实现
- 跨机实验时，一端 Rust 一端 Python 也能跑

### 5.3 Rust 版模块

| 模块 | 职责 | 预算 |
| --- | --- | ---: |
| `main.rs` | CLI、模式选择（serve / run / tui） | 80 |
| `topology.rs` | 读 TOML → `Vec<Stage>` | 120 |
| `protocol.rs` | 6 种消息编解码 | 180 |
| `worker.rs` | 子进程 spawn / 终止 / stderr 落盘 | 100 |
| `chain.rs` | 链式调用、耗时统计 | 150 |
| `tui.rs` | ratatui 界面 | 200 |
| **合计** | | **830** |

不引入 async runtime——链式调用严格串行，裸线程足够。

---

## 6. 协议

6 种消息，不变：

| 消息 | 方向 | 载荷 |
| --- | --- | --- |
| `hello` | worker → client | `n_embd`、`layers` |
| `forward` | client → worker | tokens 或 hidden，`n_past` |
| `result` | worker → client | hidden 或 token，耗时 |
| `error` | worker → client | `code`、`detail` |
| `reset` | client → worker | 清 KV |
| `bye` | client → worker | 下线 |

帧格式：`>IB`（4 字节长度 + 1 字节类型）+ payload。
hidden 是**原始 float32**，不走 JSON。

按段类型分流：

| 段类型 | 入 | 出 |
| --- | --- | --- |
| `gpu`（首段） | tokens（JSON） | hidden（float32） |
| `cpu`（中段） | hidden | hidden |
| `tail`（末段） | hidden | token（JSON） |

**已知开销**：loopback 单跳 1.7 ms（实测）。局域网预计 1–3 ms，4 次边界共 4–12 ms。

---

## 7. 配置

`topology.toml` 一个文件描述全部拓扑：

```toml
[[stage]]
model      = "F:/qlh-models/stage-split-3/stage-0-8.gguf"
host       = "127.0.0.1"
port       = 9101
layers     = 8
kind       = "gpu"
gpu_layers = 99
cpu_moe    = 0
cpus       = "0,1,2,3"
threads    = 3

[[stage]]
model      = "F:/qlh-models/stage-split-3/stage-8-24.gguf"
host       = "127.0.0.1"
port       = 9102
layers     = 16
kind       = "cpu"
gpu_layers = 99
cpu_moe    = 13
cpus       = "4,5,6,7"
threads    = 3

[[stage]]
model      = "F:/qlh-models/stage-split-3/stage-24-40.gguf"
host       = "192.168.1.x"
port       = 9103
layers     = 16
kind       = "tail"
gpu_layers = 0
cpu_moe    = 16
cpus       = ""
threads    = 4
```

`host` 是 `127.0.0.1` 则本机 spawn，否则连远端（远端需手工一条命令启动 worker）。

### 7.1 `cpu_moe` 的语义（已实测确认）

**段内语义**，不是全局。实现见 `tools/llama-bench/llama-bench.cpp:1219-1223`：

```cpp
for (int i = 0; i < n_cpu_moe; ++i) {
    patterns.push_back(llm_ffn_exps_block_regex(i));  // "blk\.<i>\.ffn_*_exps"
    merged.push_back({ patterns.back().c_str(), ggml_backend_cpu_buffer_type() });
}
```

`i` 是本段文件内部的行号（从 0 开始）。16 层工件配 `cpu_moe=13` = 前 13 层专家留 CPU。

---

## 8. 不做的事

| 不做 | 理由 |
| --- | --- |
| 自动模型分发 | 一晚实验，手工拷一次更简单 |
| 心跳 / 重连 / 重试 | 局域网内、实验期间，失败直接看报错 |
| 状态机 / 租约 / 事务 / 数据库 | 无状态链式调用，不需要 |
| 动态切分 | 切点在配置里写死，实验要固定变量 |
| 跨框架（vLLM / PyTorch / NInfer） | 首期专注纯 llama 链路 |
| OpenAI SSE 接口 | 与机制验证无关 |
| ROG 显卡 | 需重编 sm_120，本期不值 |
| 安卓 / Kotlin | 不在本期范围 |

---

## 9. 验收

### 9.1 数值（硬性）

首段 hidden 与整模对应层输入**逐位相同**。已在本机验证（16 token）：

```
max abs diff = 0.000000e+00
cosine       = 1.000000119
```

跨机后需重验，确认 TCP 传输无精度损失。

### 9.2 机制（硬性）

三段能正确串联，逐 token argmax 与整模一致。

### 9.3 性能（参考，非判据）

| 方案 | tg |
| --- | ---: |
| 单机单进程 | 28.02 |
| 三机三段 | 预期 ~17.5 |

**不以"超过单机"为判据**，因为数学上不成立。真正要拿到的是：
- ROG 每层耗时（决定未来切分点）
- 真实 RTT
- SLIM-ARC 在"装得下"时是否退化到持平
