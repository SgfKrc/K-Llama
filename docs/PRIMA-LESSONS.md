# prima.cpp 借鉴结论

> 状态：**已核实**
> 更新日期：2026-10-29
>
> 分析对象：`H:\prima.cpp`（344 源文件 / 175,783 行）。只读，未改动。
> 详细证据见 `PRIMA-ANALYSIS.md`。

---

## 1. 它和我们是同构的

prima 的分布式路径**就是按层切 + 显式传 hidden**：

- rank0 单独承载 embedding 子图（`src/llama.cpp:11018-11030`）和 output 子图（`11191-11199`）
- 其余 rank 各一个连续层窗口
- 通信是 ZMQ PUSH/PULL 环（`20492-20527`）+ 两条旁路（末段直送 rank0、只传 STOP）
- hidden 帧：`["sub_gf_out"][dims][float32]`，无压缩

**这与我们的 keep-head shim 是同一件事**，只是它改在 llama.cpp 内部，我们改在外部。

**ggml-rpc 不是它的路线。** 仓库里有 `ggml-rpc.cpp` 和 `GGML_USE_RPC` 分支，
但那是上游 `--rpc` 的张量切分路径，`--world/--rank` 路径一行都没调用。

---

## 2. 它跑不了 MoE —— 这是我们的空间

`src/llama.cpp:17003`：

```c
GGML_ASSERT((model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_QWEN2) && "this model is currently no...");
```

该断言在 `switch` 之前，对所有架构生效。所以 prima 的分布式路径**只支持 LLaMA 与 Qwen2**。
`switch` 内部虽有 `LLM_ARCH_GRANITE_MOE` 分支（L17008），但到不了——前面的断言会先失败。

**Qwen3.6-35B-A3B 的 GGUF 在 prima 上加载即失败。**

---

## 3. "内存压力 <10%" 是怎么来的

机制极简：mmap 惰性 + `MADV_WILLNEED` 无脑预取下一段（`18571`）。

- `init_mappings(false)` 不做 `MAP_POPULATE`
- `manage_graph_tensors(gf[next], MADV_WILLNEED)` 对下一 subgraph 的**全部** weight 整段预取
- **全仓没有一行 `DONTNEED`**，回收完全交给 OS
- `is_graph_loaded()`（用 mincore 算驻留比）是**死函数**，从未被调用

所以"<10%"是论文里的实验测量值，**不是机制保证**。

### 3.1 这个机制对 MoE 是坏交换

预取过滤只按名字含 `"weight"` 且 buffer 非 cuda（`18164`、`18173`），**不认识"这个专家本 token 没激活"**。

| 模型类型 | 预取利用率 | 结论 |
| --- | --- | --- |
| dense | 读 100% 用 100% | 等价变换，无浪费 |
| **MoE（8/256 激活）** | 读 100% 用 **3%** | **32 倍 I/O 放大** |

我们的 MoE 场景不需要这个交换。

### 3.2 与 SLIM-ARC 的对比

| 维度 | prima | SLIM-ARC |
| --- | --- | --- |
| 预取策略 | 无脑下一段全量 | 6 GiB 门槛 + 专家级选择性 |
| 预算控制 | 无 | memory_budget |
| 压力反馈 | 无（死函数） | pressure 状态机 |
| 显式回收 | **无** | 有 |

**SLIM-ARC 更成熟**，但它同样不理解 MoE 的激活稀疏性（其收益前提是模型远大于内存）。

---

## 4. 控制面：约 4,000 行，其中 3,551 行我们不需要

| 模块 | 行数 | 我们要不要 |
| --- | ---: | --- |
| profiler（设备探测） | 2,745 | **不要** |
| HiGHS 整数规划 | 806 | **不要** |
| 编排 / 参数 / 工具 | 260 | 参考 |
| 分布式运行时（`src/llama.cpp` 内） | ~2,000 | 部分参考 |

**这 3,551 行是为"未知异构 + 无配置"付的税。** 我们的段工件固定、三机固定，
这笔税完全不必交——**1000 行 Rust 控制面因此是宽裕的**。

### 4.1 它的调度是纯静态

- 层分配：`-lw "16,16,16,16"`（逗号分隔按 rank 序，`arg.cpp:707-740`）
- 或启动时 profile → gather → 一次 HiGHS 求解 → bcast
- **推理中不重分配**

这与我们的 `topology.toml` 静态配置是同一思路，只是它的表达更啰嗦。

### 4.2 唯一值得抄的复杂东西：拓扑重建

`20684-20829`：设备被剔除后环跨过它重接，被跨节点降级为 `FORWARDER` 只转发不断链。
三态 `WORKER` / `FORWARDER` / `EXIT`。**做容错时这是现成模型**，但首期不做容错。

---

## 5. "15x faster" 的口径

有正式论文（ICLR 2026，OpenReview `h0LjpOG1jq`），核心是 PRP（把磁盘 I/O 藏到计算后）
与 Halda 调度器。

但基线是"同一批设备里的**一台**、而且**装不下**"：

```
Llama3-70B：prima 674 ms vs llama.cpp 10120 ms
```

README 自认小模型会退化成 `32:0:0:0`（全部放一台），**此时两者 15 ms 打平**。

**这与我们实测完全一致**：单进程 28.02 t/s，拆两进程 ~17 t/s。
拆分不提速，价值只在装不下时。

---

## 6. 三档结论

### 直接借鉴

| 项 | 说明 |
| --- | --- |
| hidden 帧格式 | `[key][dims][f32]`，我们的协议已同构 |
| 三通道分离 | data / master（末段直送）/ signal（只传 STOP） |
| 环上 ACK barrier | 防 pipeline 失控 |
| `base_port + rank` 端口约定 | 比逐个写端口简洁 |
| `-lw "16,16,16"` 格式 | 比我们的 TOML 更轻，但 TOML 可读性更好 |
| device_info 字段表 | 直接用作 TUI 列 |
| `NODE_TYPE` 三态 | 做容错时用 |

### 参考思路

- PRP 多圈 k（把 I/O 藏到 k 圈之后）
- 拓扑重建
- HiGHS 目标函数形状 `Σ(a·w + b·n)`

### 不适用

- 每节点全量模型文件（我们按段切）
- 强制 `SPLIT_MODE_LAYER`
- 全局 offload 开关
- 无 TUI（我们要 TUI）
- 无脑全量预取（MoE 下坏交换）

---

## 7. 一个我们需要补的测量：MoE 层间耗时方差

prima 的 Halda 调度器假设**每层耗时是常数**（`common/common.cpp:906-918`）。

但 MoE 不同 token 路由到不同专家，**同一层的耗时会抖动**。串行接力里
总耗时 = Σ各段耗时，所以抖动会被直接放大成 TPOT 方差。

**建议在 `MEASUREMENTS.md` 增加一列：每段 decode 耗时的 p50 / p99 比。**
比值大说明该段是瓶颈候选，切分点应该避开。

现有数据（两进程，未按段拆分）：

```
run 1: median 62.1 ms, p90 84.6 ms   → p90/p50 = 1.36
run 2: median 55.3 ms, p90 61.1 ms   → p90/p50 = 1.10
run 3: median 54.3 ms, p90 64.9 ms   → p90/p50 = 1.20
run 4: median 65.2 ms, p90 82.2 ms   → p90/p50 = 1.26
```

这已显示 10–36% 的尾延迟，需要按段拆开看是哪一段贡献的。
