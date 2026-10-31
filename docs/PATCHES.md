# 两代补丁，以及与原生 RPC 的分界

> 状态：已核对源码
> 更新日期：2026-10-31

本项目是 K-Llama 的第二代。两代的差别不在协议层，而在**模型怎么装进显存和内存**：
第一代靠"裁好的层段工件"，第二代靠"整模 + 层范围"。llama.cpp 侧的补丁因此完全不同。

---

## 第一代（K-Llama / QLH）：层输出通道

改动集中在一处：让上下文多出一个"第 N 层输出"的槽位，供段间接力取数。

- `src/llama-context.cpp`：`layer_inp` 的槽位多分配一个（`n_layer + 1`），
  索引 `lid == n_layer` 即"第 n_layer 层的输入" = **末层输出（`output_norm` 之前）**；
  `set_embeddings_layer_inp` 的断言同步放宽。
- 在 `output_norm` **之前**登记该槽位，覆盖 9 个架构：
  `qwen2` / `qwen35` / `gemma4` / `llama` / `openai-moe` / `qwen3` / `qwen3moe` /
  `qwen35moe` / `qwen3next`。
- 对外是 4 个导出符号，shim 里用到的是
  `llama_set_embeddings_layer_inp()` / `llama_get_embeddings_layer_inp()`。

**为什么不能复用 `t_h_nextn`**：各架构的挂点不一样。`qwen2` 的 `t_h_nextn` 在
`output_norm` 之前，而 `qwen35` 在**之后**（那里的消费方是 MTP head）。拿 `nextn` 当层段
上游，在 Qwen3.5 上会多过一次 RMSNorm，接力的数值首步就分叉。

代价是这套改动要跟着每个架构走，9 个架构就有 9 处挂点。

## 第二代（K-Llama v2）：层范围加载

不再需要"层段工件"，改成直接打开整模、只把属于自己那段读进来。8 个文件，+227/−13：

| 文件 | 改动 |
| --- | --- |
| `include/llama.h` | 新增 `llama_context_params::embeddings_pre_norm` |
| `src/llama-cparams.h` | 内部结构同步该字段 |
| `src/llama-context.cpp` | 默认 `false`，从参数拷贝，打日志 |
| `src/llama-model-loader.h` | `layer_lo` / `layer_hi`、`n_layer_all`、`needs_tok_embd()` / `needs_output()` |
| `src/llama-model-loader.cpp` | `LLAMA_LAYER_RANGE` 解析、`get_weight` 重映射、全局张量的 lazy 标记 |
| `src/llama-model.cpp` | `done_getting_tensors` 传范围、`init_mappings` 按范围、host-only 预读、`size_data` 重算 |
| `src/models/qwen35.cpp` | 在 `build_norm` 之前 tap 出 `t_embd` |
| `src/models/qwen35moe.cpp` | 同上 |

三个关键机制：

1. **`LLAMA_LAYER_RANGE=lo:hi`**：覆盖 `block_count` 与 `nextn`，让架构以为自己只有 `hi-lo`
   层；`get_weight` 把本地 `blk.i` 映射回源文件的 `blk.lo+i`。
2. **`embeddings_pre_norm`**：接力需要的是 `output_norm` **之前**的末层输出。这个开关把
   tap 点前移，替代第一代的 `layer_inp` 槽位。只改了 qwen35 和 qwen35moe 两个架构。
3. **lazy 全局张量**：head 段不需要 `output`、tail 段不需要 `token_embd`、middle 段两者都不需要。
   不创建就不读，middle 段因此省 1030.62 MiB。

**为什么第二代反而更小**：第一代要 9 个架构各自登记挂点，第二代把"取哪一层的输出"变成
"只加载哪几层"——后者是加载器的事，与架构无关。架构侧只剩 2 处 tap。

---

## 第三代：SLIM-ARC prefetch（2026-10-31）

上游 SLIM-ARC 是给 Linux 写的，本机是 Windows，所以是**带回移植的接入**，只接它两级里的
第一级（`prefetch`）。第二级（on-demand 加载器）仍未接，理由见
[`SLIM-ARC-接入计划.md`](SLIM-ARC-接入计划.md)。

新增 5 个源文件：`slim-arc-prefetch.{h,cpp}` 加它依赖的
`page-range` / `expert-reclaim` / `expert-residency` / `expert-transition`。
`src/CMakeLists.txt` 用 `SKIP_UNITY_BUILD_INCLUSION` 把它们挡在 unity 批次之外——这批文件带
`WIN32_LEAN_AND_MEAN` 和 POSIX 名字，混进 unity 会污染同批的翻译单元。

两个挂点：

| 文件 | 位置 | 作用 |
| --- | --- | --- |
| `llama-model-loader.cpp` | `init_mappings()` 末尾 | 把每个权重的 `(mmap 基址 + 偏移, 大小, 层号)` 注册进调度器 |
| `llama-context.cpp` | `graph_compute()` 调 `sched_graph_compute_async` 之前 | 通知"现在要算第 N 层"，调度器据此预取 N+1..N+window |

**层号怎么来**：上游从图节点名里读 `blk.N`，在本树里取不到——GGML 的 scheduler 在计算前把
每个节点重命名成 `node_%d`，真正的张量名留在**操作数**上，所以改成读 `src[0..]` 的名字。
按上游写法接时 `calls` 恒为 0；改成读 src 后 `calls=5`（3 个 token 各一次通知加上 prefill），
链路确认通了。

win32 移植面（3 个函数 + 1 个锁包装）：

| POSIX | Windows | 说明 |
| --- | --- | --- |
| `madvise(WILLNEED)` | `PrefetchVirtualMemory` | 语义相同：把这些页读成驻留 |
| `madvise(DONTNEED)` | `DiscardVirtualMemory` | 内容作废、页丢弃，最接近的等价物 |
| `madvise(NORMAL/RANDOM/SEQUENTIAL)` | 空操作 | 只影响内核预读启发式，Windows 没有这个接口 |
| `mincore` | 不支持，返回"未知" | 唯一消费方是专家热锁 LRU，未知即跳过该区间 |
| `sysconf(_SC_PAGESIZE)` | `GetSystemInfo().dwPageSize` | |
| `mlock` / `munlock` | `VirtualLock` / `VirtualUnlock` | |

另有两处 MSVC 兼容：`NOMINMAX`（否则 `windows.h` 的 `max` 宏会把 `std::max` 和
`numeric_limits<T>::max()` 一起打掉），以及用 `_umul128` 拼出 64×64→128 的乘积替代
`unsigned __int128`（MSVC 没有该类型，用在专家 LRU 的打分比较上）。

**当前生效范围**：默认关闭，`SLIM_ARC_PREFETCH=1` 才装调度器。整模在页缓存里时
`issued` 为 0——它只在"段装不下、真的在读盘"的降级形态里才有事可做，与
`SLIM-ARC-接入计划.md` 的触发条件一致。

---

## 与 llama.cpp 原生 RPC 的分界

原生 RPC（`-DGGML_RPC=ON`）也能跨机，但两者解决的不是同一个问题。RPC 的协议命令里，
`SET_TENSOR` / `SET_TENSOR_2D` 上传张量数据，`GRAPH_COMPUTE` 让远端执行子图，`GET_TENSOR`
取回结果。也就是说：

**RPC 让远端当"远程 GPU"，主机仍然要有完整的模型。** 权重从主机通过网络发到远端
（`SET_TENSOR`），主机维护全局计算图，远端只负责执行被派过去的那部分 op。

**本项目让每个节点当"一段模型"，没有中心。** 整模文件在每个节点本地，各自 mmap 后只
touch 自己那几层，段间只传 `n_embd` 个浮点。

| 维度 | 原生 RPC | 本项目 |
| --- | --- | --- |
| 权重从哪来 | 主机通过 `SET_TENSOR` 发给远端 | 各节点本地读整模的对应层 |
| 首次跨机流量 | 权重总量（GB 级） | 0，只要本地有整模 |
| 主机必须装下整模 | 是 | 否（只装自己那几层） |
| 每步流量 | 激活 + 图指令 | hidden 一个向量（FP16 4 KB） |
| 计算图 | 主机维护全局图 | 每段是独立进程，无全局图 |
| 异构 | 需要相同的 ggml backend | 自定义 TCP 协议，可跨不同 CPU/GPU |
| 层范围加载 | 不支持，要自己切模型 | `LLAMA_LAYER_RANGE` 原生支持 |

一句话：RPC 是**把设备虚拟化**，模型仍然是一份；本项目是**把模型分段**，每段各自持有。
前者适合"一台主机 + 若干算力节点"，后者适合"几台都装不下整模的机器"。

代价也要说清楚：本项目的段是黑盒，段内不能跨机器切分算子；每一段必须能单独装下。
而 RPC 可以把任意 op 派到远端，粒度更细。
