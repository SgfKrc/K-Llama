# prima.cpp 深度分析：对「层段分布式推理原型」的可借鉴度

> 对象：`H:\prima.cpp`（llama.cpp 的分布式 fork，commit 288df6a）。方法：**只读代码走查 + README/论文**，
> 未编译运行。标注：`[实测]` = 本仓源码可核对；`[推断]` = 由代码结构推出、未运行验证。

## 0. 结论先行

prima.cpp 与我们是**同构路线**（按层切分 + 显式 hidden 接力），**不是** ggml-rpc 的张量切分。
分布式运行时约 2,000 行；控制面约 4,000 行，其中 **3,551 行（2,745 探测 + 806 LP 求解）全花在
"自动探测设备 + 自动分配层"**——这对我们价值很低，我们已有确定拓扑与预生成段工件。
真正值得抄的是：**协议帧、环上 ACK barrier、三通道分离、device_info 字段表**。

## 1. 分布式架构

### 1.1 按层切分，与我们的方案同构

| 维度 | prima.cpp | 我们 |
| --- | --- | --- |
| 切分单位 | 连续层窗口 `n_layer_window[]` | 连续层段 `[0,8) [8,24) [24,40)` |
| 每台加载 | **同一份完整 gguf**，只 create 自己窗口的张量 | **各段独立工件**（完整模型子集） |
| 接力单位 | hidden `F32 [n_embd, n_tokens]` | hidden float32 |
| head/tail | rank0 = embedding 子图 + output 子图 | stage-0 带 `token_embd`，末段带 `output.weight` |

证据：`src/llama.cpp:11018-11030` 给 rank0 单建 `inp_embd` 子图；`11191-11199` 给 rank0 单建
`out_embd`+`output_norm`+`lm_head` 子图；其余 rank 各一个 subgraph，非本层则 `inpL = inpB` 跳过
（`11042-11054`）。`inpB = llm_build_backend_embd(...)`（`11033`）就是"从网络收到的 hidden"。
**我们的 keep-head shim 做的是同一件事**——prima 选了更侵入的改法，不代表 shim 是权宜之计。

### 1.2 没有独立控制面进程

控制面完全嵌在 llama-cli / llama-server 内，靠 `--world/--rank/--master/--next`（`common/arg.cpp:679-754`）
+ ZMQ 消息完成。无守护进程、无中央调度器、无 TUI。rank0 既是请求入口又是环上一站，其余节点跑 `llama-cli`
当被动 worker（`examples/server/server.cpp:715-734` 显示只有 draft 模型不切分，主模型走同一套环）。

### 1.3 通信：ZMQ PUSH/PULL 环 + 两条旁路（`src/llama.cpp:20492-20527`）

- `send_socket`(PUSH) → 下一 rank 的 `data_port+rank`；`recv_socket`(PULL) 本地 bind；
  `signal_socket`(PULL) 独立端口，只传 `"STOP"`（`20888-20910`）。端口约定 `base+rank`
  （`20467-20469`），默认 data 9000 / signal 10000，**静态约定非动态分配**。
- `master_socket`(PUSH) 仅 rank 1..n-2 持有，用于**末段把结果直送 rank0**，不必绕满一圈。

### 1.4 hidden 显式传递，与 ggml-rpc 无关

帧格式（`src/llama.cpp:18031-18052`，ZMQ multipart）：

```
["sub_gf_out"][ne[2]×int64][f32 buffer]  ["inp_pos"][ne[0]×int64][i32 buffer]
```

key/dims/data 三元组，**无压缩、无量化、无校验**。`18558-18564` 每个 subgraph 算完后
`ggml_backend_tensor_get_async` 拉回 CPU 再 send；`18507-18510` 下个 subgraph 前 recv 进 `ubatch->backend_embd`。

**关于 ggml-rpc：** 仓库确有 `ggml/src/ggml-rpc.cpp`(47 KB) 与 `examples/rpc/rpc-server.cpp`，
`src/llama.cpp:3498-3560` 有 `GGML_USE_RPC` 分支——但那是**上游自带的 `--rpc` 张量切分路径**，
`--world/--rank` 路径**一行都没调用它**。两套机制并存互不相干，印证你的判断：
ggml-rpc 要求同构构建 + 张量切分，与按层接力是不同路线。

## 2. "内存压力低于 10%" 的真相

### 2.1 机制：mmap 惰性 + 粗粒度 madvise + 完全交给 OS

1. 加载：`ml.init_mappings(false, ...)`（`9366`）——**prefetch=false**，不用 `MAP_POPULATE`。
2. 预取：`manage_graph_tensors(gf[next], POSIX_MADV_WILLNEED, force)`（`18571`），对**下一个 subgraph
   的全部 weight leaf** 取 mmap 区间、按页对齐、合并相邻段、整段 `posix_madvise`（`18152-18218`）。
3. `--force` 再逐页触读（`18209-18216`，每 4 页碰一字节）。
4. **回收：没有任何代码。** 全仓 `madvise` 只有 `WILLNEED`/`RANDOM`（`1971,1979,18207`），
   无 `DONTNEED`、无 munmap、无预算。

所以"<10%"不是被测出来控出来的，而是**"只 mmap 不 assign → 页由 OS 按压力回收"的自然结果**。
`is_graph_loaded()`（`18131-18150`，用 `mincore` 逐页统计驻留比）是**死函数**——全仓唯一出现即定义处，
从未被调用。即**代码里没有任何运行时内存压力度量或反馈回路**。论文口径"TPOT 674 ms with <6%
memory pressure"（ICLR 2026 abstract）是实验测量，不是运行机制。

### 2.2 与 SLIM-ARC 的异同

| | prima.cpp | SLIM-ARC（`slimarc-trial/src/`） |
| --- | --- | --- |
| 触发 | `--prefetch`（默认关） | `model_runtime_admitted()`：mmap + 映射数>0 + **size>6 GiB**（`slim-arc-runtime.h:12-15`） |
| 粒度 | subgraph 级、**全部 weight 无差别** | tensor 级 `tensor_prefetch_info` + 专家级 `prefetch_experts()` |
| 预算 | **无** | `memory_budget`/`expert_budget`/`admission_threshold`，有拒绝与节流统计 |
| 压力反馈 | **无** | 状态机 `missing/normal/high/critical`（`slim-arc-prefetch.h:90-93,415`） |
| 回收 | 交给 OS | kv-eviction / expert-reclaim / cgroup-memory 显式回收 |

一句话：**prima 是"无脑 WILLNEED 下一段"，SLIM-ARC 是"带预算与压力反馈的选择性预取"。**
前者的简单在 dense 顺序扫描下刚好够用（这正是它的设计前提）；后者的复杂是为 MoE 稀疏访问付的。

### 2.3 对 MoE 不适用（代码级证据）

`manage_graph_tensors` 的过滤只有两条（`18164,18173`）：名字含 `"weight"`、buffer 名不含 `"cuda"`。
**它不认识"这个专家本 token 没被激活"。** MoE 层的 `ffn_gate_exps/up_exps/down_exps` 是
`[n_embd, n_ff_exp, 256]` 的 3D 张量，会整块被 WILLNEED——而单 token 只路由 8 个专家。
**8/256 利用率下预取量放大 32 倍，命中的那 8 个专家只占 3%。**

## 3. 控制面设计（用户最关心的部分）

### 3.1 有什么

- **拓扑描述**：无文件、无 YAML/TOML（全仓仅 `examples/server/bench/prometheus.yml` 等无关物）。
  拓扑 = 每台机器命令行上的 `--world N --rank i --master IP --next IP`，**手工填 IP，无节点发现**。
  README:78 的 "Automatic device selection" 是"剔除太慢的设备"，不是发现新设备。
- **层分配**两条路：① 静态 `-lw "16,16,16,16"`，逗号分隔按 rank 顺序，上限 32 项（`common/arg.cpp:707-740`）；
  ② 启动时各节点自测 → gather 到 rank0 → **一次 HiGHS 整数规划** → bcast 结果。
- **调度时机：纯静态**。`auto_schedule = params.n_layer_window[0] == 0`（`common/common.cpp:1717`），
  profile 与求解只发生在 `llama_init_from_gpt_params` 内（`1740-1804`），**推理过程中不重分配**。
- **UI**：命令行 + `printf` ASCII 表（`common/profiler.cpp:1712` `device_print_props`）。
  **无 TUI、无 HTTP 控制 API、无配置文件**；唯一 HTTP 是 llama-server 推理 API 与 `/v1/cancel`。

### 3.2 HiGHS 调度器（806 行，`common/common.cpp:855-1590`）

整数规划：变量 `w[m]`=每设备层窗口、`n[m]`=其中上卡层数；目标 `k·Σ(a[m]·w[m]+b[m]·n[m])+κ+k·Σc`
（`1266-1280`）；约束 `Σw=W`、`n[m]≤w[m]`、每设备 RAM/VRAM 上限（`1292-1358`）。`k` 是环的圈数、
`L=k·W`，只在 `n_layer` 的因子里搜（`1054,1256-1258`）；`a[m]/b[m]` 由实测 per-quant FLOPs + memcpy
延迟算出（`900-972`）；`master_priority` 给 rank0 加权（默认 1.01）。**它解决"未知异构 + 无配置"，
我们两样都有，这 806 行可以不要。**

### 3.3 拓扑重建（最值得看，`src/llama.cpp:20684-20829`）

设备被剔除（分到 0 层）后环需跨过它重接：找下一个有效节点（`20723-20730`）→ 逐跳 `try_connect`
试连（`is_port_open`，`src/network-utils.cpp:8-25`）→ 被跨过的节点降级 `NODE_TYPE_FORWARDER`
（`20749,20791-20795`），**保留在环上只转发不断链** → 全网视图沿环广播后各节点收敛（`20797-20816`）。
节点三态 `WORKER/FORWARDER/EXIT`（`include/llama.h:467-471`）。用"环上广播全网视图"代替"中心下发"，
增减设备只影响相邻两跳——**做容错时这是现成模型**。

### 3.4 代码量（`[实测]`，按行统计）

| 模块 | 位置 | 行数 |
| --- | --- | ---: |
| socket/环/barrier/拓扑重建 | `src/llama.cpp:20460-20930` | 471 |
| 消息序列化 send/recv | `17860-18080` | 221 |
| prefetch/madvise/页驻留检测 | `18079-18218` | 140 |
| decode 分布式主循环改造 | `18229-18620` | 392 |
| 层映射（`this_layer_is_mine`/`local_id`/`subgf_id`） | `3838-3883, 16941-16951` | 57 |
| `build_llama`/`build_qwen2` 多 subgraph 改造 | `11000-11200, 12736-12900` | ~400 |
| per-rank 张量加载改造 | `7300-7460, 7670-7800, 9356-9560` | ~300 |
| **分布式运行时小计** | | **~2,000** |
| HiGHS 调度器 | `common/common.cpp:855-1660` | 806 |
| 启动编排 | `1714-1904` | 187 |
| 设备探测 profiler | `common/profiler.cpp`(2356)+`.h`(389) | 2,745 |
| CLI 参数 / 对外 API / 工具 | `arg.cpp:676-792` / `llama.h:467-509` / `profile_tool.cpp`+`network-utils.cpp` | ~260 |
| **控制面小计** | | **~4,000** |

prima 自研 ≈ **6,000 行**（另有 vendor cppzmq 3,206 行）；`src/llama.cpp` 全文 24,559 行，其余 17 万行是上游本体。

## 4. 性能数据怎么来的

- **有正式论文**：ICLR 2026，《Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous and Low-Resource
  Home Clusters》（OpenReview `h0LjpOG1jq`）。贡献：**PRP（pipelined-ring parallelism，用环结构把磁盘 I/O 藏到
  计算/通信后，并解决 mmap offload 的 "prefetch-release conflict"）** 与 **Halda（异构调度器）**。
- **"15x" 口径**：README Table 2/3 的 llama.cpp 列是**单机**数据。Llama3-70B：10120 ms vs 674 ms（15.0x）；
  QwQ-32B：224 vs 89 ms（2.5x）；论文写 "5–17× lower TPOT"。基线是"同一批设备里的**一台**、而且**装不下**"。
  README:67-69 自认设备选型后小模型退化成 `32:0:0:0`、"would become llama.cpp when serving small models"，
  此时 15 ms 打平——**15x 全部来自"从 swap/换页地狱里救出来"，不是"同样装得下时更快"**。
  这与你已有实测一致（单进程 `-ncmoe 32` 28.02 t/s，拆两进程 ~17 t/s）；`[推断]` prima 在我们的
  35B-A3B 单机装得下场景同样不会更快。
- 可信度：4 设备异构表（M1 / i9+3070 / i9+2080Ti / Kirin9000，RAM 2.4–9.7 GiB）自洽；
  但**没有任何 MoE 数据**（见 §6.2，代码层面也不支持）。

## 5. 三档结论

### 5.1 直接借鉴

- **hidden 帧格式**：`key/dims/data` 三元组多帧一次 send（`src/llama.cpp:18031-18052`）→ 我们的
  `forward`/`result` 可直接用 `[tag][int64 dims[2]][f32 payload]`，省掉自设计变长编码。
- **三通道分离**：data 环 / `master_socket` 直达 / `signal` 旁路（`20497-20512`）→ 我们的 `STOP`、`HELLO`
  走独立端口，避免与 hidden 抢同一条 TCP 的队头阻塞。
- **环上 broadcast + ACK barrier**（`20590-20682`）→ `JOIN/READY` 握手照抄：rank0 发 → 沿环转发 → 末段 ACK → 回 rank0 即集群就绪。
- **端口约定 `base+rank`**（`20467-20469`）→ 拓扑配置只写 `base_port`。
- **`-lw` 逗号分隔按 rank 顺序**（`common/arg.cpp:707-740`）→ TUI「层段表」的一行式编辑格式现成可用。
- **`device_info` 字段表**（`common/profiler.h:334-361` + `profiler.cpp:1712`）→ 直接当 TUI 拓扑面板列：
  device_name / os / cpu.name / cores / flops / ram.total / ram.avail / disk.seq_bw / disk.rnd_bw / gpu.name / vram。
- **`NODE_TYPE` 三态**（`include/llama.h:467-471`）→ 调度器状态机可直接用 worker/forwarder/exit 建模。

### 5.2 参考思路（机制像，实现不能照抄）

- **mmap 惰性 + WILLNEED 下一段**：前提是"每台都有完整模型文件"。我们每台只有自己那份工件，
  段内是顺序扫描，**本来没有跨段换页问题**；叠一层 madvise 只是给 OS 添噪声。段内页优化交给
  SLIM-ARC（有预算与压力反馈，prima 没有）。
- **PRP 的环上多圈 `k`**：`map_layer_to_local_id` 的 `cycle_offset/cycle_size`（`src/llama.cpp:3863-3882`）
  让少量设备跑多圈。我们的"一进程一工件"下段数=进程数、`k≡1`；**只有将来"3 台跑 6 段"时回来抄这 20 行**。
- **拓扑重建 / forwarder**（§3.3）：思路极好，但为"设备被自动剔除"设计；静态三机暂时用不上，做容错再抄。
- **HiGHS 整数规划**：3 节点 + 已知 t/s，最优解就是"按 1/t 加权 + 手工微调"。但**目标函数形状值得抄**：
  `Σ(a·w + b·n)` 即"层数成本 + 上卡成本"，我们的分配器可用同样两项。
- **设备 profile（2,745 行）**：fio 磁盘带宽、per-quant FLOPs 微基准、内存带宽探测。设备固定手填 4 个数字即可，**别花在 1000 行里**。

### 5.3 不适用

- **全量模型文件 per 节点**：22 GB × N 份磁盘 + 每节点都要能寻址整份文件。段工件正是其正解（多 1.04 GB 重复全局张量，换掉 N×22 GB）。
- **强制 `LLAMA_SPLIT_MODE_LAYER`、禁 `NONE/ROW`**（`src/llama.cpp:21106-21109` 直接抛异常）：把切分方式绑架进运行时。
- **MoE 支持**：见 §6.2，代码级不支持。
- **`--keep-out-in-cuda`/`--gpu-mem` 这类全局开关**：我们段内 offload 用原生 `-ngl/-ncmoe`，粒度更细。
- **无 TUI / 无配置文件**：这是它 4,000 行控制面里最该被我们**替换**的部分，不是借鉴。

## 6. 关键判断：dense 折寿不明显，MoE 呢？

**(a) dense 每段计算量均匀、总耗时≈各段之和 —— 基本成立，一个例外。** 每层 FLOPs 相同，按层数分配即按算力
分配。例外是 **rank0 额外背 embedding + output**：`lm_head` 是 `n_vocab×n_embd`，解码时只有 1 token，是纯访存
操作，可能比一整层还慢（prima 默认把 output 放 CPU，README:232，另有 `--keep-out-in-cuda`）。`[推断]` 这也是
我们末段（ROG 纯 CPU）的真实风险——它除了 16 层还要扛 `output.weight`。

**(b) MoE 是否产生"专家跨段调用"开销 —— 不会，你的理解正确。** 每段工件是完整模型的**层子集**：第 `i` 层的
`ffn_gate_inp`（router）与全部 256 个 `ffn_*_exps` 都在**同一个** stage gguf 里、**同一台**设备上；激活哪 8 个专家
由该设备内部决定，段间只传 `hidden[2048]`（float32 ≈ 8 KB/token）。**无跨段专家路由、无"专家不在本段"的 fallback**，
与 `ARCHITECTURE.md:13-27` 的"两条路径正交"一致。

**(c) prima 的机制对此有何不同 —— 不是"通信"不同，是"预取"不同。这是最易被忽略的一点。**

| | dense（prima 目标） | MoE（我们） |
| --- | --- | --- |
| 每 token 触达权重 | 本段**全部** | 本段 **~3%**（8/256 专家）+ 全部非专家张量 |
| 全量 WILLNEED | **恰好正确**（都要读） | **32× I/O 放大**（`18164` 只按名字过滤） |
| page cache 访问模式 | 顺序、可预测 | 稀疏、token 间随机 |
| "prefetch-release conflict" | 轻微 | 严重（预取挤掉还要用的页） |

更准确的说法不是"prima 用 dense 所以折寿不明显"，而是：**prima 的"低内存压力"是拿"全量顺序预读"换来的。
这个交换在 dense 上是等价变换（读 100% 用 100%），在 MoE 上是坏交换（读 100% 用 3%）。我们的方案不需要这个
交换——段工件只含自己层的权重，段内用 `-ngl/-ncmoe` +（CPU 节点）SLIM-ARC 专家级预取，粒度与稀疏性匹配。**

**(d) 还有一个 prima 没解决、我们也要面对的差异：MoE 的层间耗时方差。** Halda 假设"每层耗时是常数"
（`common/common.cpp:906-918` 按 FLOPs 求和）。但 MoE 下不同 token 路由不同专家，同一层耗时会波动，而环上每个
token 的 TPOT 由**最慢那段**决定 → pipeline 气泡；dense 没有这个问题。`[推断]` 这是我们原型真正会踩的坑，且
prima 帮不上忙（连 MoE 都跑不了）。建议 `MEASUREMENTS.md` 加一列：每段 decode 耗时的 p50/p99 比。

### 6.2 prima 到底能不能跑 MoE？不能，且是编译期断言级别

```cpp
// src/llama.cpp:17003
GGML_ASSERT((model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_QWEN2)
            && "this model is currently not supported.\n");
```

该断言在 `switch (model.arch)` 之前执行、对所有架构生效。虽然 `case LLM_ARCH_GRANITE_MOE` 会走带 MoE 分支的
`build_llama()`（`11149-11166`），但被这行先拦死；`QWEN2MOE/OLMOE/DEEPSEEK2` 走 `push_back(单 graph)`，
**没有多 subgraph 结构可接力**。架构枚举里**没有 `LLM_ARCH_QWEN3MOE`**（`227-247`）——该 fork 的 llama.cpp
版本早于 Qwen3，`qwen3moe` 架构的 gguf 加载即失败；即便补上架构，上面这行断言也会在 build_graph 阶段拦死。

## 7. 给我们项目的一句话建议

> **把 prima.cpp 当作"协议与握手流程的参考实现"，不要当作"性能或内存机制的参考"：
> 抄它 130 行消息帧（key/dims/f32-data）、环上 ACK barrier、data/master/signal 三通道分离、以及
> `device_info` 那张 TUI 字段表；不要抄它 2,745 行设备探测与 806 行 HiGHS——那是为"未知异构设备 +
> 无配置"付的税，而我们的段工件与固定三机让这笔税完全不必交。同时把 `src/llama.cpp:17003` 那行断言
> 当作 moat：prima 的分布式路径对 MoE 是关闭的，这个开源方向目前没有现成对手。**

## 附：证据边界

- `[实测]` 架构/协议/调度/内存机制/代码量/MoE 支持范围均已逐行核对；**未运行** prima.cpp，
  §4 性能数字全部转引自 README 与 ICLR 2026 abstract（`https://iclr.cc/virtual/2026/poster/10008093`）。
- **未细读**：`src/llama.cpp` 中 40+ 个其它 `build_*` 架构的改造（只精读 `build_llama`/`build_qwen2`），
  以及 `profiler.cpp` 2,356 行里设备微基准的实现细节。
