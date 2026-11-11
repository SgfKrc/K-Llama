# prima.cpp 分析：同一机制的另一种实现

> 状态：**已验证**（文中引用的 v2 工具链——`chain_client` / `stage_worker` / `shim/` / `koakuma`——已退场，归档在 `slimarc/prototype`，命令与路径只作历史口径）
> 更新日期：2026-11-03
> 适用范围：prima.cpp 与本项目在层过滤、内存预取、通信、控制面上的逐项对照；读者为架构决策者。
> 对象：`H:\prima.cpp`（llama.cpp 分布式 fork，commit 288df6a）。方法：**只读代码走查**，未编译运行。
> 前提：本项目是"**同一份整模 + `LLAMA_LAYER_RANGE` 加载时选层**"（`ARCHITECTURE.md` §0）。
> 标注：`[实测]` 源码可核对；`[推断]` 由代码结构推出、未运行验证。

---

## 0. 结论先行

两边做的是同一件事：**每节点持同一份整模、只读自己那几层**。由此三条结论：

1. **分界线不在"切不切文件"，而在"在哪一层过滤层号"**：prima 在 **build graph 时跳过**
   （`src/llama.cpp:11042-11066`），我们在 **model-loader 里改写 `block_count`**（`docs/PATCHES.md` 第二代）。
   前者必须改每个架构的 build 函数，代价是**只能支持 LLaMA/Qwen2**（`17003` 一票否决）；
   后者与架构无关，代价只有 2 处 tap、8 文件 +227/−13。**prima 是我们这条路线优势的反证。**
   这也意味着**它的代码无法搬到我们的树上**（§8）——换掉底层 llama.cpp 等于删掉 prima。
2. **prima 的内存机制也是我们的问题**：我们也靠 mmap 惰性 + 按层预取。区别只是它按 subgraph
   全量 WILLNEED，我们按层窗口 + 预算。它有两个细节值得看，一个值得抄、一个**不能抄**（§2）。
3. **另外三点成立**：prima 跑不了 MoE（我们的护城河）；控制面 4,000 行里 3,551 行不需要；
   "15x" 来自"装不下"——这个前提同样适用于我们（GPU 只装得下 6 层，整模 40 层）。

## 1. 同一机制的两种实现（逐项对照）

| 维度 | prima.cpp | 我们（K-Llama v2） |
| --- | --- | --- |
| 模型文件 | 每节点同一份整模 | 每节点同一份整模（`ARCHITECTURE.md` §0） |
| 层过滤 | 运行时：`for il < n_layer { if (!this_layer_is_mine) continue; }`（`11042-11054`） | 加载期：改写 `block_count` 与 `nextn`（`PATCHES.md` 第二代） |
| 层号映射 | `map_layer_to_local_id(il)` → 本地 0..w-1（`3854-3883`） | `get_weight`：`blk.i` → `blk.lo+i`（`PATCHES.md` 第二代） |
| 层区间表达 | **累积窗口数组** `n_layer_window[]`，按 rank 顺序累加（`3841-3852`） | **显式** `layer_range = "lo:hi"` |
| 全局张量 | rank0 硬编码建 `tok_embd` / `output` | `needs_tok_embd()` / `needs_output()`，中段省 1030.62 MiB（`PATCHES.md` 第二代） |
| 接力取值 | subgraph 输出 `sub_gf_out`（`18528`） | `output_norm` **之前**（`embeddings_pre_norm`，`PATCHES.md` 第二代） |
| 段间载荷 | float32 `n_embd` = 8 KB | float32 `n_embd` = 8 KB（`ARCHITECTURE.md` §4） |
| 架构覆盖 | **2**（LLaMA / Qwen2，`17003` 断言） | 所有架构（`ARCHITECTURE.md` §2 为架构无关） |
| 补丁规模 | ~2,000 行，distributed 逻辑散在 `src/llama.cpp` 与每个 build 函数 | 8 文件 +227/−13（`PATCHES.md` 第二代） |

**这张表里最值钱的一行是"层过滤"。** prima 选了"让架构照常跑 40 层、中途跳过不属于我的"，于是每一处
`for (il = 0; il < n_layer; il++)` 都要插两个钩子：`11043` 跳过别人的层、`11066` 把源层号映射成本地
索引；`build_qwen2` 同理再来一遍（`12779,12802`）。每加一个架构就复制一遍这套改动。我们反过来：让架构
**以为**模型只有 `hi-lo` 层，于是所有 build 函数一行不改。`[实测]` 这个选择直接决定了能力边界：prima 的
`switch` 里虽然有 40+ 个 `case`，但 `17003` 的断言把它们全部挡在分布式路径之外——**它没有能力做"第 3 个
架构"，不是没写，是路线决定了写不动**；而我们支持 `qwen35moe` 只花了 2 处 tap。

## 2. 内存与预取：prima 的两个细节，一个可抄一个不可抄

### 2.1 机制对照

| | prima | 我们 |
| --- | --- | --- |
| 惰性来源 | mmap 不 `MAP_POPULATE`（`init_mappings(false)`，`9366`） | 同（`ARCHITECTURE.md` §1）+ 未被 create 的层**从头到尾没 touch** |
| 预取触发 | 每个 subgraph 算完，对**下一个 subgraph**（`18570-18574`） | `graph_compute` 前通知"要算第 N 层"，预取 N+1..N+window（`llama-context.cpp:2778-2783`，window=3） |
| 预取集合 | 遍历 graph leaf，名字含 `"weight"` 且 buffer 非 cuda（`18161-18174`） | 加载期按 (mmap 基址+偏移, 大小, **层号**) 注册（`llama-model-loader.cpp:1570-1584`） |
| 区间合并 | 排序 + 合并相邻段（`18189-18202`） | `coalesce_page_ranges()`，含溢出检查与统计（`slim-arc-page-range.cpp:78-116`） |
| 预算 | **无** | `memory_budget` / `expert_budget` / `admission_threshold` |
| 压力反馈 | **无**（`is_graph_loaded` 是死函数，`18131-18150`） | 状态机 `missing/normal/high/critical` |
| 回收 | 无 DONTNEED，交 OS | `DiscardVirtualMemory` 等价物 + 专家 reclaim |

**粒度上我们更细**（层窗口 vs 整段），**合并上我们更完整**（有溢出保护与计数），所以 §3 的"直接借鉴"不含"再抄一遍预取"。

### 2.2 可抄：`[推断]` 排除"已经上卡"的张量

prima 在选预取集合时**跳过了 CUDA buffer 上的张量**（`18168-18174`：buffer 名含 `cuda` 就 `continue`）——
已经常驻显存的权重不需要再 madvise。

我们的注册在 `init_mappings()`（`llama-model-loader.cpp:1566-1587`）里一次性完成，
遍历 `weights_map`，**不区分**该张量最终是上卡还是留内存。`[实测]` 代码里没有这个过滤。
后果：对 `-ngl` 上卡的层，我们仍会对它们的 mmap 区间发 `PrefetchVirtualMemory`——
这些页在 offload 后已不再被读，属于**纯噪声**（虽然 hint 不致命）。

`[推断]` 影响量级：按 `topology-cross-mtp.toml` 的 `-ngl 32 -ncmoe 24`，head 段 32 层中
注意力和 dense 全部上卡，噪声比例可能相当可观。**但这只在读盘场景才真正花钱**
（页缓存命中时 `issued` 恒 0，见 `PATCHES.md` 第三代），所以优先级取决于弱设备是否接入。

### 2.3 **不能抄**：Windows 驻留检测

prima 的 `is_tensor_loaded()` 在 Windows 上用 `VirtualQuery` + `State != MEM_COMMIT` 判断"未加载"
（`18083-18096`）。看起来正好补上我们的缺口——`PATCHES.md` 第三代说我们"`mincore` 不支持，返回未知，
唯一消费方是专家热锁 LRU，未知即跳过"。

`[实测/推断]` **但这个近似大概率无效**：file-backed section 的页在 `VirtualQuery` 里通常一直是
`MEM_COMMIT`（section 已提交），即使页已被 trim 出工作集。也就是说它会**恒返回"已加载"**，
比"未知"更糟——LRU 会把所有专家都当成热的。

正确做法是 `QueryWorkingSetEx` / `K32QueryWorkingSetEx`（PSAPI），查 `VirtualAttributes.Valid`（该页是否
在进程工作集）。prima 没这么做。**这一条是"prima 的坑"，不是"prima 的宝"。**

## 3. 三档结论

### 3.1 直接借鉴

- **hidden 帧格式** `[key][dims][f32]`（`18031-18052`）：我们已经同构（4 字节长度 + 1 字节类型 + payload），prima 的形态可作交叉验证，无需改动。
- **环上 ACK barrier**（`20590-20682`）：启动时 rank0 发 → 沿环转发 → 末段回 ACK → 回 rank0 才算就绪。
  我们的 worker 拉起是"拓扑里 `start_cmd` 逐台启 + 连上就绪"，**没有集群级 barrier**——
  段多了以后"某台已起但还没 ready"会被误判为可用。可借用这个最小协议。
- **`device_info` 字段表**（`common/profiler.h:334-361`）：TUI 拓扑面板的列清单仍可直接用。
- **`NODE_TYPE` 三态**（`include/llama.h:467-471`）：我们目前的 `kind = gpu/cpu/tail` 是"角色"，
  不是"健康状态"；做容错时需要 worker/forwarder/exit 这组正交状态。
- **端口 `base + index`**：我们的 `topology.toml` 逐段写 port，跨机时还要改 IP。
  `port = base + index` 能省一行样板——**低价值，顺手可做**。

### 3.2 参考思路

- **`[推断]` PRP 的"多圈 k"与"通信/预取重叠"—— 不要学，但要知道它是什么。** `k` 从 `n_layer` 的
  因子里搜（`common/common.cpp:1054,1256-1258`），`map_layer_to_local_id` 的 `cycle_offset/cycle_size`
  （`3863-3882`）把一台设备的层拆成 k 个**不连续**窗口——它**不减少**持有量（仍持 `k·w[m]` 层），
  只是切碎窗口让每圈驻留集合更小。v2 的链"段数 = 节点数"、`k ≡ 1`；v2.1 本地链虽可一台机器
  多段，但每段仍是连续层范围，k 仍 1——只有真按设备把层切成不连续窗口才值得回看。
  它"send 之后立刻预取下一段"（`18566-18575`）把预取塞进"等下游"的窗口，我们确实缺这个重叠；
  但页缓存命中时 `issued` 恒 0（`PATCHES.md` 第三代），现在无页可补——**记录，不实现**。
- **拓扑重建 / FORWARDER**（`20684-20829`）：价值较高。我们的链是直线式的（v2 每节点一段、v2.1
  允许同机多段），任一段掉线整条链就断；prima 的"跨过失效节点重接 + 降级为纯转发"是现成模型。
- **HiGHS 目标函数形状** `Σ(a·w + b·n)`（`common/common.cpp:1266-1280`）：仍是"层数成本 + 上卡成本"两项，
  我们若要做自动切分可以借用这个形状。

### 3.3 适用性判定：两处不成立，三处仍不适用

- **每节点持整模**：不是缺陷，是现行做法。没有"多 1.04 GB 重复全局张量"的代价（该代价随
  层段工件废弃而消失）；换来的是 MTP/nextn 头这类裁不出来的张量能被正常加载
  （`topology-local-32-8.toml:7-8`），以及换切分只改数字。
- **"我们的段工件让 prima 的 mmap 机制无必要"**：不成立。我们同样依赖 mmap 惰性，差别只在预取粒度。
- **强制 `SPLIT_MODE_LAYER` / 全局 offload 开关**（`21106-21109`）：我们用 `layer_range` +
  原生 `-ngl/-ncmoe`，粒度反而更细（`ARCHITECTURE.md` §5）。
- **MoE**：prima 侧根本没有这条路径（§5）。
- **无 TUI / 无配置文件**：仍是我们该**替换**的部分（我们已有 `koakuma.py` TUI 与 TOML 拓扑）。

## 4. 通信：结构上我们已经等价

prima 是环（`--next` 串成圈）+ `master_socket` 让末段直达 rank0（`20497-20506`）；
我们是链（head → tail → 回 head），tail 本来就直连 head、不绕行。载荷两边都是
8 KB float32，**都在 1280 MTU 下分成 6 个包**——prima 的 ZMQ 也是 TCP，躲不开。

`[推断]` 所以我们每步那 ~24 ms（`MEASUREMENTS.md` §8）**不是实现问题，是链路问题**，prima 没有方案。
**唯一可比的是"控制通道独立"**：prima 的 `STOP` 走独立 `signal_socket`；
我们的控制消息与 hidden 共用一条 TCP，大 hidden 排队时会拖慢控制消息。多段长链时值得分开。

## 5. MoE：prima 没有这条路径

`src/llama.cpp:17003` 的断言在 `switch (model.arch)` **之前**执行：

```cpp
GGML_ASSERT((model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_QWEN2) && "this model is currently not supported.\n");
```

且该 fork 的架构枚举里**没有 `LLM_ARCH_QWEN3MOE`**（`227-247`）——`qwen3moe`/`qwen35moe` 的 gguf
加载即失败，连单机都进不去。我们的模型恰好是 `qwen35moe`，**prima 全部 2 万行分布式代码对我们直接不可用**。

`[推断]` 这不是"它忘了加"，而是 §1 那条路线选择的必然结果。我们的做法**天然兼容 MoE**——
`ARCHITECTURE.md` §6 的硬约束（切点须为 `full_attention_interval` 的倍数）是这条路线唯一的代价。
**且 MoE 会让 prima 的内存机制更糟**：它的预取按名字含 `"weight"` 过滤（`18164`），
不认识"这个专家本 token 没激活"，`ffn_*_exps` 会整块 WILLNEED；8/256 激活下预取量放大 32 倍。
我们的 SLIM-ARC 有专家级预取（`prefetch_experts()` + 只预取连续两 token 都激活的稳定专家），设计上就对了。

## 6. 性能口径：同样适用于我们自己

- prima："15x" 的基线是"同一批设备里的**一台**、而且**装不下**"（Llama3-70B 10120 ms vs 674 ms）。
  README:67-69 自认小模型退化成 `32:0:0:0` 时两者 15 ms 打平。
- 我们：本机单卡整模 `-ngl 6` 只有 10.5 tok/s（显存只够 6 层），跨机两段 12.2、加 DFlash 20.9
  （`README.md:159-164`）。**也是"装不下 → 拆分才有收益"。**

`[推断]` 两者是同一个经济学：跨机不是免费加速，是"把显存/内存从一台扩展到多台"。
5–17× 与 1.16× 的差别不在机制，在**基线有多惨**——它拿"swap 到磁盘的单机 70B"当基线，
我们拿"能跑但只上 6 层的单机"当基线。**所以别引用 prima 的倍数暗示拆分效率**，
只引用它的定性结论——而那条我们已经用自己的数字独立验证过了。

## 7. 控制面：约 4,000 行里 3,551 行不需要

**3,551 行（profiler 2,745 + HiGHS 806）我们不需要**——那是为"未知异构设备 + 无配置"付的税，
我们设备固定、层区间是配置项。

## 8. 能不能把 prima 底下的 llama.cpp 换成我们改过的？

**结论：不能。因为 prima 的"分布式"不是模块，而是它对 llama.cpp 内部的深度改造；
换掉底层 = 删掉 prima。**

### 8.1 两边不是同一个代际的代码

| | prima（2024-08 前后） | 我们（K-Llama v2，2025 末） |
| --- | --- | --- |
| `src/llama.cpp` | **单文件 24,559 行 / 1.07 MB**，所有 `build_*` 内联其中 | **22,670 行**，只剩 API 外壳 |
| 架构实现 | 内联在 `llama.cpp`（`build_llama` 在 `11000`） | `src/models/` **172 个文件**（`qwen35moe.cpp` 592 行，`build_arch_graph` 在 `149`） |
| 图构建 | `llm_build_context` 内联 | `llama-graph.cpp` 166 KB + `llama-graph.h` 55 KB |
| 执行入口 | `llama_decode_internal`（`18229`） | `llama_context::process_ubatch`（`llama-context.cpp:1528`）→ `graph_compute`（`2757`） |

**没有任何一对文件可以直接替换**。这不是"打补丁有冲突"，是两个不同的软件。

### 8.2 prima 的分布式逻辑挂在哪些内部结构上

它改的是**库内部**，不是公开 API：

- `llama_cparams` 加 9 个字段：`n_world`/`rank`/`node_type`/`n_worker`/`worker_rank`/
  `original_next_rank`/`n_layer_window[32]`/`prefetch`/`force`（`2597-2606`）；新版 `llama-cparams.h` 一个都没有。
- `llama_ubatch` 加 4 个字段：`backend_embd`/`out_embd`/`activate_input`/`activate_output`（`2796-2804`）；
  `[实测]` 新版 `llama-batch.h` 无这些——**它们是 prima 的自定义扩展**。
- `llama_init_sockets(llama_context * ctx, ...)` 直接读写 `ctx->send_socket`/`ctx->cparams`，
  不走 `include/llama.h`。**它无法被抽成独立库。**

### 8.3 三条路的成本/收益

| 路线 | 实际含义 | 结论 |
| --- | --- | --- |
| A. 把 `H:\prima.cpp\src` 换成 `H:\llamacpp-c\src` | 新版有 172 个架构文件、没有 `n_world` / `llama_context_n_layer_window`；`examples/main/main.cpp` 全部调用老 API | **编译不过；能过也不再是 prima** |
| B. 把 prima 的分布式层移植到新版 | 重写 socket/环/序列化/拓扑（~900 行，可按设计重写）+ 在 `process_ubatch` 前后挂 I/O | **技术上可行，收益 ≈ 0**（§8.4） |
| C. 只抽控制面设计（barrier / 拓扑重建）到 Python | 约 150 行 Python | **推荐**（§3 已列） |

### 8.4 为什么 B 的收益接近零

prima 的 2,000 行分布式运行时里，约 1,500 行在解决**"如何让一个进程内的架构跳过不属于自己的层"**
（改每个 build 函数 + `map_layer_to_local_id` + 每处 `for il` 插钩子）。**这个问题我们已经用
`LLAMA_LAYER_RANGE` 在 model-loader 层解决了，且架构无关**（§1）。剩下约 500 行是通信与拓扑，
而通信这块我们已经付过账：一跳 loopback **1.7 ms**（`MEASUREMENTS.md` §4.4）、
worker 协议往返比直调只多 **0.31 ms**（`MEASUREMENTS.md` §8.2）、跨机每步 **~24 ms 且省不掉**
（Tailscale 6 个包，prima 的 ZMQ 同样是 TCP）。

`[推断]` 换成"llama.cpp 内部直接发"最多省 1–2 ms/跳，而每步 157 ms 的大头是计算与链路 RTT。
**用 900 行 C++ 侵入换 1–2 ms，且把"架构无关性"重新绑死到具体架构上，是负收益。**

真要进程内接力（两段同进程、省掉 TCP），正确做法**不是**搬 prima，而是在我们自己的
`llama-context.cpp` 里：`process_ubatch` 之前写 hidden、之后从 `llm_graph_result` 取——新版结果结构比
老版干净。**这条路 v2.1 已经走了**：`--split-tail-range` 把同机两段放进同一进程（tail 是附着的 context，没有 TCP），同机多段是一等形态、不是设计错误。

## 9. 一句话建议

> **「prima 与我们不同构」不成立：我们就在做它做的事。
> 真正该记住的是它替我们试出来的那条死路：把层过滤放在 build graph 里，就必须改每个架构，
> 于是永远只能支持两个。我们把它放在 model-loader 里，只花 8 个文件 +227 行就换来了架构无关性，
> 这是我们最大的技术资产，比任何性能优化都值得保护。
> 剩下的可抄项按性价比排序：集群 ACK barrier（防"已起未就绪"）> `[推断]` 预取排除已上卡张量
> （只在读盘时花钱）> 拓扑重建降级（做容错时）> 端口 `base+index`（顺手）。
> 明确不要抄的：PRP 多圈 k、Windows `VirtualQuery` 驻留检测（在 file-backed mmap 上恒真）、
> **以及任何"把 prima 的 llama.cpp 换过来"的尝试——那是删掉 prima，不是复用 prima。**

## 附：证据边界

- `[实测]` 两边的机制、行号、结构体成员、代码规模均逐行核对；**未运行** prima.cpp。
- **未细读**：prima `src/llama.cpp` 中其余 40+ 个 `build_*`；本项目 Python 控制面实现（只读了 `docs/` 与 `topology-*.toml`）。
- 参考：`H:\prima.cpp\README.md`、ICLR 2026 abstract（`https://iclr.cc/virtual/2026/poster/10008093`）、
  `docs/{ARCHITECTURE,PATCHES,MEASUREMENTS}.md`、`README.md`。
