# 分段改造的已知边界

> 状态：**现行**
> 更新日期：2026-11-06

这份文档记录的是**原型阶段有意接受、产品阶段要付代价处理**的限制。不是 TODO 列表，是边界说明：
每条写清现象、根因、影响范围，以及修的时候要付什么代价。

当前状态（2026-11-06，核对到 `H:\llamacpp-c` 的 `f563a3c`）：上一版记为问题的六条，逐条核实后的现状见
§1；本周新增的边界集中在投机与远程链的取舍上（§2）。下面没有一条是「上游没做」——归因都在本仓。
代码引用一律写符号（函数/字段名），不写会随改动漂 100–300 行的 `文件:行`。

README 的六条边界与本文章节的对应：① 架构限定 → §2.1、② 远程链单序列 → §2.4、③ 远程链不做
投机 → §2.2、⑤ 投机加速比测不出 → §3.2；④ 多模态与 ⑥ 核显属输入/部署边界，分别在
[`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §7 末尾与 [`IGPU-SLOWDOWN.md`](IGPU-SLOWDOWN.md)。

---

## 1. 已实现：上一版记为问题的六条

| 上一版记的 | 现行实现 | 位置 |
|---|---|---|
| 远程链没有 KV 管理 | `llama_memory_split` 直接持有 `llama_split_remote*`，`seq_rm` / `clear` 逐个下发；`llama_split_attach_remote()` 装上聚合视图 | `src/llama-memory-split.h` 的 `llama_memory_split::remotes` / `seq_rm()` / `clear()`；`llama_split_attach_remote()` |
| 远程 `pos` 只传标量 | `split_expand_pos()` 把每个 token 的 4 个 M-RoPE section 全部下发，`req_header::n_pos` 让两端对齐 stride | `split_expand_pos()`（`src/llama-context.cpp`）、`req_header::n_pos` |
| `split-stage` 的段角色恒为 tail | `--role relay\|tail` 必填，用 `llama_model_n_layer_src()`（源层数，不是本地层数）校验角色与层范围 | `tools/split-stage/main.cpp` 的参数校验：`relay` 要求 `layer_hi < n_layer_src`、`tail` 要求 `layer_hi == n_layer_src` |
| 远程 logits 只有 1 行 | 本地链的 tail 只为 `batch.logits` 标记的行出 logits，一个都没标时兜底最后一行；**远程 tail 自协议 v5 起普通 decode 回候选而不是整行，只有走整行回退的请求才整行回传**（尺寸见 §2.2） | `llama_context::decode_split()` 的 `ask_logits`；`common_sampler_candidates_check()` |
| `state_write` 只有段数 | 段数 + 每个远程段 `{pos_max, n_bytes, blob}`；保存时各段 `pos_max` 必须等于链首，恢复时每段不得落后于快照 | `llama_memory_split::state_write()` / `state_read()`（`src/llama-memory-split.cpp`） |
| 本地链 prompt cache 只写链首 | `state_seq_*` 走 `get_memory()`（聚合视图）而不是 `memory`（链首） | `llama_context::state_seq_write_data()` / `state_seq_read_data()` |

最后一条的实测：3103 token 历史的整段重算是 19.85 s，第二轮只算 38 个新 token 是 0.71 s
（原记「3137 token」，日志里是 3103 + 16 生成；两个数的出处、口径与「修复前的形态是报错
而不是变慢」见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §1）。修复前的表现是链首恢复了、
尾段还停在旧位置，于是下一次截断式 `seq_rm` 在尾段失败。

---

## 2. 仍然成立的边界

### 2.1 `embeddings_pre_norm` 是本仓自造字段，只覆盖两个架构

**现象**：relay 段要的是**过 `output_norm` 之前**的末层 hidden。这个开关由本 patch 加进公共 API
（`llama_context_params::embeddings_pre_norm`，`include/llama.h`），实现只在 `src/models/qwen35.cpp`
与 `src/models/qwen35moe.cpp` 的 `embeddings_pre_norm` 分支（`t_embd = cur` 挪到 `build_norm` 之前）
——`src/models/` 下 171 个模型实现文件里的 2 个。

**根因**：出口是逐架构手写的，没有公共挂点。`t_layer_inp` 是上游现成的通用层输入出口，换过去会丢掉
「relay 段顺带不裁 LM head」这个好处（省一次近 515 MiB 的 `output` 张量），而且第一代就是走那个槽位、
当年因为要给 9 个架构各登记一次挂点而放弃。所以这不是能力缺失，是我们只做了两个架构。

**影响**：这一周把**失败方式**从「静默数值偏移」改成了「启动即拒」。`split_arch_exports_pre_norm()`
（`src/llama-context.cpp`）只认 `LLM_ARCH_QWEN35` / `LLM_ARCH_QWEN35MOE`，三处建链检查都是 `return -1`：
`split_attach_remote()` 的远程链首、`split_attach()` 的本地链首、`split_attach()` 段循环里的每个非最后段
（最后一段只出 logits，不查架构）。**远程段自己的架构在链首这端看不见**：`OP_HELLO` 握手（协议 v4 引入、v5 仍在）
只报 range / 宽度 / 层深（`qlh_split::stage_id`），不报架构。`--role` 必填、`split-stage` 另查角色与层范围
（`tools/split-stage/main.cpp`），只拦「段摆错」。**只覆盖两个架构这个事实不变**，变的是后果：以前
下一段再 norm 一次、argmax 常常仍对得上，现在建链时就失败。

**修的时候要付的代价**：两条路，各有缺口。按模型补 `embeddings_pre_norm`：机械，但要在每个架构上
逐个验证 `output_norm` 的位置与 norm 类型，且公共 API 里从此多一个只有本仓认识的字段；改走
`t_layer_inp[n_layer]`：载体是上游现成的，但要在每个架构补一行登记，并放弃裁 LM head 的收益。
真正的解法是抽到 `llm_graph_result` 层的公共出口，那要改上游公共代码、还要确认所有架构的 norm
位置一致——那是产品阶段的事。

### 2.2 远程链不做 DFlash / MTP 投机，瓶颈是「每帧要回多少行」

**现象**：本地双段链上两者都已接上——MTP 的 draft context 建在 tail 的模型上（NextN 头只在 tail），
DFlash 的 8 路 tap 按 head/tail 分别解析；远程链上两者都不做。

**根因不是延迟，是每帧要回多少行**。候选回传之后这条的形态变了一半：

- **普通 decode 已经不是问题了**。协议 v5 加了候选通道：tail 本地跑完
  `logit_bias → penalties → top_n_sigma → top_k`，只回 40 个候选（**352 B/token**，原 993 KB），
  server 从 `top_k` 之后接着跑。见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §8。
- **投机仍然是问题**。验证要 k+1 行的 **argmax**，而 argmax 需要完整一行的 logits——候选通道给不了。
  k=3 一帧是 4 行 × 993 KB ≈ 4 MB，千兆上 3–4 s，比本地把这 4 行算出来（0.33 s）还慢。
- **候选通道要求 `penalties` 在 tail 跑**（它在 `top_k` 之前），而投机验证的采样逻辑在 server 侧，
  两者叠起来要额外设计——「改公共协议比改带宽难」这部分没变。

**影响**：远程链的普通 decode 每步只回 40 个候选（走整行回退的请求除外），不投机。要做就得让验证的 argmax 也在 tail 算，或者回传 k+1 行
完整 logits（4 MB/帧，千兆以下不成立）。代价是协议再加一条路径 + 版本升 + 所有 stage 重编译。

### 2.3 级联投机（DFlash 块 + MTP 尾巴）在 server 原生路径上不做

**现象**：qlh 的 Python 驱动链上有 `--cascade-mtp N`（`chain_client.py` 的 `DFlashDriver.step()`，只在
**整块全接受**时触发），实测 +6%（`SPECULATION-MATRIX.md` §4）。llama-server 的原生投机循环里没有这个
级联：DFlash 与 MTP 是两个独立的实现，验证器只有一个。

**根因**：v1 的触发条件是「整块全接受」，而跨 ubatch 与深草稿让这个条件迅速变稀：server 日志按
`accepted k/k` 数，k=3 时 96/226 轮（42.5%）、k=15 时 5/177 轮（2.8%）；把 1/1、2/2 这类块尾也算成全
接受是 98/226（43.4%）与 8/177（4.5%），两个口径见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §7。
DFlash 的价值恰恰在深草稿。放开成「至少接受 1 个」也只剩 +2~4%，代价是 4 个文件 330–450 行、还要改 server 主循环。

**影响**：想要 DFlash 的深度与 MTP 的便宜尾巴叠加，只能在 Python 驱动那条链上做，server 原生路径
拿不到。

**这条不违反「能力对等」**：llama.cpp 上游本身没有级联投机，所以不做它不构成与上游的差距。

### 2.4 远程链是单序列（设计决定）、append-only 的

**现象**：`split_attach_remote()` 拒绝 `n_seq_max > 1`（`-np > 1` 在 `tools/server/server.cpp` 的启动
检查里先被拒）；远程链只导出**checkpoint 那一档**序列状态（`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`，即
recurrent 部分），全量 state（含 KV）由 `split_state_unavailable()` 判掉、`state_get_size()` 返回 0。

**这是设计决定，不是缺口。** 技术上做得到、约 105 行（协议加每行 `seq_id`、client/stage 逐行
传递），且 checkpoint 本来就是 per-seq 的，不用重做恢复顺序。不做的理由：

- 远程链解决「一台机器装不下」或「借用算力」，不是多租户；
- llama.cpp 的并发是**连续批处理**而非高并发，`-np` 的意义是「别让请求排队」，不是扛 QPS；
- 真要并发，**在 head 那台机器起多个 server、各走自己的链**更简单，也更容易隔离故障——单条链
  上任何一段出问题会拖垮所有 seq。单机内则用本地链（已验证 `-np 4` 并发与串行逐字节相同）。

**影响与代价**：slot 复用在远程链上退化为从头重算；追加式复用仍全命中，但**聊天历史被模板
重渲染时一个 token 都复用不上**——这是 append-only 的边界。全量 state（含 KV）仍不可用，那是
另一笔过网成本。决策是「不修这条，改部署形态」；将来真有远程多用户需求，105 行路径是清楚的。

### 2.5 远程链的 checkpoint 每段约 12.6 MiB 过网

**现象**：checkpoint 覆盖全链之后，每次保存要把每个远程段的序列状态取回来。35B 的 8 层 qwen35moe
段，每段 blob 实测 **13,172,908 B = 12.56 MiB（恒定）**——不是早先按 6.4 MiB 推算的值，差 2 倍
（早先那个数是 2B 的 8 层段：6,733,996 B，见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §1）。
整份 checkpoint 62.813 MiB = head 本地 50.2 + 远程 12.56。两个远程段、每请求两个 checkpoint → **约 25 MiB 过网**。

**跨机实测**（本机 head + ROG tail，Tailscale 直连，裸 TCP 16.06 MB/s）：两次 remote blob
共 26.3 MiB，**1.91 s**，纯传输约 1.64 s；单次 checkpoint 创建 1.18 / 1.07 / 0.84 s（`build-split/xhost/REPORT.md`）。

**根因**：这是 checkpoint 的正确性成本，不是实现浪费——只存链首会让尾段停在旧位置。省不掉，
只能选择「不 checkpoint」或「付这笔带宽」。

**影响**：**盈亏实测是赚的**。同一 slot 连发两个请求，第二个命中 136 token（`prompt_n=10`，
2830.7 ms）；不命中要按 122 ms/token 重算约 146 token ≈ 17.8 s，checkpoint 花 1.9 s →
**净赚约 14.7 s**。盈亏平衡点约 15–20 token 复用前缀（1.9 s ÷ 0.122 s/token ≈ 16），比早先按 6.4 MiB 估算的更靠前。

**修的时候要付的代价**：要压缩就得改 blob 编码（现在直接搬 recurrent 状态），且两端版本必须同步升级。
短期没有比「容忍 12.6 MiB/段」更划算的做法。

### 2.6 `--split-layer-range` 单独跑：两道检查都拒了，边界移到「标记由谁置真」

**现象**（现行）：一个层范围单独起 server，两种误配都在模型加载后、建上下文前报错退出，判据都在
`common_init_from_params()`（`common/common.cpp`）里：范围在模型中间结束、又没给 `--split-tail-range` /
`--split-remote`；或者范围不从 0 开始（`20:24` 这种「就是尾巴」的形态）而 `common_params::split_range_is_stage` 为假。

**根因**：server 进程内加载本地 tail 时用的正是 `20:24` 这个形态，所以区分「合法的一环」与「误配」
只能靠一个内部标记：`split_range_is_stage` 默认 false（`common/common.h`），只有 llama-server 加载
tail 时置真（`tools/server/server-context.cpp` 的 `tail_params.split_range_is_stage = true`）。两道检查
都读**从模型读回的**范围（`llama_model_layer_range()` / `llama_model_n_layer_src()`）而不是参数，所以
`LLAMA_LAYER_RANGE` 那条入口也一起覆盖了。

**残留边界**：判据是「本进程是不是在拼链」，不是「这个范围是否合法」。任何在进程内构造
`common_params` 又把标记置真的调用方都会跳过第二道检查；`split-stage` 不经过
`common_init_from_params()`，它有自己的角色/层范围校验（§2.1）。这两种误配的后果已经从「能跑、返回
200、输出通顺但错」变成「启动即错」——**只对这一条加载路径成立**。修的话要把标记换成「范围 + 链
身份」的显式描述（谁在哪一环、上一环是谁），那是参数/协议层的一次重构，现在不值得。

### 2.7 链路停摆：`/health` 仍 200；超时曾杀 head，现在链死而进程活

**跨机实测**（挂起 stage；`build-split/xhost/health_probe.json`、`build-split/chain_death_hello.out`）：
`/health` 24 次采样全部 200；`/completion` 精确 120211.65 ms（+120.2 s）失败；head 在
`seq_rm ... recv failed` 之后走 `common_context_seq_rm()` 的 `GGML_ABORT("failed to remove sequence …")`
→ **进程退出**（现在的代码只对本地链走到这一行）。早先「变成普通请求失败」的说法是错的。

**已改**：失败置 `remote_failure` 而不 abort，`decode_split()` 算之前就查它并返 `-3`——后续请求全
失败，绝不假装成功；链死后不再碰网络（第二个请求 0.01 s 失败）。`/health` 仍 200 是刻意的（存活
语义），要看链问 **`?deep=1`**：链死快速 503、链活真发 1-token 请求、忙时 `200 {"deep":"busy"}`。

**自愈**：下一个从位置 0 开始的请求重建（有界 reconnect → 整链 clear → 恢复超时），失败保持 dead
+ 5 s 节流（`kRebuildThrottleMs`，`src/llama-memory-split.cpp`）。闸门是结构性的：重建后唯一一致
状态是「全空」，空链只能从 0 写，所以「KV 空但接着算」不可能。实测只重启 stage：200、
`prompt_n`=完整长度、`cache_n=0`、输出逐字节相同。

**握手**（`PROTO_VER` 5 里的 `OP_HELLO`，v4 引入）：重连必须重握手，接错的 stage 让重建失败；防接错不防恶意，
也区分不了同形状不同权重的 GGUF——不做模型哈希的代价。

### 2.8 `can_shift=false`：对 qwen35 忠实，对本地纯 KV 链过度保守

**现象**：聚合 memory 的 `get_can_shift()` 恒返回 false（`llama_memory_split::get_can_shift()`）。

**对 qwen35/qwen35moe 这条是忠实的**：两者是 IMROPE（`llama_model_rope_type()`，`src/llama-model.cpp`），
`n_pos_per_embd() = 4`（`llama_hparams::n_pos_per_embd()`），而 `llama_kv_cache::get_can_shift()` 对
`n_pos_per_embd() > 1` 就返回 false（`src/llama-kv-cache.cpp`）——**整模同样不支持 ctx_shift**。所以这条差距为 0，不该记成欠账。

**欠账在别处**：本地纯 KV 链（非 MROPE 模型，`n_pos_per_embd() == 1`）本来可以 shift，聚合视图
却一律报 false，server 于是退化到「重算前缀」。这段是过度保守。

**修的时候要付的代价**：`can_shift` 要变成「所有段的 `can_shift` 的与」，并且 shift 要真的广播到
每一段。前者一行，后者是 `seq_add` 的跨段语义——现在 `seq_add` 对远程段是直接拒绝的
（`llama_memory_split::reject_remote("seq_add")`），所以这条只对本地链有意义。

---

## 3. 浮点与测量的边界（不是 bug，但会让验收变成假阳性）

### 3.1 跨 ubatch 的「逐位相同」不成立

**现象**：同一个 prompt 在 `n_ubatch` 装不下时被切成两个 ubatch，GEMM 的累加顺序随之改变，
head 自己的 tap 与整模的差可以有 4e-4（29 token / `n_ubatch=12`，`tests/dflash-taps.cpp` 的头注释），
同一轮测量里最大到 1e-1 量级（**未落盘**：只在当时的测量记录里，日志中没留下这一行）。

**根因**：浮点加法不满足结合律。切 ubatch 就换了归约树，这是性质不是缺陷，两边都没有错。

**影响**：所有「逐位验收」必须在**单 ubatch 内**做。长 prompt 的逐位对比如果不控制这一点，
会稳定地给出假阳性，然后被误判成分段引入的数值错误——本仓为此排查过不止一次。

**修的时候要付的代价**：不修。改成「容差比较」反而更糟：分段链的正确性判据只能靠逐位，
容差会把真实的接线错误一起放过。折中是**把每次对比的 ubatch 边界写进结论**，而不是放宽判据。

### 3.2 投机加速比随负载浮动，k 的拐点随模型

**现象**（2026-11-06 实测，口径都是贪心、量到 ms/token）：

- MTP k=2 在纯代码 prompt 上 1.33×，含中文的混合负载 1.06×，纯中文 1.00×（**本周未能复现**，见下）
- DFlash k=3 在 CPU 上 +23.7%——那是 CPU 的数，不能外推到 GPU
- 2B 上 k 的拐点复现不出来：k=3 优于 k=2（25.09 vs 28.32 ms/token；早先记的 25.87 **未复现**），接受率只从 97.7% 掉到 96.6%
- `-np > 1` + MTP 明显变慢（整模加 MTP：22.56 → 28.37 ms/token）

**这四条的数字出处与重测结果见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §2 与 §2.3。**
其中第一条的 1.33× / 1.06× / 1.00× **未能复现**：同一副链、同样的贪心口径，真实代码 / 混合 /
中文 prompt 上 k=2 是 0.87× / 0.69× / 0.63×（接受率 47% / 27% / 19%），1.32× 只出现在模型
能复读的退化 prompt 上。上面「按负载选 k」的结论不变，但**不要再用这三个数当证据**。

**根因**：节省来自「一轮前进多少 token」与「每行的边际成本」两笔账的比值，而两者都随负载与模型变。
中文的接受率低、代码的接受率高；DFlash 的深草稿只在接受率高时才回本。
`-np > 1` 的变慢像 draft context 的输出缓冲按 `n_seq_max` 放大（`llama_context::output_reserve()`
的 `n_outputs_max = max(n_outputs, n_seq_max())`），但机理**没有定论**。

**影响**：**链上默认不开投机**（llama-server 的 `--spec-type` 默认空，日志是
`no implementations specified for speculative decoding`）。开之前要按负载选源和 k，并且
「35B 上 k=2 是上限」这类结论**不能跨模型搬**——2B 上 k=3 更好，而且只有一个 prompt 的样本，还不足以定论。

但这里有个更重要的限制，比上面所有数字都强：**本机双进程测不出投机收益**。head 与 tail 跑在
同一台机器上时，两者走内存与管道、**没有 25–30 ms 的跨机往返可省**（而削减往返正是投机唯一
该拿的好处），同时两个进程抢同一套 CPU 与内存带宽，draft 的额外算力被全数计入——付出成本、
拿不到好处，这是个退化配置。多机的账反过来算：省 `(N − N/3.2) × 25–30 ms` 往返，多花 tail 上
一次 draft 前向，差一个量级。**真实多机上大概率划算，这一点没有测过**，不能用本节数字反驳。
详见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §0.1。

**修的时候要付的代价**：不修。要做的是把「按负载选」做成配置项而不是默认值，这需要一套负载分类与实测表，比投机本身的工作量大。

---

## 4. 一条给未来的提醒

上面这些问题有一个共同形状：**它们多数不会报错，只会算错或悄悄退化**——能拒的只有架构白名单、
角色/层范围（含 `split_range_is_stage`）、远程单序列（§2.1、§2.6、§2.4）；其余都是「看起来在正常
工作」，错误只体现在数值或长期行为上。

所以验证必须比 logits 而不是比生成的文字（argmax 一致经常是巧合），长会话要盯 KV 占用而不是只看
前几个请求，跨 ubatch 的对比要先确认 ubatch 边界。原型里我们用 `tests/split-smoke.cpp` 和
`tests/dflash-taps.cpp` 做逐位对照，这个习惯在做产品时要保留。
