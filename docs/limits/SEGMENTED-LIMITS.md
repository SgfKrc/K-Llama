# 分段改造的已知边界

> 状态：**现行**（文中引用的 v2 工具链——`chain_client` / `stage_worker` / `shim/` / `koakuma`——已退场，归档在 `slimarc/prototype`，命令与路径只作历史口径）
> 更新日期：2026-11-11
> 适用范围：原型阶段有意接受、产品阶段要付代价处理的限制。每条写现象、根因、影响范围与修复代价；
> 实测数据与逐位证据归 [`SPLIT-MEASUREMENTS.md`](../measure/SPLIT-MEASUREMENTS.md) 及四份专题：[`PROTOCOL-V9.md`](../design/PROTOCOL-V9.md)、
> [`MODEL-SWAP.md`](../measure/MODEL-SWAP.md)、[`SPECULATION-DATA.md`](../measure/SPECULATION-DATA.md)、[`CROSS-MACHINE.md`](../deploy/CROSS-MACHINE.md)。

代码核对到 `H:\llamacpp-c` 的 `4acf03c8b`（基底 `09338b85e`，55 文件 / +20345 −245）。除 §2.2 的 batch invariance 一段（上游主体）与 §2.9 的 hybrid
`seq_rm` 短路（上游缺陷，本仓只做「失败可见 + 调用方降级」）外，每条的归因都在本仓。代码引用写符号，不写会漂的 `文件:行`。

**35B 的权重在 2026-11-08 换成官方 `UD-Q4_K_M`**：§2.2 的「整模 0.910 / 链上 0.688（旧路径；keep-KV 0.788；catch-up 每批 0.910）」是官方版的数，ROG / Surface / 回环收益与 §3.2 的
DFlash、MTP
加速比是旧蒸馏版上的数，两份权重不能混用。口径见 [`MODEL-SWAP.md`](../measure/MODEL-SWAP.md)。

README 六条边界对应：① 架构 → §2.1，② 单序列 → §2.4，③ 链上投机 → §2.2，⑤ 加速比随部署形态走 → §3.2；④
多模态与 ⑥ 核显属输入 / 部署边界，在 [`SPLIT-MEASUREMENTS.md`](../measure/SPLIT-MEASUREMENTS.md) §7 与 [`IGPU-SLOWDOWN.md`](../measure/IGPU-SLOWDOWN.md)。

## 0. 与上游的关系（rebase 到 `09338b85e` 之后）

fork 整体 rebase 到上游 193 个提交之后的 `09338b85e`，本仓「自己的」代码少了 17 个文件——它们现在与上游逐字节相同，`git diff 09338b85e 4acf03c8b`
里不再出现。补丁统计与明细见 [`PATCHES.md`](../deploy/PATCHES.md) 与 [`PATCHES-DETAIL.md`](../deploy/PATCHES-DETAIL.md)。

### 0.1 16 个文件与上游逐字节相同

| 类别 | 文件 | 为什么 |
|---|---|---|
| 「两次对齐」时期手工适配过的 12 个 | `llama-kv-cache.{cpp,h}`、`llama-kv-cache-dsv4.{cpp,h}`、`deepseek32/4/41`、`dots3note`、`glm-dsa`、`minimax-m3`、`mistral-large4`、`qwen4exp` | 那些改动本来就是把上游新 API 抄回来（per-layer rotation `fb5997274`、`llm_attn_mask` 去 `cnv` `ca8daad19`）；上游自己带着这些提交，适配没有存在理由 |
| 只承载 `shifts` 机制的 4 个 | `llama-graph.h`、`llama-memory-recurrent.{cpp,h}`、`delta-net-base.cpp` | 删掉 `build_rs_shift_snapshots` 之后只剩上游代码 |

`src/models/dflash.cpp` 是例外：rebase 那一版与上游逐字节相同，2026-11-11 这一版因 device-buffer 诊断输出（§0.4）重新进入变更集，所以逐字节相同的是 16 个、54
文件并集含它。8 个被逐个核对过的与上游逐字节相同，不是「差几行」；`src/llama-graph.cpp` 只剩 `build_inp_embd` 的 relay 段（`tok_embd == nullptr`
时不建 `GET_ROWS`），因为 relay / tail 没有 token embedding。

### 0.2 删掉 `build_rs_shift_snapshots`：同一问题的两种解法

- **原做法**：recurrent 快照按 ubatch 往前挪（shift），使跨 ubatch 的回退也能读到对应位置。
- **上游做法**：`ab3768cdc` 相反——**拒绝**超出本 ubatch 快照的回退（`cell.rs_max = n_seq_tokens − 1`），并用 `split_equal()` 的 `n_keep_tail` 保证整帧回退落在同一 ubatch 内。
- **为什么整块删**：两套语义互不相容（shift 要求旧槽位保持旧内容，上游按本 ubatch 行号写），并存会让人以为 shift 在补上游的洞。删除范围：`shifts` 字段、`can_reuse` 比较、`get_rs_shift()` 与全部调用点。
- **影响**：「跨 ubatch 回退」现在由上游拒绝，不是由本仓支持；§2.9 的 `seq_rm` 边界有一半是上游的设计选择。

### 0.3 整帧回退在上游 `rs_max` 下永远不合法

- **现象**：验证帧写 `n_draft + 1` 行 → `rs_max = n_draft`。删 `n_draft` 行的部分接受合法，删 `n_draft + 1` 行的整帧放弃**永远必被拒**；而
  `frame_needs_ckpt = n_draft + 1 > n_rs_seq` 在 `n_draft <= n_max` 内恒假，整帧回退于是落到 `seq_rm` → 被拒 → 请求 500。
- **修法**：三个谓词各司其职——`frame_needs_target_ckpt(slot, n_draft)` 回答「要不要捕获」（`FULL`；或 `RS` 且 `has_grammar || has_reasoning_budget`；或 `RS` 且
  `n_draft + 1 > n_rs_seq`）；`partial_uses_ckpt` 回答「部分接受能否不用 ckpt」（`FULL || n_draft > n_rs_seq`）；`frame_has_ckpt = !slot.spec_ckpt.empty()`
  回答「本帧到底有没有捕获到」。整帧回退用最后一个、**不重新求值**——两次求值一旦漂移就走错路径（实测出过 500）。配套 `spec_ckpt.clear()` 让 `empty()` 精确等于「本帧没捕获过」；没有 grammar / 预算的请求保持零捕获的便宜路径。
- **为什么藏得住**：`verify/verify.sh` 组 8/9/10 不走这两条路径。

### 0.4 device-buffer 的可用性判定：谓词回答的问题不对

- **现象**：上游 `2cbca389f` 之后 DFlash 优先从 device 的 `layer_inp_src` 读。跨机与 loopback 都崩、整模不崩，
  `GGML_ASSERT(g.layer_inp_src != nullptr && g.layer_inp_src->ne[0] == n_embd_inp)` 挂在第二个合取项。
- **根因**：本仓 `supports_embd_row()` 原来只答「buffer 存在、读得到吗」，于是分段链上它答「支持」，而 buffer 只装了 head 那段的 6 路 extract layer（12288 列），dflash 的 `fc` 要 8
  路（16384 列）。整模宽度恰好等于 `target_layers × n_embd`，所以整模不崩。
- **修法**：可用写成全条件——`存在（非 null）且宽度 == n_embd_inp_enc() 且层顺序一致`（`build_layer_inp_dev` 升序 + 构造期 ascending 检查）。上游没有分段链，隐含「列数 = target_layers
  数量」这个前提，从未被检查。修法细节、组 11 的 8 项与「为什么不按 `is_split_chain` 判」见 [`PATCHES-DETAIL.md`](../deploy/PATCHES-DETAIL.md)。
- **未验证边界**：宽度条件会允许「head 覆盖全部 extract layer」的切法（如 head `0:39` + tail `39:40`）走 device 路径，这是从代码推出来的、没有实测过。

## 1. 已实现：此前记为问题的六条

| 此前记的 | 现行实现 | 位置 |
|---|---|---|
| 远程链没有 KV 管理 | `llama_memory_split` 直接持有远程指针，`seq_rm` / `clear` 逐个下发；`llama_split_attach_remote()` 装聚合视图 | `src/llama-memory-split.h` 的 `remotes` / `seq_rm()` / `clear()` |
| 远程 `pos` 只传标量 | `split_expand_pos()` 下发每个 token 的 4 个 M-RoPE section，`req_header::n_pos` 对齐 stride | `split_expand_pos()`、`req_header::n_pos` |
| `split-stage` 段角色恒为 tail | `--role relay\|tail` 必填，用 `llama_model_n_layer_src()` 校验：relay 要求 `layer_hi < n_layer_src`、tail 要求 `layer_hi == n_layer_src` | `tools/split-stage/main.cpp` |
| 远程 logits 只有 1 行 | 本地 tail 只为 `batch.logits` 标记的行出 logits；远程 tail 自 v5 起普通 decode 回候选、整行回退才整行传，验证帧走 v7 逐行 argmax | `decode_split()` 的 `ask_logits`；`common_sampler_candidates_check()` |
| `state_write` 只有段数 | 段数 + 每段 `{pos_max, n_bytes, blob}`；保存时各段 `pos_max` 必须等于链首，恢复时不得落后 | `llama_memory_split::state_write()` / `state_read()` |
| 本地链 prompt cache 只写链首 | `state_seq_*` 走 `get_memory()`（聚合视图）而非链首 `memory` | `state_seq_write_data()` / `state_seq_read_data()` |

最后一条实测：3103 token 历史整段重算 19.85 s，第二轮只算 38 个新 token 是 0.71 s（原记 3137 token，日志是 3103 + 16 生成；出处见
[`SPLIT-MEASUREMENTS.md`](../measure/SPLIT-MEASUREMENTS.md) §1）。修复前的表现是链首恢复了、尾段停在旧位置，下一次截断式 `seq_rm` 在尾段失败。

### 1.1 段级 offload：`-ngl N` 数「本段最后 N 块」

- **语义**：`src/llama-model.cpp` 的 `load_tensors()` 里 `i_gpu_start` 按 `layer_range_active()` 分叉——整模是 `n_layer_all + 1 - N`（与上游逐位不变），range 载入是
  `n_layer_all - N`（N 是本段块数，输出槽不算）。
- **输出槽**：段内只有 `N >= 本段块数` 才把 logits 槽带上卡。tail 的本地块数含 NextN（`32:40` 是 9 块），`-ngl 8` 只搬 33..40，整段上卡要 `-ngl 9` 或 `all`。
- **两条边界**：① 整模路径必须逐位不变（`verify/verify.sh` 组 2）；② `-ngl == 本段块数` 曾让 `devices.at()` 越界，分配比例已钳到 `act-1`。

实测 tail `32:40` 全量上卡后段内 tg 35.7 → 1754.2；语义与实测见 [`SEGMENT-OFFLOAD.md`](../measure/SEGMENT-OFFLOAD.md) §2、§3。

### 1.2 每段独立 offload 是部署能力

- **形态**：`--split-remote` 的每个端点都是独立 `split-stage` 进程，各带 `--layer-range` 与 `-ngl`；head 是 `llama-server`，用自己的 `-ngl`，四种摆法都合法。
- **边界**：显存与预算按段独立核算；`op_offload` 默认值按角色给（split 角色默认关，见 [`IGPU-SLOWDOWN.md`](../measure/IGPU-SLOWDOWN.md) §⑤）；head 与 tail 不在同一台机器时，两边 `-ngl`
  只能各按本机显存上限定，没有全局最优解。

## 2. 仍然成立的边界

### 2.1 `embeddings_pre_norm` 只覆盖两个架构

- **现象**：relay 段要过 `output_norm` 之前的末层 hidden，开关是 `llama_context_params::embeddings_pre_norm`，实现在 `src/models/qwen35.cpp` 与 `qwen35moe.cpp`
  的 `embeddings_pre_norm` 分支——`src/models/` 176 个模型实现里的 2 个。
- **根因**：出口逐架构手写，没有公共挂点。换到上游通用的 `t_layer_inp` 会丢掉「relay 段顺带不裁 LM head」这个好处（省一次近 398 MiB 的 `output` 张量），且第一代就走那个槽位、因要给 9 个架构各登记一次而放弃。
- **影响**：失败方式从「静默数值偏移」变成「启动即拒」——`split_arch_exports_pre_norm()` 只认 `LLM_ARCH_QWEN35` / `LLM_ARCH_QWEN35MOE`，三处建链检查都 `return -1`；
  远程段的架构在链首看不见，`OP_HELLO` 只报 range / 宽度 / 层深。
- **代价**：按模型补是机械工作，但要逐个验证 `output_norm` 位置与 norm 类型；改走 `t_layer_inp[n_layer]` 要每架构补一行登记并放弃裁 LM head。真正解法是抽到 `llm_graph_result` 层的公共出口，
  要改上游公共代码——产品阶段的事。

### 2.2 远程链的投机：能做，收益中位约 +6%，随负载从 +8.4% 到 −1.2%

- **结论**：能做，收益中位约 +6%，随负载从 +8.4% 到 −1.2%，弱机器（Surface −40%）与回环（0.67×）上是负的。实现在协议 v7 验证通道、v8 远程 MTP draft、v9 `layer_inp` tap 三层，见
  [`PROTOCOL-V9.md`](../design/PROTOCOL-V9.md)；五类 prompt 与弱机器表见 [`CROSS-MACHINE.md`](../deploy/CROSS-MACHINE.md)；逐 prompt、k 扫描、结构成本见
  [`SPECULATION-DATA.md`](../measure/SPECULATION-DATA.md)。
- **为什么负**：每帧两次 hop；MTP draft 不看 grammar / rbudget，接受率从 2.76 掉到 1.46–1.73 token/帧；回退率 9–19%（`toolcall.gbnf` 实测 11/16 帧被拒），每次回退 ≈ 一个普通 round；
  回环上 993 KB 只要约 1 ms，省带宽这个唯一好处不存在。

**固定折扣：链上 MTP 接受率比整模低——keep-KV 默认 64 档 13%、512 档 4%；旧路径 64 档 24%、512 档 23%**（官方 35B：整模 0.9098 / mean len 2.81，双段链**旧路径** 0.6879 /
2.37，现行默认 0.7877（115/146），数值逐字节相同）。根因是本仓加的两处：`speculative::process()` 的 catch-up 条件是 `!is_mem_shared && !is_chain && !is_remote`（上游只有
`!is_mem_shared`），`speculative::draft()` 在 chain 下每轮 `llama_memory_clear` 草稿 KV。两者构成自洽循环：不 catch-up → 草稿没有 prompt 的 KV →
必须每轮清空 → 即使 catch-up 也读不到 → 跳过 catch-up 是对的。旧路径的代价是接受率封在 0.688。

**现行路线**（`QLH_CHAIN_DRAFT_KEEP_KV=0` 关掉回到旧路径：请求开始时清一次 + 每轮按 target 的接受量 `seq_rm`，让草稿看到本次请求已生成的 token，而非 prompt），2026-11-10
起是默认：35B 链上 64 档 0.688 → 0.788、512 档 0.582 → 0.724（keep-KV 默认比整模低：64 档 13%、512 档 4%；旧路径 64 档 24%、512 档 23%），生成文本逐字节不变（见 [`MODEL-SWAP.md`](../measure/MODEL-SWAP.md) §11.4）。

**判据**：llama.cpp 不承诺 batch invariance，CPU + 量化落在上游 `DETERMINISM.md`（llama.cpp 的 `docs/` 下）的覆盖空白里，所以判据是无回退时链上投机 == **整模投机**、逐字节相同（能过）；
带回退时只保证到漂移翻掉的第一个近似并列处；「投机 == 不投机」降为 INFO。三个 ggml 分界、上游 PR / issue 与 bit-exact 的改造代价见 [`PROTOCOL-V9.md`](../design/PROTOCOL-V9.md) §4。

**仍然成立的结构事实**：采样投机不做；MTP 只能跟末段；异构弱网上张量并行是纯负收益；ngram 在 chat 模板下全程 0 draft——数字与判据见 [`SPECULATION-DATA.md`](../measure/SPECULATION-DATA.md) §4。35B 这一档内部稳：
官方版重扫 k=1..8 甜点仍是 2，与蒸馏版同档。

**三条判据**：某个 draft 源在某个 prompt 上快，先问 prompt 是否专为它构造；收益必须按部署形态报；判据先问这份代码承诺过什么。

### 2.3 级联投机（DFlash 块 + MTP 尾巴）在 server 原生路径上不做

- **现状**：Python 驱动链有 `--cascade-mtp N`（`chain_client.py` 的 `DFlashDriver.step()`，只在整块全接受时触发），实测
  +6%（[`SPECULATION-MATRIX.md`](../measure/SPECULATION-MATRIX.md) §4）；`llama-server` 的原生投机循环没有级联。
- **根因**：触发条件「整块全接受」随跨 ubatch 与深草稿迅速变稀——k=3 时 96/226 轮（42.5%）、k=15 时 5/177 轮（2.8%），把 1/1、2/2 块尾也算上是 43.4% / 4.5%。
- **代价**：放开成「至少接受 1 个」只剩 +2~4%，代价是 4 个文件 330–450 行并改 server 主循环。想要 DFlash 的深度与 MTP 的便宜尾巴叠加，只能在 Python 驱动链上做；上游本身没有级联。

### 2.4 远程链是单序列、append-only

- **现象**：`split_attach_remote()` 拒绝 `n_seq_max > 1`（`-np > 1` 在 `tools/server/server.cpp` 的启动检查里先被拒）；远程链只导出 checkpoint
  那一档序列状态（`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`），全量 state（含 KV）由 `split_state_unavailable()` 判掉、`state_get_size()` 返回 0。
- **这是设计决定**：远程链解决「一台装不下」或「借用算力」；llama.cpp 的 `-np` 是连续批处理，意义是别让请求排队；真要多租户，在 head 起多个 server、各走自己的链更简单也更好隔离故障。
- **影响**：slot 复用在远程链上退化为从头重算；追加式复用全命中，但聊天历史被模板重渲染时一个 token 都复用不上。
- **代价**：技术上约 105 行可做（协议加每行 `seq_id`），决策是「不修这条，改部署形态」。

### 2.5 远程链的 checkpoint：blob 12.56 MiB 留在 stage 内，v6 起过网 8 字节

- **现象**：35B 的 8 层 qwen35moe 段，每段 blob **13,172,908 B = 12.56 MiB（恒定）**（2B 的 8 层段是
  6,733,996 B = 6.42 MiB）。**blob 留在 stage 进程内**，与历史长度无关。
- **过网量**：v5 及更早是整 blob 往返（跨机实测两次 26.3 MiB、1.91 s）；**v6 起每段 8 字节 handle**
  （`STATE_FLAG_HANDLE`，`src/llama-split-proto.h:224-235`），被消除的是 0.8–1.2 s/次。
- **handle 过期**是 `ST_STALE`，可恢复：stage 在应用前就拒绝，链停在原地，调用方重跑 prompt 即可。
- **详版与盈亏账**：`report/06-局限与结论.md` §1.9。

### 2.6 单独层范围的两道检查：边界在「标记由谁置真」

- **现象**：一个层范围单独起 server，两种误配都在加载后、建上下文前报错，判据在 `common_init_from_params()`：范围在模型中间结束又没给 `--split-tail-range` / `--split-remote`；或范围不从 0
  开始（`20:24` 这种「就是尾巴」）而 `common_params::split_range_is_stage` 为假。
- **根因**：server 进程内加载本地 tail 时用的正是 `20:24`，区分「合法一环」与「误配」只能靠内部标记（默认 false，只有 llama-server 加载 tail 时置真）。
  两道检查读从模型读回的范围（`llama_model_layer_range()` / `llama_model_n_layer_src()`），`LLAMA_LAYER_RANGE` 入口一并覆盖。
- **残留边界**：判据是「本进程是不是在拼链」，不是「范围是否合法」；任何在进程内构造 `common_params` 又把标记置真的调用方都跳过第二道检查。
- **代价**：修它要把标记换成「范围 + 链身份」的显式描述，是参数 / 协议层重构，现在不值得。

### 2.7 链路停摆：`/health` 仍 200；链死而进程活

- **现象**：跨机实测（挂起 stage）——`/health` 24 次采样全部 200；`/completion` 精确 120211.65 ms（+120.2 s）失败；head 在 `seq_rm ... recv failed` 之后曾走
  `GGML_ABORT("failed to remove sequence …")`，进程退出（`seq_rm` 改成 `bool` 后这一行已不存在）。
- **已改**：失败置 `remote_failure` 而不 abort，`decode_split()` 算之前就查它并返 `-3`（链死后第二个请求 0.01 s 失败）。`/health` 仍 200 是刻意的（存活语义），要看链就问 `?deep=1`：链死快速
  503、链活真发 1-token 请求、忙时 `200 {"deep":"busy"}`。
- **自愈**：位置 0 的请求重建（有界 reconnect → 整链 clear → 恢复超时），失败保持 dead + 5 s 节流（`kRebuildThrottleMs`）；重建后唯一一致状态是「全空」，所以「KV 空但接着算」不可能。只重启 stage
  恢复后输出逐字节相同。
- **握手**：`OP_HELLO`（v4，`PROTO_VER` 到 9）区分不了同形状不同权重的 GGUF——不做模型哈希的代价。

### 2.8 `can_shift=false`：对 qwen35 忠实，对本地纯 KV 链过度保守

聚合 memory 的 `get_can_shift()` 恒返回 false。

- **对 qwen35/qwen35moe 是忠实的**：两者是 IMROPE，`n_pos_per_embd() = 4`，而 `llama_kv_cache::get_can_shift()` 对 `n_pos_per_embd() > 1` 返回 false——
  整模同样不支持 ctx_shift，差距为 0。
- **欠账在别处**：本地纯 KV 链（`n_pos_per_embd() == 1`）本来可以 shift，聚合视图一律报 false，server 退化到重算前缀。
- **代价**：`can_shift` 要变成「所有段的与」，且 shift 要真广播到每一段；后者是 `seq_add` 的跨段语义（现在 `reject_remote("seq_add")` 直接拒绝），所以只对本地链有意义。

### 2.9 `seq_rm` 失败显式化：上游短路、本仓改成 `bool`、两条边界判据

- **现象**：投机评测发现「同一位置回退前后算出的 `h` 不同」，根因是 `seq_rm` 失败被静默忽略——调用方以为行删掉了，继续在「prompt 已回退、KV 没回退」的状态上算下一个 token，不报错、只算错。
- **根因（上游）**：`llama_memory_hybrid::seq_rm` 先删 recurrent，失败就 `return false`，**attention KV 的删除被整体跳过**；hybrid 在 server 默认 `n_rs_seq = 0` 下
  partial 分支要求 `rollback <= n_rs_seq` 不成立。上游已有 open PR #28007 与 issue #31656，本仓不提新 issue。本仓原先把失败吞掉了（这一半已修）；能修的是「失败可见 + 每个调用点按场景降级」。
- **改法**：`common_memory::seq_rm` / `common_context_seq_rm` 从 `void` → `bool`，五个调用点按场景退化：

| 场景 | 失败后 |
|---|---|
| prompt 预处理（`p0`，cache 复用裁尾） | `prompt_clear` + **用 task 的 tokens 重建 `prompt.tokens`** + return，下一轮从 pos 0 全量重算 |
| context shift | `stop = STOP_TYPE_LIMIT`，跳过 `seq_add` 与 prompt 重写 |
| cache shift / 验证帧回退 | `send_error` + `prompt_clear` + `release` |
| spec trim（部分接受的 ckpt 回退、普通接受裁草稿行） | 先发完手上的 token，再结束响应 + 全清 |

**边界一：链死时不能结束请求。** 把「链死」也走「失败 → `send_error` + `release`」会让自愈永久失效：`try_rebuild` 只在一次从 pos 0 开始的 decode
里触发，结束请求会跳过那次 decode，链就永远死。现行做法：链死只记 ERROR、不结束请求。

**边界二：`p0 > pos_max` 是成功。** 让所有 partial `seq_rm`
返回 false（含空删除），降级调用方在同一位置无限重试、5 次后超时；`llama_kv_cache::seq_rm` 与 `llama_memory_recurrent::seq_rm` 在 `pos_in`
不匹配时跳过并**无条件 `return true`**（`QLH_FORCE_SEQ_RM_FAIL` 已改成只在 `p0 <= pos_max` 时拒绝）。

**已知未覆盖**：`p0` 退化分支是 code-review-verified, not run；shared-prefix 路径不调用 partial `seq_rm`，`verify/verify_seq_rm_shared.py` 的 `[INFO] injection never fired`
表示这次跑没测到东西。验收见 [`PROTOCOL-V9.md`](../design/PROTOCOL-V9.md) §5。

### 2.10 同机构造测不出 head offload 的收益

- **现象**：head + tail 同机时给 head 加 `-ngl 8`，**head 自己的 plain 图帧反而更慢**：`QLH_SPLIT_TRACE` 逐帧分解 80.30 ms vs 全 CPU 76.01 ms（慢 5.6%），抖动 52~160 vs
  63~100 ms，端到端 tg 8.44 vs 9.34；同一配置在段级 probe 上是 47.6 vs 63.7 ms，快 25%。
- **根因**是流水线形态：probe 在单个 range 上连续 decode，GPU kernel launch / sync 能与 CPU 线程池跨步流水；server 每帧一次 decode 再等 TCP 往返，两端争同一批 CPU 线程。
- **判据**：head offload 的收益只能用部署形态测（head 独占本机、tail 在 LAN），那一档是 `-ngl 8` **+19.6%**（另一次 +34%）、`--ncmoe` + `--moe-cache-mib 1024` **+40.9%**；
  同机构造的数字只能当压力 / 回归数据，方向甚至相反。

## 3. 浮点与测量的边界

### 3.1 跨 ubatch 的「逐位相同」不成立

- **现象**：同一 prompt 装不进 `n_ubatch` 时切成两个 ubatch，GEMM 累加顺序随之改变：head 自己的 tap 与整模的差可以有 4e-4（29 token / `n_ubatch=12`，`tests/dflash-taps.cpp`
  头注释），同批测量里最大到 1e-1 量级（**未落盘**）。
- **根因**是浮点加法不满足结合律，切 ubatch 就换归约树，是性质不是缺陷。
- **影响**：所有逐位验收必须在单 ubatch 内做；长 prompt 的逐位对比不控制这一点会稳定给出假阳性。
- **代价**：不修。容差比较更糟——分段链的正确性判据只能靠逐位；折中是把每次对比的 ubatch 边界写进结论。

### 3.2 投机加速比随负载浮动，k 的拐点随模型

**现象**（2026-11-06 实测，贪心，ms/token）：

- MTP k=2 在纯代码 prompt 上 1.33×、中英混合 1.06×、纯中文 1.00×（**未能复现**：真实代码 / 混合 / 中文 prompt 上 k=2 是 0.87× / 0.69× / 0.63×，接受率 47% / 27% / 19%，1.32×
  只出现在模型能复读的退化 prompt 上——**不要再用这三个数当证据**）。
- DFlash k=3 在 CPU 上 +23.7%（不能外推到 GPU）。
- 2B 上 k 的拐点复现不出来（k=3 优于 k=2：25.09 vs 28.32 ms/token，早先记的 25.87 **未复现**，接受率只从 97.7% 掉到 96.6%）。
- `-np > 1` + MTP 明显变慢（整模加 MTP 22.56 → 28.37 ms/token）。

- **根因**：节省来自「一轮前进多少 token」与「每行边际成本」两笔账的比值，两者都随负载与模型变。
- **影响**：链上默认不开投机（`--spec-type` 默认空），这是启动参数默认值，不是「不划算」的结论。35B 官方版重扫 k=1..8 甜点仍是 2（0.910 / 2.81），可跨两份权重复用、不能搬到 2B。k 还随草稿源结构变：markov 头在 k=8、
  `n=256` 上 mid/long 档比 DFlash 高 +15.3 / +14.5 点，中文改写只剩 +4.1 点，k=2 上 −0.5 ~ −0.9 点（见 [`SPECULATION-DATA.md`](../measure/SPECULATION-DATA.md) §3）。
- **本机双进程测不出投机收益**：没有 25–30 ms 跨机往返可省，同时抢同一套 CPU 与内存带宽；本机负数不能外推到多机。
- **代价**：不修；要做的是把「按负载选」做成配置项，需要负载分类与实测表，比投机本身工作量大。

## 4. 一类共同的失败形状

这些边界多数不报错，只算错或悄悄退化。能拒的只有架构白名单、角色 / 层范围、远程单序列；其余都是「看起来在正常工作」，错误只体现在数值或长期行为上。所以验证必须比 logits 而不是比生成的文字（argmax 一致经常是巧合），长会话要盯 KV 占用，跨 ubatch
的对比要先确认 ubatch 边界。原型里用 `tests/split-smoke.cpp` 与 `tests/dflash-taps.cpp` 做逐位对照，这个习惯做产品时要保留。

四个具体形状：

1. **新加的 per-frame 状态没跟上 reset 约定**——`split_remote_reset()` 会清掉新加的 h_nextn 行，原来「先存行、后存 logits」让行被 `store_logits()` 的 reset 抹掉，stage 回 2048
floats、head 收到 0 行，draft 静默消失；凡是新增 per-frame 字段，先问它会不会被哪个 reset 清掉。
2. **更严谨的失败处理堵死自愈入口**（§2.9 边界一）：新增失败分支时，除了问「失败后状态一致吗」，还要问「这个分支会不会跳过唯一能恢复的那条正常路径」。
3. **一帧只能带一种 payload，两个消费者争同一通道**——decode 回复只能声明一个 `n_embd` 宽度，nextn 行（8 KB）与 tap 段（prefill 时 8 MB）不能同发；verify / candidate 是 tap 优先、plain
分支曾是 nextn 优先，DFlash 链的 prefill 帧因此把 tap 吞掉，接受率掉到 0.111（能跑、只是低 13%，不崩）。通道复用要先写死优先级并让三处分支共用同一判定。
4. **谓词回答的问题不对**（§0.4）：`supports_embd_row()` 检查「buffer 在不在」，使用者以为它回答「内容对不对」；修复要把假设写进谓词并让所有入口问同一句，且**每次使用时现算**（构造期答案会因 buffer 重分配而陈旧）。
