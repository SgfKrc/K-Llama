# 远程链支持多序列：一条链承载 N 个序列的 prefill 与 decode

> 状态：**方案**（代码未实现）
> 更新日期：2026-11-13
>
> 适用范围：v2.2 远程链的多序列与 prefill/decode 同批；读者为要落地这块改动的开发者。
> 相关：协议现状 [`PROTOCOL-V9.md`](PROTOCOL-V9.md)；单序列边界 [`SEGMENTED-LIMITS.md`](../limits/SEGMENTED-LIMITS.md) §2.4；旧口径与实测
> [`ARCHITECTURE.md`](ARCHITECTURE.md) §11；接线纪律 [`THREE-SEGMENT.md`](../deploy/THREE-SEGMENT.md) §3；`stage_id` 加字段的建议 [`OPEN-ISSUES.md`](../OPEN-ISSUES.md) B5；**四个投机通道的 per-seq 形态 [`V22-PERSEQ-SPEC.md`](V22-PERSEQ-SPEC.md)**。
>
> 待定项：§10 的 3 条。

---

**证据分级**：**实测**（本机跑出，有脚本或日志）／**代码**（读自 `H:\llamacpp-c` 的 `slimarc/v2.2` 树，带行号）／**引用**（转述他文档口径）／**推断**（估算与预判，不是证据；「约 N 行」一律属此类）。行号以 2026-11-13 的树为准。

## 0. 一句话

**把「一帧一个 `seq_id`」换成「一帧每行一个 `seq_id`」，删掉驱动侧两道拒绝；调度层一行不动——它本来就是多序列混合批。**

## 1. 范围

### 1.1 本版的起点

v2.2 已 rebase 到新基底 `5676dfbe0` 并通过本机验收（**实测**：53 PASS / 0 FAIL / 1 SKIP，SKIP 是尚未 rebase 的 KVMem 组）。上游把 18 个
NextN / layer-input / embd-row / state-row API 从 `src/llama-ext.h` 转正进 `include/llama.h`，并把 `llama_set_embd_state_src` 改名
`llama_set_ctx_state_src`；这些不触碰本文档要动的三层。

### 1.2 三层的改动边界

| 层 | 现状（代码） | v2.2 做什么 |
| --- | --- | --- |
| 协议 `src/llama-split-proto.h` | `PROTO_VER = 9`（`:79`）；`req_header::seq_id` 是单数，一帧描述一个序列（`:545`）；尾段回复只有**一行** logits（`layout_of()` 里 `logits_bytes = n_vocab * 4`，与行数无关） | 升 v10：每行 `seq_id` + 每行「要不要 logits」+ 回复可带 N 行 logits |
| 驱动 `src/llama-context.cpp` | `split_attach_remote()` 拒 `n_seq_max > 1`（`:2379-2386`）；`split_admit_batch()` 拒混合序列的批（`:3022-3029`）；`split_drive_remote()` 取首行 seq 发给全帧（`:3166`） | 删前两道，改第三道；帧体带逐行 seq |
| 调度 `tools/server/server-context.cpp` | `server_batch::token` 自带 `id_slot`（= seq）、`is_prompt`、`output`；`pre_decode()`（`:4293` 起）把所有 slot 的 token 加进同一个 batch | **不动**（§2.1） |
| 启动 `tools/server/server.cpp` | 远程链 `-np` auto 取 1（`:129-141`）；显式 `-np > 1` 直接拒（`:146-158`） | auto 保持 1；显式 `-np N` 放行，并校验 `n_seq_max` 一致（§4） |

### 1.3 「prefill/decode 分离」在本版指什么

不是把链拆成 prefill 链与 decode 链（那是 v3 的拓扑问题）。指**同一条链、同一个 batch 里，一个序列在 prefill、另一个在 decode**。上游 `server_batch` 已经这样混装，`token::is_prompt` 只用于统计（`server-context.cpp:170`）；`use_mixed_batch()` / `is_mixed`（`:1705`）说的是「模型要求 prompt 落在一个 ubatch 内」，与这里无关。

### 1.4 不改

本地链（已支持 `-np 2/4`，组 5 验收通过）、N 个请求打包一帧与链式拓扑（§9）。四个投机通道从「每帧一份」变「每行一份」另见 §7.2 指向的文档。

## 2. 依据

### 2.1 调度层已经是多序列混合批（代码）

`server_batch::token` 带 `id_slot` 与 `output`（`server-context.cpp:165-175`），`pre_decode()` 用 `batch.add(slot.id, cur_tok, pos, output, is_prompt)` 往**同一个** `tokens` 里加（`:4293` 起）。所以「多序列 + prefill/decode 同批」在上游是既有形态：v2.2 不需要新批处理层，只需要让线上帧能表达同一件事。

### 2.2 本地链已经走过这条路径（代码 + 实测）

`split_admit_batch()` 的注释写明本地链「a batch may mix sequences」，并记下「两个 seq 各一行、per-seq logits 与单序列逐位相同」的验证
（`:3012-3019`，**代码**，注释口径，脚本本次未复核）；`split_add_hop_row()` 逐行搬运 `tok.seq_ids`（`:2818-2874`）；`split_logit_plan`
已从 `token::output` 算出「尾段欠哪几行 logits」（`:2770-2810`）。组 5（`-np 2` / `-np 4`，十个不同长度请求全 200、进程存活、prefix cache 生效）是**实测**通过项，不是性能项。

### 2.3 远程链缺的正好是三件（代码）

| 缺口 | 证据 |
| --- | --- |
| 逐行 `seq_id` | `req_header` 只有一个 `seq_id`；stage 侧整帧写同一个值：`batch.seq_id[i][0] = req.seq_id`（`split-stage/main.cpp:2242`） |
| 逐行「要不要 logits」 | 线上没有 output 位图；stage 靠「最后一行」约定：`batch.logits[i] = ... i + 1 == n_tok`（`:2255`），回复也只发一行（`:2520`） |
| 回复多行 logits | `logits_bytes = n_vocab * 4`；head 侧只存一行（`split_frame_answer::logits_row`，`llama-context.h:454`），`get_logits_ith()` 对任何 `i != logits_row` 返回 null（`llama-context.cpp:1166`） |

第三件是代码注释里「Feasible, about 105 lines」（`:2367`）没算进去的：那句只说「one seq_id per OP_DECODE row」。**推断**：实际 290–495 行（§5 表逐项相加）。

## 3. 协议 v10

### 3.1 帧体

```
OP_DECODE = header | positions (n_tokens*n_pos*4) | hidden (n_tokens*n_embd*4) | rows (n_tokens*8)
```

`rows` 是 v10 追加的逐行记录，放在帧体**最后**：

```c
struct row_req { int32_t seq_id; uint8_t flags; uint8_t _pad[3]; };  // 8 B; flags bit0 = 要这一行的 logits
static_assert(sizeof(row_req) == 8);
```

一帧 512 行 = 4 KB，同帧 hidden 是 8 MiB（n_embd 4096，**推断**按该宽度算）。

### 3.2 为什么放帧尾，而不是 positions 之前

1. **拒绝必须发生在整帧消费之后**。stage 的既有规则是「先读完帧体，再校验 `seq_id`，拒绝时连接仍可解析」（`split-stage/main.cpp:1395-1410`）；放最前面只能省掉读，代价是要么破坏这条规则，要么还是得读完。
2. 与 v5 的 `cand_req`、v9 的 `layer_inp_req` 同一惯例：附加段追加在 hidden 之后，读者按「定宽大块在前、自描述小段在后」解析。
3. `positions` 的 section-major 布局（`pos[j*n_tokens+i]`）与 `n_pos` 的 stride 关系不动。

### 3.3 为什么总是带，而不是加一个 flag

v5/v7/v8 的 flag 都表示**能力或通道选择**，布局由 `PROTO_VER` 保证两端一致；逐行 seq 是**每帧都必需**的，没有它 stage 无法建 batch。省下的 4 KB 不值得换一条「两端对 flag 解读必须一致」的新路径；RLE 压缩（变长 → 需自描述）省得更少，还要多一套解析。

### 3.4 回复：logits 段从 1 行变 `n_logits` 行

- `rsp_header` 新增 `uint32_t n_logits`（32 → 36 字节；`static_assert`（`:572`）与 `verify/verify_seq_id.py` 的 `RSP` 结构一起改）。备选：借 `pos_max`（`OP_DECODE` 回复里恒 0）——不推荐，字段一名一义比省 4 字节重要。
- `layout_of()` 增加 `n_logits` 参数：`logits_bytes = n_logits * n_vocab * 4`，行按**帧内行号升序**排列。
- head 侧：`split_answer` 的 `logits` 从一行变 `n_logits` 行 + 一个升序行号表，`get_logits_ith(i)` 在表里查，不在表里仍返回 null。
- 四个投机通道的语义**不变**，但它们的「每帧一份」要变成「每行一份」——同一份 v10 的事，见 §7.2。

### 3.5 为什么必须是版本 bump

`PROTO_VER` 的规则写在文件里：「Bumped whenever a frame layout or an op number changes. There is no negotiation」（`:25-27`）。v9 的 stage
读完 positions 与 hidden 就会 `llama_decode` 并回复，多出来的 `n_tokens*8` 字节会留在 socket 里，被当成**下一帧的头**——这正是
v5/v7/v8/v9 各自 bump 的理由（`:27-78`）。v10 有两处布局变化：帧尾多了 `rows` 段；`rsp_header` 多 4 字节且 `n_logits > 1` 时 body 变长。
`9 → 10`，两端不一致在首帧干净报错。

### 3.6 被否决的替代：按 seq 拆帧

驱动按 seq 分组、每序列一帧，协议**零改动**（`header.seq_id` 就够）。否决理由：段内不再合批——两个序列的行分别进两次 `llama_decode`，正好丢掉 continuous batching 买到的全部东西；每段还多 N-1 次往返。它只胜在实现最省（**推断** 30–50 行）。

## 4. `n_seq_max` 怎么协商

### 4.1 不一致的后果（代码）

- 段的 KV 池按 `n_ctx_seq = n_ctx / n_seq_max` 切（`llama-context.cpp:414-431`；不整除时向下取整到 256 的倍数并 warn）。
- hybrid 模型的 recurrent 缓存按 **seq 数**分配：`recurrent_kv_size = max(1, n_seq_max)`（`llama-model.cpp:3505`、`:3525`），cell 按 `seq_id` 索引（`llama-memory-recurrent.cpp:177`）。即 `n_seq_max = 1` 的段**没有第二个 cell**：第二个序列只能靠下一条的边界检查拦住，漏掉就是错。
- 段侧已有边界检查：`check_seq_id()` 对越界 `seq_id` 答 `ST_BAD_REQ`，帧体已消费、连接继续可用（`split-stage/main.cpp:1401`）；多序列下逐行检查，同一条路径。

### 4.2 三条路

| 方案 | 何时发现 | 报错可读性 | 代价 |
| --- | --- | --- | --- |
| **A. `stage_id` 加四个身份字段，attach 时比对** | attach（模型加载后、首个请求前） | 高：数字 + 谁该改 | `stage_id` 28 → 40 字节（§4.3）；两个新参数 |
| B. 只在 stage 加参数，head 不查 | 首个越界行（请求中） | 低：`ST_BAD_REQ` → "stage rejected the request" | 0 |
| C. 每帧带 `n_seq_max` | 首个越界行 | 中 | 每帧 4 字节，无收益 |

**结论：A**。这四个数与 `n_ctx` 一样是**链的身份**（决定 KV 切分、recurrent cell 数、回退深度、输出行上限），不是可选能力；v4 的 HELLO 正是为「重连到错的段」设计的。B 作为兜底保留（判据 5）。这也与 [`OPEN-ISSUES.md`](../OPEN-ISSUES.md) B5 的既有建议一致。

### 4.3 加什么、比什么

- `stage_id` 加四个 `uint32_t`：`n_seq_max`、`n_ctx_seq`、`n_rs_seq`、`n_verify_rows`。`_pad[2]` 换成 `uint16 n_seq_max`（上限 256），三个 4 字节字段接在后面；`sizeof` 从 28 到 40（断言 `:214`）。
- head 侧 `head_expect` 带这四个值，在 `identity_mismatch()` 里比较，消息写明「两端 `-np` / `--n-seq-max`、`-c`、`--split-rs-seq`、`--split-verify-rows` 必须一致」。
- **不**比 `n_ctx` 原始值：两端 `n_ctx` 不同但每序列容量相同是合法的。
- 四个字段同一理由：不一致的后果都是**硬故障**，不是性能差——`n_rs_seq` 偏小让回退被拒、偏大让 `rs_idx` 越界断言（[`V22-PERSEQ-SPEC.md`](V22-PERSEQ-SPEC.md) §4.3）；`n_verify_rows` 偏小让 tail 的 `output_reserve` abort（`split-stage/main.cpp:1036` 注释写明「cannot grow the graph later」）。
- `n_rs_seq`（每序列回滚深度，可大于帧行数）与 `n_verify_rows`（帧总行数**上限**，§4.5）各司其职。比 `n_verify_rows` 比的是**上限**、不是实际行数——两端各自按自己的值分配 buffer，上限不同意味着一边可能装不下。

### 4.4 `n_ctx` 纪律与 `kv_unified`

v2.1 的已知坑：`n_ctx` 必须各段一致，否则位置边界静默错位（[`THREE-SEGMENT.md`](../deploy/THREE-SEGMENT.md) §3 第 1 条，**引用**，历史口径）。多序列下这条纪律变成两条：各段 `-c` 与 head 的 `-c` 相同 → 每序列容量相同；且 **head 必须 `--no-kv-unified`**——`kv_unified` 时 head 的 `n_ctx_seq = n_ctx`（整池给一个序列用，`llama-context.cpp:416`），段侧没有对应物，两边每序列容量差 N 倍，症状是请求中途 `ST_NO_SPACE`。由于 `-np` auto 在普通路径上会开 `kv_unified`（`server.cpp:139`），远程链的 auto 保持 1；**显式 `-np N` 时未加 `--no-kv-unified` 必须启动即拒**（不静默替用户改写配置——静默改写比报错更难查）。

### 4.5 `n_verify_rows` 从常量变启动参数

`MAX_VERIFY_ROWS = 17`（`:101`）与 `MAX_MTP_DRAFT = 16`（`:109`）改成 `--split-verify-rows N`（stage 侧参数，默认 17），后者变运行时的
`draft.n_max`（每序列的 k，**不是**帧总量 `n_verify_rows - 1`，也不是 `n_rs_seq - 1`——`--split-rs-seq` 会把 `n_rs_seq` 抬高而帧行数不变；
`OP_MTP_DRAFT` 的 `k` 校验跟着改，`:395`）。生效路径不新造：`n_outputs_max` 本来就是 context 参数（`llama-context.cpp:358`，`params.n_outputs_max == 0` 时取 `n_batch`），`split-stage/main.cpp:1040` 改成 `max(n_verify_rows, n_seq_max)`，`output_reserve()` 按它一次分配（`:3924`）。使用点逐个换：`llama-context.cpp:3220`、`server-context.cpp:2365`/`:5578`、`split-stage/main.cpp:1040`/`:2215`/`:2522`。

**`n_verify_rows` 是上限，不是精确值**——它决定 buffer 大小，够用即可：

    n_verify_rows ≥ n_seq_max × (draft.n_max + 1)

两个量分工：`n_verify_rows` ← 每序列**实际帧行数**（`draft.n_max + 1`）× 序列数，是 buffer 大小的唯一依据，**多出的余量只是浪费、不是错误**；
`n_rs_seq` ← 每序列**能回滚多深**（`max(draft.n_max + 1, split_rs_seq)`，`common/common.cpp:1947`），是回滚校验的依据，可大于帧行数、**不参与 buffer 计算**。

| 配置 | 需要的行数 | 默认 17 够吗 |
|---|---:|---|
| N=1、k=3 / k=16 | 4 / 17 | 都够（分别余 13 行、刚好） |
| N=4、k=3 | 16 | **够**（余 1 行；15.9 MB < 单序列默认的 16.9 MB） |
| N=4、k=4 | 20 | 不够，要调大 |
| N=4、k=16 | 68 | 不够，要调到 68（67.6 MB，对比 ubatch 2048 的 3.3 GB 约 2%） |

所以 **tail 不需要知道 head 的 `draft.n_max`**：默认 17 在 `N ≤ 4`、`k ≤ 3` 时已经够用，`-np 4` 加默认投机可以什么都不配就跑。head 在 attach 时检查
`n_verify_rows ≥ n_seq_max × (draft.n_max + 1)`，不满足就拒、消息给三个数（现值、需要值、「调大 `--split-verify-rows` 或调小 `--spec-draft-n-max` / `-np`」）；
满足就放行，允许多余余量。buffer 仍是一份共享的（`n_outputs_max` 是 context 级单值）。

## 5. 驱动改动清单

| 文件 | 位置 | 动作 | 推断行数 |
| --- | --- | --- | --- |
| `src/llama-split-proto.h` | `:79`、`:101`、`:109`、`req_header`、`rsp_header`、`layout_of` | 9→10；加 `row_req`；`rsp_header` 加 `n_logits`；`layout_of` 加参数；两个常量改运行时；`stage_id` 加四个字段 | 70–105 |
| `src/llama-split-remote.h/.cpp` | `decode()` | `seq_id` → `const std::vector<qlh_split::row_req> &`；发帧尾段；按 `n_logits` 收多行 | 50–80 |
| `src/llama-context.cpp` | `:2356-2386` | 删注释块与 `n_seq_max > 1` 检查（按裁决只写「支持多序列」，不保留旧理由）；新增 `QLH_SPLIT_SKIP_IDENTITY` 开关（仅测试用） | −30 |
| 同上 | `:3011-3030` | 删 `split_has_remote() && split_batch_mixes_seqs()` 分支；`split_batch_mixes_seqs()`（`:2716`）随之可删 | −15 |
| 同上 | `:3161-3180`、`:1149-1172` | `seq` 改逐行记录；接 `n_logits` 行；`:3220` 的判据改用 `n_verify_rows`；`get_logits_ith()` 按行号表查 | 55–95 |
| `src/llama-context.h` | `split_frame_answer`（`:450`） | 一行 logits → `n_logits` 行 + 行号表；`head_expect`（`:408`）加四个值 | 20–30 |
| `tools/split-stage/main.cpp` | `:2203`、`:2242`、`:2255`、`:2520` | `batch_guard(..., n_seq_max)`；逐行 seq 与 output；回复发 `n_logits` 行 | 60–100 |
| 同上 | 参数表 | 加 `--n-seq-max N`（默认 1）与 `--split-verify-rows N`（默认 17） | 25–40 |
| `tools/server/server.cpp` | `:129-158` | auto 保持 1 但改文案；显式 `-np N` 放行 + 要求 `--no-kv-unified`；透传 `--split-verify-rows` | 25–40 |
| `verify/verify_seq_id.py` | `:37` | `PROTO_VER = 10`；若脚本发 `OP_DECODE` 则补帧尾段 | 5–10 |

合计**推断 290–495 行**（含注释），协议与 stage 占一半以上。「105 行」只覆盖请求侧：逐行 seq 本身。

删掉的两道拒绝留下三处文案（`llama-context.cpp:2380`、`server.cpp:152`、`split_admit_batch()` 的 `-3` 消息）。它们描述的失效模式不再存在，随检查一起删除，**不**改写成别的场景：新的「不一致」只剩四处配置对不上，由 §4.3 的 HELLO 消息负责。

## 6. 状态语义

### 6.1 `seq_rm` 与 checkpoint 都是 per-sequence（代码）

`OP_SEQ_RM` 带 `seq_id` + `pos0/pos1`，stage 侧 `llama_memory_seq_rm(mem, req.seq_id, ...)`（`split-stage/main.cpp:1470-1485`）。多序列下每次只影响一个序列，**不需要改**；head 侧本来就按 slot 发，组 7 的「远程段 KV 没被 server 裁」判据已在测这条路径。

`llama_state_seq_get_data_ext(ctx, seq_id, ...)` 按序列取；链的 state 由 `llama_memory_split::state_write(io, seq_id, flags)` 把 head 与各段的记录拼起来（`llama-memory-split.cpp:406-470`），段侧记录是 8 字节 handle（v6）。多序列下每个序列各有自己的 blob，互不引用。

### 6.2 为什么 restore 顺序不用重想

一个 checkpoint 只属于**一个** `seq_id`：blob 里没有跨序列的共享资源，恢复序列 B 不读写序列 A 的任何记录。序列内的顺序仍是 head 先、段按链序后（`state_read` 同一个循环），每段拿 `pos_expected` 与自己的 `seq_pos_max` 比（`split-stage/main.cpp:1612-1670`）——这条比较只看本序列的位置。

**残余一处**：`state_size()` 的记忆是单值的（`size_query_seq` / `size_query_flags` / `size_query_bytes`，`llama-memory-split.h:197-200`）。两个序列交错做「先量后写」时，第二个序列的量会覆盖第一个，`state_write()` 的布局自检因 `size_query_seq != seq_id` 而**跳过**（`llama-memory-split.cpp:456`）——不是误拒，是保护面变小；llama-server 的 checkpoint 在 slot 自己的处理里同步完成，**推断**不会交错。

## 7. 判据

### 7.1 新组 14：远程链多序列（现有 57 项零覆盖）

组 7 跑远程链但没给 `-np`（auto = 1），组 5 跑 `-np 2/4` 但走本地链；远程链 `-np 2` 因 `:2379` 从未跑过。

| # | 判据 | 怎么判 |
| --- | --- | --- |
| 1 | 远程链 `-np 2`，两个请求并发，输出与串行逐字节相同 | `temp 0`、固定 prompt 与 `max_tokens`；并发结果与两次单独请求拼起来逐字节比 |
| 2 | 混合批（一个在 prefill、一个在 decode）逐字节相同 | **确定性构造**：先发 A，等它产出第一个 token（`/v1/chat/completions` 的流式首 token 到达）再发 B——此时 A 必在 decode、B 刚开始 prefill。两个长度约束：A 的 prompt 短于 `n_ctx_seq` 的一半（给 B 留位置），B 的 prompt ≤ `n_ubatch`（在一帧内 prefill 完，否则 B 自己跨多帧、A 的 decode 插在它中间，那是另一种混合形态）。与串行基线比 |
| 3 | KV 隔离：对 A 做 `seq_rm` 不影响 B | 长度交替触发 slot 复用，B 的输出与不触发时逐字节相同；段日志里 `OP_SEQ_RM` 的 seq 是 A 的 |
| 4 | 多序列下的 checkpoint save/restore | 两序列各存一次、各恢复一次，恢复后继续生成的 token 与不中断时逐字节相同；handle 失效时答「无 checkpoint」而不是链死 |
| 5 | 身份字段不一致必须干净报错 | head `-np 2` + stage `--n-seq-max 1`：attach 阶段就拒（方案 A），消息带两个数字。新增 `QLH_SPLIT_SKIP_IDENTITY=1`（读在 head 的 attach 里，跳过 `identity_mismatch()` 比对，**只**为测这条降级路径存在）时退化为首帧 `ST_BAD_REQ`，**不得**出现「输出像正常但错」；`--split-verify-rows` 不一致同理 |
| 6 | `n_ctx` 不被 `n_seq_max` 整除时可见 | head / stage 启动日志都有「rounding down」warn，且两端取整后的 `n_ctx_seq` 相同 |
| 7 | 远程链 `-np 2` + 投机，两个请求的输出与**各自串行（带投机）**逐字节相同 | 并发一轮取输出；同一对 prompt 各跑一次 `-np 1` 带投机；两两逐字节比。基线也必须是带投机的串行 |
| 8 | 两个序列的投机参数不同时互不污染 | A 用 `top_k 5` / `n_max 3`，B 用 `top_k 40` / `n_max 7`；输出分别与各自串行基线逐字节相同。**k 不同是关键**：同时验行数分配与行映射 |
| 9 | 一个序列在投机、另一个在普通 decode 时两条都正确 | A 带投机、B 不带。B 的输出与「B 单独跑」逐字节相同；A 的接受 token 数与「A 单独带投机跑」相同 |

判据 1–4 的「逐字节相同」是本项目既有口径（组 1/3/6/7 同型），不是为这一版新发明的强判据。判据 7–9 的细化（行映射、k 降级）在 [`V22-PERSEQ-SPEC.md`](V22-PERSEQ-SPEC.md) §6。

**每序列容量约束（对全部判据生效）**：多序列下每序列容量是 `n_ctx / n_seq_max`（`llama-context.cpp:414-431`，**代码**），判据里每条 prompt + `max_tokens` 必须小于它，否则撞上下文墙而不是判语义。本表按 `-c 4096` / `-np 2`（每序列 2048）写，prompt ≤ 1024、`max_tokens` ≤ 256。

### 7.2 多序列帧上的投机通道

四个通道（候选 / 验证 / h_nextn / tap）的 per-seq 形态见 [`V22-PERSEQ-SPEC.md`](V22-PERSEQ-SPEC.md)。三条边界在此固定：**帧内序列数 > 1 不再是「关闭通道」的理由**；整帧输出行上限（`n_verify_rows`）仍在，按 `Σ(k_i + 1) + Σ(普通 decode) 1 ≤ n_verify_rows` 算；tap 的「哪些层」是链级 sticky，与序列数无关。通道级 bug 的形态是「静默走整行路径」或「读错行的 token」，不是报错——判据 10 专打后者。

## 8. 风险与边界

| 风险 | 后果 | 处置 |
| --- | --- | --- |
| 每段 `n_seq_max` 不一致 | 偏小：recurrent 没有第二个 cell，第二个序列只靠 `check_seq_id` 拦住（漏掉就是错）；偏大：每序列 KV 变小，中途 `ST_NO_SPACE` | HELLO 比对（§4）；stage 侧逐行 `check_seq_id` 兜底 |
| `n_ctx` 的切分与 `kv_unified` | 两端每序列容量不同 → 中途 `ST_NO_SPACE`；不整除时静默取整；head 开 `kv_unified` 时每序列拿整池而段侧拿 `n_ctx/N` | §4.4 两条纪律；HELLO 比 `n_ctx_seq`；warn 进判据 6；远程链要求 `--no-kv-unified` |
| hybrid recurrent 的 per-seq cell | 段的 recurrent 池只有 `max(1, n_seq_max)` 个 cell；`-rs-seq` 快照仍是一份、按 `rs_idx` 分序列 | 与 KV 同一条纪律；`-rs-seq` 在多序列下的语义见 per-seq 文档 §4 |
| 并发 prefill 的 0.70× 劣化 | 两个 prefill 同时到达时合批让每个 chunk 更慢（CPU 线程已满） | **既有现象，不是本文档引入**。v2.1 实测（2B 本地双段 `0:12+12:24`，944-token prompt ×2，`temp 0`）：端到端 0.99× / 0.81× / 0.81×，纯 decode 重叠 1.12×，并发 prefill 0.70×（[`ARCHITECTURE.md`](ARCHITECTURE.md) §11，**引用**）。本文档**不承诺**任何吞吐或延迟收益 |
| `n_verify_rows` 是上限、不是精确值 | 默认 17 覆盖 `N≤4`、`k≤3`（N=4、k=3 时 16 行 15.9 MB）；k 或 N 更大时要显式调大（N=4、k=16 → 68 行 67.6 MB） | §4.5：head 在 attach 时检查 `≥ n_seq_max × (draft.n_max + 1)`，不够就拒并给三个数；多出的余量只是浪费 |
| 重连 / 重建的起点判据 | `split_starts_at_zero()` 已是逐序列（`:2980`），但重建要求**所有**序列都在 0 | 行为不变，只是拒绝得更频繁；报错文案已说明要重跑 prompt |

## 9. 未做（v3）

- **N 个请求打包一帧**：本文档的一帧对应一次 `llama_decode` 的 batch，不是一个 HTTP 请求；跨请求的批由 server 的 slot 调度决定。**链式拓扑**（星形以外的串联）与 **`kv_unified` 下的远程链**（需段侧有等价统一池语义）同样未做。

## 10. 待定

1. `stage_id` 四个字段的宽度（`uint16 n_seq_max` 塞原 `_pad[2]` + 三个 `uint32` = 40 字节，推荐；四个都用 `uint32` = 44 字节），以及 `n_logits` 的落点（`rsp_header` 加宽 4 字节，推荐；或借 `pos_max`）。
2. `row_req` 是否需要第 2 个 flag 位（例如「这行是 prefill」）——stage 目前不需要，但调试与日志有用。
3. v2.2 是否把新组 14（判据 1–9）纳入结题口径。
