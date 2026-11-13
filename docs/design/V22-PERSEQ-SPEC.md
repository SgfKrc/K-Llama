# 远程链多序列：四个投机通道的 per-seq 形态

> 状态：**已实现**（v2.2）
> 更新日期：2026-11-13
>
> 适用范围：v2.2 远程链在多序列帧上启用候选 / 验证 / h_nextn / tap 四个投机通道；读者为要落地这块改动的开发者。
> 相关：多序列主干 [`V22-MULTISEQ.md`](V22-MULTISEQ.md)（协议 v10 的 `rows` 段、`stage_id` 四个身份字段、`n_verify_rows` 参数、判据 1–9）；
> 协议现状 [`PROTOCOL-V9.md`](PROTOCOL-V9.md)。
>
> 待定项：§8.2 的 4 条。

---

**证据分级**：**实测**（本机跑出，有脚本或日志）／**代码**（读自 `H:\llamacpp-c` 的 `slimarc/v2.2` 树，带行号）／**引用**（转述他文档口径）／
**推断**（估算与预判，不是证据；「约 N 行」一律属此类）。行号以 2026-11-13 的树为准。

## 0. 一句话

**把「一帧一份通道参数」换成「一帧每行一份」；四个通道的代价差三倍——验证与 h_nextn 只加 flag 位，候选要加段，tap 的协议一行不改。**

## 1. 范围与分工

主干定协议 v10 的四组事：帧尾 `rows` 段的逐行 `seq_id`、回复的 `n_logits` 行、`stage_id` 四个身份字段（`n_seq_max` /
`n_ctx_seq` / `n_rs_seq` / `n_verify_rows`）、`--split-verify-rows` 参数。本文定同一份 v10 里四个投机通道的逐行形态。

**两者必须一次定型**：`PROTO_VER` 无协商（`src/llama-split-proto.h:25-27`，**代码**），v10 之后再补字段就是 v11、全链重编一次。
不改：本地链的四个通道（`-np 2/4` 已通过组 5）、`OP_MTP_DRAFT` 的帧格式（§3.3 论证不必改）、tap 的「问哪些层」（§3.4）。

## 2. 依据：现在是帧级单份，而且是四层单份

### 2.1 协议层：三个通道的「要什么」没有逐行表达（代码）

| 通道 | 帧上带什么 | 现状 |
| --- | --- | --- |
| 候选 | `cand_req` 40 B + `n_bias` 对 + `n_hist` 个 int32，追加在 hidden 之后（`:497-500`） | **一份**，无行号字段（`:450-464`） |
| 验证 | 什么都不加（`:502-504`） | 帧级 bool；回复是 `n_tokens` 个 int32，隐含「每行都要」 |
| h_nextn | 什么都不加（`:506-508`） | 帧级 bool；行数由「是不是 verify 帧」推出 |
| tap | `layer_inp_req` 16 B + `n_taps` 个源层 id（`:510-513`、`:413-419`） | **一份**，且与帧无关（链级 sticky） |

### 2.2 驱动层：arm 状态是 context 级单份（代码）

`split_frame_answer::armed` 是 `uint32_t` 位域，三位 `ARMED_CANDS` / `ARMED_VERIFY` / `ARMED_NEXTN`（`llama-context.h:477-481`）；
`split_cand_req` / `_bias` / `_hist` 也是 context 成员各一份（`:590-592`）。四个 arm API **都没有 seq 参数**：
`split_request_candidates(...)`（`llama-context.cpp:6505`）、`split_request_verify_rows(bool)`（`:6590`）、`split_request_nextn_rows(bool)`
（`:6640`）、`split_set_layer_inp(ids, n)`（`:6690` 附近）。一个 llama-server 进程是一个 context、N 个 slot，N 个请求共用这一份。

### 2.3 stage 层：整帧一个 seq、一行答案（代码）

`batch.seq_id[i][0] = req.seq_id`（`split-stage/main.cpp:2242`）、`batch.logits[i] = want_verify || i+1 == n_tok`（`:2255`）——
两个循环变量都从帧头的一个 `seq_id` 与一个 bool 来。回复侧 `answer_plain` 只发最后一行 logits（`:2520`），`answer_relay` 只发 hidden（`:2329`）。

### 2.4 服务层：三个 arm 点各自假设「帧属于一个 slot」

- 候选：`request_remote_candidates(off)` 每帧只挑**一个** slot（`server-context.cpp:5482-5530`），条件是 `slot.i_batch == last_row`。
- 验证：要求 `slot.spec_i_batch.front() == off && off == 0`，`off != 0` 直接 `continue`（`:5572-5580`）。**多序列下 `off != 0` 是常态，
  这一条会让验证通道静默不 arm**——不是报错，是永远走整行路径。
- h_nextn：任一 slot 需要就 arm 全帧（`:5590-5622`）；行数规则在 stage 侧：「verify 帧全部行，否则最后一行」
  （`split-stage/main.cpp:2346-2366`、`llama-context.cpp:3292-3294`）。

### 2.5 输出行上限是帧级总量（代码）

`n_verify_rows`（运行时值，默认 17，见 [`V22-MULTISEQ.md`](V22-MULTISEQ.md) §4.5）判在 `llama-context.cpp:3220`——现在是 `n_tok <= MAX_VERIFY_ROWS`，改成参数后是
`n_tok <= n_verify_rows`。`n_tok` 是**整帧**行数，不是每序列。stage 侧按同一个数分配 compute buffer（`n_outputs_max`，`split-stage/main.cpp:1040`）。
这条决定 §3.5 的 k 上限。

## 3. 四个通道的 per-seq 形态

### 3.1 候选：每序列一份，段变多份

**不能共用一份**：`cand_req` 带采样参数（`top_k`、`top_n_sigma`、`penalties` 三个系数、`n_steps`/`steps`）与惩罚历史窗口
（`:450-464`），全是 per-request 的——两个 slot 的 `top_k` 与惩罚历史不同，一份装不下。`llama-ext.h:206-210` 还写明「request
是每帧的、不是模式」，因为一个回退的请求不能污染下一个；多序列下这条理由变成「不能污染**另一个序列**」。

**协议**：`rows` 段 flags 加 `ROW_FLAG_CANDS`（这一行要候选）。带候选的行在 `rows` 段之后追加候选段：`uint32_t n_cand_rows`，
然后 `n_cand_rows` 个 `{int32_t row, uint32_t n_bytes}` 索引，再是连续排列的 `cand_req` / bias 对 / hist——保持「定宽大块在前、
自描述小段在后」的惯例。**互斥要下沉到行**：`DECODE_FLAG_CANDIDATES` 与 `DECODE_FLAG_VERIFY_ROWS` 现在帧级互斥（`:270-271`），
per-seq 后改成逐行互斥（同一行不能既要候选又要 token id，不同行可以），两端断言：一行两个 flag → `ST_BAD_REQ`。

**驱动侧**：`split_request_candidates` 加 `llama_seq_id`；三个 `split_cand_*` 成员从单份变按 seq 索引的 map；`armed` 位域变
`std::map<llama_seq_id, uint32_t>`；回复侧 `split_answer.cands` 从单份变 `map<row, cands>`，`split_take_candidates()` 的 `row` 出参查表。

**stage 侧**：最大的一块。现在整帧一个 bool、只对最后一行跑候选（`:2335`、`:2382` 起）；per-seq 后按候选行循环，每个候选行用
**它自己的** `cand_req` 重建惩罚状态。注释已写明「每帧从历史重建、不做增量」（`:443-449`），所以「每行重建」是同一套代码的循环。

### 3.2 验证：协议几乎不用改，驱动改得最多

**最省的原因**：答案本来就逐行——`RSP_FLAG_VERIFY_ROWS` 的 body 是 `n_tokens` 个 int32、一行一个 argmax（`:331-339`），head 侧
`split_verified_token(idx)` 按行号取（`llama-context.cpp:6599-6607`）。所以「每行要不要 token id」只需 `rows` 段一个 `ROW_FLAG_VERIFY`
位，回复段从「全帧 n_tokens 个」变「按行号表排列的 m 个」。

**真正的工作在驱动侧**：`split_verified_token(idx)` 的索引语义要从「帧内行号」改成「batch 行号」，或由调用方把 `off + 局部行号`
算好再查。`server-context.cpp:5572-5580` 现在用 `off != 0 → continue` 绕开这个问题，per-seq 化就是把这处绕开换成真的行映射：
`slot.spec_i_batch` 的每个元素是 batch 行号，验证答案按行号表回填。**这是本块唯一有「读错行 → 输出错 token」风险的地方**（判据 10）。
**stage 侧**：`batch.logits[i] = want_verify || ...`（`:2255`）改成读 `flags[i] & ROW_FLAG_VERIFY`，逐行 argmax 的循环不动（`:2382` 起）。

### 3.3 h_nextn：只加 flag 位，但要改行数规则

**现状的行数规则是帧级的**：`row0 = want_verify ? 0 : last`（`split-stage/main.cpp:2346`），head 侧反推 `nextn_row0 = n_tok - n_rows`
（`llama-context.cpp:3292-3294`）。多序列下这条规则直接失效：一个序列在验证（要它那 k+1 行）、另一个在普通 decode（只要最后一行），
一个 bool 表达不了。**协议**：`rows` 段的 `ROW_FLAG_NEXTN`。不加段、不加参数——h 行宽度就是帧头的 `n_embd`，行集由 flag 表达；
回复侧 `RSP_FLAG_NEXTN_ROWS` 从「n_rows 行」变「按行号表排列的 m 行」。

**`OP_MTP_DRAFT` 不用改**：它的帧本来就带 `seq_id`，且注释写明 stage 内 draft context 是 per-sequence（`llama-split-proto.h:515-519`，
「a frame that named a different sequence would draft over another request's KV」），所以 h 行按 seq 归属后 draft 帧天然对得上。
要改的是 `split_mtp_ready()` 那句断言——「One seq per batch on a remote chain, so the draft always belongs to the last stage」
（`llama-context.cpp:6638`）——多序列下它不再成立，变成「draft 属于帧里最后一行的 seq」，而一帧可能两个序列都要 draft。

**驱动侧**：`split_request_nextn_rows` 加 seq 参数；`split_take_nextn` 的 `row0` 出参改成行号表；`request_remote_nextn_rows()`
（`server-context.cpp:5590`）从「任一 slot 需要就 arm 全帧」改成逐 slot 置位。

### 3.4 tap：协议零改动，只有回传行的归属要按 seq 解释

**「哪些层」是链级属性，与 seq 无关**——有代码直接支持：`split_tap_ids` 放在 `split_answer` **之外**，注释写明「a property of the
chain, not of the frame, so it lives outside split_answer and survives begin_frame()」（`llama-context.h:597-601`）；
`split_set_layer_inp` 的注释同样写明「*not* armed per frame」（`:362-370`），设置点只有一处：draft 模型自己的 `target_layers`
（`common/speculative.cpp:1252`）。所以「问哪些层」不需要 per-seq。

**要按 seq 解释的只有回传的 row 归属**：回传是 `[row][tap][n_embd]`（`split-stage/main.cpp:2329`），head 侧按 row 复制成每 tap 一块
（`llama-context.cpp:3257-3268`）、`tap_rows = n_tok`。多序列下 draft 的 KV 注入是 per-seq 的（每个序列各一份 draft 上下文），
head 必须知道第 r 行属于哪个序列才能注入对的那份。**这不需要协议改动**：v10 的 `rows` 段已逐行带 `seq_id`（[`V22-MULTISEQ.md`](V22-MULTISEQ.md) §3.1），
head 从行记录就能推出每行的 seq。所以 tap 的改动全在**消费方**：`common/speculative.cpp` 的注入循环按行 seq 分组。

### 3.5 k 的上限：帧级总行数约束（放宽不等于取消）

一个验证序列每帧要 `k+1` 行，而整帧的输出行上限是 `n_verify_rows`（运行时值，见 [`V22-MULTISEQ.md`](V22-MULTISEQ.md) §4.5）。所以：

```
Σ(验证中的序列) (k_i + 1) + Σ(普通 decode 的序列) 1  ≤  n_verify_rows
```

`n_verify_rows` 取默认 17 时，最坏情况（全部在验证）是 `N * (k+1) ≤ 17`：

| 序列数 N | 每序列最多行数 | k 上限 |
|---:|---:|---:|
| 1 | 17 | 16（= `MAX_MTP_DRAFT` 的默认值） |
| 2 | 8 | 7 |
| 3 | 5 | 4 |
| 4 | 4 | 3 |

普通 decode 的序列每帧也占 1 行，所以上表**保守**（混合帧还能多 1–3 行）；它假设每个序列都用满，比实际更严。

**`n_verify_rows` 是上限，不是精确值**（[`V22-MULTISEQ.md`](V22-MULTISEQ.md) §4.5）：条件是 `n_verify_rows ≥ n_seq_max × (draft.n_max + 1)`——
**用 `draft.n_max + 1` 而不是 `n_rs_seq`**：`--split-rs-seq` 会把 `n_rs_seq` 抬到高于帧行数（`common/common.cpp:1947`），帧本身还是 k+1 行，
按 `n_rs_seq` 算会白分配（N=4、k=3、`--split-rs-seq 8` → 32 行 31.7 MB，实际只需 16 行）。上表是「用常数 17」时的结果：`N ≤ 4`、`k ≤ 3`
已经够用（默认什么都不配就能跑），`k` 或 `N` 更大才要调大；多出的余量只是 buffer 浪费、不是错误。
**buffer 仍是一份共享的**：`n_outputs_max` 是 context 级单值（`split-stage/main.cpp:1040`），`output_reserve()` 按它一次分配
（`llama-context.cpp:3924`，且不能长大——`:1036` 注释）；per-seq 化**不引入每序列一份 buffer**，否则 N=4、k=3 时是 4×15.9 MB。

### 3.6 回复的行号表（候选与 h_nextn 共用）

`n_logits`（[`V22-MULTISEQ.md`](V22-MULTISEQ.md) §3.4）只说明「这一回复答了几行」，没说「哪几行」。多序列下三个通道各答不同行集，所以 v10 的回复需要一张**通用行号表**：
紧跟 `rsp_header` 的 `n_rows` 个 int32、按升序；`n_logits` 的语义扩成「logits 段答了表里前 m 行」。候选段复用同一张表（§3.1 的
`{row, n_bytes}` 索引可以并进去），验证段与 h_nextn 段按表顺序排。这样 `layout_of()` 只有一个「行集」概念，四个通道各自声明「我答哪几行」，
与 stage 侧 `reply_parts()` 的「builder 说*哪些段*、*在哪*，不说加多少字节」（`split-stage/main.cpp:1997-2009`）同一套思路。

## 4. `n_rs_seq` 的 rollback 语义

### 4.1 `n_rs_seq` 是每序列的量（代码）

三处注释与三处断言互相印证：字段定义「number of recurrent-state snapshots **per seq** for rollback; tensors are widened to
(1 + n_rs_seq) groups」（`llama-memory-recurrent.h:74-75`）；图侧 `K = n_rs_seq + 1` 是每个 conv 层、每个序列的 slot 数
（`llama-graph.cpp:4190-4193`）；索引侧 `rs_idx.assign(n_seq_max, 0)`、`assert(n_seq_max == rs_idx.size())`、
`GGML_ASSERT(idx <= n_rs_seq)`（`llama-memory-recurrent.cpp:37`、`:426`、`:429`）——每序列一个索引，各自受同一个 `n_rs_seq` 约束；
cell 侧 `cell.rs_max = n_seq_tokens <= n_rs_seq + 1 ? n_seq_tokens - 1 : 0`（`:676`），cell 按 seq 一个。

取值来自投机类型：`need_n_rs_seq()` 返回 `draft.n_max + 1`（`common/common.h:459-482`，**代码**），注释写明「one extra snapshot row
**per layer per sequence**」（`:477`），理由是一个 grammar 拒绝的验证帧要**整帧**放弃，即 `n_max + 1` 个位置一次回退。`--split-rs-seq` 可以把它抬得更高（`cparams.n_rs_seq = max(need_n_rs_seq(), split_rs_seq)`，`common/common.cpp:1947`）——那是回滚余量，不是帧行数（§3.5）。

### 4.2 多序列 + 投机：`need_n_rs_seq()` 不用改

它是每序列的量，与帧里有几个序列无关。多序列下的变化是**总量**：stage 的 recurrent 池按「每序列 `(1 + n_rs_seq)` 个 slot ×
`n_seq_max` 个序列」分配——现有布局本来就按 seq 一份（`llama-kv-cache-dsv4.cpp:1863` 的 `rs_max[seq_id] = min(n_seq_tokens[seq_id],
n_rs_seq)`、`:1109` 的 `rs_idx[seq] > n_rs_seq` 检查），不需要新机制。回退的合法性也是 per-seq 的：`rollback <= cell.rs_max`
（`llama-memory-recurrent.cpp:212-213`），cell 属于一个序列，所以「A 回退 3 个位置、B 不动」天然支持。

**一处要盯**：`split_equal(n_ubatch, true, n_rs_seq + 1)`（`:455`）要求每个序列尾部 `n_rs_seq + 1` 个 token 落在**同一个 ubatch**；
多序列时按序列各算一次，某序列在 ubatch 里的 token 数超过 `n_rs_seq + 1` 时 `rs_max` 变 0（`:676`），该序列不能 rollback——
不是全局失败，是该序列降级。投机帧每序列正好 `k+1 ≤ n_rs_seq` 行，满足；混合帧里普通 decode 的序列只占 1 行，更满足。

### 4.3 所以 `stage_id` 必须带它（与 [`V22-MULTISEQ.md`](V22-MULTISEQ.md) §4.3 同一处改动）

两端 `n_rs_seq` 不一致是**硬故障**：head 按自己的值回退 k 个位置，stage 的 `rollback <= cell.rs_max` 用它的值——偏小时拒绝
（请求失败），偏大时 `rs_idx` 越界（`:429` 的 `GGML_ASSERT`，进程死）。两者都是「attach 时就该拒」的形态，不是「首个请求才发现」。
`n_rs_seq` 与 `n_seq_max` **正交**：前者是「每份状态能回退几个位置」，后者是「有几份状态」，取值来源不同（`-np` / `--split-rs-seq`
或投机类型），一起加不冲突。语义上它是「每序列每层」而非「每序列总数」，这个精确化不影响正交性，但 `stage_id` 的注释要写准。
它与 `n_verify_rows` 的关系是 **`n_verify_rows ≥ n_seq_max × (draft.n_max + 1)`**，且**不能用 `n_rs_seq` 代换**——`--split-rs-seq` 会把
`n_rs_seq` 抬到高于帧行数（§3.5）。两者一起进 `stage_id` 各有用途：一个查每序列回滚深度、一个查帧总行数。

## 5. 改动清单

| 文件 | 位置 | 动作 | 推断行数 |
| --- | --- | --- | --- |
| `src/llama-split-proto.h` | `:259`、`:271`、`:295`、`:320` | 三个 `DECODE_FLAG_*` 从帧级变逐行：`rows` 段加 `ROW_FLAG_CANDS` / `ROW_FLAG_VERIFY` / `ROW_FLAG_NEXTN`；互斥下沉到行 | 30–45 |
| 同上 | `:413-464`、`:497-513` | 候选段从一份变「索引 + 多份」；回复行号表 | 20–30 |
| `src/llama-context.h` | `:450-497`、`:585-601` | `armed` / `cands` / `nextn_row0` 从单值变按 seq / 按行表 | 25–40 |
| `src/llama-context.cpp` | `:3208-3220`、`:3257-3294`、`:3324-3330` | 四个 `want_*` 变逐行；tap / nextn 按行表存；候选按行存 | 40–60 |
| 同上 | `:6505`、`:6590`、`:6640`、`:6638` | 三个 arm API 加 `seq_id`；`split_mtp_ready()` 的「one seq per batch」断言改写 | 20–30 |
| `src/llama-split-remote.h/.cpp` | `decode()` | 发逐行 flags 与多份候选段；收行号表 | 30–50 |
| `tools/split-stage/main.cpp` | `:2203`、`:2242`、`:2255` | `batch_guard(..., n_seq_max)`；逐行 `seq_id` 与 flags | 20–35 |
| 同上 | `:2329`、`:2346-2366`、`:2382`、`:2520` | 候选按行循环；h_nextn 按行集；verify 按 flags；回复带行号表 | 45–70 |
| `tools/server/server-context.cpp` | `:5482-5530`、`:5572-5580`、`:5590-5622` | 三个 arm 点从「挑一个 slot」变「逐 slot 置位」；去掉 `off != 0 → continue` | 35–55 |
| 同上 | `:2365` | `spec_draft_cap = min(draft.n_max, n_verify_rows / n_seq_max - 1)`（每序列 k，且受帧总行数约束；**不用** `n_rs_seq - 1`，后者会被 `--split-rs-seq` 抬高） | 10–15 |
| `common/speculative.cpp` | `:1252` 附近 | tap 行按 seq 分组注入；候选 / 验证的按 seq 取用 | 20–35 |

合计**推断 295–465 行**（含注释）。加上主干文档的 290–495 行，v2.2 多序列 + per-seq 投机合计**推断 585–960 行**。两者都不含测试脚本（判据落地另计）。

## 6. 判据

[`V22-MULTISEQ.md`](V22-MULTISEQ.md) §7.1 的 7–9 是「多序列 + 投机」的粗判据，本文档补两条通道专项，不重复那三条。

| # | 判据 | 怎么判 |
| --- | --- | --- |
| 10 | 验证帧的行映射不串行 | A 投机（`spec_i_batch` 不从行 0 开始）、B 普通 decode；A 的每个接受 token 与串行基线逐个相同。**这条专打 §3.2 的「读错行」**：行映射错了会出一个错 token，而不是变慢 |
| 11 | `k` 超过上限时是干净降级 | 按 §3.5 的上限（默认 `n_verify_rows 17`、`-np 2` 时是 7）配一个超限的 `n_max 12`：启动日志有 per-slot 上限的 warn，请求仍成功；输出与「`n_max 7` 串行」比**只比 token 不比接受率**（降级改块长、不改贪心结果）；**不得**出现 `n_tokens > n_verify_rows` 的 `ST_BAD_REQ` |

**每序列容量约束（对全部判据生效）**：多序列下每序列容量是 `n_ctx / n_seq_max`（`llama-context.cpp:414-431`，**代码**），判据里每条
prompt + `max_tokens` 必须小于它，否则撞上下文墙而不是判语义。本表与 [`V22-MULTISEQ.md`](V22-MULTISEQ.md) §7.1 默认按
`-c 4096` / `-np 2`（每序列 2048）写，prompt ≤ 1024、`max_tokens` ≤ 256。

## 7. 风险与边界

| 风险 | 后果 | 处置 |
| --- | --- | --- |
| 验证答案的行映射错位 | 读到**另一行**的 argmax → 输出一个错 token，不报错 | §3.2 的行号表；判据 10 专打；两端断言「行号表升序且 < n_tokens」 |
| `k` 按 `n_seq_max` 定死，某些序列不用满 | 混合帧浪费行额度（§3.5 表是保守值） | 可接受：只影响块长，不影响正确性；不为此做逐帧动态 |
| 退回整行路径不是免费降级 | 那一轮 k+1 行全要 logits，带宽回到 v9 前 | §3.5 的兜底判断；WARN 已有（`:3224-3227`） |
| 候选段变长 | 每候选行 40 B + bias + 64-token 窗口（`:443-449`），N=4 时最多 4 份 | 相对 hidden（8 MiB / 512 行）可忽略；不做压缩 |
| `n_rs_seq` 两端不一致 | 偏小：回退被拒（请求失败）；偏大：`rs_idx` 越界断言（进程死） | §4.3：`stage_id` 带它，attach 时拒 |
| 某序列 `rs_max` 变 0 | 该序列不能 rollback，投机降级为「不投机」 | `llama-memory-recurrent.cpp:676` 的既有行为，不是新故障 |
| 每序列一份 compute buffer | N=4、k=3 时 4×15.9 MB；k=16 时 4×67.6 MB | §3.5：共享一份 `n_outputs_max`，不按 seq 分 |

## 8. 未做与待定

### 8.1 未做

- **每帧按实际行数动态定 k**：需要在 propose 前知道本帧有几个序列在验证，而那由调度决定。
- **跨序列共享候选**：两个序列参数相同也不合并——合并要求两端对「参数相同」有同一判据，比省下的 40 B 贵。
- **tap 的逐序列层集**：同一链上两个序列读不同的 draft 层，属多 draft 模型拓扑，与 v3 的链式拓扑一起考虑。

### 8.2 待定

1. `n_verify_rows` 的余量是否值得 warn（例如 N=4、k=3 只用 16/17 行）——不影响正确性，只是让运维看见 buffer 浪费。
2. 回复行号表的形状：紧跟 `rsp_header` 的独立 int32 数组（推荐），还是并进各段的索引里。
3. 候选段的排列：索引 + 连续载荷（推荐），还是每候选行自带 `cand_req`（解析更简单，但每份都要对齐）。
4. 判据 10–11 是否与 [`V22-MULTISEQ.md`](V22-MULTISEQ.md) §7.1 的 1–9 一起进 v2.2 结题口径（那份文档 §10 第 4 条是同一问题）。
