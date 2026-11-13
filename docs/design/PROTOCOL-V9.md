# 协议 v5 / v7 / v8 / v9：投机跨进程的通道设计与逐位验证

> 状态：**现行**（文中引用的 v2 工具链——`chain_client` / `stage_worker` / `shim/` / `koakuma`——已退场，归档在
> `slimarc/prototype`，命令与路径只作历史口径）
> 更新日期：2026-11-11
> 适用范围：远程链的候选通道（v5）、验证通道（v7）、远程 MTP draft（v8）、`layer_inp`
> tap（v9）的设计与验收，以及 `seq_rm` 精确性探针与 batch invariance 判据。收益数据归[`SPECULATION-DATA.md`](../measure/SPECULATION-DATA.md)
> 与 [`CROSS-MACHINE.md`](../deploy/CROSS-MACHINE.md)；实现位置归 [`PATCHES-DETAIL.md`](../deploy/PATCHES-DETAIL.md)。

口径：35B 为官方 `Qwen3.6-35B-A3B-UD-Q4_K_M`（`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`，21.11 GiB），2B 为
`qwen35-2b-Q4_K_M`，贪心（`temperature 0`、`top_k 1`），`-np 1`，纯 CPU `build-split`；逐位对比只在单 ubatch 内成立（
[`SEGMENTED-LIMITS.md`](../limits/SEGMENTED-LIMITS.md) §3.1）。

## 1. 候选通道（v5，v8 仍在用）

**为什么等价**：`common_sampler` 的链顺序是
`penalties → dry → top_n_sigma → top_k → typ_p → top_p →min_p → xtc → temp`，**`top_k` 之后的所有步骤只在 k 个候选内作用
**。所以 tail 本地跑完 `logit_bias → penalties → top_n_sigma → top_k`，回传 40 个 `(id, logit)`，server 从
`cand_first`（`top_k` 在链里的下一个节点）接着跑，与回传整行数学等价。回退是白名单预判（`common_sampler_candidates_check()`，每请求一次），不是运行时兜底
。

**惩罚历史按帧重放**：server 每帧带最近 `min(penalty_last_n, n_prev)` 个 token，tail 从零 `reset` + 重放；代价 256 B/帧，换来重连、
`seq_rm`、`state_restore`、slot 复用之后两端自动一致（增量式一旦对「哪些 token 被 accept」有分歧就会静默错）。

**帧大小**：`rsp_header` 32 B + 40 × `token_cand` 8 B = **352 B/token**（两个 `static_assert` 锁了 32 与 8），对整行
**993 KB** 是 2821 倍。

**等价性实测**：4 prompt × 64 token、penalties 三参、`top_n_sigma`、xtc、temp=0、`top_k=1`、`logit_bias`、ignore_eos、
post-sampling logprobs、cache 复用两轮、400 token 长生成、10 组随机参数 fuzz——全部逐 token 相同。

**带宽**（TCP 代理计数）：同一 64-token 请求整行 67,932,452 B vs 候选 4,383,012 B，每 token 省 992,960 B。**loopback 延迟**：
57.92 → 52.92 ms/token（+8.6%）；loopback 等效 ~200 MB/s，所以只值 5 ms；按跨机 LAN 的 24 ms/0.95 MiB 折算，跨机 decode 从
4.67 / 5.38 / 6.05 预计到 **6.6 / 8.1 / 9.7 tok/s**（整模的 66–87%，原 45–54%）——**折算，不是实测**。

**没做**：prefill 阶段的 tail 仍整行回传，prompt 处理的带宽未优化。这一节的数字只出现在 fork 的提交说明里（`c5bd787b` 的候选通道一节），不在
`build-split/` 日志里，按「未落盘到日志」对待：机制与数据可信到提交说明这一层，独立复算没做过。

## 2. v7 验证通道与 v8 远程 MTP draft

**v7**：tail 对验证帧的**每一行**回一个 argmax，回复体是 `n_tokens × int32`。贪心验证只需要 token id——`temp=0` 时
`common_sampler_sample_and_accept_n` 逐行等价于 argmax。单帧回复（4 行）**993,312 B → 48 B（20,694×）**。帧内行号与 batch 绝对下
标只在帧首对齐时相同，所以 `request_remote_verify_rows` 硬性要求 `off == 0`，不满足就不 arm（报错，不静默错位）。

**旧版误算过一次**：曾写「验证要 k+1 行完整 logits，k=3 一帧 4 MB，千兆上 3–4 s」——4 MB 在千兆是 32 ms、按本仓实测 16.06 MB/s 是 250 ms；而且贪心验
证不需要整行，argmax 在 tail 侧算，每行只回一个 token id。

**v8**：`OP_MTP_DRAFT`（op 8）带一行 hidden 与 `mtp_req{token,pos,k}`，回 k 个 int32，**k 步递归在 stage 内部完成**；decode 回
复可追加 `n_rows × n_embd` 个 float（`DECODE_FLAG_NEXTN_ROWS` / `RSP_FLAG_NEXTN_ROWS`）。`stage_id.has_nextn` 占原
`_pad` 字节，**故意不参与身份比较**——「能力」不是「身份」。`n_rows` 只能取 `rsp_header.n_tokens`，不能从字节数反推（否则帧边界错位）。

**grammar**：head 侧先做单 token 自检（`common_sampler_grammar_rejects()`，内部是 `llama_grammar_apply_impl` 只掩码不推进），
被拒就返回 `LLAMA_TOKEN_NULL` → **整帧回退**（撤 KV + prompt + sampler，含 grammar 状态）→ `skip_draft_once` 防同一条路径空转。
**rbudget**：`common_sampler_rbudget_rejects()` → `common_reasoning_budget_rejects()`，纯读谓词（不调 `apply`，状态机在
`accept` 里推进）；预算耗尽不是放开而是**收紧**——整词表只剩 forced token。

**验收**（`verify/verify_mtp.sh`）：同一行 h（先证明与整模 `h_nextn` 逐位相同）、同一 token、同一 pos，远程 tail 内部递归 3 步的结果与「整模 + MTP
」**逐个相同**（11751 / 13 / 271 全中）；`verify/verify_mtp_e2e.sh`：远程链三组（不投机 / ngram / MTP）输出**逐字节相同**。
`verify/verify.sh` 十四组 77 项、`verify/verify_chain_death.py` 36、`verify/verify_seq_id.py` 9、
`verify/verify_mtp.sh` 6、`verify/verify_mtp_e2e.sh` 6、`verify/verify_grammar_spec.sh` 23、
`verify/verify_rbudget_spec.sh` 36 在 v9 之后全过。

## 3. v9 的 `layer_inp` tap

### 3.1 为什么需要这条通道

三种草稿源在链上的损失完全不同（35B 官方，k=2，`n_predict 64`）：

| draft | 整模 | 本地链 | 远程链（v9 之前） |
| --- | --- | --- | --- |
| MTP k=2 | 0.9098 / 2.81 | 0.6879 / 2.37（旧路径；现默认 0.7877） | 0.6879（与本地链同一条 `!is_chain` 分支，keep-KV 在远程链不启用） |
| DFlash k=2 | 0.8298 / 2.65 | 0.8298 / 2.65 | **没有通道** |
| DSpark-5L k=2 | 0.7986 / 2.60 | 0.7986 / 2.60 | **没有通道** |

MTP 走 **nextn 通道**，被本仓加的 `!is_chain && !is_remote` 压到 0.688（2026-11-10 起草稿 KV 续写成为默认，把它抬到 0.724–0.788，逐字
节不变，见 [`MODEL-SWAP.md`](../measure/MODEL-SWAP.md) §11.4）——那是自洽设计（草稿看不到 prompt 的 KV，所以每轮清空，见
[`MODEL-SWAP.md`](../measure/MODEL-SWAP.md) §11.4）。

DFlash / DSpark 走 **`layer_inp` tap**（`dflash.target_layers = [2,7,12,17,23,28,33,38]`，每层导出一份 hidden），本地
链与整模逐位相同；但远程链原本没有 tap 通道，于是三种源里只有损失最大的那个能用。

### 3.2 协议设计

- `PROTO_VER = 9`；`DECODE_FLAG_LAYER_INP = 0x08`、`RSP_FLAG_LAYER_INP = 0x08`。
- 请求体在所有现有段之后追加 `layer_inp_req{ n_taps, n_embd }` + `n_taps` 个 **int32 源模型层号**。
- 回复体**最后**追加 `n_taps` 个 id + `n_rows × n_taps × n_embd` 个 f32。
- `rsp_header` 把原 `_pad` 的一个字节改成 `n_taps`（**尺寸仍是 32**），前四段（ids / h_nextn / cands / logits）的解析与 v8 **逐字相同
  **。`MAX_LAYER_INP_TAPS = 16`。

**三个设计理由**：① 复用 decode 回复而不是新开 op——tap 是「这一帧 tail 隐藏行的附带品」，语义上必须和产生它的那一帧绑定；独立 op 要么逼 tail 缓存上一帧（多一份状态 +
一条错误路径），要么多一个 round trip。② 用源模型层号而不是 stage 本地层号——stage 自己用 `layer_lo` 减即可，于是同一个请求可以发给链上每个远程段，head 不需要知道链在
哪切，三段链（relay 拥有一部分 tap）自然成立。③ 必须升版本号——`n_taps` 只挪了 padding，v8/v9 帧长一样，唯一防线是 `PROTO_VER` 的前置检查。

### 3.3 传输量：大头是 prefill，不是 draft

| 帧 | DFlash（2 路远程 tap） | DSpark（1 路） |
| --- | --- | --- |
| 1 行 decode | 16 KB | 8 KB |
| k=2 verify（3 行） | 48 KB | 24 KB |
| prefill 一个 ubatch（512 行） | **8 MB** | **4 MB** |

对照：一行 logits 是 993 KB。实测（本机回环，1192 token 的 prompt）：DFlash 远程 33151 ms vs MTP 远程 33078 ms，**相差 0.2%**——回环下
tap 开销可忽略。

真跨机（Tailscale → Surface，1102 token）：DFlash 远程 prefill ~32.1 s、MTP ~29.05 s，**多 3.05 s（9–10%）**，与按 9.5 MB/s
估的「25 MB ≈ 2.6 s」吻合；没有 10 s 级低效，说明 tap 是成批走的、不是逐 token 往返（全表见 [`CROSS-MACHINE.md`](../deploy/CROSS-MACHINE.md) §3）
。

### 3.4 tap 缺失的退化：停投机，不碰链

某帧 tap 缺失 / 行数不够 → **本请求停掉投机**（`dp.drafting = false`），不 `send_error`、不算链死。请求的 token 完全不变，只是回到无投机的速度。这与
v7 验证帧的 `spec_blocked` 同一套哲学：能力缺失降级成「不投机」，不是降级成「不可服务」。

### 3.5 验收：组 10

`verify/verify.sh` 组 10 对 DFlash / DSpark 两个草稿源各 10 项：整模 tap 落盘、tap 全部逐位、**wire 上那几路 tap** 逐位、远程链 logits
逐位、续写 token 逐 token、1 行帧的 tap 逐位、k=2 verify 帧（3 行）逐行 argmax、verify 帧的 3 行 tap 逐位、tap 传输字节数与预期一致、stage 收到
layer-input 请求。

实测：完整树 dflash **8/8 路**、dspark **5/5 路**与整模逐位相同；远程链 logits **9/9 个位置**、续写 token **8/8**；tap 字节数 **245760
B** 与预期一致。纯分段树 `0c307d82a` 上也是 dflash 8/8（wire 2/2、245760 B）、dspark 5/5（wire 1/1、122880 B）。

**环境坑**：`dflash-taps.exe: Device or resource busy`——整模那趟进程刚退出、文件锁未释放，日志里只有这一行，表现是组 10 的 dflash 十项全红、
dspark 全绿；看日志第一行，是 `Device or resource busy` 而不是 `[FAIL]` 就重跑这一半。

### 3.6 两个接线缺口

**缺口一：`ctx_tail_remote` 只对 MTP 置位。** `tools/server/server-context.cpp` 里原来只有
`if (spec_mtp && ctx_tail == nullptr)` 这一处置位；DFlash / DSpark 走 `has_draft` 路径、标志为 false，于是所有 tap 被解析到
head 上下文——而 DFlash 的源层 33/38 超过 head 的 32 层：

```
src/llama-context.cpp:1425: GGML_ASSERT(lid <= model.hparams.n_layer()) failed
```

修法是加一条并列判断：`has_draft && ctx_tail == nullptr && llama_context_split_has_remote(ctx_tgt)`→
`ctx_tail_remote = true`。

**缺口二：plain 帧的发送优先级反了。** `tools/split-stage/main.cpp` 的三种回复分支里，verify 和 candidate 是 tap 优先，只有 plain 是
nextn 优先（`if (want_nextn) ... else if (tap_flag) ...`）。一帧只能声明一个 `n_embd` 宽度，nextn 行（8 KB）和 tap 段（prefill
时 8 MB）不能同发；DFlash 链的 head 会同时请求两者，于是 prefill 帧回了 nextn、把 tap 吞掉。后果精确对应观察：prefill 帧掉 tap →prompt 的位置从未注入
draft KV → draft 从空 KV 起跳 → **接受率掉到 0.111，而不是崩**：

```
head decode: n_tok=35 n_tap_ask=2 verify=0      ← prefill 帧，head 请求了
rsp: flags=0x04 n_taps=0 n_bytes=1001472        ← 回复是 NEXTN_ROWS 形状，没有 tap 段
tail tap: n_tok=35 ask=2 owned=2 ok=1 flag=0x08 ← tail 明明准备好发
```

修复后 plain 分支改成 tap 优先、nextn 兜底；DFlash / DSpark 远程链**零损失**：逐 prompt 完全相同，`draft_n` 回到 141/144，输出与整模、本地链
6/6 逐字节相同。

**判据**：一帧只能带一种 payload，两个消费者争同一个通道，两侧的优先级约定必须一致；如果当初统一成 nextn 优先，verify 帧也会掉 tap、直接崩（draft 永远拿不到特征）。

## 4. 判据：为什么不是「与不投机逐字节相同」

**契约层**：llama.cpp 从不承诺 batch invariance。上游 `DETERMINISM.md`（llama.cpp 的 `docs/` 下，本机副本
`H:\llamacpp-c\.qlh_tmp\DETERMINISM_upstream.md`）把 `--deterministic` 的覆盖范围限定在 RMSNorm（全 backend）+ MatMul(
CUDA) + Attention(CUDA)，原文写着 "Determinism for attention on non-CUDA backends … planned"、"Quantized
matmul is unchanged for now (stretch goal)"。**CPU + 量化正落在这片空白里**——「1 行与 k+1 行必须算得一样」不是这份代码承诺过的性质。

**三个分界**（都在 ggml 的 CPU 代码里）：

| 分界 | 形状效应 |
|---|---|
| `ggml/src/ggml-cpu/ops.cpp` 的 `use_split_kv_path` | **单 token + KV ≥ 512** 才走 split-KV（切段并行 + `reduce_partials` 重加权），多 token 走 `f16_one_chunk` 整条累加 |
| `ggml/src/ggml-cpu/repack.cpp` 的 `nrows > 3` | `> 3` 行走 GEMM，1–3 行走逐行 GEMV |
| `ggml/src/ggml-cpu/tiled/tiled.cpp` 的 `ggml_tiled_min_batch` | `rows >= 8` 才启用 16×16 tile |

**实测**（整模、不分段、纯 embeddings 模式）：1 行 vs 2 行的 h **2048 个元素全部不同**（`max|Δ| = 0.273`）；`rows=2` 与 `rows=3` 差异完全
相同、`rows=4` 跳变——正是 `nrows > 3` 那条。**v2 当年也没做到**：`chain_client.py` 的
`forward_batch(clients, block, n_past+1)` 同样是 k+1 行一次 forward；它声称的逐位一致来自 `probe_perfect_draft.py`，而那个
probe 只测了 2 行 + 30 token。

**判据分三段**：**(a) 无回退时**，链上投机 == **整模投机**，必须逐字节相同（严格，能过）；**(b) 带回退时**，只能保证到「漂移翻掉的第一个近似并列处」；**「投机 == 不投机」降为
INFO**，它测的是 batch invariance，不是接线。

**上游状态**：PR #28108（已关闭）——FA 宽度 3/4 不能 bitwise 一致（`n_tokens 1、2 pass；3、4 FAIL`，正是 k+1 = 3/4），修法代价 nb=3
**+16.5%**、nb=4 **+15.5%**，把 1..4 钉到一个实例会让 nb=1 慢 **+84.5%**；Issue #28111（open）——IQ 量化在 `n=5` / `n=8` 不
batch-invariant，定位到 MMVQ 内部 per-`ncols` 特化改了累加顺序；Issue #25618（open）——量化 target 上 ngram 无损、`draft-mtp` 分叉、
bf16 target 一致；vLLM 把 batch invariance 当必做项（改写全部 GEMM/GEMV + 开关 + 专文），llama.cpp 当 stretch goal。

要在 CPU 上 bit-exact 需三处一起改（FA split-KV 门控、repack 分界固定、MoE 每专家形状钉死），代价主要在第一项——长 context 下单 token decode 的
attention 只能单线程扫 KV，而那正是 split-KV 存在的理由。**我们不改上游，只记录。**

## 5. `seq_rm` 的精确性（`verify/verify_seq_rm_exact.py`）

**判据为什么成立**：`seq_rm(p, -1)` 只删 p 之后的 KV，第 p 行输入不变——它的注意力覆盖 `0..p`（含自己），第 p 个 token 也没变，所以重算第 p 行必须得到**逐位
相同**的 logits。所有「回滚后重算」的路径都建立在这个前提上：投机验证帧被拒后的回退、prefix cache 复用、checkpoint 恢复。

口径：2B 整模、纯 CPU、`n_threads=4`、p=29、batch 里 14 个 prompt token、vocab 248320；只用 llama API，无 split、无 server、不
占端口。探针是 `build-split/bin/seq_rm_exact.exe`（源 `.qlh_tmp/seq_rm_exact.cpp`）；日志
`.qlh_tmp/pure-split-acceptance/seq_rm_exact_2026-11-09_rebase.log`（rebase 后重跑，1 pass / 5 known-fail / 0
unexpected）。

| 形态 | `n_rs_seq` | k（回退量） | 结果 |
|---|---:|---:|---|
| A1 FULL | 0 | 1 | **红**：`seq_rm=false`，`pos_max 29 -> 29` |
| A2 FULL | 0 | 3 | **红**：`seq_rm=false`，`pos_max 31 -> 31` |
| B1 RS | 3 | 1 | **红**（rebase 后翻的）：`seq_rm=false`、一行未删 |
| B2 RS | 3 | 3，重放 1 行（投机回退的真实形状） | **红**，同上 |
| B3 RS | 3 | 3，重放 3 行 | **红**，同上 |
| C1 RS | 3 | 4 > 快照窗口 | 绿：**明确拒绝**（false） |

`--expect-known-fail` 下是 `1 pass / 5 known-fail / 0 unexpected`、退出 0；不带开关退出 1。

**B 组为什么从绿变红**：上游 `ab3768cdc` 把「允许回退多少」从 `rollback <= n_rs_seq` 改成 `rollback <= cell.rs_max`，而
`rs_max = 本 ubatch 写入的快照数 − 1 = n_seq_tokens − 1`；本 probe 是逐 token decode（`llama_batch_get_one(&t, 1)`），
所以 `rs_max = 0`、任何回退都必须被拒。这是上游「只保证最后一个 ubatch 的快照」的设计后果，不是本仓缺陷。

**没有改 probe**：用多 token ubatch 能让 B 组变绿，但「重放 logits 逐位相同」会因 1 行 vs k+1 行的 kernel 差异（0.27 量级）失效——等于把可靠判据换成不
可靠的；所以更新 known-fail 基线并把原因写进探针文件头。

**FULL 形态的原始输出**（`seq_rm_exact.exe <model> 0 29 1 1`）：

```
[seq_rm_exact] n_rs_seq=0 p=29 k=1 k_replay=1 prompt_tokens=14 n_vocab=248320
[seq_rm_exact] seq_rm=false pos_max 29 -> 29
[seq_rm_exact] replay rc=-1 pos_max=-1
[seq_rm_exact] argmax first=27 second=-1
[seq_rm_exact] RESULT FAIL
```

读法：`seq_rm` 返回 false 而 `pos_max` 一点没降——尾部 KV 一行都没删（hybrid 的短路：recurrent 拒绝后 attention 那半被整体跳过）。随后重放
`rc=-1`：decode 拒绝在「KV 比 prompt 长」的位置上写，于是没有第二份 logits 可比（`second=-1`）。**红的是「删除」这一步，不是「重算」**——只要删除真的发生，重算
就逐位相同（这一半在上游改动之前由 B 组直接证过，现在由 grammar / rbudget 两组的回退形态间接证明）。

B 组 rebase 前的原始输出（`seq_rm_exact.exe <model> 3 29 3 1`，历史对照）：`seq_rm=true pos_max 31 -> 28`、
`replay rc=0 pos_max=29`、`argmax first=27 second=27`、`logits diff=0/248320`、`RESULT PASS`；rebase 之后同一命令是
`seq_rm=false`、一行未删。

**C 组为什么必须绿**：回退量超过快照窗口时必须**明确拒绝**而不是删一半，删一半比拒绝更糟——调用方拿到 true，以为位置已经一致。

**产品形状下的回退仍被逐位证明**：投机验证帧把 `n_draft + 1` 行放进一个 ubatch，回退 `n_draft` 行落在 `rs_max` 之内，由
`verify/verify_grammar_spec.sh`（4 组「带回退且逐字节相同」）与 `verify/verify_rbudget_spec.sh`（整帧回退 `seq_rm` 到 tail 成功、
回退 6 次仍逐字节相同）覆盖，两组全绿。

**shared-prefix 形态**（`verify/verify_seq_rm_shared.py`）：`n_cmpl=2` 让 parent 与 child 共享前缀，再注入
`QLH_FORCE_SEQ_RM_FAIL=1`，比较注入前后两份 completion——两次各 2 条、逐字节相同、`health` 都是 200，但日志里只有
`[INFO] injection never fired: this shape does not reach a partial seq_rm (the comparison above then proves nothing about isolation)`
：这个形状不经过 partial `seq_rm`，所以「`prompt_clear` 误伤另一个 slot」不可达。**看到这行就说明该次运行没测到东西**，脚本按 INFO 通过。

**结论**：server 默认 `n_rs_seq = 0` 的 hybrid 上「删尾部」永远失败，这是上游缺陷（open PR #28007 / issue #31656）；调用方必须检查返回值，五个调
用点已按场景降级，见[`SEGMENTED-LIMITS.md`](../limits/SEGMENTED-LIMITS.md) §2.9。
