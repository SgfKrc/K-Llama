# 分段改造的已知边界

> 状态：**现行**
> 更新日期：2026-11-08

这份文档记录的是**原型阶段有意接受、产品阶段要付代价处理**的限制。不是 TODO 列表，是边界说明：
每条写清现象、根因、影响范围，以及修的时候要付什么代价。

当前状态（2026-11-08，核对到 `H:\llamacpp-c` 的 `c5bd787b`）：上一版记为问题的六条，逐条核实后的现状见
§1；§2 是仍然成立的边界。投机那一条（§2.2）翻过三次，现在是终态：远程链**能**投机，但收益中位
约 +6%、随负载从 +8.4% 到 −1.2%，弱机器与回环上是负的；而且「投机 == 不投机」不能当判据。除
§2.2 的 batch invariance 一段（归因在上游 llama.cpp）与 §2.9 的 hybrid `seq_rm` 短路（上游缺陷，
本仓只做「失败可见 + 调用方降级」）之外，下面每条的归因都在本仓。代码引用一律
写符号（函数/字段名），不写会随改动漂 100–300 行的 `文件:行`。

**35B 的权重在 2026-11-08 换成了官方的 `UD-Q4_K_M`**（旧的蒸馏版已弃用）：§2.2 里「整模
0.910 / 链上 0.688」是官方版上的数，而 §2.2 的 ROG / Surface / 回环收益与 §3.2 的 DFlash、MTP
加速比仍是旧版权重上的数。**两份权重不能混用**，口径与全表见
[`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §11。

README 的六条边界与本文章节的对应：① 架构限定 → §2.1、② 远程链单序列 → §2.4、③ 远程链的投机
能做但收益随负载走 → §2.2、⑤ 投机加速比随部署形态走 → §3.2；④ 多模态与 ⑥ 核显属输入/部署边界，
分别在 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §7 末尾与 [`IGPU-SLOWDOWN.md`](IGPU-SLOWDOWN.md)。

---

## 1. 已实现：上一版记为问题的六条

| 上一版记的 | 现行实现 | 位置 |
|---|---|---|
| 远程链没有 KV 管理 | `llama_memory_split` 直接持有 `llama_split_remote*`，`seq_rm` / `clear` 逐个下发；`llama_split_attach_remote()` 装上聚合视图 | `src/llama-memory-split.h` 的 `llama_memory_split::remotes` / `seq_rm()` / `clear()`；`llama_split_attach_remote()` |
| 远程 `pos` 只传标量 | `split_expand_pos()` 把每个 token 的 4 个 M-RoPE section 全部下发，`req_header::n_pos` 让两端对齐 stride | `split_expand_pos()`（`src/llama-context.cpp`）、`req_header::n_pos` |
| `split-stage` 的段角色恒为 tail | `--role relay\|tail` 必填，用 `llama_model_n_layer_src()`（源层数，不是本地层数）校验角色与层范围 | `tools/split-stage/main.cpp` 的参数校验：`relay` 要求 `layer_hi < n_layer_src`、`tail` 要求 `layer_hi == n_layer_src` |
| 远程 logits 只有 1 行 | 本地链的 tail 只为 `batch.logits` 标记的行出 logits，一个都没标时兜底最后一行；**远程 tail 自协议 v5 起普通 decode 回候选而不是整行，只有走整行回退的请求才整行回传；投机验证帧另走 v7 的逐行 argmax 通道**（尺寸见 §2.2） | `llama_context::decode_split()` 的 `ask_logits`；`common_sampler_candidates_check()` |
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

### 2.2 远程链的投机：能做，但收益随负载从 +8.4% 到 −1.2%，弱机器与回环上是负的

**现象（终态，2026-11-08）**：这一节翻过三次。最初写「远程链不做投机，缺的是验证通道不是
带宽」；v7/v8 落地后改成「能做、只做贪心、k=2 是甜点、+15%」；按五类真实负载重测之后是现在
这条：**能做，收益中位约 +6%，随负载从 +8.4% 到 −1.2%，弱机器（Surface）与回环上是负的**。
同时「投机 == 不投机逐字节相同」这个判据不成立——不是实现没做好，是 llama.cpp 在 CPU + 量化
上根本不承诺 batch invariance（见下「判据」）。三次反转的教训在节末。

**做了四层**：

| 层 | 内容 |
|---|---|
| **v7 验证通道** | tail 对验证帧**每行**回一个 argmax，一帧 k+1 行 = 4(k+1) 字节。单帧回复（4 行）993,312 B → **48 B（20,694×）** |
| **v8 MTP draft** | `OP_MTP_DRAFT`（op 8）带一行 hidden 与 `mtp_req{token,pos,k}`，**k 步递归在 stage 内完成**（每步输入是上一步的输出，不能批处理）；`DECODE_FLAG_NEXTN_ROWS` / `RSP_FLAG_NEXTN_ROWS` 让 decode 回复追加 `n_rows × n_embd` 个 float；`stage_id.has_nextn` 占原 `_pad` 字节且**故意不参与身份比较**——「能力」不是「身份」 |
| **grammar 兼容** | head 侧先做**单 token 自检**（`common_sampler_grammar_rejects()`：`llama_grammar_apply_impl` 只掩码不推进），被拒就返回 `LLAMA_TOKEN_NULL` → **整帧回退**（撤 KV + prompt + sampler，含 grammar 状态）→ `skip_draft_once` 防止同一条路径空转 |
| **rbudget（reasoning budget）兼容** | `common_sampler_rbudget_rejects()` 走 `common_reasoning_budget_rejects()`，是**纯读谓词**（不调 `apply`，状态机在 `accept` 里推进）；预算耗尽不是放开而是**收紧**——整词表只剩 forced token |

**ROG 实测（head `0:32` + tail `32:40`，35B，LAN，贪心）五类 prompt × 三配置，tok/s**：

| prompt（token） | 不投机 | ngram | MTP k=2 | MTP 变化 |
|---|---:|---:|---:|---:|
| 散文 882 | 8.18 | 7.33 | 8.47 | +3.6% |
| 代码 1205 | 8.74 | 6.92 | 9.47 | **+8.4%** |
| JSON 2379 | 9.47 | 7.09 | 9.36 | **−1.2%** |
| agent 2508 | 7.51 | 5.09 | 7.62 | +1.5%（当时无 draft） |
| 人为重复 `A+B+A+B+A` 89 | 9.73 | 7.75 | 10.75 | +10.5% |

**中位约 +6%，不是 +15%。**（**这一组与下面的 Surface / 回环数字都测在旧蒸馏版上**；官方
`UD-Q4_K_M` 的接受率基线是整模 k=2 的 0.910、链上 0.688，见下面「固定折扣」段。）历史上那个
「+15%（10.20 tok/s）」来自一个 27-token 的高确定任务，
不是真实负载；把它当结论就会得出「跨机投机稳赚」这种错判。ngram 五类**全都低于不投机**，
原因见下「ngram 的真相」。

**弱机器上全部转负**：Surface（4C8T，纯 CPU）MTP k=2 散文 **5.21 vs 不投机 8.65（−40%）**。

**本机回环上带 grammar 的投机也是负收益**：不投机 **20.82 tok/s** / 投机 k=2 **13.95 tok/s**。

**为什么弱机器与回环上是负的**，四个真实原因：

1. **每帧两次 hop**：`OP_MTP_DRAFT` 出草稿、`OP_DECODE` 再验证，一轮要走两趟链。
2. **MTP draft 不看 grammar / rbudget**：接受率从无 grammar 时的 **2.76 token/帧** 掉到
   **1.46–1.73**。
3. **回退率 9–19%**：`toolcall.gbnf` 这种严格语法实测 **11/16 帧被拒**，而每次回退 ≈ 一整个
   普通 round（撤 KV + 重放 prompt + sampler 回滚）。
4. **回环上 993 KB 只要约 1 ms**，所以「带宽省四个数量级」这个唯一的结构性好处根本体现不
   出来——省下的是不存在的成本，付出的是真实的 draft 算力与第二趟 hop。

**还有一个与部署无关的固定折扣：链上 MTP 的接受率比整模低 24%。**

官方 35B（`unsloth` 的 `UD-Q4_K_M`）上，同一个 k=2、同一批 prompt：整模是
`0.9098 / mean len 2.81`，本地双段链 `0:32 + 32:40` 是 **`0.6879 / 2.37`**（远端链走的是同一个
`!is_chain && !is_remote` 跳过分支，所以它同样没有 prompt 历史；**远程链上的接受率没有单独
测**）。**数值没有退化**：3/3 prompt × 64 token 与整模**逐字节相同**（含 `reasoning_content`，
250/244/249 字符）。所以这不是链算错了，是链上的草稿更差。

根因是本仓自己加的两处，不是上游行为：`speculative::process()` 的 catch-up decode 条件是
`!is_mem_shared && !is_chain && !is_remote`（上游 `e0f142d5d` 只有 `!is_mem_shared`，`&& !is_chain
&& !is_remote` 是 `c5bd787b` 新加的），而 `speculative::draft()` 在 chain 下每轮
`llama_memory_clear` 草稿 KV。两处构成一个**自洽的循环依赖**：

```
不 catch-up  →  草稿没有 prompt 的 KV  →  必须每轮清空（否则静默读脏）
            →  即使 catch-up 也读不到   →  所以跳过 catch-up 是"对的"
```

在当前这套草稿 KV 策略下，catch-up 建出来的 prompt 历史**没有任何读者**，跳过它是正确且必要
的。代价是把链上的接受率封在 0.688——**这是设计选择，不是接线错误。**

**它被小模型掩盖过**：`draft()` 那段注释自己写的是「2B 上接受率仍在 70–93%」，到 35B 掉到
0.688 才暴露。qwen35moe 的 NextN 块**带 attention**（MTP 图里有 `cache_k_l8` / `FLASH_ATTN_EXT`），
所以「草稿看不到 prompt 的 KV」是真实差异，不是理论洁癖。

**要做的话**必须**同时**改两处（catch-up 的 `is_chain` 判断，以及 `draft()` 的 chain 分支不再
清空、改成按 target 的 `seq_rm` 严格同步），只改一处会让「历史建了没人读」与「读到别的 token
的状态」二者必居其一。收益**是估算不是实测**：短 prompt + 短输出 ≈ 0，512 token 的生成省
10–14 s，2000 token 的 prompt 净负（catch-up 本身 30–60 s）。**结论：将来做成 opt-in 开关，
不是现在默认打开。** 口径、日志与逐位复算见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §11。

而且 **FORCING 期间 tail 回的未掩码 argmax 基本每轮都被拒**，有约 **2 倍白跑**。

**判据：为什么不是「与不投机逐字节相同」**（这一条比上面所有数字都重要）。

**契约层**：llama.cpp 从未承诺 batch invariance。上游 `docs/DETERMINISM.md` 把
`--deterministic` 的覆盖范围限定在 **RMSNorm（全 backend）+ MatMul(CUDA) + Attention(CUDA)**，
原文写着 "Determinism for attention on non-CUDA backends … planned"、
"Quantized matmul is unchanged for now (stretch goal)"。**我们的 CPU + 量化正落在这片空白里**，
所以「1 行 vs k+1 行算出来必须一样」在上游就不是一条可依赖的性质。（本机副本：
`H:\llamacpp-c\.qlh_tmp\DETERMINISM_upstream.md`。）

**三个具体分界**，都在本仓的 ggml CPU 代码里：

- `ggml/src/ggml-cpu/ops.cpp` 的 `use_split_kv_path`：**单 token + KV ≥ 512** 才走 split-KV
  （切段并行 + `reduce_partials` 重加权），多 token 走 `f16_one_chunk` 整条累加——同一条 KV
  在两种形状下用两种归约顺序。
- `ggml/src/ggml-cpu/repack.cpp` 的 `nrows > 3`：`> 3` 行走 GEMM，1–3 行走逐行 GEMV。
- `ggml/src/ggml-cpu/tiled/tiled.cpp` 的 `ggml_tiled_min_batch`：`rows >= 8` 才启用 16×16 tile。

**实测**（整模、不分段、纯 embeddings 模式）：1 行 vs 2 行的 h **2048 个元素全部不同**
（`max|Δ| = 0.273`）；`rows=2` 与 `rows=3` 的差异**完全相同**、`rows=4` 跳变——正是
`nrows > 3` 那条分界。

**v2 当年也没做到**：`chain_client.py` 的 `forward_batch(clients, block, n_past+1)` 同样是
k+1 行一次 forward；它声称的逐位一致来自 `probe_perfect_draft.py`，而那个 probe 只测了
**2 行 + 30 token**。

所以本仓的判据是：

- **(a) 无回退时**：链上投机 == **整模投机**，必须逐字节相同（严格判据，这一条能过）；
- **(b) 带回退时**：只能保证到「漂移翻掉的第一个近似并列处」——之后分叉是浮点归约序，不是
  接线错误；
- **「投机 == 不投机」降为 INFO**：它测的是 batch invariance，不是我们的接线。

**上游状态**（说明这不是我们能修的）：

- **PR #28108（已关闭）**：FA 宽度 3/4 不能 bitwise 一致，`n_tokens 1、2 pass；3、4 FAIL`
  ——正是 k+1 = 3/4 的位置。修法代价 nb=3 **+16.5%**、nb=4 **+15.5%**；把 1..4 钉到一个实例
  会让 nb=1 慢 **+84.5%**。
- **Issue #28111（open）**：IQ 量化在 `n=5` / `n=8` 不是 batch-invariant，定位到 MMVQ 内部
  的 per-`ncols` 特化改了累加顺序。
- **Issue #25618（open）**：量化 target 上 ngram 无损、`draft-mtp` 分叉、bf16 target 一致。
- **vLLM 把 batch invariance 当必做项**（改写全部 GEMM/GEMV + 给开关 + 写专文），llama.cpp
  把它当 stretch goal——同一个问题，两种投入。

**结论**：要在 CPU 上做到 bit-exact，需要三处一起改（FA split-KV 门控、repack 分界固定、MoE
每专家形状钉死），代价主要在第一项——长 context 下单 token decode 的 attention 只能单线程扫
KV，而那正是 split-KV 存在的理由。**我们不改上游，只记录。**

**ngram 的真相**（这一节也翻过）：chat 模板下 ngram **全程 0 draft**。机制是「复制上下文里
最后 12 个 token 的后续」，而 chat 模式下模型对重复文本的反应是**评论**
（"It looks like you've pasted the same paragraph…"）而不是续写，于是永不命中；代码/JSON 生成
的是新内容，同理。**只有 raw completion + 人为重复文本才有收益**：同一 prompt 上 ngram
**27.85 vs none 9.59 tok/s = 2.9×**、接受率 **1.0**——这就是历史那个「4.3×」的全部来源，而
那次用的 prompt 是 `A + B + A + B + A`。上面 ROG 表里 ngram 五类全部更低，是同一件事的另一面。

**顺带记一条实现遗漏：`split-stage` 的 `-t`。** `GGML_DEFAULT_N_THREADS = 4`，**且
`n_threads_batch` 也是 4、不会自动跟随 `n_threads`**，所以 Surface（4C8T）上原来只用到了 4
个线程。现在补了 `-t/--threads`（同时设两者）、`-tb/--threads-batch`、`-ub/--ubatch-size`。
Surface 实测：不投机 **7.41 → 7.67（+3.5%）**；MTP **5.12 → 5.48（+7%）**。**只设
`n_threads` 拿不到收益，batch 那一半必须一起设。**

**仍然成立的结构事实**（与上面的收益结论无关）：

- **链上 MTP 的接受率低 24% 是设计选择。** k=2 时整模 0.910 / 链上 0.688（35B 官方模型），
  根因是 `process()` 跳过 catch-up 与 `draft()` 在 chain 下每轮清草稿 KV 这一对自洽的策略；
  输出与整模逐字节相同。要做 catch-up 必须两处一起改，收益只在长生成上成立（估算）。详见
  上面那段「固定折扣」与 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §11。
- **采样投机仍不做。** rejection 要比 p 与 q 两个分布、每行需要完整 logits：k=5 一轮 5–7 行
  ≈ 5–7 MB，在实测 5–16 MB/s 链路上 250–800 ms/帧。v2 的采样路径正是这么做的，实测比不投机
  慢 1.9×。远程链的投机是**贪心专用**，上面所有表都是贪心口径。
- **MTP 只能跟末段。** MTP 不是「草稿头小部件」，是**完整的一层 transformer**（`blk.40` 共
  856.36 MiB：MoE 专家 820.18 MiB 占 95.8%、attention 27.63 MiB、NextN 胶水 8.52 MiB），全
  文件只有 1 个 `output.weight` 与 1 个 `token_embd.weight`——它不自带 lm_head、也不自带
  embedding。输入是 `t_h_nextn = RMSNorm(最后一层输出, output_norm)`（中间层的 hidden 是未
  训练的），`needs_output() = hi >= n_layer_src`，loader 写死
  `nextn_keep = (hi >= n_layer_src) ? nextn_all : 0`。硬挪中段只能反着数据流走（tail 算完 →
  下发 hidden → mid 跑 MTP → draft 上行 → tail 验证），k=2 实测每轮省 75.5 ms、而两次 RTT
  已 50 ms，**净负**；复制一份到每段是 856 MiB 起步，且 96% 是 MoE 专家。
- **DFlash 在远程链上仍缺一块。** 8 路 tap 天然分布在两段，tail 那两路（33/38 层）要跨段的
  `layer_inp`，协议还没有这个 op（`OP_LAYER_INP`，约 96 KB/轮下行）。不是「不做」，是「没有
  做」：MTP 已经证明验证通道够用，DFlash 差的是取数通道。
- **异构弱网上张量并行是纯负收益**，不是「慢一点」：TP 每 token 要 960 KiB、80 次同步，本
  项目这种流水线只要 8–16 KiB、1–2 次——差 60–120 倍。TP 通信 ∝ 层×token，PP 通信 ∝
  段数×token，这个差别在 5 MB/s 上是 196.6 ms/token vs 1.6/3.3 ms。

**旧版算错过一次，留个记号**：原写「验证要 k+1 行完整 logits，k=3 一帧 4 MB，千兆上 3–4 s」，
两处都错——4 MB 在千兆是 32 ms、按本仓实测 16.06 MB/s 是 250 ms；而且贪心验证根本不需要整
行，argmax 在 tail 侧算，每行只回一个 token id。

**三次反转的教训**：（1）「某个 draft 源在这个 prompt 上很快」先问 prompt 是不是为它构造
的——`A+B+A+B+A` 之于 ngram、27-token 高确定任务之于 MTP，都属这一类；（2）收益必须按**部署
形态**报（LAN/回环、强/弱机器）：同一个功能在 ROG 上的 +8.4%、在 27-token 任务上的 +15%、在
Surface 上的 −40% 都不矛盾，它们不是同一个配置；（3）**判据要先问「这份代码承诺过什么」**
——拿 batch invariance 当接线判据，会把上游的空白当成自己的 bug。

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
→ **进程退出**。（`seq_rm` 改成 `bool` 之后这一行已不存在：本地链也只返回 false，由调用方按场景
降级——`prompt_clear` 后全量重算、结束响应或让请求失败——见 §2.9。）早先「变成普通请求失败」
的说法是错的。

**已改**：失败置 `remote_failure` 而不 abort，`decode_split()` 算之前就查它并返 `-3`——后续请求全
失败，绝不假装成功；链死后不再碰网络（第二个请求 0.01 s 失败）。`/health` 仍 200 是刻意的（存活
语义），要看链问 **`?deep=1`**：链死快速 503、链活真发 1-token 请求、忙时 `200 {"deep":"busy"}`。

**自愈**：下一个从位置 0 开始的请求重建（有界 reconnect → 整链 clear → 恢复超时），失败保持 dead
+ 5 s 节流（`kRebuildThrottleMs`，`src/llama-memory-split.cpp`）。闸门是结构性的：重建后唯一一致
状态是「全空」，空链只能从 0 写，所以「KV 空但接着算」不可能。实测只重启 stage：200、
`prompt_n`=完整长度、`cache_n=0`、输出逐字节相同。

**握手**（`OP_HELLO`，v4 引入；`PROTO_VER` 现在到 8）：重连必须重握手，接错的 stage 让重建失败；防接错不防恶意，
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

### 2.9 `seq_rm` 失败不再静默：上游短路、本仓改成 `bool`、两个坑

**这一节记的不是本仓引入的缺陷**：上游 hybrid memory 让「删尾部」在 server 默认配置下永远失败，
本仓原先把失败吞掉了（这一半已修）。能修的是「失败可见 + 每个调用点按场景降级」；
`seq_rm` 本身的行为要等上游修。

**现象**：上线前的一次投机评测发现「同一个位置，回退前后算出的 `h` 不同」。追下去不是投机接错，
而是 `seq_rm` 失败被静默忽略了：调用方（验证帧回退、prompt-cache 复用、checkpoint 恢复）以为行
删掉了，继续在「prompt 已回退、KV 没回退」的状态上算下一个 token——**不报错，只是算错**。

**根因（上游）**：`llama_memory_hybrid::seq_rm` 先删 recurrent，失败就 `return false`——
**attention KV 的删除被整体跳过**。而 hybrid 在 server 默认配置下 `n_rs_seq = 0`，partial 分支要求
`rollback <= n_rs_seq` 直接不成立，于是「删尾部」永远失败。更糟的是 `common_context_seq_rm` 在 split
链上**只 flush 日志就返回**（`void` 没有错误通道），调用方连「没删成」都拿不到。

**这是上游的问题，不是我们的**：`src/llama-memory-hybrid.cpp` 在 193 个提交里零改动，`seq_rm` 与我们
逐字节相同。上游已有 **open PR #28007**（机制描述一致：「`llama_memory_hybrid::seq_rm` short-circuits:
when the recurrent part refuses, the KV part is not truncated either」）和 **open issue #31656**（同一条
abort 路径），所以本仓**不提新 issue**，只在调用方把失败显式化。

**改法**：`common_memory::seq_rm` / `common_context_seq_rm` 从 `void` → `bool`，五个调用点按场景退化：

| 场景 | 失败后 |
|---|---|
| prompt 预处理（`p0`，cache 复用的裁尾） | `prompt_clear` + **用 task 的 tokens 重建 `prompt.tokens`** + `return`，下一轮从 pos 0 全量重算 |
| context shift | `stop = STOP_TYPE_LIMIT`，跳过 `seq_add` 与 prompt 重写（滑动是全有或全无） |
| cache shift / 验证帧回退 | `send_error` + `prompt_clear` + `release` |
| spec trim（两处：部分接受的 checkpoint 回退、普通接受的裁草稿行） | 先把手上这批 token 发完，再结束响应 + 全清（保住已生成内容） |

**关键细节**：只 `prompt_clear()` 是**不够**的。prompt 处理循环不会从空 prompt 重建内容（那发生在
`launch_slot_with_task`），清空后 fall-through 会「从零生成」——比 500 更糟。必须把 task 的 tokens
装回去，才是真的「全 prompt 重算」。

**坑一：链死时不能结束请求。** 第一版把「链死」也走「失败 → `send_error` + `release`」，结果
**自愈永久失效**（`verify_chain_death.py` 的 D 组红了：只重启 stage 之后 `/completion -> 500`、
`prompt_n = None`）。原因：`llama_memory_split::try_rebuild` 只在**一次从 pos 0 开始的 decode** 里
触发，而「结束请求」会跳过那次 decode——**一次也不 rebuild，链就永远死**。最终形态：链死只记
ERROR、不结束请求，让正常路径失败并释放 slot，下一个从 pos 0 的请求触发重建。

这条的教训值得单独说：一个「看起来更严谨」的失败处理（失败就结束）反而堵死了系统里唯一的自愈入口。

**坑二：`p0 > pos_max` 是成功，不是失败。** 做故障注入时最初让所有 partial `seq_rm` 都返回 false
（含空删除），结果降级调用方在同一位置无限重试，5 次循环后超时。查实现：`llama_kv_cache::seq_rm`
在 `pos_in` 不匹配时跳过、末尾**无条件 `return true`**；`llama_memory_recurrent::seq_rm` 同样。所以
空删除是成功。注入（`QLH_FORCE_SEQ_RM_FAIL`）已改为只在 `p0 <= pos_max` 时拒绝。

**已知未覆盖**：

- `p0` 那条退化分支标为 **code-review-verified, not run**：它与 spec trim 那条结构相同（后者已端到端
  跑过），但没有测试能构造出「cache 复用形态下真的走一次 `p0 <= pos_max && p0 > 0` 的 partial
  `seq_rm`」。
- shared-prefix 隔离：`copy_prompt_to` / `copy_state_to` / `wait_shared` 这条路径**不调用** partial
  `seq_rm`，所以「`prompt_clear` 误伤另一个 slot」不可达。脚本 `verify_seq_rm_shared.py` 留下
  `[INFO] injection never fired` 记录——**看到这一行说明这次跑没测到东西，别当绿灯**。

**新增验收**（口径见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §10）：

- `verify_seq_rm_exact.py --expect-known-fail`：纯 llama API、无 split、无 server，断言「删掉 p 之后
  的 KV 后重放第 p 行，logits 必须逐位相同」。当前 **A（FULL 形态）两条红、B（RS 形态）三条绿、
  C 一条绿**，退出 0。上游修好 FULL 缺陷后它会变成「0 红」，那时把开关关掉。
- `verify_seq_rm_shared.py`：shared-prefix 形态。

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
`no implementations specified for speculative decoding`）——这是启动参数的默认值，不是
「不划算」的结论。开之前要按负载选源和 k，并且「35B 上 k=2 是上限」这类结论**不能跨模型搬**
——2B 上 k=3 更好，而且只有一个 prompt 的样本，还不足以定论。

**顺带记一条：35B 这一档自己内部是稳的**——换到官方 `UD-Q4_K_M` 之后重扫 k=1..8，甜点仍是
**2**（接受率 0.910、mean len 2.81，tok/s 最高），与蒸馏版落在同一档。所以「k=2」可以在 35B
的两份权重之间复用，仍然不能搬到 2B 上（见
[`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §11）。

但这里有个更重要的限制，比上面所有数字都强：**本机双进程测不出投机收益**。head 与 tail 跑在
同一台机器上时，两者走内存与管道、**没有 25–30 ms 的跨机往返可省**（而削减往返正是投机唯一
该拿的好处），同时两个进程抢同一套 CPU 与内存带宽，draft 的额外算力被全数计入——付出成本、
拿不到好处，这是个退化配置。这条只说明「本机的负数不能外推」，**不是「多机一定为正」**：
多机实测的中位是 +6%，见下。

**跨机已经测了（2026-11-07/08，协议 v7/v8）**：ROG 上 head `0:32` + tail `32:40`、LAN、贪心，
五类真实 prompt × 三配置：MTP k=2 相对不投机是 **+3.6%（散文）/ +8.4%（代码）/ −1.2%（JSON）/
+1.5%（agent，当时无 draft）/ +10.5%（人为重复）**，中位约 **+6%**；ngram 五类全部更低。所以
「真实多机上划算」只能写成条件句：**LAN + 强 head + 接受率高的负载**。同一副链换切点、换机器
（Surface 上 MTP k=2 散文 **5.21 vs 8.65，−40%**）或换到回环（带 grammar 时**不投机 20.82 /
投机 13.95 tok/s**）都会翻号。数据见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §9；本机
双进程为什么测不出来，见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §0.1。

**修的时候要付的代价**：不修。要做的是把「按负载选」做成配置项而不是默认值，这需要一套负载分类与实测表，比投机本身的工作量大。

---

## 4. 一条给未来的提醒

上面这些问题有一个共同形状：**它们多数不会报错，只会算错或悄悄退化**——能拒的只有架构白名单、
角色/层范围（含 `split_range_is_stage`）、远程单序列（§2.1、§2.6、§2.4）；其余都是「看起来在正常
工作」，错误只体现在数值或长期行为上。

所以验证必须比 logits 而不是比生成的文字（argmax 一致经常是巧合），长会话要盯 KV 占用而不是只看
前几个请求，跨 ubatch 的对比要先确认 ubatch 边界。原型里我们用 `tests/split-smoke.cpp` 和
`tests/dflash-taps.cpp` 做逐位对照，这个习惯在做产品时要保留。

**第二个形状是「新加的 per-frame 状态没跟上既有的 reset 约定」**（2026-11-07，协议 v8 落地时
撞到的）：`split_remote_reset()` 清掉所有 per-frame 槽位，**包括新加的 h_nextn 行**。原来的顺序
是「先存行、后存 logits」，于是行被 `store_logits()` 里的 reset 抹掉 → stage 回 2048 floats、
head 收到 0 行 → **draft 静默消失**：不报错、不变慢，只是不投机。修法是把 `store_nextn` 放到
三个回答分支之后执行，判据是 wire 上的 h_nextn 与整模逐位相同。**这类 bug 的共同点是「约束是
一个没写下来的顺序」**——凡是新增 per-frame 字段，先问它会不会被哪个 reset 清掉。

**第三个形状是「更严谨的失败处理堵死了自愈入口」**（2026-11-08，§2.9 的坑一）：把「链死」也
当成「请求失败就结束」，看起来更干净，却让 `try_rebuild` 唯一触发的那次 pos 0 decode 永远不
发生——**失败处理本身把恢复路径关掉了**。所以新增一条失败分支时，除了问「失败后状态一致吗」，
还要问「这个分支会不会跳过唯一能恢复的那条正常路径」。
