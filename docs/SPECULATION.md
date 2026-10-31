# 投机解码：四个方向的判定

> 状态：已实测，未接入
> 更新日期：2026-10-30

本文记录四种投机方案在本项目上的可行性判定。每条都带实测依据；没有实测的会明确标注为估计。
适用前提：Qwen3.6-35B-A3B（Q4_K_XL，40 层），两段跨机（本机 0:32、ROG 32:40），
当前 **140 ms/token**。

---

## 0. 成本模型：k 决定一切

一轮投机的位置操作账（两段链，每段各算一次）：

```
一轮 = tail 单步(1) + head 批量(k) + tail 批量(k) + head 预写(1)
一个 k 行 batch 的实测成本 = 1 + 0.178 × (k-1) 个单行
  （0:32 段 72 / 81.8 / 105 / 161.3 / 269.3 ms 于 k=1/2/4/8/16）
plain 生成 2 个 token = 4.00 个单位
```

| k | 成本 | 每草稿需接受的盈亏线 | 接受率 85% 时 |
| ---: | ---: | ---: | ---: |
| 1 | 4.00 | **1.00（即 100%）** | 2.19 单位/token，**慢 9%** |
| 2 | 4.36 | 0.09 | 1.61，快 1.24× |
| 4 | 5.07 | 0.13 | 1.15，快 1.74× |
| 8 | 6.49 | 0.16 | 0.83，快 2.40× |
| 16 | 9.34 | 0.17 | 0.64，快 3.13× |

**k=1 是结构性亏损**：成本 4.00 与 plain 生成 2 个 token 完全相同，却只多前进 `r ≤ 1` 个，
只有草稿 100% 正确才持平。实测 83% ⇒ 必然慢 9%。

**k ≥ 2 才有意义**，门槛很低（每草稿接受 9-17% 即回本）。这是 MTP 与 DFlash 的分水岭，
也是早期把 MTP 排在 DFlash 前面时漏掉的一笔账。

## 1. 判定汇总

| 方向 | 判定 | 依据 |
| --- | --- | --- |
| MTP（模型自带） | **部件已通，不接主链路** | 单 head ⇒ k=1，§0 结构性亏损；实测慢 13-14% |
| DFlash（外挂 0.24 GB） | **唯一值得投入** | k=8/16 在盈利区间，每草稿接受 16% 即回本 |
| PLD / n-gram | 按负载开关 | 中文对话 m=1.05（无效），代码续写 m=2.56–3.37 |
| 浅层投机（前 k 层） | 不做 | 实测 α=0.00–0.08，E[accept]≈1.0 |
| dSpark | 不做 | 开源权重只有 DeepSeek-V4 系，无 Qwen 版 |

---

## 2. 前置条件：recurrent state 回滚

hybrid 架构（`LLL F` 交替）的 linear 层有 recurrent state，而它**没有位置概念**。
验证 k 个候选后只接受 j<k 个时，必须把位置 j 之后的 KV **和** state 一起退回去。
shim 目前把 `cparams.n_rs_seq` 显式设成 0（`qlh_keep_head_gpu.c` 里的注释是
"no recurrent-state rollback"），上下文日志可直接确认：

```
llama_context: n_rs_seq = 0
llama_memory_recurrent: size = 50.25 MiB (1 cells, 32 layers, 1 seqs 0 rs_seq)
```

**这一条不解决，任何投机方案都无法实现。**

内存代价已实测（`SA_N_RS_SEQ=4`，35B）：head 0:32 → **251.25 MiB**（R 11.25 + S 240）；
tail 32:40 → **62.81 MiB**（R 2.81 + S 60）。S 随 `n_rs_seq` 线性增长。本机 22 GiB 毫无压力；
tail 那 63 MiB 塞进 ROG 的 0.82 GiB 可行，但吃掉七分之一余量。

回滚也已验证：不开快照时 `qlh_kh_seq_rm` 返回 **-6**，开到 4 后返回 0。与 llama.cpp #31368
一致——只放行最近一批快照内的回退。shim 侧 `SA_N_RS_SEQ ≥ k` 即可，无需改 llama.cpp。

---

## 3. 验证成本：k 个候选只需一次绕链

`qlh_kh_forward_embd_tokens` 接受 k 行 hidden、返回 k 个位置的 argmax，所以验证 k 个候选只走
**一次**本机→ROG→回的往返。但摊薄不均匀：**lm_head（248320×2048）按行线性增长**（35B tail 段
实测 23.5 / 27.9 / 34.7 / 58.3 / 90.3 ms 于 k=1/2/4/8/16，边际 4.4-5.0 ms/行），层计算却能
被 batch 摊薄（MoE 专家共享）。**这是所有投机方案在本项目的成本上限。**

tail 与 head 的批量都与逐位置逐位一致（见 §9）。

---

## 4. MTP：首选

手上有一份带 MTP 的蒸馏变体
（`H:\qlh_models\qwen3-6-35b-a3b-opus47-distill-apex-mtp\`）：

```
qwen35moe.block_count          = 41      ← 40 主层 + blk.40
qwen35moe.nextn_predict_layers = 1
blk.40.nextn.{eh_proj,enorm,hnorm,shared_head_norm}
```

`common/speculative.cpp:1503-1506` 说明了它的档位：

```cpp
//   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
//   neither (qwen35 / qwen35moe): a single trained MTP head.
```

**qwen35moe 是单 head，一次只能预投 1 个 token，即 k=1。** 按 §0 的成本模型，k=1 是结构性
亏损（成本等于 plain 生成 2 个 token，却只多前进 r ≤ 1 个）。**所以 MTP 单头不值得接**——
这与早期判断相反，早期那笔账漏算了轮内的四次位置操作。

**接受率已实测**（head 0:32 + tail 32:40 + MTP，贪心，三个 prompt）：

| prompt | 接受 |
| --- | ---: |
| 中国的首都是 | 12/19 = 63.2% |
| 用一句话解释什么是层段推理 | 13/15 = 86.7% |
| `def quicksort(arr):` | 14/15 = 93.3% |
| **合计** | **39/49 = 79.6%** |

代码类最高（可预测），对话居中偏上，与 Qwen 官方的 60-80% 一致。

**subagent 实跑确认**：位置账修好后（先验证后写入、不回滚），两个 prompt 逐位一致、接受率
83.3% / 91.7%，但稳定**慢 13-14%**——正是 §0 预测的方向。

**部件已接通**：`qlh_kh_load` 在 `SA_MTP=1` 且模型带 NextN 时创建第二个 context
（`ctx_type = LLAMA_CONTEXT_TYPE_MTP`），配套 `qlh_kh_last_hidden` 取主图
`t_h_nextn`、`qlh_kh_mtp_draft` 用 `llama_batch_ext` 提交 `(token, hidden)` 并 argmax。
loader 侧的两个障碍也已修（`nextn` 保留、`n_layer_all` 减去 MTP 层）。

---

## 5. DFlash：可行，但第二步

权重可用：[z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)
（Q4_K_M 的 GGUF 仅 0.24 GB），llama.cpp 已支持该架构（`src/models/dflash.cpp`、
`LLM_ARCH_DFLASH`）。

从 GGUF 读到的关键参数：

```
dflash.block_count    = 6
dflash.block_size     = 16                              ← 一次出 16 个候选
dflash.target_layers  = [2, 7, 12, 17, 23, 28, 33, 38]  ← 需要目标模型的 8 个中间层
```

两个硬障碍：

1. **8 路中间层 hidden 横跨两段**。33 和 38 在 ROG 段，本机拿不到。要么让 ROG 回传，
   要么改 shim 支持多层输出（第一代 K-Llama 的 `layer_inp` 能做，第二代简化掉了）。
2. **llama.cpp 的 `draft-dflash` 要求 target 与 draft 在同一个 `llama_context`**。它直接读
   target 的中间层和 KV，没有"跨进程分段 target"的概念。这条路要自己在 Python 侧重写编排。

按实测成本估算（本机 batch=16 是 269 ms、tail 段 90.3 ms、传输 16×8 KB 约 40 ms）：

```
验证 16 个 ≈ 269 + 135(ROG 含内存压力) + 40 = 444 ms
接受率 20% → 3.2 个 → 139 ms/token（持平）
接受率 30% → 4.8 个 → 92 ms/token（1.5×）
接受率 40% → 6.4 个 → 69 ms/token（2.0×）
```

社区线索的接受率是 0.30–0.50。按 §0 的模型，k=8 且每草稿接受 16% 即回本：

```
接受 4/8 (50%)  → 1.30 单位/token，快 1.54×
接受 6.8/8 (85%) → 0.83 单位/token，快 2.40×
```

**这才是本项目的真机会**：MTP 单头的 k=1 结构上赚不了，DFlash 的 `block_size=16` 天生在
盈利区间。代价是工程量高一个量级（8 路中间层横跨两段、需要自己编排 target/draft）。

---

## 6. PLD / n-gram：按负载开关

llama.cpp 内置 5 种（`ngram-simple` / `ngram-map-k` / `ngram-map-k4v` / `ngram-mod` /
`ngram-cache`），核心是 `common/ngram-map.cpp:49` 的 `common_ngram_simple_draft`——纯函数，
不依赖 llama 上下文，Python 侧 20 行可移植。

在 2B 的真实 gold 序列上模拟（tokenizer 同为 248320，重复模式可迁移到 35B）：

| 负载 | 平均每步前进 m | 判定 |
| --- | ---: | --- |
| 中文自由对话 | **1.05–1.08** | 无效 |
| Markdown 表格 | 1.36 | 无效 |
| 中文代码生成 | 1.23 | 无效 |
| 代码续写补全 | **2.56–3.37** | 1.6–2.0× |
| 原样重复 / 抄写 | **2.67–4.92** | 1.5–2.6× |

**m 由文本重复性决定，与模型规模几乎无关**，盈亏线 m≥2.5。落地取**影子模式**：每步只统计
n-gram 命中与命中后能接受多长（零开销），确认某类请求 m>2.5 后再对它启用。

---

## 7. 两个被否掉的方向

**浅层投机（取前 k 层当草稿）**：用 2B 切出 head4/8/12，把 `token_embd` 单行反量化后经
embd 注入，逐位置 teacher forcing 与完整链的 gold 比较：

head4/8/12 的 α 分别 0.083 / 0.042 / 0.062，代码生成全为 0.000，E[accept]=1.00（等于没有
草稿）；对照组完整 24 层是 0.969 / 0.906，证明注入路径本身没问题。未训练的截断层接上最终的
`output_norm` + `lm_head`，分布是错位的。LayerSkip 那类效果来自训练时的 early-exit loss 加
layer dropout。

**dSpark**：机制是外挂草稿（DFlash 骨干）+ Markov 头 + 置信度调度，llama.cpp 用
`markov_w1.weight` 张量识别。开源权重只有 DeepSeek-V4-Pro/Flash 系，没有 Qwen 版本，
自训成本远超收益。

---

## 8. 落地顺序

1. **`n_rs_seq` 可配 + 按序回滚**。已完成：`SA_N_RS_SEQ`、`qlh_kh_seq_rm`。
2. **批量验证链路**。已完成：`hidden_tokens` op 贯通到 `StageClient.forward_tokens_out`。
3. **MTP 部件**。已完成（loader 保留 nextn、独立 context、`qlh_kh_mtp_draft`）。**但不接主链路**：
   k=1 结构性亏损。
4. **修两个数值障碍**（见 §9）。不修则任何 k≥2 的投机都无法保证输出一致。
5. **PLD 影子模式**统计真实负载命中率，达标后才对该类请求启用。
6. **DFlash**：k=8 天生在盈利区间，是唯一值得投入完整工程的方向。

## 9. 两个数值障碍

位置账已修好（`checks/probe_spec_wip.py`，先验证后写入模式下 2 个 prompt 逐位一致、
接受率 83-92%），但**真投机（批量验证 + 拒稿回滚）卡在两个底层行为上**：

1. ~~head 的多 token 批量 ≠ 逐位置~~ —— **已推翻，批量验证是干净的**。

   hidden 确实差 2.5e-2~3.8e-2（相对 4e-3），但**不翻转 argmax**。用「完美草稿」（块的第 2 行
   填单步真实会产出的 token，KV 无污染）跑 30 个 token，2B 与 35B 都**逐位一致**，块的第 1 行
   （`tb[0]`，验货用）与第 2 行（`tb[1]`，免费 token）**都对**。之前两次「批量出错」都是实验
   设计问题：一次 filler 塞了重复 token 污染 KV，一次 `rebuild()` 重放时 token 序列不一致。
   `checks/probe_perfect_draft.py` 是干净的端到端版本（30 token 全序列对比）。
2. **`qlh_kh_seq_rm` 不是干净回滚** ← **这是唯一剩下的障碍**。单 token 写完 → rm → 同输入
   重写同一位置，hidden 差 0.13~0.62（相对 2~10%），token 偶发翻转（8 个位置里 3 次）。
   残余误差会污染后续所有 hidden，把接受率从 88% 打到 50%。

   机制已梳理：`llama_memory_recurrent::seq_rm` 只放行 `rollback ≤ n_rs_seq` 且要求
   `rs_idx == 0`（快照索引 single-use，`s_copy` 读完即清零）；快照本身由 graph 在每次
   decode 时把当前状态拷进 `rs_idx * size` 起始的槽位。hybrid 的分发是正确的（先 recurrent、
   失败即中止，再 attention）。**所以问题不在"要不要回滚"，而在快照恢复出来的状态是否与
   当初逐位相同**——这正是要查的。

3. **官方投机的假设在这里不成立**。`common/speculative.cpp:2726` 把 `n_rs_seq` 设为 **0**，
   因为它的 target 是纯 transformer（KV 按位置删就够）。而 `qwen35moe` 的 tail 段 8 层里
   **6 层是 linear**（32/33/34/36/37/38，只有 35/39 是 full attention），recurrent state
   没有位置索引。**hybrid 模型 + 分段 target 的投机是官方没有覆盖的场景**，没有现成答案。

## 10. 已知限制

- **只对 temp=0 无损**。shim 暴露 argmax 而非 logits 分布，无法做标准 rejection sampling。
- **ROG 内存余量**：0.82 GiB 可用，`n_rs_seq` 占 100 MiB（§2）。
- **未实测**：DFlash 真实接受率、k≥2 端到端收益。
