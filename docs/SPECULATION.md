# 投机解码：四个方向的判定

> 状态：已实测，未接入
> 更新日期：2026-10-30

本文记录四种投机方案在本项目上的可行性判定。每条都带实测依据；没有实测的会明确标注为估计。
适用前提：Qwen3.6-35B-A3B（Q4_K_XL，40 层），两段跨机（本机 0:32、ROG 32:40），
当前 **140 ms/token**。

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
| MTP（模型自带） | **代码场景 1.19×** | k=1 亏；**递归 k=2** 在代码上接受 1.60，达盈亏线 1.18 |
| DFlash（外挂 0.24 GB） | **唯一值得投入** | k=8/16 在盈利区间，每草稿接受 16% 即回本 |
| PLD / n-gram | 按负载开关 | 中文对话 m=1.05（无效），代码续写 m=2.56–3.37 |
| 浅层投机（前 k 层） | 不做 | 实测 α=0.00–0.08，E[accept]≈1.0 |
| dSpark | 不做 | 开源权重只有 DeepSeek-V4 系，无 Qwen 版 |

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

**qwen35moe 是单 head。k=1 结构性亏损**（成本等于 plain 生成 2 个 token，却只多前进 r ≤ 1 个），
实测慢 28%。但**递归可以让它投多个**：同一个 hidden 反复喂自己的输出，`(h_p, d1) -> d2`。

递归的接受率按层衰减，实测（50 token，k=2）：

| 场景 | 第 1 层 | 第 2 层 | 平均接受 | 成本 4.36 ÷ 前进 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| **代码**（`def quicksort`） | 85.1% | 87.5% | **1.60** | 1.68 | **快 1.19×** |
| 文档（Markdown 表头） | 62.2% | 65.2% | 1.03 | 2.15 | 慢 7% |
| 对话（`中国的首都是`） | 60.0% | 73.3% | 1.24（k=4） | >2.0 | 慢 |

k=2 盈亏线是 **1.18 个草稿**，只有代码场景达标。第 3 层掉到 36%、第 4 层 17%——**递归上限
就是 2**。MTP 的定位是「代码续写 1.19×」，不是通用加速，必须按负载开关（§6）。

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

按 §0 的模型，k=8 且每草稿接受 16% 即回本。社区线索的接受率是 0.30–0.50：

```
接受 4/8 (50%)  → 1.30 单位/token，快 1.54×
接受 6.8/8 (85%) → 0.83 单位/token，快 2.40×
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
3. **MTP 递归 k=2**。部件已完成；代码场景实测 1.19×，按负载开关接入（§4）。
4. **回滚已修**（§9）。剩下的跨形状 kernel 差异不影响 argmax。
5. **PLD 影子模式**统计真实负载命中率，达标后才对该类请求启用。
6. **DFlash**：k=8 天生在盈利区间，是唯一值得投入完整工程的方向。

## 9. 数值验证结论

位置账在 `checks/probe_spec_wip.py`（先验证后写入模式下 2 个 prompt 逐位一致、接受率 83-92%）。
两个曾经的障碍都已澄清或修掉：

1. ~~head 的多 token 批量 ≠ 逐位置~~ —— **已推翻**。hidden 差 2.5e-2~3.8e-2 但不翻转 argmax：
   用「完美草稿」跑 30 个 token，2B 与 35B 都逐位一致，块的第 1 行（验货）与第 2 行（免费
   token）都对（`checks/probe_perfect_draft.py`）。
2. ~~`qlh_kh_seq_rm` 不是干净回滚~~ —— **已修复**，详见
   [`ISSUE-recurrent-snapshot-rollback.md`](ISSUE-recurrent-snapshot-rollback.md)。根因是
   `n_rs_seq` 快照**只有「单 ubatch 内」语义、跨 ubatch 不滚动**：单 token decode 每步只写
   slot 0，`seq_rm` 后重写就读到陈旧的 slot 1。修法是在 `build_rs` 里插移位节点让快照跨
   ubatch 连续。实测同形状回滚 **0.000e+00**。跨形状仍有 ~1e-2，是 GDN 两套 kernel
   （CH/AR）的算术差异，与回滚无关、不影响 argmax。

   **官方从不跨 ubatch 回滚**（verify batch 一次写 `n_draft+1` 个 token，回滚量总落在刚写的
   槽里；draft 端 `n_rs_seq=0`、整段重跑），所以这个缺陷只有「分段 + 单 token draft」才暴露。

3. **tail 段 8 层里 6 层是 linear**（32/33/34/36/37/38，只有 35/39 是 full attention），
   recurrent state 没有位置索引——这是上面那条缺陷只在本项目暴露的根本原因。

## 10. 接入状态（未完成）

已接进 `chain_client.py`（`--spec-k N`），`forward_batch` 一次绕链验证整个候选块。本机两段
（`topology-local-32-8.toml`）实测：**正确性通过**（k=2 与 k=0 的 token 序列逐位相同），接受率
83.3%，全接受时一轮前进 4 个。**但速度没优势**（baseline 119-125 ms/token，投机持平或更慢），
**且 k=2 约半数运行会在第一次 `forward` 前让 gpu 段 worker 退出**，同配置 baseline 4/4 稳定，
根因未定位；`回滚失败 2` 稳定复现、长跑（gen=32）会崩。

**崩溃与回滚都已修**（`qlh_kh_seq_rm` 漏声明 ctypes 签名 → 64 位句柄溢出打死 worker；
`n_rs_seq` 从 k-1 改成 k）。修完 6/6 + 1/1 通过，token 与 baseline 逐字节相同。

**速度仍未测出**：本机 head 段 WDDM 溢出（baseline 就掉了一半性能），跨机又卡在传输——
局域网 SSH 通了（48 MB/s），但本机走 **WLAN**，22 GiB 连续传输跑几块就停；且 `copy /b`
合并出的文件**大小对、内容错**（256 MiB 切片哈希：head 相同，mid/tail 不同）。ROG 上那份
损坏的模型会让 tail 段恒定输出 token 0（本机同配置正常）。

**所以投机仍不进主链路**，`--spec-k` 默认 0。缺的是一个**有线网络 + 单机装得下 22 GiB** 的
运行环境，不是代码。

## 11. 已知限制

只对 temp=0 无损（只暴露 argmax，做不了 rejection sampling）；ROG 只有 0.82 GiB，`n_rs_seq` 占
100 MiB；DFlash 接受率与 k≥2 端到端收益未实测。
