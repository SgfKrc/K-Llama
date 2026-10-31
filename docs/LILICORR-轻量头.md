# 给 DFlash 加一个重排头（LiLiCorr）

> 状态：**方案（未实施）**；推理侧已上传，训练侧在本机不可复现
> 更新日期：2026-10-31
>
> 适用范围：DFlash 草稿侧"块内不一致"的改造路径。
> 与 [`SPECULATION.md`](SPECULATION.md) 的分工：那篇回答"投机该不该做、哪个方向"，
> 这篇回答"DFlash 之后还有没有便宜的第二刀"。论文全文与交叉验证材料在
> `local_docs/research-lilicorr/`。

---

## 0. 结论

1. **推理侧已经就绪。** 上游 PR [ggml-org/llama.cpp#30506](https://github.com/ggml-org/llama.cpp/pull/30506)
   已于 2026-10-15 合入（13 文件 +649/−22），本项目 fork（`H:\llamacpp-c`）里全部代码都在。
   用法仍是 `--spec-type draft-dflash`，不需要新的投机类型。
2. **没有可用的权重。** HuggingFace 上 13 个 `lilicorr` 仓库全是 NVIDIA Nemotron 系；
   Qwen 系只有一个第三方 `Qwen3.8-27B` 的 epoch0 中间产物，自述未做 serving 验证。
   `Qwen3.6-35B-A3B` 没有任何 LiLiCorr 权重。
3. **训练在本机不可行，瓶颈不在头。** 头只有约 30–37M 参数，但论文用 704 张 H100 训
   （88 节点 × 8）、1.4M 对话、6 epoch；第三方用 `Qwen3.8-27B` 复现的配方是 8×A100 跑
   约 17 天 / 2 epoch、数据 65 GB。真正吃资源的是**用 35B target 采集 hidden**。
4. **k 扫描已经做完（2026-10-31，本链路实测）。** 最优是 **k=5**：64-token 生成下
   5.82 token/轮、20.9 tok/s、接受 96.4%，比原先默认的 k=3（3.94 token/轮、17.0 tok/s）快
   约 20%。再往上崩塌：k=8 接受率 31.1%、k=15 只有 11.1%，两者都比 k=3 慢。
5. **所以现在不训、不接。** LiLiCorr 的收益集中在深 k，而深 k 在本链路上已经被验证成本
   吃掉了：k=5 之后每多一行验证的边际成本，超过它带来的接受概率。权重等社区，训练等租卡。

---

## 1. 问题：逐位置训练，块内不相关

DFlash 一次前向出 16 个位置，每个位置单独算 cross-entropy。这约束了每个位置的**边缘分布**，
没有约束位置之间的**联合分布**。本项目的实测曲线正好是这个缺陷的形状
（同一 prompt、`llama-speculative-simple`、CPU）：

| n_max | 代码接受 | 对话接受 |
| ---: | ---: | ---: |
| 3 | 98.0% | 61.1% |
| 5 | 96.4% | — |
| 15 | 36.0% | 20.3% |

深 k 下接受率断崖，不是模型不会预测，是**块内第一行就不一致**，后续全被验证器否掉。

## 2. LiLiCorr 做什么

不重新预测 token，而是**给已有候选重排**：

1. **候选格**：保留草稿每位置 top-K。论文取 B=16（15 个草稿位）、K=8 → 120 个节点。
2. **单次打分**：每个候选的输入是四项等宽向量之和——target 冻结的 token embedding、
   DFlash 在该位置的 hidden、草稿分布的 5 维特征（概率、相对 logprob、名次、是否第一）、
   位置与名次的 embedding。一个 2 层 Transformer 对整格做全注意力，一次前向出结果。
3. **两向量耦合**：每个候选输出一个 out 向量与一个 in 向量，相邻候选 `a→b` 的分数是
   `cos(a.out, b.in) × σ`。整块 904 个相似度 `((B-2)K² + K)` 全是并行矩阵乘。

解码时按分数贪心走格，**不再调用网络权重**。所以增量成本与 target 的 40 层无关。

**论文的几何**（Table 4）：d_e=1024、σ=8.0、2 层、hidden 1024、8 heads、FFN 2048、
头参数量 36.76M（Qwen3-8B target）/ 32.04M（4B）。折算到本项目的 `n_embd=2048`
（8B 的一半，三个输入投影各减半），约 **30M 参数、fp16 约 60 MB**。

**训练目标**：真值前驱条件下，K 个候选的相似度过 softmax 做交叉熵；
监督只在真值仍在候选池内时生效。论文是**联合训练**——DFlash 的 cross-entropy 权重 1.0、
重排目标权重 γ=0.25，所以 backbone 同时被更新，不是冻结点。

## 3. llama.cpp 侧已就绪（源码证据）

| 位置 | 内容 |
| --- | --- |
| `src/llama-arch.cpp:422-429` | 8 个元数据键 `lilicorr.{hidden_size,candidate_topk,num_heads,num_layers,vector_eps,logit_scale,factor_dim,mlp_hidden_size}` |
| `src/llama-arch.cpp:829-841` | 13 个张量名，含 `lilicorr.layers.%d.attn.in_proj` |
| `src/models/dflash.cpp:36-47` | 元数据解析与校验（`n_embd % num_heads`、`topk*(topk+1) <= n_embd`） |
| `src/models/dflash.cpp:216-273` | 全部张量形状 |
| `src/models/dflash.cpp:739-897` | `build_lilicorr_scores()`：候选格、特征拼接、2 层 Transformer、out/in 向量、相似度打包 |
| `src/models/dflash.cpp:1201` | 触发条件：`lilicorr_pass_hidden_proj && ubatch.embd_state && n_tokens == block_size * n_seqs` |
| `common/speculative.cpp:1070-1079` | 读 `lilicorr.candidate_topk`，按 `n_seq × (block_size+1)` 分配跨序列槽位 |
| `conversion/nemotron.py:19-114` | `LiLiCorrDraftModel(DFlashModel)`，写全部 lilicorr 键 |
| `gguf-py/gguf/gguf_writer.py:1291-1313` | 8 个 `add_lilicorr_*` 写入器 |

三个落地约束：

- **必须走 `--target-model-dir`**，因为词表与 target 的 token embedding 从 target 取
  （`conversion/nemotron.py:65`）。
- **转换器吃的是已合并的 HF checkpoint**，即训练阶段就要把 backbone 与头存成一份权重。
  上游没有"DFlash 权重 + 独立头文件"的拼装脚本。
- **必须关 backend sampling**（`common/speculative.cpp:1125` 的 `!is_lilicorr` 条件）。

## 4. 权重现状

| 来源 | 内容 |
| --- | --- |
| `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` 等 | 13 个仓库，全部 Nemotron target，配不到本项目 |
| `shanjiaz/Qwen3.8-27B-LiLiCorr-block8-epoch0` | 第三方、Qwen target、epoch0 中间产物，未做 serving 验证 |
| `Qwen3.6-35B-A3B` | **无**。精确检索 `35B-A3B-LiLiCorr` 命中 0 |
| 我们自己 | 只有纯 DFlash 的 Q4_K_M GGUF（`H:\qlh_models\dflash\`），无 `lilicorr.*` 张量 |

## 5. 训练成本

**论文口径**（Table 4）：AdamW、lr 6e-4、weight decay 0、4% warmup、grad clip 1.0、
global batch 64、序列长 3120、每序列采 512 个 anchor、6 epoch、
1,397,048 条对话（Nemotron Post-Training V2 的 code/math/STEM/chat，每个 assistant turn
用 target 重新生成）、130,968 优化步、bf16 + fp32 master、**88 节点 × 8 H100**。
论文没给 wall-clock。

**第三方 Qwen 配方**（`vllm-project/speculators` 的 lilicorr 分支，`Qwen3.8-27B` 全套可复现）：
8×A100-80GB、7.65–8.15 s/step、约 8.3 天/epoch、2 epoch ≈ 17 天、约 185k 步、
trainer 显存 43–45 GiB/GPU、数据集 67 个 arrow 文件约 65 GB。

**为什么本机不行**：头本身是 30M 参数、单卡训得动；卡住的是 **target 前向采集**。
训练 loss 硬性需要 target 的 hidden（`hf_lilicorr.py:216`）——论文取的是产生最后一个
已验证 token 的那组 hidden，跨五个捕获层 collapse；distractor 项还要 target 对候选的 logits，
可以从捕获的 hidden 投影出来，但前提是把 target 的 `lm_head` 权重也带上。
35B-A3B 即使 Q4 也约 20 GiB，本机 4060 装不下，CPU 推理 3–8 tok/s；按 2,000 条 × 800 token
采集是数天量级，缓存还要 3–6 GB。数据量比论文少两个数量级时，接受率增益没有依据。

## 6. 收益判据

**实测的 k 曲线**（2026-10-31，`scripts/check_p4_cross.py`，32:40 链路，prompt 为
`def quicksort(arr):`，只计解码阶段、不含 prefill 与协议开销，同一 prompt 同一 worker）：

| k | 生成 64 token | | | | 生成 128 token | | | |
| ---: | 接受率 | token/轮 | ms/轮 | tok/s | 接受率 | token/轮 | ms/轮 | tok/s |
| 3 | 98.0% | 3.94 | 232 | 17.0 | 74.2% | 3.23 | 263 | 12.2 |
| **5** | **96.4%** | **5.82** | **278** | **20.9** | **70.3%** | **4.52** | **338** | **13.4** |
| 8 | — | — | — | — | 31.1% | 3.49 | 364 | 9.6 |
| 15 | — | — | — | — | 11.1% | 2.67 | 473 | 5.6 |

两件事同时成立：

- **最优 k 是 5，不是更大。** 每多验证一行，tail 段的 `lm_head`（248320×2048）就多一份
  线性成本；k=8 之后多出的草稿几乎全被拒，成本照付。
- **接受率强烈依赖生成区段。** 同一 prompt 前 64 token 是 96–98%，写到 128 token 时
  掉到 70–74%——代码的开头可预测、后段不可预测。所有"接受率"数字都必须带生成长度口径，
  否则 98% 和 74% 会被当成矛盾。

**LiLiCorr 落在哪一格**：它的收益集中在深 k（论文自己的 ablation 也是 K=8、B=16）。
在本链路上，能受益的区间（k≥8）本来就已经输给 k=5 一大截：

| 假设 | k=5（现状） | k=8 + 论文上限 +19% AL |
| --- | ---: | ---: |
| token/轮 | 4.52 | 3.49 × 1.19 = 4.15 |
| 吞吐（128 token 口径） | 13.4 | ~11.4（验证成本不变） |

即使按论文最好的数字，重排后的 k=8 仍然不如不训的 k=5。要让重排赢，它得把 k=8 的
接受率从 31.1% 拉到 45% 以上——那是 +45% AL，超出论文给的 +9~19% 区间。

论文口径（相对同数据同预算训练的 vanilla DFlash）：接受长度 +9~19%、贪心吞吐 +4~13%
（Qwen3-8B/4B，单 H100，concurrency 1）。本项目验证成本结构更差（tail 段 `lm_head` 按行
线性），收益只会更低。

## 7. 最短路径

1. ~~零成本的 k 扫描~~ —— **已完成**，结论见 §6：默认改为 k=5，
   `scripts/check_p4_cross.py` 的 `SA_P4_BLOCK` 默认值已同步改掉。
2. **要训就租卡**，不在本机硬扛：1–2 张 80 GB 卡做 target hidden 抽取（采集与训练分离，
   `hf_lilicorr.py` 有 `data.mode: offline` 开关），单卡训头。
3. **数据**：采我们自己的负载（代码 + 中文对话），不必复刻 Nemotron 语料；
   但量级要贴近论文才有可比性，样本量差两个数量级时只能看方向。
4. **合成与转换**：把训好的 `model.lilicorr.*` 与 DFlash backbone 存成一份 HF 权重，
   按 `nemotron.py` 的 `LiLiCorrDraftModel` 分支适配 Qwen config 名，
   `convert_hf_to_gguf.py --target-model-dir <35B HF 目录>`。
5. **验收**：逐位判据不适用（重排改变草稿序列，但验证器仍决定最终 token）。
   判据是接受率与吞吐；正确性靠"贪心输出与 k=0 逐位一致"守。

## 8. 不做什么

- **不做 DSpark**：backbone 就是 DFlash，但 anchor-first 的训练布局不同，要从头训 backbone，
  8 卡 + 38 TB 缓存（`SPECULATION.md` §7）。
- **不做 Domino**：逐位置 correction 网络，把并行草稿的延迟优势抵消掉，与 LiLiCorr 是
  同一问题的对立取舍。
- **运行时不留 PyTorch**：训练是离线步骤，产出是 GGUF；链路仍然只有 llama.cpp + shim。
- **不在 k=3 上做**：上限 +1.5%，低于测量噪声。
- **不复刻论文规模**：704 张 H100 的配方在本项目是背景信息，不是执行计划。
