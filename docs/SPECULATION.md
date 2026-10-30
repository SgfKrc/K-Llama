# 投机解码：四个方向的判定

> 状态：已实测，未接入
> 更新日期：2026-10-30

本文记录四种投机方案在本项目上的可行性判定。每条都带实测依据；没有实测的会明确标注为估计。
适用前提：Qwen3.6-35B-A3B（Q4_K_XL，40 层），两段跨机（本机 0:32、ROG 32:40），
当前 **140 ms/token**。

---

## 1. 判定汇总

| 方向 | 判定 | 依据 |
| --- | --- | --- |
| MTP（模型自带） | **首选** | 单 head，验证只需 batch=2，均摊 61–69 ms（2.0–2.3×） |
| DFlash（外挂 0.24 GB） | 可行，第二步 | block 16 个候选，验证 444 ms，接受率 ≥20% 即回本 |
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

内存代价已实测：单快照 50.25 MiB（本机 0:32 段的 24 层 linear），按层数线性缩放。

| 快照数 | 本机（24 层 linear） | ROG（6 层 linear） |
| ---: | ---: | ---: |
| 8 | 402 MiB | 100 MiB |
| 16 | 804 MiB | 201 MiB |

本机 22 GiB 可用毫无压力；ROG 只剩 0.82 GiB，100 MiB 能承受但会加剧已有的内存压力
（见 `docs/DEPLOYMENT.md` 的跨机实测）。llama.cpp 侧机制齐备（PR #26499/#28118/#31350/#31368），
shim 只需开 `n_rs_seq ≥ k` 并导出按序列回滚。

---

## 3. 验证成本：k 个候选只需一次绕链

`shim/qlh_keep_head_gpu.c` 的 `qlh_kh_forward_embd_tokens` 接受 k 行 hidden、返回 k 个
位置的 argmax，实测与逐位置调用逐位一致。所以验证 k 个候选走**一次**本机→ROG→回的往返。

但成本的摊薄并不均匀。**lm_head（248320×2048）按行线性增长**，这是 35B tail 段实测：

```
k=1    23.5 ms
k=2    27.9 ms   (+18.7%)
k=4    34.7 ms   (+47.6%)
k=8    58.3 ms   (+147.5%)
k=16   90.3 ms   (+283.8%)     边际 4.4-5.0 ms/行
```

层计算能被 batch 摊薄（MoE 专家共享），lm_head 每行都要过一次完整词表投影，因此保持线性。
**这是所有投机方案在本项目的成本上限。**

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

**qwen35moe 是单 head，一次只能预投 1 个 token。** 但这恰好是最划算的形态——
验证只需 batch=2：

```
验证成本 = 本机 batch=2 (81.8 ms) + ROG (约 28 ms) ≈ 110 ms
接受率 60% → 前进 1.6 个 → 69 ms/token (2.0×)
接受率 80% → 前进 1.8 个 → 61 ms/token (2.3×)
```

**技术障碍**：`LLAMA_LAYER_RANGE` 目前强制把 `nextn` 覆盖成 0（loader 日志可见），
即 C 方案把 MTP 层关掉了。要让 tail 段保留 MTP，需要改 `llama-model-loader.cpp` 的
range 解析：当 `hi == n_layer_all` 时不覆盖 `nextn`。

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

社区线索的接受率是 0.30–0.50，即 **1.5–2×**。收益与 MTP 相当，但工程量高一个量级。
**先做 MTP，跑通基础设施后，DFlash 只需替换草稿器这一层。**

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

**m 由文本重复性决定，与模型规模几乎无关。** 盈亏线 m≥2.5（因为验证成本里 lm_head 占大头）。

落地形态取**影子模式**：每步只统计 n-gram 是否命中、命中后接受多长（零开销），
确认某类请求的 m>2.5 后再对它启用。最坏情况零收益，不会因为"命中但第 1 个就被拒"变慢。

---

## 7. 两个被否掉的方向

**浅层投机（取前 k 层当草稿）**：用 2B 切出 head4/8/12，把 `token_embd` 单行反量化后经
embd 注入，逐位置 teacher forcing 与完整链的 gold 比较：

| 草稿 | 中文对话 α | 代码 α | E[accept] |
| --- | ---: | ---: | ---: |
| head4 | 0.083 | 0.000 | 1.00 |
| head8 | 0.042 | 0.000 | 1.00 |
| head12 | 0.062 | 0.000 | 1.00 |
| 完整 24 层（对照） | 0.969 | 0.906 | — |

对照组 0.97 证明注入路径本身没问题。未训练的截断层接上最终的 `output_norm` + `lm_head`，
分布是错位的。LayerSkip 那类效果来自训练时的 early-exit loss 加 layer dropout。

**dSpark**：机制是外挂草稿（DFlash 骨干）+ Markov 头 + 置信度调度，llama.cpp 用
`markov_w1.weight` 张量识别。开源权重只有 DeepSeek-V4-Pro/Flash 系，没有 Qwen 版本，
自训成本远超收益。

---

## 8. 落地顺序

1. **开 `n_rs_seq` + 导出按序列回滚**（约 30 行 C）。这是所有方案的前置条件。
2. **批量验证接口贯通**（shim 已有，Python 侧缺 `hidden_tokens` op 与 tail 段的 list 返回，
   约 180 行）。
3. **换带 MTP 的整模 + 改 loader 保留 nextn**，接 `draft-mtp`。预期 2.0–2.3×。
4. **PLD 影子模式**统计真实负载命中率，达标后才对该类请求启用。
5. **DFlash** 作为最后一步，需要多层 hidden 输出与自定义编排。

## 9. 已知限制

- **只对 temp=0 无损**。shim 暴露的是 argmax 而非 logits 分布，无法做标准 rejection sampling。
  项目默认贪心（`chat.py`），当前不构成问题。
- **ROG 的内存余量**。0.82 GiB 可用，`n_rs_seq` 要占 100 MiB，会加剧已有的 22.7 ms 压力。
- **未实测项**：ROG 的 batch 曲线（按本机外推）、DFlash 的真实接受率（只有社区线索）、
  `n_rs_seq` 非零值在现 DLL 上能否正常初始化。
