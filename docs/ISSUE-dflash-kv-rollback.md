# ISSUE：DFlash 多轮生成在第二轮撞 KV 位置（seq_rm 返回 -6）

> 状态：已定位到"回滚与占位不一致"，但两个观测互相矛盾，未收敛
> 更新日期：2026-10-31
> 影响面：P4 验收阻塞（P1/P2/P3 均已通过）

## 现象

两段拓扑（head 0:32 + tail 32:40）上的 DFlash 轮，**单轮完全正常**：

```
head layer_inp (6, 5, 2048)
tail layer_inp (2, 5, 2048)
concat rows = (5, 16384)
draft_inject rc=0
draft_block -> 15 drafts
verified 16 tokens, accepted prefix = 1
PASS
```

但把它循环起来（`check_p4_generate.py`），**第二轮**就崩：

```
stage 0 (gpu): forward failed rc=-2: llama_decode failed rc=-1
   (tokens: n_tokens=1, n_past=21)
```

## 两个互相矛盾的观测

**观测 A**：`llama_decode` 在 `n_past=21` 失败，说明 head 的 KV 里**位置 21 已被占用**。

**观测 B**：紧接着 `seq_rm(21)` 在两个 stage 上都返回 `-6`：

```
head: [seq_rm] p0=21 rc=-6
tail: [seq_rm] p0=21 rc=-6
```

`-6` 来自 shim 的 `qlh_kh_seq_rm`：`llama_memory_seq_rm(...)` 返回 false，即**那段范围里没有可删的行**。

如果位置 21 真的被占用（A），`seq_rm(21)` 就该删掉它（B 应为 0）。两者不可能同时成立，除非
**"占用"和"可回滚"指的不是同一段 KV**，或者 `n_past` 的记账与 `batch.pos` 的实际落点错位。

## 已确认正确的部分

- 单轮成功，说明 8 路拼接、注入、出块、验证、接受判定整条链路都对
- `n_past` 记账经过一次修正（`keep = n_past + accept`，见 `check_p4_generate.py` 的注释）
- `seq_rm` 的参数顺序正确：`qlh_kh_seq_rm(handle, seq_id=0, p0=n_past)`，且 ctypes argtypes 已声明
- 两段都调了 `seq_rm`（早先只回滚 tail 时症状相同，所以不是"漏了一段"）

## 下一步该查什么

1. **`forward_tokens(drafts, n_past)` 的实际落点**：在 shim 里打印 `batch.pos[0]` 和
   `batch.pos[n-1]`，与调用方传的 `n_past` 对齐
2. **`qlh_kh_forward` 是否真的把 15 行提交进 KV**：`llama_decode` 的返回码在 head 段是
   被吞掉的（`binding.tokens` 只看 rc），确认它到底是 0
3. **`llama_memory_seq_rm` 的 `-6` 语义**：在 hybrid 模型（Gated Delta Net + full attention）
   上，linear-attention 层的状态回滚与 KV 回滚是不是两套机制——DFlash 这条链上
   `SA_N_RS_SEQ` 的快照回滚可能才是真正管 linear 层的那条路
4. 怀疑点：**head 段是 `kind="gpu"` 且开了 `SA_NEEDS_LOGITS=1`**，它的 dense hidden 出口
   与 `cparams.embeddings` 互斥；如果 `forward_tokens` 只有 hidden 出口失效但 KV 写入成功，
   那 A 成立而 B 的 -6 就另有原因（比如位置用的是 tail 的坐标系）

## 工作区状态

- `check_p4_generate.py`：多轮驱动，卡在第二轮
- `check_p3_two_seg.py`：单轮，**PASS**（这是当前可信的基线）
- 所有 Python 侧改动都已提交（`735eb7f4`），shim 侧 `89cb4253`
