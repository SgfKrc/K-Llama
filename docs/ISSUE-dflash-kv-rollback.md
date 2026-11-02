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


## 2026-10-31 续：又前进两层，仍卡在 DFlash 自己的 KV

### 已修复

**1. 主链的 seq_rm 现在生效**（根因：`n_rs_seq` 不够）

```
llama-memory-hybrid.cpp:144   recurrent 删不掉 -> 整体 return false（attention 也不删）
llama-memory-recurrent.cpp:211  部分回滚要求 rollback <= n_rs_seq
chain_client.py:539            SA_N_RS_SEQ 默认 = max(1, spec_k) = 1，而 DFlash 要回滚 15
```

设 `SA_N_RS_SEQ=16` 后：`seq_rm p0=21 removed=1`（原 0），第二轮 KV 轨迹连续正确。

**2. `draft_open` 里 `dparams.n_rs_seq = 16`**（DFlash 自己的 ctx 也要够）

**3. `draft_rollback` 的 -6 视为 no-op**（与 `seq_rm` 同一约定）

**4. `draft_kv` 记账**：draft 只接收**注入**的行（prompt + 每轮 anchor），草稿不入 KV；
   回滚目标应与 tail 一致 = `n_past + accept`，而不是 `draft_kv - drop`（后者第二轮变成 -8）

**5. `draft_rollback` 必须每轮都做**：`draft_block` 自己会往 draft KV 写 `block_size` 行
   （anchor + 15 mask），不因"全接受"而跳过——跳过时 KV 停在 36，下轮 inject 撞 21

### 当前卡点

修完上面 5 条后：

```
[kh.draft_rollback] n_keep=21 rc=0 handle=True        ← 返回成功
- the last position stored in the memory module ... is X = 36   ← 但 KV 仍在 36
```

`qlh_kh_draft_rollback` 返回 0（即 `llama_memory_seq_rm` 返回 true），但 attention 那半的
KV 没有被截断。DFlash 的 memory 类型需要确认：如果它走 `llama_memory_hybrid`，那么
`seq_rm` 里 `mem_recr->seq_rm` 返回 true 之后必须再调 `mem_attn->seq_rm`——目前的结果
说明后者没生效，或者 draft 用的是别的 memory 实现。

### 下一步

1. 在 `qlh_kh_draft_rollback` 里分别打印 `mem_recr` / `mem_attn` 的 `seq_rm` 结果
   （需要 `llama_memory_hybrid` 的 getter，或者直接看 `llama_get_memory` 的类型）
2. 若 DFlash 的 attention 子缓存确认没被截断，考虑改用 `llama_memory_seq_rm` 之后
   再显式 `llama_memory_clear` 或重建 draft ctx（代价可接受：draft ctx 只有 256 上下文）
3. 备选：把 draft 的 KV 上下文压到一轮的量级（每轮重建），牺牲增量但保证正确
