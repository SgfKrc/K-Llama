# ISSUE：DFlash 的 warmup 空跑必崩（`--no-warmup` 完全正常）

> 状态：已定位到"是 warmup 特有的问题、不影响正常前向"，但**根因未完全定位**
> 更新日期：2026-10-31
> 影响面：低（我们的链不走这条路径）

## 现象

把 target 的 `token_embd.weight` + `output.weight` 合并进 DFlash GGUF 之后，draft 模型能
独立加载了（不再需要 `ctx_other`）。但用 `llama-cli` / `llama-server` 跑的时候：

```
I common_init_: warming up the model with an empty run - please wait ... (--no-warmup to disable)
<进程 AV，exit = 0xC0000005 (-1073741819)>
```

**加 `--no-warmup` 就完全正常**：

```
EXIT=0
> hi
[Prompt: 215.0 t/s]
```

## 已排除的假设

| 假设 | 验证方式 | 结果 |
|---|---|---|
| 合并后的 GGUF 结构坏了 | 69 个原张量逐张量 sha256 比对；data 区整块 sha256 比对 | **全部相同**，否决 |
| 张量宽度不匹配（`n_embd_out` 2048 vs `n_embd_inp_enc` 16384） | `common_batch_set_zero_state` 改用 `llama_model_n_embd_inp_enc` | **仍然崩**，否决 |
| warmup 的 `embd_state` 不该设给 draft 模型 | 对 `n_embd_inp_enc > 0` 的模型跳过 `set_zero_state` | **仍然崩**，否决 |
| `n_ctx` 太小 | `-c 512` / `-c 2048` 都试过 | 都在 warmup 处崩，否决 |

## 已知的精确位置

`-v` 日志里，崩溃前最后一行是：

```
graph_reserve: reserving a graph for ubatch with n_tokens = 128, n_seqs = 1, n_outputs = 128
input tensor 'inp_pos' [i32, ne = { 128, ... }]
```

也就是说崩在 **`graph_reserve`/图构建**阶段，紧随 `inp_pos` 之后、`build_attn` 之前。

而 `--no-warmup` 走同样的 `graph_reserve`（也有 `n_tokens=128` 那次）**却不崩**。两者的差别只剩
warmup 那次 decode 的 **batch 内容**（BOS + EOS，2 个 token）与位置。

## 为什么可以先不修

1. **`--no-warmup` 是官方参数**（`common_params.warmup`），不是绕过补丁。
2. **本项目的链完全不经过它**：`stage_worker.py` 用 shim（`qlh_kh_load` + 显式 forward），
   不走 `common_init_from_params` 的 warmup。
3. `examples/speculative-simple` 与 `llama-server` 的 draft 路径也都正常（实测 95% 接受率）。

## 如果以后要修

下一刀应该切在 `src/models/dflash.cpp` 的 decoder graph：查 `n_tokens != block_size` 时
（warmup 给 2）与 `n_tokens == block_size`（16）时的分支差异，特别是
`model.lilicorr_pass_hidden_proj && ubatch.embd_state` 那个条件（L1031）——warmup 恰好会
带上 `embd_state`，而正常前向不带。这条线索目前是**唯一还没试过**的方向。

## 顺带补的公开 API

`llama_model_n_embd_inp_enc()`：DFlash/EAGLE3 的 draft graph 吃
`target_layers × n_embd`（8 × 2048 = 16384）宽的行，既不是 `n_embd_inp`（2048）也不是
`n_embd_out`（2048）。之前只能靠猜，猜成 `n_embd_out` 会让 warmup 的零状态宽度不匹配。
这个 accessor 本身是对的，留着有用（哪怕它没修好 warmup）。
