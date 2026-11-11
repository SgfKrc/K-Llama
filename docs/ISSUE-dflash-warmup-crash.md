# ISSUE：DFlash 的 warmup 空跑必崩（`--no-warmup` 完全正常）

> 状态：**已验证**（文中引用的 v2 工具链——`chain_client` / `stage_worker` / `shim/` / `koakuma`——已退场，归档在 `slimarc/prototype`，命令与路径只作历史口径）
> 更新日期：2026-11-03
> 适用范围：DFlash draft 模型独立加载后走 `common_init_from_params` 的 warmup 路径；读者为内核维护者。影响面低——本项目的链不经过这条路径。

合并 target 的 `token_embd.weight` + `output.weight` 到 DFlash GGUF 后，draft 模型能独立
加载，但 `llama-cli` / `llama-server` 的 warmup 空跑必崩；加 `--no-warmup` 完全正常。
崩溃点已定位到图构建阶段，四条假设已实证排除。**未定部分**：warmup 与正常前向在
`n_tokens != block_size` 时的分支差异没有验证（见〈如果以后要修〉），根因尚未确认。

---

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

## 定位：崩在图构建阶段

`-v` 日志里，崩溃前最后一行是：

```
graph_reserve: reserving a graph for ubatch with n_tokens = 128, n_seqs = 1, n_outputs = 128
input tensor 'inp_pos' [i32, ne = { 128, ... }]
```

也就是说崩在 **`graph_reserve`/图构建**阶段，紧随 `inp_pos` 之后、`build_attn` 之前。

而 `--no-warmup` 走同样的 `graph_reserve`（也有 `n_tokens=128` 那次）**却不崩**。两者的差别只剩
warmup 那次 decode 的 **batch 内容**（BOS + EOS，2 个 token）与位置。

## 影响面：本项目的链不经过 warmup

1. **`--no-warmup` 是官方参数**（`common_params.warmup`），不是绕过补丁。
2. **本项目的链完全不经过它**：`stage_worker.py` 用 shim（`qlh_kh_load` + 显式 forward），
   不走 `common_init_from_params` 的 warmup。
3. `examples/speculative-simple` 与 `llama-server` 的 draft 路径也都正常（实测 95% 接受率）。

## 如果以后要修

下一刀应该切在 `src/models/dflash.cpp` 的 decoder graph：查 `n_tokens != block_size` 时
（warmup 给 2）与 `n_tokens == block_size`（16）时的分支差异，特别是
`model.lilicorr_pass_hidden_proj && ubatch.embd_state` 那个条件（L1031）——warmup 恰好会
带上 `embd_state`，而正常前向不带。这是唯一未经实验验证的方向。

## 相关 API：`llama_model_n_embd_inp_enc()`

DFlash/EAGLE3 的 draft graph 吃 `target_layers × n_embd`（8 × 2048 = 16384）宽的行，
既不是 `n_embd_inp`（2048）也不是 `n_embd_out`（2048）。猜成 `n_embd_out` 会让 warmup
的零状态宽度不匹配。该 accessor 本身正确，与 warmup 缺陷无关，独立可用。
