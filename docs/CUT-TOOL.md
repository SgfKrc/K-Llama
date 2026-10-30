# 裁切工具：已内联并验证

> 状态：**已完成，`cut.py` 359 行**
> 更新日期：2026-10-29

---

## 1. 结论

新原型有了自己的裁切工具 `cut.py`（359 行），**不再依赖旧仓库**。
全部 8 项必需能力保留，三种模式交叉验证一致，fail-closed 全部生效。

超出 200 行目标 159 行，原因是保留了三处根因注释与 10 条校验的完整错误文案。
判断：**清晰报错比行数更值**，不压。

---

## 2. 交叉验证结果（与旧工具逐项对比）

用 `qwen3-5-9b-local-Q4_K_M.gguf`（5.38 GB，`block_count=33`、`n_layer=32`、`interval=4`）：

| 模式 | 参数 | 旧工具 | 新工具 | 结果 |
| --- | --- | --- | --- | --- |
| head | `--keep-head 4` | 55/387 | 55/387 | ✅ 一致 |
| middle | `--k 8 --end 24` | 214/228 | 214/228 | ✅ 一致 |
| tail | `--k 8` | 336/106 | 336/106 | ✅ 一致 |

（数字为 保留张量数 / 丢弃张量数）

---

## 3. fail-closed 验证

三种非法配置都被拒绝，退出码 2：

```
--keep-head 6        → FAIL: --keep-head N=6 不是 full_attention_interval(4) 的整数倍
--k 8 --end 22       → FAIL: --end=22 不是 ... 的整数倍
                       FAIL: 段内层数 end-K=14 不是 ... 的整数倍
--k 8 --end 14       → FAIL: 段内层数 end-K=6 不是 ... 的整数倍
```

每条错误都指向具体参数与原因，不是笼统的"参数错误"。

---

## 4. `output.weight` 取舍验证（`#147` 的核心）

| 段 | kept_block_count | output.weight | can_serve_tail | llama.cpp 加载 |
| --- | ---: | --- | --- | --- |
| head `0-4` | 4 | False | False | ✅ |
| middle `8-24` | 16 | False | False | ✅ |
| tail `24-32` | 9 | True | True | ✅ |

**非末段正确省略 `output.weight`，末段正确保留。** 末段 kept=9 而非 8，是因为
`n_layer=32` 但 `block_count=33`（多一个 MTP 层），符合 MTP 处理逻辑。

---

## 5. manifest 双口径验证

`head 4` 工件的 manifest：

```
顶层 block_count   = 33   ← 源模型
顶层 n_layer       = 32
kept_block_count   = 4    ← 本段
source_layer_range = 0..4
tensors            = 55 / 387
generator          = cut.py v3
```

`artifact_contains`：

```
token_embd     = True
output_norm    = True
output_weight  = False     ← 非末段正确
can_serve_tail = False
```

**口径分离正确**：顶层描述源模型（33），`kept_*` / `source_layer_range` 描述本段（4）。
这是最容易做错的地方，实测存量工件的 manifest 顶层写 33 而 GGUF 内是 4。

---

## 6. 修复的两个 bug

### 6.1 字典字面量导致 head 模式崩溃

```python
# 错：dict 字面量会立即求值所有 value，head 模式下 args.end 是 None
detail = {"head": ..., "middle": f"blk.{args.k}..blk.{int(args.end) - 1}"}[mode]
# TypeError: int() argument must be ... not 'NoneType'
```

改为 `if/elif/else` 分支。

### 6.2 manifest 计数重复求 len

```python
# 错：调用方传的是 Count，函数内又 len() 一次
"tensors_kept": len(kept)      # TypeError: object of type 'int' has no len()
"tensors_kept": kept           # 正确
```

---

## 7. 保留的能力清单

1. 三种模式：`head`（不重编号）/ `middle` / `tail`（重编号为 `blk.0..`）
2. 切点 4 倍数校验，`K` / `end` / `end-K` 三者都查
3. MTP 层归零（`<arch>.nextn_predict_layers`）
4. 必需张量按层类型断言（linear / full 后缀，仅 `qwen35*`）
5. manifest 双口径
6. KV 按层数组切片（实测复制 34 个字段）
7. `omit_gguf_output_weight`（`qwen35` + `qwen35moe`）
8. `--dry-run`
9. **拒绝覆盖已存在的目标**（防止误删已有工件）

---

## 8. 一个我验证过并被否证的猜想

**猜想**：中段和末段各带 `token_embd`（515.3 MiB × 2 = 1.03 GB），若能省掉，
三段总重从 21.85 GB 降到 20.71 GB，可能让本机三点同载可行。

**实测否证**。`qwen35moe.cpp:46-50`：

```c
tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
//                                                                                ^ 无标志 = 必需
output   = create_tensor(tn(LLM_TENSOR_OUTPUT,    "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
//                                                                                ^ 可选
```

`token_embd` **无条件必需**，`output.weight` 才是可选的。**这 1.03 GB 是拆分的结构性成本。**

### 8.1 实际 35B-A3B 工件核对

| 段 | block_count | token_embd | output.weight | tensors |
| --- | ---: | --- | --- | ---: |
| `stage-0-8` | 8 | ✅ 515.3 MiB | ❌ | 148 |
| `stage-8-24` | 16 | ✅ 515.3 MiB | ❌ 正确省略 | 294 |
| `stage-24-40` | 16 | ✅ 515.3 MiB | ✅ 515.3 MiB | 295 |

---

## 9. 旧工具里砍掉的能力

| 参数 / 功能 | 用途 | 处置 |
| --- | --- | --- |
| `--source-model-id` + `--tokenizer-sha256` | 供给系统的模型身份 | 砍 |
| `--verify-manifest` | 对照 manifest 校验已有工件 | 砍（`check_correct.py` 做数值校验） |
| `--hf-config` | 层类型序列逐位校验 | 砍（需额外读 config.json） |
| `_validate_model_preflight_metadata` | 供给准入 | 砍 |
| `_sha256` 全文件哈希 | 工件指纹 | 砍（20 GB 文件算哈希太慢） |
| 进度条 / 彩色输出 | UI | 简化为 print |


---

## 2. 旧工具确实完善 —— 三个值得保留的设计

### 2.1 切点必须是 `full_attention_interval` 的整数倍（fail-closed）

`cut_layers.py:144-186`。三种模式都校验：`K`、`end`、`end-K` 三者**都**必须是 interval 的倍数。

**根因**（必须保留在注释里）：llama.cpp **不读** HF 的 `layer_types`，而是按重编号后的
**本地层号取模**重推层类型（`qwen35.cpp:21-28`）。所以切点错位必然是
`missing tensor`，而不是"能加载但算错"。

实测：`--keep-head 6` 被拒（6 不是 4 的倍数）。

### 2.2 manifest 双口径

`cut_layers.py:327-435`。这是最容易做错的地方：

| 字段组 | 描述对象 |
| --- | --- |
| 顶层 `block_count` / `n_layer` / `nextn_predict_layers` | **源模型** |
| `kept_block_count` / `source_layer_range` / `artifact_contains` | **本段** |

实测证据：存量 `q35-9b-local-head4.gguf` 的 manifest 顶层写 `block_count: 33`，
而 GGUF 内实际是 4。下游找工件只认 manifest 三对齐
（`source_layer_range` + `source_model_sha256` + 实算 `artifact_sha256`），
不猜文件名（`llama_engine.py:701-782`）。

### 2.3 `output.weight` 的省略（`#147` 修复记录）

`gguf_tensor_selection.py:30-33`：

```python
_QWEN35_EMBEDDING_FALLBACK = frozenset({"qwen35", "qwen35moe"})

def omit_gguf_output_weight(architecture, end_layer, decoder_layers) -> bool:
    return architecture in _QWEN35_EMBEDDING_FALLBACK and end_layer < decoder_layers
```

`#147` 的教训值得记：

> 原先只列 `qwen35`，漏了 MoE 变体 `qwen35moe`，于是 35B-A3B 的**每个中段都白白带走
> `output.weight`（515.3 MiB）**。实测 `[20,32)` 应 6085 MiB 纯层权重，
> 实际 7066 MB——多出的约 981 MiB 正是 `token_embd` + `output.weight`。
>
> MoE 与 dense 在这个判据上**没有区别**……因此按架构家族整体判定，而不是逐个枚举变体 ——
> 否则每加一个 MoE 变体就会再漏一次。

---

## 3. 一个我验证过的死亡猜想

**猜想**：中段和末段各带着 `token_embd`（515.3 MiB × 2 = 1.03 GB），如果能省掉，
三段总重会从 21.85 GB 降到 20.71 GB，让本机三点同载变得可行。

**实测否证**。`qwen35moe.cpp:46`：

```c
tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
//                                                                                ^ 无标志 = 必需
output   = create_tensor(tn(LLM_TENSOR_OUTPUT,    "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
//                                                                                ^ 可选
```

`token_embd` **无条件必需**，`output.weight` 才是可选的。所以：

- `stage-8-24` 无 `output.weight` —— **正确**，旧工具已处理
- `stage-8-24` 有 `token_embd` —— **无法避免**

**这 1.03 GB 是拆分的结构性成本，省不掉。** 三段同载仍需可用内存 ≥29 GB。

### 3.1 实际工件核对

| 段 | block_count | token_embd | output.weight | tensors |
| --- | ---: | --- | --- | ---: |
| `stage-0-8` | 8 | ✅ 515.3 MiB | ❌ | 148 |
| `stage-8-24` | 16 | ✅ 515.3 MiB | ❌（正确省略） | 294 |
| `stage-24-40` | 16 | ✅ 515.3 MiB | ✅ 515.3 MiB | 295 |

---

## 4. 旧工具里我们不需要的能力

| 参数 / 功能 | 用途 | 处置 |
| --- | --- | --- |
| `--source-model-id` + `--tokenizer-sha256` | 供给系统的模型身份 | 砍 |
| `--verify-manifest` | 对照 manifest 校验已有工件 | 砍（我们的 `check_correct.py` 做数值校验） |
| `--hf-config` | 层类型序列逐位校验 | 砍（需要额外读 config.json） |
| `_validate_model_preflight_metadata` | 供给准入 | 砍 |
| `_sha256` 全文件哈希 | 工件指纹 | 砍（20 GB 文件算哈希太慢） |
| 进度条 / 彩色输出 | UI | 简化为 print |

---

## 5. 必须保留的清单

1. 三种模式：`head`（不重编号）/ `middle` / `tail`（重编号为 `blk.0..`）
2. 切点 4 倍数校验（fail-closed）
3. MTP 层归零（`<arch>.nextn_predict_layers`）
4. 必需张量按层类型断言（linear 12 后缀 / full 11 后缀，仅 `qwen35*`）
5. manifest 双口径
6. KV 按层数组切片
7. `omit_gguf_output_weight`（`qwen35` + `qwen35moe`）
8. `--dry-run`（测试纪律依赖它）

---

## 6. 验收方式

**交叉验证**：同一模型、同一参数，新旧工具报的保留/丢弃张量数必须一致。

```powershell
& $py G:\C\PYT\qlh\scripts\cut_layers.py --src <小模型> --keep-head 4 --dry-run
& $py cut.py --src <小模型> --keep-head 4 --dry-run
```

**加载验证**：新工具产出的工件必须能被 `llama-bench` 加载。

**测试只用小模型**（5.5 GB 的 qwen3-5-9b），不用 20 GB 的 35B-A3B——
后者单次裁切就会把内存打满。
