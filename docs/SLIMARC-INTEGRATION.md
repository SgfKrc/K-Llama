# SLIM-ARC 接入的实际障碍

> 状态：**已验证，未实施**
> 更新日期：2026-10-28
>
> 结论先行：**SLIM-ARC 当前不支持 `qwen35moe`**，需要改补丁才能接入。改动不大，但必须先决定改不改。

---

## 1. 验证方法

把 `G:\C\PYT\qlh\android\app\src\main\cpp\llama.cpp\src\` 复制到
`H:\slimarc-proto\slimarc-trial\src\`，跑：

```
python SLIM-ARC-main/scripts/apply-slim-arc.py H:\slimarc-proto\slimarc-trial
```

## 2. 结果

前 7 个补丁函数全部成功：

```
transform_model         ✅
patch_model_loader      ✅
patch_model             ✅
patch_qwen3next         ❌ 这里失败
patch_context           （未执行）
patch_kv_cache          （未执行）
patch_cmakelists        （未执行）
```

错误：

```
RuntimeError: required Qwen3-Next expert count anchors not found
```

## 3. 根因

`apply-slim-arc.py:258-260` 的判定：

```python
pristine_expert_count = "n_expert, n_expert_used,"
...
if content.count(pristine_expert_count) != 2 or screened_expert_count in content:
    raise RuntimeError("required Qwen3-Next expert count anchors not found")
```

它要求 `src/models/qwen3next.cpp` 里 `n_expert, n_expert_used,` **恰好出现 2 次**。

实际情况：

| 文件 | 出现次数 |
| --- | ---: |
| `qwen3next.cpp`（SLIM-ARC 的目标） | **1**（L554） |
| `qwen35moe.cpp`（我们的模型） | **2**（L512、L690） |

**SLIM-ARC 的作者测的是 Qwen3-Next，我们的模型是 Qwen3.5-MoE，架构不同。**

## 4. 但适配是可行的

两处调用的代码结构**完全一致**：

```cpp
build_moe_ffn(cur,
    <gate_inp>, <up_exps>, <gate_exps>, <down_exps>,
    nullptr,
    n_expert, n_expert_used,          // ← 就是这个锚点
    LLM_FFN_SILU, true,
    hparams.expert_weights_scale,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
    nullptr, <gate_up_exps>);
```

差异只在第一个参数：`qwen35moe.cpp` 第一处用 `model.layers[il].*`，第二处用 `layer.*`。
参数个数、顺序、`build_moe_ffn` 的签名都相同。

因此这个补丁做到 `qwen35moe.cpp` 里在技术上没有障碍。

## 5. 改动方案

在 `apply-slim-arc.py` 里做三处小改：

1. 把 `patch_qwen3next(src/models/qwen3next.cpp)` 换成对**两个文件**都调用
2. `transform_qwen3next` 的"恰好 2 次"断言改为参数化（`expected_count`）
   - `qwen3next.cpp` → 1
   - `qwen35moe.cpp` → 2
3. 其他锚点（`#include <cstring>`、`#include <limits>` 等）需确认 `qwen35moe.cpp` 里都有

**预计改动 15–25 行。** 但要重新验证全部 18 个锚点，以及编译通过。

## 6. 需要决定的事

### 选项 A：改补丁，让 SLIM-ARC 支持 qwen35moe

- 代价：改 15–25 行 + 重新验证锚点 + 编译 + 跑 A/B
- 收益：满足用户"接 SLIM-ARC 作兜底"的要求
- 风险：`qwen35moe` 的 MoE 路径若与 `qwen3next` 有实质差异（如 gating 函数、expert_weights_scale 语义），补丁即使应用成功也可能行为不对

### 选项 B：不改补丁，退到 `qwen3next` 架构的模型

- 需要换一个 Qwen3-Next 系列的模型
- 但用户已明确"支持优先考虑 qw 系 35B-A3B，3.5-3.6 没有架构差异"
- **与用户要求冲突**

### 选项 C：这次不接，记为已知限制

- 明天实验用纯 llama.cpp
- SLIM-ARC 作为"未来单段超过内存时再处理"的待办
- 代价：不满足用户"接 SLIM-ARC"的明确要求

**我倾向 A**，但需要用户确认值得投入。因为按 §4.2 的分析，SLIM-ARC 在"装得下"时本来就该退化持平，
改补丁的主要价值是**验证它确实会退化**，而不是提升性能。

## 7. 另一个重要发现

我们的 llama.cpp 不是上游，而是 **QLH 分支**（commit `784cbbbbe`，
`feat(api): 层段接力上游新增「末层输出（norm 前）」槽位并扩展到 9 个架构（QLH）`）。

这意味着 SLIM-ARC 是在一个**已经被改过的** llama.cpp 上再打补丁。两层补丁叠加的
冲突面比"上游 + SLIM-ARC"更大，上面 7 个成功的补丁函数只说明这些锚点还没被 QLH 改掉。
