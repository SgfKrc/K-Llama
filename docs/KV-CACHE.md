# KV cache 复用与回滚（第七期调研）

结论先给：**DSH 每轮重放整段前缀，根因不在 KV cache 技术，在 server 每个请求无条件
reset**。KV 这一层能做的三件事里，只有一件量级正确。

## 1. 为什么会重放

`server.py:228-229` 每个请求都走一遍

```python
for client in self._clients:
    client.reset()
```

无条件，不看上一轮写了什么。所以 DSH 每发一轮，整段 prompt 从位置 0 重算。实测代价：
2B 上 1856 token 的 prompt + 600 token 输出，prefill 加生成共 **73 s**。

`koakuma.align()` 那条兜底与 DSH 无关：它只在 `stray() > 0`（投机块在轮末被截断）时
返回 True，而 server 路径 `--spec-k` 默认 0、没有投机块，`stray()` 恒 0，align 永不触发。
把它当成「DSH 重放的原因」会指错方向。

## 2. `seq_rm` 到底发生了什么

三层各说各话，这是「裁不动」的真实形状：

| 层 | 行为 | 位置 |
|---|---|---|
| llama.cpp | `llama_memory_hybrid::seq_rm` 先删 recurrent，失败即返回 false，attention 那半一行都不删 | `src/llama-memory-hybrid.cpp:143`（H:\llamacpp-c） |
| shim | 量 `pos_max` 前后值再判，`return removed ? 0 : -6`；`SA_DEBUG_KV=1` 打印 `[kv] seq_rm p0=.. removed=.. pos_max X->Y` | `shim/qlh_keep_head_posix.c:1856` |
| worker | `if rc not in (0, -6): raise` —— **-6 直接放过** | `keep_head_worker.py:488` |

关键在第三层：shim 已经把「本来就没有可删的行」和「有行但删不掉」分开埋了点，worker 却在
返回值上把它们合并了，Python 侧永远看不到 `removed=false`。

而 hybrid 拒绝回滚的条件是 `1 <= rollback <= n_rs_seq`。**非投机链 `n_rs_seq=0`**
（快照深度只在 rs>0 时提升到 `SA_KH_RS_SEQ=16`），所以哪怕只删最后一个位置
（`rollback=1`）也会被拒——hybrid 一旦拒，attention 那半也没截断，KV 从此与 `n_past`
不一致，下一次 prefill 死在 `X < Y` 上。

所以 `koakuma.align()` 的选择（永远重放、绝不裁剪）是对的，但要认清它是被 worker 层那句
「-6 当 no-op」逼出来的，不是 hybrid 不支持裁剪。

## 3. 能做什么（按收益排序）

### 3.1 【已实现，实测零命中】server 走 token 前缀续写

代码在 `server.py` 的 `Chain._reusable` / `_remember`，判据是 **token id 前缀**：只有 KV
里恰好是本轮 prompt 的真前缀时才续写；分叉、或 KV 比 prompt 长（要裁剪，正是 hybrid 拒绝
的操作）都退回全量重放。账本说不清的时候一律清空——慢是看得见的，错不是。

**但在当前模板 + DSH 组合下命中率是 0**，根因不在实现：

- generation prompt 末尾是 `<|im_start|>assistant\n<think>\n`（`thinking=False` 时是
  `<think>\n\n<｜end▁of▁thinking｜>\n\n`），而历史 assistant 段渲染成 `<|im_start|>assistant\n<答案>`。
  **两者在 assistant 头之后第一个 token 就分叉**，实测字符公共前缀 59 / 总长 144，
  token 公共前缀 15 / 总长 429。
- `chat.Conversation.add_assistant` 正是为这件事存在：用
  `" " + thinking_start + "\n" + generated` 桥接生成文本与历史渲染（`chat.py:283-296`，
  注释原话「Get this wrong and every turn replays」）。`koakuma` 因此能续写；**接 DSH 时
  历史由客户端给，server 没有这条拼接，也不该有**。

**而且收益上限本身就不够**（2B 实测，这条更关键）：

| | prompt | 生成 | 总时长 |
|---|---|---|---|
| 轮1 | 15 | 400 | 38.2 s |
| 轮2 | 429 | 400 | 42.6 s |

多出的 414 个 prompt token 约 4.4 s，即 **prefill 只占总时长的 10%**，生成是大头。
即使前缀 100% 命中，也只能省这一成。

要真正吃到它，得让 server 篡改渲染输入（给历史 assistant 补 `thinking_start`），风险高于
收益。**所以这条不再投入**；代码保留是因为判据是保守的，哪天客户端回传能对齐它会自己生效，
`reused_prompt_tokens` / `kv_ledger` 两个字段会立刻显示出来。

- 动的不变量：`n_past ≈ 已写 token 数` 从一个请求内的局部量升级成跨请求的持久账本。少一条
  判定就是静默错上下文（不报错，只是答得不对）。
- 并发注意：DSH 的多个请求共用一把锁串行，不同会话交错会毁掉前缀局部性。

### 3.2 【可做】让 `seq_rm` 的失败可见（~25 行）

shim 加一个 `qlh_kh_seq_pos_max(handle, seq)` 导出（C 3 行 + `.def` 1 行），
`keep_head_worker.seq_rm` 改成返回 rc 并核对 `pos_max`；`align()` 从「永远重放」改成
「先试图裁，`pos_max` 真落到目标才继续，否则重放」。

- 收益：**只对 TUI + 投机路径**，DSH 用不上。而且 `n_rs_seq` 已被 `SA_KH_RS_SEQ=16`
  覆盖投机块深度，修完多数情况本来就能裁成功。
- 风险：低。收益小，属于顺手补洞。

### 3.3 【备选】跨请求 state checkpoint（~1.5 人天）

`llama_state_seq_get_data_ext` / `set_data_ext`（`PARTIAL_ONLY`）在每段各存一份 state，
命中前缀就换入。这是官方 `llama-server` 的做法——它的「回退」靠 checkpoint 恢复，而不是
`seq_rm`。

- 成本：2B 每段几十 MiB（10–30 ms 换入），35B 每段数百 MiB（约 100 ms）。
- 只有「多会话共享一段很长的系统提示词」时才划算。改 shim 4 个导出 + 链路上的 state 账本
  + 两段同步。

## 4. 明确不做

| 不做 | 理由 |
|---|---|
| 每段 Radix 前缀树 / 跨请求共享前缀 | 要求所有段同时命中、同时失效，否则 `n_past` 分叉成静默错；还要 `n_seq_max>1`，35B 每段多几百 MiB，路径 A 3–4 人天 |
| KV 量化（`-ctk/-ctv`） | 只省内存、不改 prefill 速度（还多一次 dequant），shim ABI 也没暴露。目标是 35B 装更长 ctx 时再单独评估 |
| PagedAttention / LMCache 移植 | 我们每段独立进程、各自整模，没有共享 KV 池可分页；等于重写引擎层 |
| 每会话一条独立链 | 每段直线多一份 KV + RS（640 MiB/段起），换来的只是并发不抢锁 |

## 5. 为什么 prefix caching 在这套形态里「一半失效」

失效的是**跨请求共享**那一半，不是续写那一半。每段的 KV 是同一 token 序列在该段上的
投影，只要各段在同一个 `n_past` 上对齐，「续写增量」本身就是每段各自的 prefix cache——
这条路 3.1 就是。做不了的只是 Radix/LMCache 那种「多个不同会话复用同一前缀」，因为它
要求所有段同时命中同一前缀，任何一段没命中，`n_past` 就分叉，而分叉是静默的。

补一句 3.1 实测之后的话：**「同会话内续写」也不是自动成立的**。它要求模板渲染历史
assistant 的方式与生成时逐 token 一致，而末尾注入的 thinking 引导破坏了这个一致性。
所以 prefix caching 在这套形态里两边都要额外条件才成立。

## 6. 待验证

- `llama-memory-hybrid.cpp:143` / `llama-memory-recurrent.cpp:192` 的回滚条件来自调研时的
  阅读，未在本仓库复跑；上一轮实测只到 `[seq_rm] p0=236 rc=-6` 这一层。
- 3.1 的一成收益上限是在 2B + 400 token 生成下量的。生成更长（比如 2000 token）时
  prefill 占比会更低，结论只会更偏向「不值得」；生成极短时反过来，未测。
- 3.1 若要让命中率变成 100%，路径是 server 给历史 assistant 补 `thinking_start`。这等于
  改写客户端给出的历史，未评估对 35B 与不同 checkpoint 的影响，也未做。
