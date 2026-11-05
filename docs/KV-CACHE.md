# KV cache 复用与回滚（第七期调研）

结论先给：**DSH 每轮重放整段前缀，根因不在 KV cache 技术，在 server 每个请求无条件
reset**。KV 这一层能做的三件事里，只有一件量级正确。

> 本文写的是 **v2 的 `server.py` 驱动链**。v2.1 把链下沉进 `llama_context` 之后，这条根因
> 已经消失：`llama-server` 的 slot / prompt cache 直接作用于整条链（checkpoint 覆盖全链，
> 3103 token 历史第二轮只解 38 个新 token）。本文的结论只用于解释 v2 的历史行为，不是
> v2.1 的现状。

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

### 3.1 【做成】server 走 token 前缀续写 —— 命中 88%

`Chain._reusable` / `_remember` 记一份「KV 里逐位置的 token id」账本：下一个请求 tokenize
后与它比公共前缀，命中就只写新增（`generate_stream` 的 `prefix_ids` / `n_past`），分叉、
KV 比 prompt 长、投机、`kv_tail`、计数对不上都清账本退回全量重放。

**曾经判过一次死刑，理由是错的。** 第一次实测命中率 0，我归因于「Qwen 模板在
generation prompt 末尾注入 `<think>`，与历史 assistant 段的渲染不一致」，认为这是模板的
固有性质。真正的差别在别处：

- `chat.Conversation.add_assistant` 用 `" " + thinking_start + "\n" + generated` 把标记
  **拼进 content**（`chat.py:294`），下一轮模板渲染历史 assistant 时就和 KV 对齐了。
- `server.py` 把客户端回传的 content 原样塞进模板，**没有这一拼**，于是历史 assistant 少了
  `<think>\n`，前缀在 assistant 头之后立刻断。
- 也就是说：**模板不是不对称的，是我们少写了两个 token**。DeepSeek 那边没这问题，因为它是
  自己的服务端在渲染。

修法是 `_close_assistant_turn`：渲染前给**每一条** `role=assistant` 且 content 不以标记开头
的 turn 前缀 `" " + thinking_start + "\n"`（前导空格被模板 trim 吃掉）。只补最后一条是错的
——真实追问最后一条永远是 user，要补的历史 assistant 在中间。

**实测（2B，`max_tokens=48` 逐轮截断）**：

| | prompt | reused | 占比 |
|---|---|---|---|
| 轮1 | 15 | 0 | 首轮无可复用 |
| 轮2 | 77 | 0 | 见下 |
| 轮3 | 141 | **124** | **88%** |

轮2 断在 61 是因为轮1 的回复被 `max_tokens=48` 从中间切断，解码出的文本重新 tokenize 与
生成时的 token 不再逐一相等。自然结束或按字符边界截断时不会这样。

- 动的不变量：`n_past ≈ 已写 token 数` 从请求内的局部量升级成跨请求的持久账本。少一条判定
  就是静默错上下文（不报错，只是答得不对），所以 `_remember` 见疑即清。
- 并发注意：多个请求共用一把锁串行，不同会话交错会毁掉前缀局部性。

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

补一句 3.1 实测之后的话：**「同会话内续写」不是自动成立的**。它要求模板渲染历史 assistant
的方式与生成时逐 token 一致，而这一步靠调用方在历史里补回 thinking 标记才成立（服务端的
历史来自客户端，所以由服务端补）。跨请求共享那一半仍然做不了，理由如上一段。

## 6. 待验证

- `llama-memory-hybrid.cpp:143` / `llama-memory-recurrent.cpp:192` 的回滚条件来自调研时的
  阅读，未在本仓库复跑；上一轮实测只到 `[seq_rm] p0=236 rc=-6` 这一层。
- 3.1 的 88% 是 2B、48 token 生成、逐轮截断下量的。35B（实际生产模型）未测：拿不到
  ROG 那台机器时只能在 2B 上验证机制，收益数字要在 35B 上重测。
- 3.1 的 12% 未命中来自「截断切在多字节字符中间」。真实对话若被 `max_tokens` 从中间截断，
  同样会断；按字符边界结束则不会。这一条没有修，因为它取决于模型的停止时机。
