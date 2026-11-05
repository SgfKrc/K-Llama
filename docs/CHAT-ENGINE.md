# 接入 llama.cpp 的 `common/chat.cpp`

> 状态：**方案**
> 更新日期：2026-11-05
> 分支：`slimarc/chat-engine`
> 适用范围：模板渲染、工具调用、多轮消息构造。读者是实现者与审查者。

## 1. 为什么

`chat.py` 的 `apply_chatml` 手写字符串拼接，**硬编码了某个模板版本的输出**。只做推理时这是
最省事且可验证的做法（`checks/verify_equiv.py` 钉住它与模板逐字一致），代价在接 agent 生态
时暴露：

- 模板里的条件分支（`tools` / `enable_thinking` / `role: tool` / `developer`）一个都没实现
- 工具调用完全没有
- 只支持 Qwen ChatML 一种格式
- 没有语法约束

上游 `common/chat.cpp` 之所以有几千行，正是因为它**解释模板**而不是复刻输出——模板改了什么
它自动跟上。我们要拿回的是这个性质。

## 2. 方案：独立 C++ DLL，零 llama 依赖

不链进 shim（shim 是 2274 行 C 的推理热路径，且 posix 构建靠 Win32 调用的文本替换），不写
Python 扩展（项目现有原生边界全是 ctypes，没有扩展先例）。做 `qlh_chat.dll`，C ABI + JSON
字符串进出，C++ 对象绝不过边界。

实测（不是估算，在 `F:\qlh_probe` 真编真链）：

| 项 | 结果 |
| --- | --- |
| 20 个 .cpp 独立编译 | 通过，只需 4 个 include 目录 |
| 体积 | 80 KB（裁剪）/ 173 KB（零分叉） |
| 导入表 | 只有 MSVC 运行时 + KERNEL32 + CRT，**零 llama.dll / ggml.dll** |

三个决定实现形态的结论：

1. `common_chat_templates_init` 传 `nullptr` model + 非空 `chat_template_override` 可用，
   DLL 不需要模型句柄
2. PEG 解析器不消费 tokens（`grep "\.tokens"` 零命中），纯文本喂进去无损，不需要 tokenizer
3. `jinja/` 是自研纯 STL 实现，零第三方依赖

**取零分叉（173 KB）**：裁剪省 93 KB，代价是永久维护 3 处上游分叉；`common/` 每周都在动，
`cp` 的成本远低于 rebase。

## 3. 排期

| 期 | 内容 | 判据 | 状态 |
| --- | --- | --- | --- |
| **一** | `shim/chat/`（vendored 上游 60 文件 + `UPSTREAM.txt`）、`shim/qlh_chat.cpp`（C ABI 五函数）、两个构建脚本、`checks/verify_chat.py` | dll 编出来（1,732 KB）；PE 导入表零 llama/ggml 符号；golden 模板渲染逐字节对上；P0 那份真实输出能被解析出正确 tool_calls。**63 PASS / 0 FAIL** | **完成** |
| **二** | `chat.py` 换成薄封装（删 `apply_chatml` 与 `next_turn`）、`verify_equiv --mode prompt`、`test_contract` 规则倒置（任何 .py 不许出现 `<|im_start|>`） | 71 项单测全过；`verify_multiturn` 在 2B 上与接之前**逐位相同**（三轮 `ids`/`text` 全 `==`）；`test_contract` 扫 9 个根 `.py` 得 0 处字面量 | **完成** |
| **三** | `server.py` 工具调用（非流式）：读 `tools`/`tool_choice`、产出 `tool_calls` + `finish_reason: "tool_calls"`、`role: tool` 回传 | `verify_openai_tools.py` 28 PASS；2B 端到端自己调出 `get_weather`（prompt 278 / gen 66 / 9.35 tok/s），回传 `role:tool` 后答出「晴，25 度」；非法 `arguments` 历史 → 400。**DSH 验收推迟到第四期** | **完成** |
| **四** | 流式 `delta.tool_calls`（按 index 累积）、`delta.reasoning_content`、`finish_reason` | 2B 端到端 48 个 chunk：`finish_reason: "tool_calls"`、`reasoning` 87 字独立成段、`id` 只宣告 1 次、拼回 `{"city":"北京"}` 与一次性解析逐字节相同；无 tools 的流式 122 chunk 仍是原样文本（无 `reasoning_content`） | **完成** |
| 五 | **多模态**：验证已完成（视觉活着、未漂移）；实现走原生还是旁路待定 | 一张图能进去并被正确描述 | 验证已过，实现待定 |
| 六 | grammar 约束（可选） | 出现真实的格式错误样本才做 | 未开始 |
| 七 | **KV cache 复用**：调研完成，前缀续写已实现并实测——**当前模板下命中率 0，收益上限也仅一成**，结论见 [KV-CACHE.md](KV-CACHE.md) | 不再投入；要多会话共存才走多序列 | 已量过，不做 |

期与期之间不并行：每一期的判据依赖上一期的产物。

### 第五期：验证已通过

2026-11-05 实测：**`opus47-distill` 的视觉活着，且未漂移。**

- mmproj 用 `unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf`（857.6 MiB，sha256 与 HF
  LFS ETag 一致），放在 `H:\qlh_models\mmproj\`
- 五个输入五个相符输出：红圆（位置/颜色/形状全对）、大字 “QLH-42”（OCR 读出 QLH42，
  连字符省略）、三色条（「蓝黄绿」顺序对）、2×2 网格（四角映射全对）、纯白图（答「空白」）
- **阴性对照双向成立**：纯白图不编造内容、不给图就答「我没看到图片」——排除「顺着 prompt 编」
- `prompt_tokens` 随图变化：纯文本 23 / 512² 图 281 / 768×384 文字图 318 / 600×200 色条 139，
  说明 image token 真实进了上下文而非固定占位

运行参数（实测可交互）：`-ngl 99 -cmoe`（MoE 专家留 CPU）→ 显存 3.5 GiB、加载 5.3 s、
生成 7–21 tok/s、单次问答 8–40 s。

**注意**：llama.cpp 警告 Qwen-VL 建议 `--image-min-tokens 1024`（grounding 精度）。本次不加
也能正确 OCR，将来做 bounding box / 小字识别时要加上复测。

### 实现期的真实难点（不在模型，在链路）

1. **M-RoPE 让位置与 token 数脱钩**：图像 token 的 `t` 恒定、x/y 是网格，推进量是
   `max(nx,ny)`=32 而非 1024。而本链路处处假设 `n_past ≈ 已写 token 数`（`stream.py:192`、
   `keep_head.py:198`、`speculate` 的 `n_past+1+j`、`seq_rm`）
2. 图像 token **必须走 embd 输入**（M-RoPE 只允许 embedding 输入的位置重叠）
3. **4 维位置要传到每一段**（每段都持整模、各算自己层的 RoPE）
4. ~~`keep_head.py:196` 硬拒 `n_tokens > 512`~~：**已修**，`forward()` 现在按各段
   `n_batch` 切块 prefill（见 §5）。图像 token 走 embd 路径时同样要切，但每块的位置
   推进要按 M-RoPE 算，不能沿用「块大小」——这部分留给第五期。
5. 带宽：1024 行 × 2048 × 4 B = 8 MB/图，跨 Tailscale

估算 800–1500 行、跨 8 个文件，是工具调用那一期的 3–5 倍。

### 备选：旁路（成本已实测）

`llama-server --mmproj` 独立跑视觉那套，3.5 GiB 显存、7–21 tok/s、几小时可接完，**不碰
`n_past` 不变量**。代价是图由另一个推理过程描述后再喂进本链路，细节受描述质量限制。

两条路的分界：**「让 agent 能看图」用旁路；「模型自己理解图」才需要原生**。开工前必须选一个。

## 4. 明确不做

| 不做 | 理由 |
| --- | --- |
| grammar（第一期到第五期） | DSH 不发 `strict`、不发 `tool_choice`；且我们的 `<tool_call>` 是特殊 token，格式比 JSON 稳。**代价是 `tool_choice: required` 只能软约束**，文档要如实标 |
| `parallel_tool_calls` | DSH 不发 |
| `strict: true` | DSH 的 `supportsStrictMode: false`，不发 |
| `type: "custom"` 的 grammar tools | DSH 的 `supportsOpenAIGrammarTools: false` |
| `image_url` 内容块 | 我们只做文本 |
| 千问系之外的模板 | ABI 层 `qlh_chat_new` 检查模板特征，不命中即报错，不靠删上游代码实现 |

## 5. 实现要点

- **不能加 `-I<llama>/common/jinja`**：`<string.h>` 会解析到 `jinja/string.h`，MSVC 报一屏
  `memchr/strlen 不是 global namespace 成员`。所有 include 写 `"jinja/xxx.h"` 相对 common 根
- **C ABI 五个函数**：`new` / `del` / `apply` / `parse` / `str_free`。`apply` 与 `parse` 不能
  合并——前者吃 OpenAI 请求、后者吃模型输出文本，且 `parse` 要用 `apply` 存下的 PEG arena
  （C++ 对象只能活在 handle 里）
- **`parse` 用上游的 `common_chat_msg_diff::compute_diffs`** 产出增量，Python 侧不重新实现
  tool_call 增量拼接——那是这块最容易写错的地方
- 跨 DLL 边界的堆内存必须由分配方释放，`str_free` 不能省
- **两个千问检查点的思考标记不一样**：2B 模板开 `thinking`，35B 开 `thinking`。旧的
  手写 ChatML 只按 35B 写，所以它在 2B 上从第一个 token 起就与模板不一致（7047 vs
  248068）。唯一可信来源是渲染出的 `generation_prompt` 的最后一行，不是任何常量
- **读一次 `tokenizer.chat_template` 在 35B 上要 7 秒**（H: 盘、21.87 GiB GGUF），而
  `server.flatten_messages` 每个请求都读一次。`chat.read_template` 按
  `(路径, mtime, size)` 缓存；建 handle 只要 6 ms，不缓存（handle 带 parse 状态）
- **渲染 handle 必须活到 parse**：`qlh_chat_parse` 要求同一个 handle 上先 `apply`（否则
  rc=-1）。旧的 `flatten_messages` 在 `return` 时关了 handle，于是「渲染 → 生成 → 解析」
  根本走不通。现在一次请求一个 `Chat`，`with` 包住全过程
- **只在请求带 `tools` 时 parse**：`parse` 会把 `thinking` 从 content 里剥出去，一条没写
  结束标记的回答会变成 `content=""` 加一堆 `reasoning_content`。不请求工具的调用方不该
  看到这个变化
- **`parse` 不产出 `tool_calls[].id`**：`to_json_oaicompat()` 只给
  `type`/`function{name,arguments}`，`arguments` 已是 JSON 字符串。id 必须自己发
  `call_<24 hex>`——客户端要靠它把 `role:"tool"` 配对回这次调用
- **历史 `arguments` 的字符串形式并不总是被静默丢弃**：上游对
  `caps.supports_object_arguments` 的模板（Qwen3.5 命中）会自己 parse，实测字符串与
  mapping 渲染逐字节相同。仍然在 Python 侧归一，是不让行为依赖上游对模板的分析；非法
  JSON 也因此能给出点名字段的 400，而不是模板内的一行 `Failed to parse tool call arguments`
- **流式可以每个 chunk 都 parse**：全量 1372 字一次 `partial=False` 是 0.17 ms，逐片
  `partial=True` 平均 0.05 ms，而生成一个 token 要 50–150 ms。不需要「看到 `<tool_call>`
  才开始解析」这类启发式
- **流式一个调用只宣告一次 id**：`parse` 的 diff 只在第一片给 `name`，`id` 永远为空串。
  按 `tool_call_index` 记住自己发的 id，后续片只发 `{"function":{"arguments": ...}}`；
  每片都发新 id 会让客户端把一次调用拆成多个块
- **收尾要用 `partial=False` 再 parse 一次**：工具调用只有在闭合标签到达后才算完成，
  增量那一遍不会吐出这段尾巴
- **一次 forward 不能超过段声明的 `n_batch`**：`keep_head._check_context` 超了直接
  `batch too large`，整条请求在出第一个 token 之前就死。切块 prefill 是同一份算术拆着
  算（每块从上一块的落点续），`chain_client.forward()` 一处覆盖所有调用方；`n_batch`
  由 stage 在 HELLO 里报，旧 worker 不报时按 512。实测 1856 token 的 prompt 从「119 ms
  报错」变成「603 片跑完」，这也是图文那一期要先解决的东西

## 6. DSH 的实际要求（读其源码确认）

它只发 `max_completion_tokens`（**不是 `max_tokens`**）和 `stream_options.include_usage`，
带 `tools`（不带 `strict`）；会消费 `finish_reason`（**不给就抛** `"Stream ended without
finish_reason"`）、`delta.content`、`delta.reasoning_content`、按 index 归并的
`delta.tool_calls`。必须容忍 `role: "developer"`。

四处都已处理，逐条都有出处：

- `max_completion_tokens` —— `server.request_max_tokens()` 两个名字都接，新的优先。
  只读旧名时每个请求都被压回服务端默认值，看起来像模型自己早停，最难查。
- `include_usage` —— 末 chunk 带 usage；DSH 先读 `chunk.usage` 再判 `choices`，通过。
- `developer` —— Qwen 模板把它渲染成 `<|im_start|>system`，与 `system` 逐字节相同。
- `tool_calls` 的 `id` —— 层里 `to_json_oaicompat()` 不生成，由 `server.py` 补。

以上按 DSH 的报文形状在 2B 端到端跑过；**尚未在 DSH GUI 里真连一次**，那是最后一步。
