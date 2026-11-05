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
| 三 | `server.py` 工具调用（非流式）：读 `tools`/`tool_choice`、产出 `tool_calls` + `finish_reason: "tool_calls"`、`role: tool` 回传 | DSH 真机连一次，能读文件/跑命令 | 未开始 |
| 四 | 流式 `delta.tool_calls`（按 index 累积）、`delta.reasoning_content`、`finish_reason` | openai SDK `.stream()` 累积结果与非流式**逐字段一致** | 未开始 |
| 五 | **多模态**：验证已完成（视觉活着、未漂移）；实现走原生还是旁路待定 | 一张图能进去并被正确描述 | 验证已过，实现待定 |
| 六 | grammar 约束（可选） | 出现真实的格式错误样本才做 | 未开始 |

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
4. `keep_head.py:196` 硬拒 `n_tokens > 512`，1024 token 的图直接抛
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

## 6. DSH 的实际要求（读其源码确认）

它会发 `max_completion_tokens`（**不是 `max_tokens`**）、`stream_options.include_usage`、
`tools`（不带 `strict`）；会消费 `finish_reason`（**不给就抛** `"Stream ended without
finish_reason"`）、`delta.content`、`delta.reasoning_content`、按 index 归并的
`delta.tool_calls`。必须容忍 `role: "developer"`。
