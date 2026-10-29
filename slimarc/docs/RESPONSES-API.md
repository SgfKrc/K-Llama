# OpenAI Responses 协议接入计划

> 状态：未实现
> 更新日期：2026-10-29

`server.py` 现在只实现 `/v1/chat/completions`。Responses 是 OpenAI 目前主推的接口形态，
两者的请求与响应形状差别不小，不能靠改几个字段糊过去。本文件定下要做什么、不做什么、
以及验收判据。

---

## 1. 与 Chat Completions 的实际差异

| | Chat Completions | Responses |
| --- | --- | --- |
| 端点 | `/v1/chat/completions` | `/v1/responses` |
| 输入 | `messages[]`，每项带 `role` | `input`，可以是字符串，也可以是 item 数组 |
| 输出 | `choices[0].message.content` | `output[]`，文本在 `message` 项的 `content[].text` |
| 流式单位 | `chat.completion.chunk`，`choices[0].delta.content` | 29 种事件类型，主体是 `response.output_text.delta` |
| 结束信号 | `finish_reason` | `response.completed`，携带完整 Response 对象 |
| 便利访问器 | 无 | `output_text`（SDK 侧聚合，不是服务端字段） |

**关键点：流式协议完全不同。** Chat Completions 是一个扁平的 chunk 数组；Responses 是一个
带索引的、有开始/结束语义的事件序列。照搬前者的循环发不出合规的流。

---

## 2. 流式事件序列

实现纯文本问答时只需要序列的子集。参考顺序（取自
[Parallel 的 Responses 文档](https://docs.parallel.ai/responses-api/features/streaming-events.md)，
去掉 web_search / mcp 那几类本服务不产生的事件）：

```
response.created
response.in_progress
response.output_item.added          (item: message, status: in_progress)
response.content_part.added         (part: output_text, text: "")
response.output_text.delta          (可重复，每次一个增量)
response.output_text.done           (text: 全文)
response.content_part.done
response.output_item.done           (item: message, status: completed)
response.completed                  (response: 完整对象，含 usage)
```

每个事件都需要这些公共字段：

```json
{
  "type": "response.output_text.delta",
  "sequence_number": 5,
  "output_index": 0,
  "content_index": 0,
  "item_id": "msg_...",
  "delta": "你好"
}
```

`item_id` 在同一响应的所有事件里必须一致，`sequence_number` 单调递增。
少了这两个字段，官方 SDK 的解析器会拒绝。

---

## 3. 请求体要处理的字段

### 必须支持

| 字段 | 处理 |
| --- | --- |
| `model` | 回显，不校验 |
| `input` | 字符串 → 单条 user 消息；数组 → 按 item 的 `type`/`role` 拼成 prompt |
| `stream` | 决定走事件流还是整体返回 |
| `max_output_tokens` | 对应现在的 `max_tokens` |
| `temperature` / `top_p` / `seed` | 已有采样通路，直接转发 |
| `instructions` | 作为 system 提示拼到最前 |

`input` 是数组时，item 可能是：

```json
{"type": "message", "role": "user", "content": "..." }
{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "..."}]}
{"role": "user", "content": "..."}                 // type 可省略
{"type": "function_call_output", ...}              // 见 §4
```

三种 `message` 形态都要接受 —— 官方 SDK 会发第二种，手写 curl 通常是第一种。

### 明确不支持，遇到就报错

不要静默忽略。客户端以为开了工具调用却拿到纯文本，比直接失败更难排查。

- `tools` / `tool_choice`（除 §4 的单轮之外）
- `text.format` 的结构化输出（`json_schema`）
- `previous_response_id`（无状态存储，见 §5）
- `background`、`store`、`include`、`truncation`
- `reasoning`（本服务没有推理阶段）
- 内置工具：`web_search`、`file_search`、`code_interpreter`、`mcp`

返回 OpenAPI 风格的错误体，并在 message 里点明哪个字段：

```json
{"error": {"type": "unsupported_parameter", "param": "tools",
           "message": "tools is not supported by this server"}}
```

---

## 4. 工具调用：一期不做，但留好位置

本项目现在只做**单轮文本生成**，模型不带工具调用能力（argmax/采样都直接出 token，
没有 function-call 的语法约束）。所以 `tools` 一律拒绝。

但协议形状要提前想清楚，否则以后返工：

- 输出项可以有 `function_call` 类型，字段 `call_id` / `name` / `arguments`
- 客户端下一轮把 `function_call_output` 作为 `input` 项传回来
- 那时才需要 `previous_response_id` 或让客户端自己带全历史

**一期结论：拒绝 `tools`，但 `output[]` 的构造代码要允许出现非 `message` 项。**
现在只发一种，不改结构。

---

## 5. 无状态

不实现 `store` / `previous_response_id`。理由：proto 的目标是验证分布式推理，加一层
响应存储会引入过期、容量、并发写三个与目标无关的问题。

客户端要做多轮，就自己把历史带进 `input`——这和现在 chat completions 的处理方式一致
（`server.py` 的 `flatten_messages` 每次从位置 0 重新 prefill）。

`store: true` 请求要显式拒绝，而不是忽略。OpenAI 的默认值在不同版本间变过，
含糊处理会让客户端以为能取回历史。

---

## 6. 与现有代码的关系

`server.py` 现在 285 行，结构是：

```
Handler.do_POST           路由
  ├─ _full_response()     非流式
  └─ _stream_response()   流式
flatten_messages()        messages[] → 一段 prompt
Chain.generate()          拿到 Chunk 迭代器
```

接入方式：**新增端点，复用下半截**。

```
Handler.do_POST
  ├─ /v1/chat/completions → 现有两个函数
  └─ /v1/responses        → 新增两个函数
                             ├─ _responses_full()
                             └─ _responses_stream()

flatten_input()            input → 一段 prompt（新增，与 flatten_messages 并列）
Chain.generate()           不动
```

`Chain`、`stream.generate_stream`、`Chunk`、UTF-8 重组**都不需要改**。
流式差异只体现在“怎么把 Chunk 变成事件”。

预计新增 150–200 行，`server.py` 到 450 行上下。如果超过 550 行，说明抽象错了，
应该把两种协议拆成两个模块（`proto_chat.py` / `proto_responses.py`），
由 `server.py` 只做路由。

---

## 7. 验收判据

### 7.1 官方 SDK 能解析

用 Python 官方 SDK 打通，这是唯一有意义的验收：

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="x")

# 非流式
r = c.responses.create(model="slimarc", input="你好")
assert r.output_text                      # 便利访问器能取到文本
assert r.output[0].type == "message"

# 流式
got = []
for ev in c.responses.create(model="slimarc", input="你好", stream=True):
    if ev.type == "response.output_text.delta":
        got.append(ev.delta)
assert "".join(got)
```

**不能只用 curl 验收。** 手写 HTTP 请求能过，SDK 解析失败，说明事件字段不合规，
而真实客户端都用 SDK。

### 7.2 文本与 chat completions 一致

同一 prompt、同一采样参数，两个端点产出的文本应**逐字相同**。
不同则说明 `flatten_input` 拼出的 prompt 与 `flatten_messages` 不一致 —— 这是最可能出的错。

### 7.3 中文不破损

沿用 `chat_tui.py` 的做法：断在字节边界的中文不能出现替换字符。
`stream.Utf8Assembler` 已经处理，新代码**必须复用它**，不要自己再拼一次。

### 7.4 不支持字段确实报错

`tools`、`previous_response_id`、`store: true` 各发一次，必须返回 4xx 且 `param` 指对。

---

## 8. 环境与约束

- 解释器：`G:\C\PYT\qlh\.venv-test\Scripts\python.exe`
- 需要 `openai` 包（验收用）：先确认是否已装，没有就 `pip install openai`
- 测试用 2B 两段 `topology-2b.toml`（1.44 GB，不需要 ROG）
- **不要绑核**（会把单 step 从 50 ms 拖到秒级，见 `MEMORY-LIMITS.md` §3.4）
- 测试完清进程：`Get-Process python | Stop-Process -Force`
- 不 commit，改完报告

---

## 9. 参考

- 事件序列与字段：[Parallel 的 Responses 流式文档](https://docs.parallel.ai/responses-api/features/streaming-events.md)
- 请求/响应形状：[Responses Quickstart](https://docs.parallel.ai/responses-api/responses-quickstart.md)
- 官方字段级兼容说明（哪些字段 honored / ignored / rejected）：
  [OpenAI Responses Compatibility](https://docs.parallel.ai/responses-api/openai-compatibility.md)
- OpenAI 官方参考：[Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create)

⚠️ `developers.openai.com` 的页面直连返回 403，抓不到内容。上面 Parallel 的文档是
可访问的第三方实现说明，字段与事件名与官方一致，可作为实现依据。若需核对官方原文，
用浏览器打开官方链接，不要依赖脚本抓取。
