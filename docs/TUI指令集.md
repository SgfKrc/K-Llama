# TUI 指令集

> 状态：**现行**
>
> 更新日期：2026-10-28
>
> 适用范围：交互外壳聊天页（`src/tui_textual.py` 的 `ChatPane`）里以 `/` 开头的命令。命令的唯一事实源是 `src/tui_shared.py` 的 `COMMAND_SPECS`（`/help` 与执行分支同源，未实现的命令不登记）。功能屏的按键操作、启动方式与排障见 [TUI 使用指南](TUI使用指南.md)。

---

## 一、规则

1. 在聊天页输入 `/` 开头的命令后按 `Enter` 执行；`/help` 按 `COMMAND_SPECS` 生成当前可用清单。
2. 参数用位置参数与子命令，`/model load <id> [engine] [quant]` 这类子命令按源码解析。
3. 破坏性操作在执行前弹确认框：`/delete-session`、`/reset`、`/model load|unload`、`/queue clear`、`/logs delete`、`/history drop-turn`。
4. `/model` 的加载/卸载要求后端在 loopback 上：远程后端需主节点配置 `QLH_MODEL_API_TRUSTED_CIDRS`。
5. 人类账户登录、账户管理、TOTP 和 HA 角色管理已退役；本指南只记录仍属支持范围的交互命令。角色与连接配置见[静态部署决策](静态主节点部署与认证退役决策-2026-10-27.md)。

## 二、命令表（来源 `src/tui_shared.py` 的 `COMMAND_SPECS`）

### 会话

| 命令 | 参数 | 说明 |
| --- | --- | --- |
| `/new` | `[title]` | 新建并切换会话（`POST /sessions`） |
| `/resume` | `<session_id>` | 恢复历史会话并渲染其消息 |
| `/rename` | `<title>` | 重命名当前会话 |
| `/sessions` | — | 列出最近会话 |
| `/delete-session` | — | 删除当前会话及其全部消息（需确认） |
| `/reset` | — | 清空后端会话历史与 KV 缓存（需确认） |
| `/history` | `[<session_id>] [limit] \| sync-status \| info <session_id> \| drop-turn <session_id> <turn_index>` | 查看对话 / 本地持久化状态 / 会话详情 / 删单轮（需确认，删 user+assistant 两条） |
| `/clear` | — | 清空本地显示（不动后端；清后端用 `/reset`） |

### 模型与资产

| 命令 | 参数 | 说明 |
| --- | --- | --- |
| `/model` | `load <id> [engine] [quant] \| unload` | 加载/卸载模型（需确认；仅 loopback 后端可调用） |
| `/assets` | `available \| registry \| downloadable \| gguf` | 模型资产浏览（只读）：可选模型与引擎 / 已注册实验模型 / 可下载清单 / 本地 GGUF |
| `/storage` | — | 存储与数据库健康（只读） |

### 队列与路由

| 命令 | 参数 | 说明 |
| --- | --- | --- |
| `/queue` | `pause \| resume \| strategy <fifo\|mlfq> \| clear \| cancel <task_id>` | 队列控制（`clear` 需确认） |
| `/route` | `auto \| local \| distributed \| required` | 设置请求级路由偏好 |

### 推理与生成

| 命令 | 参数 | 说明 |
| --- | --- | --- |
| `/thinking` | `on \| off` | 思考内容的展示（仅 UI 显隐，不改变模型行为） |
| `/reasoning` | `on \| off \| auto` | 深度思考开关（改变模型行为）：`on` 强制思考、`off` 强制不思考（省算力，可避免 Qwen3 等输出超长 `<think>`）、`auto` 沿用模型模板默认 |
| `/cancel` | — | 取消当前生成 |

### 日志

| 命令 | 参数 | 说明 |
| --- | --- | --- |
| `/logs` | `list \| download <file> \| read <file> \| delete <file> \| nodes` | 文件列表 / 下载 / 查看 / 删除（需确认）/ 各节点汇总 |

### 界面

| 命令 | 参数 | 说明 |
| --- | --- | --- |
| `/help` | — | 显示本帮助 |
| `/quit` | — | 退出聊天页 |

## 三、维护

命令的增删改只改一处：`src/tui_shared.py` 的 `COMMAND_SPECS`，并同步 `ChatPane.on_input_submitted` 的执行分支与 `tests/test_tui_shared.py`；本文档与 `/help` 都从该表派生。
