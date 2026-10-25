# TUI 使用指南

> 状态：**现行**
>
> 更新日期：2026-10-19
>
> 适用范围：交互外壳（`qlh` / `qlh chat`）与只读单命令（`src/tui_commands.py`）的启动方式、按键与排障。按键以 `src/tui_textual.py` 的 `BINDINGS` 与每屏键提示为准，聊天页命令见 [TUI 指令集](TUI指令集.md)。

---

## 一、这是什么

交互外壳是 `src/tui_textual.py`（Textual），覆盖 **9 个功能屏 + 1 个调试兜底屏**，与 `PAGES` 一致：

| 序号 | 屏 | 内容 |
| --- | --- | --- |
| 1 | 聊天 | SSE 流式对话 |
| 2 | 状态 | 运行概览 |
| 3 | 模型 | 注册表、加载/卸载、下载、搜索、预检、登记 |
| 4 | 分布式 | 配置、容量、层段 |
| 5 | 节点 | 成员、入群、邀请、连接 |
| 6 | 队列 | 有界FIFO、准入暂停/恢复、取消等待请求及资源排空 |
| 7 | 日志 | 筛选、统计、导出 |
| 8 | 设备 | 画像、自动配置、GPU |
| 9 | 设置 | 会话参数与依赖边界 |
| — | 调试 | 未接线功能的 API 兜底（不计作产品功能覆盖） |

页面通过 HTTP 与后端 API 交互（默认 `http://127.0.0.1:8000/api`）。本机统一入口按需在**当前进程内**启动后端并显示探活阶段；远程地址只探测目标后端。分布式与容量数据取自集群只读端点（`/cluster/resources`、`/cluster/layers`、`/cluster/pipeline-capacity` 等）。支持 Windows 10+ / Linux / macOS。

后端日志写入既有日志文件与日志屏，不穿透 TUI。

只支持一种交互实现：自绘 ANSI 的标准库 TUI 已于 2026-09-17 归档（见 [TUI 重写方案](archive/tui/TUI重写方案-2026-09-16.md)）。`--tui-engine builtin` 参数保留但会明确提示"已归档"并以退出码 2 结束；未安装 Textual 时给出安装指引并以退出码 1 结束。

## 二、启动

### 统一入口 `qlh`（推荐）

```bash
qlh                                       # 交互外壳（本机后端按需在进程内启动）
qlh chat --host http://127.0.0.1:8000
qlh chat --route distributed_preferred --thinking
qlh chat --host http://100.100.52.106:8000 --log-token TOKEN
qlh status                                # 只读单命令（不启动后端）
qlh models
```

源码检出的启动器：`qlh.bat` / `qlh.sh`；`K-Llama.bat` / `K-Llama.sh` 是推荐别名，`bjtu.*` / `koakuma.*` 是兼容别名，都转发到同一个入口脚本 `qlh.py`，参数原样透传。

依赖边界：交互外壳需要 Textual（`requirements-tui.txt`，Edge 同装，见 `requirements-edge.txt`）；协议层 `src/tui_api.py` 是纯标准库，单命令与 CI 路径不需要 UI 依赖。

### 一键启动 `start_tui.bat` / `start_tui.sh`

```bash
./start_tui.sh                 # Linux/macOS（需 chmod +x）
start_tui.bat                  # Windows
```

交互模式下后端在**当前进程内**启动（`BackendSupervisor`），冷启动由启动屏承载，退出 TUI 时后端随之停止；需要常驻后端请直接运行 `python src/api_server.py`。参数原样透传，`QLH_BACKEND_PORT` 可改端口（如 `QLH_BACKEND_PORT=8100 start_tui.bat --port 8100`）。

单命令模式（`start_tui.bat status`）执行一条只读命令后退出，不启动后端；命令必须是第一个参数。

### 手动启动（排障用）

```bash
python src/api_server.py            # 或 python -m uvicorn src.api_server:app --port 8000
python qlh.py --port 8000           # 另开终端进入交互外壳
```

## 三、功能屏与按键

全局键：`[` / `]` 上下切屏、`r` 刷新、`q` 退出（`Ctrl+C` 同效）。左侧导航可直接点击切屏。每个屏在标题下显示本屏可用键。

| 屏 | 键 | 操作 |
| --- | --- | --- |
| 模型 | `l` / `u` | 加载 / 卸载模型 |
| 模型 | `d` / `v` | 创建下载任务 / 搜索仓库 |
| 模型 | `f` / `i` | 本地资产预检 / 登记资产 |
| 队列 | `p` / `c` | 暂停-恢复准入 / 取消等待请求 |
| 分布式 | `t` / `m` / `j` | 切换分布式推理 / 设置最大节点 / 填写主节点地址并连接 |
| 节点 | `j` / `b` / `k` / `o` | 连接主节点 / 生成一次性入群请求码 / 消费主节点授权 / 签发授权（需真实 OTP） |
| 日志 | `e` | 导出压缩包 |
| 设备 | `g` / `h` | 按画像自动配置 / 选择 GPU |
| 设置 | `w` | 读取当前设置并写回 |
| 调试 | `a` / `x` | 重新读取 OpenAPI 路由表 / 执行选中操作 |

写操作在提交前显示目标与影响范围，并经确认框执行。模型下载、模型搜索与日志导出在后台线程执行，避免阻塞终端。

入群请求码由 `b` 生成，主节点用 `o` 输入 Auth App/TOTP 一次性验证码与 TTL 后签发，`k` 消费授权。签发端点只接受已登录管理员的真实 OTP；未绑定 Auth App/TOTP 时后端返回 `501 auth_control_plane_unavailable`。

登录后 REST、下载与 SSE 流式聊天都带内存中的 `Authorization: Bearer <token>`；请求来源由后端按直接 TCP peer 地址判定。远程日志操作另需 `--log-token`（透传 `X-QLH-Log-Token`）。

## 四、聊天页命令

聊天页的 `/` 命令共 25 条，完整表（会话、模型资产、队列路由、推理展示、日志、账户认证、集群）见 [TUI 指令集](TUI指令集.md)。命令与实现同源：`src/tui_shared.py` 的 `COMMAND_SPECS`，`/help` 由它生成。

离线回放：`qlh chat --fixture <路径>` 由只读薄层 `src/tui_commands.py` 执行，零依赖、不联网、不经 UI。

## 五、调试兜底屏（非功能验收）

调试屏从运行中后端的 `/openapi.json` 动态读取路由，不依赖手工维护的路由副本。

- `a` 重新读取路由表；选中一行后可编辑路径、查询参数 JSON 与请求体 JSON（路径里的 `{参数}` 必须先替换）。
- `x` 执行：GET/HEAD 直接查询，POST/PUT/PATCH/DELETE 经确认。
- `/chat/stream` 与 `/chat/upload` 标为"专用界面"，避免用普通 JSON 请求破坏 SSE 或 multipart 契约。
- 远程后端不支持 `/openapi.json` 时列表显示不可用，功能屏照常可用。

调试屏用于验证新接口或临时 JSON 合同，不计入功能覆盖率；稳定业务流程只能从调试屏完成时，应登记为功能缺口。

## 六、只读单命令

`src/tui_commands.py` 提供只读子命令，执行后立即退出、不启动后端（后端未运行时提示并以退出码 1 结束）：

```bash
qlh status                 # 系统状态 + 当前模型
qlh models                 # 模型列表（* 标记当前模型）
qlh nodes                  # 集群节点列表
qlh queue                  # 请求队列与调度策略
qlh device                 # 设备画像（CPU/内存/磁盘/GPU/档位）
qlh logs                   # 聚合日志（远程需 --log-token）
qlh help                    # 只读命令一览
qlh --host 100.x.x.x status # 对远程主节点执行
qlh models --json           # 机读输出
```

参数：`--host`（默认 `127.0.0.1`）、`--port`（默认 `8000`）、`--timeout`（默认 `5.0`）、`--log-token`（默认空）、`--json`、`--fixture PATH`。

## 七、退出与后端生命周期

- 交互外壳按 `q` 或 `Ctrl+C` 退出；后端是进程内守护线程，随 TUI 退出而停止。
- 需要 TUI 退出后仍保留后端，直接运行 `python src/api_server.py` 并用 `qlh --host` 连它。

## 八、常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 启动屏长时间停在探活阶段 | 端口被占用（改 `QLH_BACKEND_PORT`）、Python 环境缺依赖，或 `.env`/数据库不可达；后端日志在 `logs/` 与日志屏 |
| TUI 显示"后端未启动" | 后端未运行或地址不对；用 `python src/api_server.py` 手动起后端，或确认 `--host` |
| TUI 报"内部错误" | 多为后端版本与 TUI 契约不一致（字段缺失或类型错误）；契约测试见 `tests/test_tui_shared.py`、`tests/test_tui_textual.py` |
| 中文乱码 | Windows：脚本已 `chcp 65001`；手动启动时先执行 `chcp 65001`。Linux/macOS：确认终端为 UTF-8 |
| 远程日志打不开 | 需 `--log-token`；本机模式不走 HTTP，直接读 `logs/` |

## 九、测试

- TUI 定向回归：`.\.venv-test\Scripts\python.exe -m pytest -q tests/test_tui_textual.py tests/test_tui_write_ops.py tests/test_tui_shared.py tests/test_tui_sse.py`。
- 端到端 flow 与演示复用见 [TUI 端到端 flow 测试与答辩演示复用计划](archive/tui/TUI端到端flow测试与答辩演示复用计划-2026-09-24.md)（已归档：F1/F2/F3 三档全部落地）。
