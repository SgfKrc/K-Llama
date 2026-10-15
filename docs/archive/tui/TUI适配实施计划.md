# TUI 适配与聊天页实施计划

> **状态**：部分实施（T1-T8 管理 TUI 与 T9.0-T9.5 聊天页已完成；T9.6 主应用包接线和 T9.6-R2 Windows 开发机 CPU/CUDA 实装门完成，外部干净机/Linux、默认入口和分布式真机验收未完成）
>
> **生命周期**：Active（T1-T8 在用）+ Candidate（T9 发布收口）——现有 7 屏管理 TUI 与网关契约继续作为稳定基线；`bjtu chat` 已在 Windows CPU/CUDA 隔离安装中可用，UP-N6.4W 跨卷保留门已完成，但外部干净机/Linux、分布式真机验收和默认入口尚未完成
>
> **更新日期**：2026-08-11
>
> **适用范围**：T1-T8 管理 TUI 网关适配与验收记录，以及 T9 简化聊天页面、流式会话、本地/分布式路由展示和可选依赖环境计划；当前日常使用见 [TUI 使用指南](../../TUI使用指南.md)
>
> **使用入口**：当前日常使用与启动方式见 [TUI 使用指南](../../TUI使用指南.md)；`bjtu chat` 已存在，`bjtu launcher/ui/tui/update/version` 由独立 Bootstrap 接线
>
> **关联文档**：[总体下一步计划](../distributed/总体下一步计划.md) · 微服务架构改造计划（§2.2 / §2.4 / §4.2 / §6.2）· [模块接口说明](../../模块接口说明.md) · [Python后端冷启动优化方案](../runtime/Python后端冷启动优化方案.md) · [TUI 使用指南](../../TUI使用指南.md) · [TUI 指令集](../../TUI指令集.md)
>
> **前置条件**：T1-T8 已完成。T9 原型可在现有网关上开始；正式接线必须先冻结 chat SSE、会话持久化和请求级路由契约。分布式真机验收依赖《总体下一步计划》L1-1/L1-2/L1-3。

> **主仓现行补充（2026-09-18）**：当前统一入口使用 `src/tui_textual.py`，旧标准库 TUI、旧聊天页和启动画面已归档到 `_to_delete/`；`qlh` 无参/`qlh chat` 使用 Textual 外壳，本机可由 `tui_backend.py` 在进程内启动后端。当前外壳为 9 个功能屏 + 1 个调试兜底屏：模型、集群、节点、日志、设备、设置等稳定流程由专用交互承载；`/openapi.json` 页面只用于尚未接线的研发/运维 JSON 合同，不计作功能覆盖。本机 `api_server` OpenAPI 快照为 152 个操作，SSE 聊天/上传仍走专用页面。本文早期关于“7 屏、标准库入口、Textual 聊天独立进程、默认入口尚未完成”的段落属于历史网关/发布验收记录，现行启动和功能口径以 [TUI 使用指南](../../TUI使用指南.md) 为准；外部干净机、Linux 打包和真实分布式仍是发布/实验验收，不阻塞主仓 TUI 使用。

---

## 一、背景与目标

### 1.1 背景

以下内容是 2026-08-02 的历史网关适配基线：当时 `tui_admin.py`（70KB，纯标准库）经 HTTP 操作集群，覆盖 7 个管理屏，另加 `ChatScreen` 共 8 屏；当时审计为 38 处调用点 / 25 个唯一端点。当前主仓 TUI 已切换到 Textual 9 个功能屏 + 1 个调试兜底屏，现行口径见本文上方补充和 [TUI 使用指南](../../TUI使用指南.md)。

- **可行性高**：全部同步 JSON、零流式/SSE/WebSocket、零 cookie、错误解析只依赖 JSON `detail` 且有容错。
- **3 类真实风险**：① 网关若按 §4.2 内部契约实现则 TUI 端点大面积 404（内部契约 ≠ 对外契约）；② DELETE 空响应/204 致 TUI 误报；③ 数值字段类型、`/api` 前缀、`X-QLH-Log-Token` 语义必须原样保留。

### 1.2 目标

1. 网关对外端点按 **api_server 现有路径/方法/字段** 原样实现，覆盖 TUI 全部 25 个唯一端点。
2. `tui_admin.py` **零改动**跑通 7 屏 × 2 角色（主节点 / 从节点视角）。
3. 以 TUI 38 个调用点为用例建立**契约测试**，纳入阶段 2 Go/No-Go 验收。

### 1.3 T1-T8 基线非目标（历史边界）

以下条目只约束已经完成的 T1-T8 网关适配阶段，不约束本文新增的 T9：

- 不改 `tui_admin.py` 代码（零改动是验收项，不是可选项）。
- 不实现 TUI 未调用的端点（如 chat/stream、models/download 等按主计划其他域处理）。
- 不做 TUI 功能增强（如新增屏幕、流式日志）。

---

## 二、TUI 依赖面审计基线（2026-08-02 复核）

### 2.1 客户端实现事实（`src/tui_admin.py`）

| # | 事实 | 位置 | 对网关的要求 |
|---|------|------|--------------|
| 1 | URL 硬编码 `base_url + "/api" + path` | :218 | **`/api` 前缀必须原样保留**，无 `base_url` 可配路径前缀 |
| 2 | query 用 `urlencode` 过滤空值 | :220-222 | 空值参数（None/""）不发送，网关不得因缺参 400（参数均有默认值语义） |
| 3 | 请求头 `Accept: application/json`；有 body 时 `Content-Type: application/json` | :224-227 | 响应必须 JSON；无 body 的 POST/DELETE 不要求 Content-Type |
| 4 | `X-QLH-Log-Token` 仅 `with_log_token and self.log_token` 才带 | :228-229 | **允许无 token 请求日志端点**；有 token 时按原语义校验 |
| 5 | 错误解析：HTTPError → 读 `detail`（dict 时取 `message`） | :234-244 | 错误体必须是 JSON `detail`；**禁止 302/纯文本/204 空体** |
| 6 | 空响应返回 `{}`；非 JSON 回退 `{"detail": text}` | :254-259 | 空响应会使 TUI 显示"—"，不崩但功能退化；**DELETE 必须返回 JSON** |
| 7 | 3s 轮询 3 个屏幕（Dashboard/Nodes/Queue） | :552、:599、:675、:992 | 网关不得引入 429/限流（urllib 无重试，直接报错） |

### 2.2 端点依赖全表（38 调用点 → 25 唯一端点）

> 行号以 2026-08-02 源码为准。`*` 标注的端点路径参数经 `urllib.parse.quote` 编码（:800、:809、:1076）。

| # | 端点（隐含 `/api` 前缀） | 方法 | 调用点 | 屏幕 | 读取的响应字段 |
|---|--------------------------|------|--------|------|----------------|
| 1 | `/health` | GET | :602、:1407 | 总览/设置 | `status=="ok"` |
| 2 | `/status` | GET | :603 | 总览 | `run_mode`、`node_role`、`node_id`、`max_nodes`、`model_loaded`、`model_name`、`active_model_id`、`current_quant`、`gpu{name,allocated_mb,total_mb,utilization}`、`kv_cache{total_tokens,allocated_pages,max_pages,estimated_memory_mb,rounds}`、`device{tier_label,tier,score,warnings}` |
| 3 | `/models/current` | GET | :604 | 总览 | `engine`、`device`、`total_params`、`gpu_allocated_gb`（float） |
| 4 | `/cluster/my-role` | GET | :605、:679 | 总览/节点 | `is_master`、`is_provisional`、`runtime_node_role`、`node_role`、`node_id` |
| 5 | `/cluster/nodes` | GET | :678 | 节点 | `count`、`online_count`、`offline_count`、`nodes[]{node_id,role,node_type,state,address,network_type,avg_rtt_ms,task_count,error_count,last_heartbeat}` |
| 6 | `/cluster/invite` | GET | :683 | 节点（主） | `master_host`、`master_port`、`node_count`、`max_nodes`、`identity_verified` |
| 7 | `/cluster/spare-master` | GET | :684 | 节点（主） | `spare_master{node_id,hostname,is_online}` |
| 8 | `/cluster/master-health` | GET | :686 | 节点（从） | `master_online`、`master_host`、`master_port`、`last_seen_seconds_ago`、`stale` |
| 9 | `/cluster/discover` | GET | :758、:848 | 节点 | `found`、`master_host`、`master_port`、`source`、`stale` |
| 10 | `/cluster/connect` | POST | :777 | 节点 | `status`、`message` |
| 11 | `/cluster/nodes/register` | POST | :789 | 节点 | `status`、`reason`、`message` |
| 12 | `/cluster/nodes/{id}/deregister` * | POST | :800 | 节点 | `status` |
| 13 | `/cluster/nodes/{id}` * | DELETE | :809 | 节点 | `status` |
| 14 | `/cluster/transfer-master` | POST | :818 | 节点 | `status`、`message` |
| 15 | `/cluster/spare-master` | POST | :827 | 节点 | `status`、`message` |
| 16 | `/cluster/spare-master` | DELETE | :833 | 节点 | `status`、`message` |
| 17 | `/cluster/config/max-nodes` | PUT | :844 | 节点 | `status` |
| 18 | `/cluster/transfer-logs` | GET | :856 | 节点 | `logs[0]{direction,from_role,to_role,related_node}`、`count` |
| 20 | `/cluster/reset-identity` | POST | :876 | 节点 | `status` |
| 21 | `/cluster/config/distributed-inference` | GET | :889 | 分布式 | `enabled`、`default` |
| 22 | `/cluster/config/distributed-inference` | PUT | :953 | 分布式 | `status` |
| 23 | `/cluster/layers` | GET | :890 | 分布式 | `total`、`strategy`、`computed_at`、`assignments[]{node_id,role,start_layer,end_layer,has_embedding,has_lm_head,score}` |
| 24 | `/cluster/layers` | PUT | :976 | 分布式 | `status`、`message` |
| 25 | `/cluster/layers` | DELETE | :982 | 分布式 | `status` |
| 26 | `/cluster/config` | GET | :891 | 分布式 | `network{server_ip,server_port,heartbeat_interval_s}`、`model{quant_type,page_size,max_page_num,max_seq_len}` |
| 27 | `/cluster/queue` | GET | :995 | 队列 | `paused`、`strategy`、`current_task`、`queue_size`、`max_size`、`q0_depth`、`q1_depth`、`q2_depth`、`completed_count`、`aging_params{q0_max_tokens,q1_max_tokens,q1_to_q0_s,q2_to_q1_s}`、`preempt_stats{count,total_overhead_ms}`、`q0/q1/q2[]{task_id,original_level,wait_seconds,max_new_tokens,is_aged,session_id}` |
| 28 | `/cluster/queue/strategy` | POST | :1053 | 队列 | `strategy` |
| 29 | `/cluster/queue/pause` | POST | :1057 | 队列 | （不读） |
| 30 | `/cluster/queue/resume` | POST | :1061 | 队列 | （不读） |
| 31 | `/cluster/queue/clear` | POST | :1067 | 队列 | `cleared` |
| 32 | `/cluster/queue/task/{id}` * | DELETE | :1076 | 队列 | `message`、`success` |
| 33 | `/device/profile` | GET | :1089 | 画像 | `os{system,release}`、`hostname`、`cpu{model/brand,physical_cores,logical_cores}`、`ram/memory{total_gb,available_gb}`、`disk{free_gb,total_gb}`、`gpus[]{name,gpu_type,cuda_available,vram_total_gb}`、`selected_gpu_index`、`tier_label`、`tier`、`score_total`、`recommendations`、`warnings` |
| 34 | `/device/select-gpu` | POST | :1157 | 画像 | `selected_gpu{name}`、`selected_gpu_index`、`warning` |
| 35 | `/device/auto-configure` | POST | :1166 | 画像 | `applied_config{description}`、`tier`、`score` |
| 36 | `/logs/recent?limit&level` | GET | :1193-1196 | 日志（远程） | `count`、`matched`、`buffer_size`、`buffer_capacity`、`logs[]{level,time/timestamp,name,message}` |
| 37 | `/logs` | GET | :1198 | 日志（远程） | `files[]{name,size,modified}` |
| 38 | `/logs/stats` | GET | :1199 | 日志（远程） | `log_dir`、`files_count`、`files_total_bytes`、`buffer_size`、`buffer_capacity`、`buffer_total_seen`、`buffer_dropped_estimate`、`levels{}`、`nodes{}` |

> 36-38 三个日志端点在**本地模式**（TUI 与后端同机）下不走 HTTP（直接读 `logs/` 目录，:1201-1230），仅**远程模式**（`--host` 指向远端主节点）走 HTTP 并带 token（:1193-1199）。

---

## 三、网关适配要求

### 3.1 五个必须守住的细节（对应 §2.1 风险）

| # | 细节 | 违反后果 | 网关实现要求 |
|---|------|----------|--------------|
| ① | `/api` 前缀原样保留 | TUI 全部请求 404 | 网关挂载路径 `/api/*`，无额外前缀配置 |
| ② | 错误体必须是 JSON `detail`；DELETE 必须返回 JSON | TUI 误报"失败"或显示内部错误 | 统一异常过滤器输出 `{"detail": ...}`；`queue/task/{id}`、`nodes/{id}`、`spare-master`、`layers` 的 DELETE 返回 `{"status": ...}`/`{"message","success"}` 等原字段，**禁止 204** |
| ③ | 数值字段保持 number | `float()` 转换抛 ValueError → 屏幕显示"内部错误" | `avg_rtt_ms`、`wait_seconds`、`gpu_allocated_gb`、`utilization`、`score`、`q*_depth`、`size` 等序列化时禁止字符串化；TS 侧用 `number \| null` 类型定义 |
| ④ | `/api/health` 返回 JSON `status=="ok"`（网关先行拓扑下亦然） | 健康检查屏显示异常 | 网关自身存活即返回 200 `{"status":"ok"}`，**不得 503/纯文本**；模型/调度未就绪是 `/status` 的 `model_loaded` 等字段的事 |
| ⑤ | `/api/status` 的 `gpu`/`kv_cache`/`device` 内嵌对象结构原样 | TUI 显示"—"功能退化 | 聚合响应时保持嵌套字段名；扁平化即视为契约破坏 |

**软要求**：① 响应必须 UTF-8 JSON（TUI 硬编码 utf-8 decode，:233）；② 3s 轮询不触发 429 限流；③ 无 token 的日志请求必须返回 JSON（允许放行或 401 JSON，**禁止 302 重定向**——urllib 会跟随重定向导致行为不可控）。

### 3.2 端点 → 后端服务映射（网关实现方式）

| TUI 端点（#） | 后端 | 网关实现方式 |
|---------------|------|--------------|
| `/health`（1） | 无 | **内嵌**：网关自身返回 `{"status":"ok"}` |
| `/status`（2） | scheduler-svc + inference-svc + 本地 | **聚合**：scheduler `/v1/status` + inference `/v1/status`（模型/显存）+ 网关本机画像缓存；嵌套字段按 §2.2 #2 原样输出 |
| `/models/current`（3） | inference-svc | 代理：`GET /v1/models/current` |
| `/cluster/my-role`、`nodes`、`invite`、`spare-master`(GET)、`master-health`、`discover`（4-9） | scheduler-svc | 代理（scheduler-svc 需按 §2.2 字段提供；内部端点形态自行决定，见主计划 §4.2 对外适配原则） |
| `/cluster/connect`、`nodes/register`、`nodes/{id}/deregister`、`nodes/{id}`(DELETE)、`transfer-master`、`spare-master`(POST/DELETE)、`config/max-nodes`、`transfer-logs`、`reset-identity`（10-19） | scheduler-svc | 代理 |
| `/cluster/config/distributed-inference`(GET/PUT)、`layers`(GET/PUT/DELETE)、`config`(GET)（21-26） | scheduler-svc | 代理；**保留 PUT/DELETE 方法语义**（对外 ≠ 内部 POST 设计） |
| `/cluster/queue*`（27-32） | scheduler-svc | 代理；`queue/task/{id}` DELETE 透传 JSON 响应 |
| `/device/profile`、`select-gpu`、`auto-configure`（33-35） | scheduler-svc（采集库 device_profiler 留 Python） | 代理 |
| `/logs/recent`、`/logs`、`/logs/stats`（36-38） | legacy-control（阶段 2）→ control-svc（阶段 3） | 代理，**透传 `X-QLH-Log-Token`**，允许无 token |

### 3.3 时序决策：日志端点的承载进程

`/logs/*` 的 `buffer_*` 字段来自 FastAPI 进程内 logging 内存缓冲（非纯文件读取），TS 侧无法在阶段 2 低成本复刻。因此：

- **阶段 2 期间**：主计划 §2.2 允许"控制面域端点暂由 FastAPI 遗留进程承载"——从 api_server 剥离出**控制面遗留进程 `legacy-control`**（端口 `QLH_LEGACY_CONTROL_PORT`=8040，仅挂 logs/review/sessions/conversations/settings/bootstrap 等控制面路由），网关对 `/logs/*` 反向代理并透传请求头。迁移域端点（chat/cluster/queue/device 等）从遗留进程移除，实现主计划 §2.5"端点壳退役"。邮件能力已裁撤，不进入 legacy-control。
- **阶段 3**：control-svc 就绪后，`/logs/*` 改代理到 control-svc，legacy-control 退役。**契约（路径/字段/token 语义）全程不变**。

---

## 四、契约测试设计

### 4.1 用例清单

测试文件：`gateway/test/tui-contract.e2e-spec.ts`（Jest + Supertest，连真实网关与各服务）

- **用例 1-38**：§2.2 全表 38 个调用点逐一用例——方法 + 路径 + 参数（含 `urllib.parse.quote` 编码的路径参数）命中 200/预期码，响应体 JSON 可解析。
- **用例 39-42（5 项细节断言）**：
  - 39：DELETE `/api/cluster/queue/task/{id}`、`/api/cluster/nodes/{id}`、`/api/cluster/spare-master`、`/api/cluster/layers` 均返回 JSON 且非空体（`Content-Type: application/json`，body ≠ ""）。
  - 40：数值字段类型断言——`/status.gpu.utilization`、`/status.device.score`、`/cluster/nodes.nodes[].avg_rtt_ms`、`/cluster/queue.q0[].wait_seconds`、`/models/current.gpu_allocated_gb` 均为 `number`（`typeof === "number"`，null 允许，字符串禁止）。
  - 41：`GET /api/health` 返回 `{"status":"ok"}`；无 `X-QLH-Log-Token` 请求 `/api/logs/recent` 返回 JSON（非 302）。
  - 42：`/api/status` 响应含 `gpu`/`kv_cache`/`device` 三个嵌套对象且字段名与 §2.2 #2 一致。
- **用例 43（错误契约）**：构造 4xx（如不存在的节点 id DELETE），断言响应体为 JSON 且含 `detail` 字段。

### 4.2 断言模板（TS）

```typescript
// gateway/test/tui-contract.e2e-spec.ts（节选）
import request from 'supertest';
import { app } from '../src/app';

describe('TUI 契约：/api 前缀与 JSON 错误', () => {
  it('用例 1: GET /api/health 返回 status=ok', async () => {
    const res = await request(app).get('/api/health');
    expect(res.status).toBe(200);
    expect(res.body.status).toBe('ok');
  });

  it('用例 32: DELETE /api/cluster/queue/task/:id 返回 JSON 非空体', async () => {
    const res = await request(app).delete('/api/cluster/queue/task/nonexistent');
    expect(res.status).toBe(404);                    // 或 200，视 scheduler 语义
    expect(res.headers['content-type']).toContain('application/json');
    expect(res.text).not.toBe('');                    // 禁止 204 空体
  });

  it('用例 40: 数值字段类型必须为 number', async () => {
    const res = await request(app).get('/api/status');
    expect(typeof res.body.gpu?.utilization).toBe('number');
    expect(typeof res.body.device?.score).toBe('number');
  });

  it('用例 41: 无 token 日志请求返回 JSON 而非重定向', async () => {
    const res = await request(app).get('/api/logs/recent');
    expect(res.status).not.toBe(302);
    expect(res.headers['content-type']).toContain('application/json');
  });
});
```

> 说明：§2.2 中"（不读）"的端点（queue/pause、queue/resume）只需断言状态码与 JSON 响应，不校验字段。

---

## 五、实施步骤（任务清单）

> 执行顺序：契约测试先行（TDD），再实现网关适配，最后双网关对比与 TUI 实测。每任务完成后 commit。

- [x] **T1 建立契约测试骨架** ✅ 2026-08-02
  - 新建 `gateway/test/tui-contract.e2e-spec.ts`，按 §4.1 列出 43 个用例（未实现网关适配前允许 skip 标记）。
  - 验收：`npm test -- tui-contract` 可运行（允许 fail/skip），测试清单与 §2.2 全表一一对应（人工核对 38 项）。

- [x] **T2 网关基础设施（若主计划 2.1 已完成则跳过）** ✅ 2026-08-02
  - NestJS + Fastify adapter、`/api` 前缀挂载、request-id 中间件、异常过滤器输出 JSON `detail`（对齐 `api_server.py:319-366`）。
  - 验收：`curl -H "Accept: application/json" localhost:8000/api/health` 返回 `{"status":"ok"}`；未匹配路由返回 JSON 404 `{"detail": ...}`。

- [x] **T3 集群/队列/分层端点代理（#4-32）** ✅ 2026-08-02
  - 实现 scheduler-svc 客户端模块 `gateway/src/clients/scheduler.client.ts`（透传 + 502 兜底 + 非 2xx 转 HttpException）；`gateway/src/modules/cluster/cluster.controller.ts`（`@All('cluster/*')` 泛化转发，保留 PUT/DELETE 方法语义）。
  - **实现方式：1:1 透传代理**——内部端点 = 对外路径去掉 `/api` 前缀（见主计划 §4.2 落地决策）；scheduler-svc 未来实现按此 + §2.2 字段对齐。
  - 测试桩 `gateway/test/fake-scheduler.ts`：§2.2 字段对齐 + 真实 DELETE 语义（queue/task 不存在 → 200 success:false；nodes/{id} 不存在 → 404 detail，对齐 `api_server.py:5715-5732/:5266-5285`）。
  - 验收：用例 4-32、39 转绿（34 passed / 10 skipped）；`contract_diff.py` 双网关对比留待阶段 2 整体验收（旧网关尚未部署）。

- [x] **T4 设备画像端点代理（#33-35）** ✅ 2026-08-02
  - `gateway/src/modules/device/device.controller.ts`：与 cluster 同模式的 `@All('device/*')` 透传代理到 scheduler-svc（采集库 device_profiler 留 Python，见 §3.2 决策）。
  - fake-scheduler 补 3 条 device 路由（profile 含 os/cpu/ram/memory/disk/gpus/tier/score 全字段，数值字段 number）。
  - 验收：用例 33-35 转绿（37 passed / 7 skipped）。

- [x] **T5 状态聚合端点（#1-3）** ✅ 2026-08-02
  - `gateway/src/clients/forward-client.ts`：公共转发基类（SchedulerClient/InferenceClient 继承，消除复制）；`inference.client.ts` 新增（QLH_INFERENCE_URL，默认 :8010）。
  - `gateway/src/modules/status/status.controller.ts`：GET /api/status 并行聚合 scheduler `/cluster/status`（run_mode/node_role/node_id/max_nodes）+ inference `/v1/status`（model/gpu/kv_cache）+ `/device/profile`（device 摘要），单源失败回落默认值不 500（网关先行语义）；`modules/models/models.controller.ts`：/api/models/current 透传 inference `/v1/models/current`。
  - fake-inference.ts 测试桩（/v1/status、/v1/models/current，数值字段 number）；fake-scheduler 补 /cluster/status。
  - 验收：用例 2、3、40、42 转绿（41 passed / 3 skipped）。

- [x] **T6 日志端点代理（#36-38）** ✅ 2026-08-02
  - **legacy-control Python 桩**：`src/legacy_control.py`（纯标准库零依赖，未来 legacy-control 进程原型；`/logs/recent`、`/logs`、`/logs/stats`，X-QLH-Log-Token 可选，stdout 探活 `LEGACY_CONTROL_LISTENING:<port>`）。**⚠️ 已随微服务叫停删除（2026-09-14），本条为历史记录。**
  - 网关 `gateway/src/modules/logs/logs.controller.ts`：`/api/logs` 与 `/api/logs/*` 均透传 legacy-control（拆两个方法——fastify adapter 下同一方法叠加多个 @All 会覆盖）；`ForwardClient` 增加 `extraHeaders` 参数透传 X-QLH-Log-Token；`clients/legacy.client.ts`（QLH_LEGACY_CONTROL_URL，默认 :8040）。
  - 测试：jest 直接 spawn Python 桩（真实网关→Python 链路）；用例 36-38 打开；用例 41 补 logs 段 status=200 + 带 token 透传断言。
  - 验收：**44/44 全部用例通过（0 skipped）**。

- [x] **T7 TUI 实测（验收主体）** ✅ 2026-08-02（自动化走查通过）
  - 主节点角色：7 屏全走查（总览/节点/分布式/队列/画像/日志/设置），动作面全执行一遍（连接、注册、注销、删除、分层覆盖、队列策略、备用主节点设置等）。
  - 从节点视角：`tui_admin.py --host <主节点> --port 8000`（或 Tailscale IP），重点验证 `/cluster/master-health`（#8）与远程日志（#36-38）。
  - 验收：**tui_admin.py 零改动**，7 屏 × 2 角色全部通过；无"内部错误"屏（数值字段类型问题会在此暴露）。
  - **自动化走查**：`scripts/tui_walkthrough.py --mode master|client`（驱动 `tui_admin.py --plain`，喂入全动作输入序列，断言屏幕齐全 + 无错误屏 + 动作结果全出现）。配套 `scripts/dev_stubs.py --client-mode`（模拟从节点身份）。实测：master 7 屏 × 23 动作全通过；client 从节点视角（Dashboard + Nodes master-health 分支 + 远程日志）通过。
  - 排障记录：① TUI 对 `distributed-inference`/`layers`/`max-nodes` 用 **PUT**（act_toggle/act_override/act_max_nodes），桩补 PUT 路由；② `act_connect` 在主节点身份下有隐藏 confirm（切换为从节点），走查序列需补输入；③ 日志屏渲染桩日志 ERROR 级记录为 `[错误] 20xx-...` 行，属合法日志内容，错误屏断言需排除时间戳格式行。

- [x] **T8 回归与收尾** ✅ 2026-08-02
  - `pytest -q`（Python 侧回归）；前端 9/9（确认网关改动未破坏 Web 面）。
  - 更新 微服务架构改造计划 §2.4 勾选状态与本文档状态为"现行"。
  - 验收：全部测试绿，提交。
  - 2026-08-02 实测：Python 全量回归 **1086 passed / 3 skipped**（112.6s）；前端 `npm test` **9/9** + 生产构建成功；网关契约测试 **63/63**（tui-contract + rest-contract）；TUI 7 屏 × 2 角色自动化走查通过（`scripts/tui_walkthrough.py`）。已提交。

---

## 六、T1-T8 验收标准（Go/No-Go）

| # | 项 | 通过标准 |
|---|----|----------|
| 1 | TUI 零改动 | `git diff src/tui_admin.py` 为空 |
| 2 | 端点覆盖 | §2.2 全表 38 个调用点全部可用（HTTP 状态 + JSON 响应） |
| 3 | 5 项细节 | §3.1 ①-⑤ 全部满足，用例 39-42 绿 |
| 4 | 双角色实测 | 7 屏 × 主/从角色走查通过，无"内部错误"屏 |
| 5 | 日志远程模式 | 无 token 可访问（JSON 响应）；带 token 行为与旧网关一致 |
| 6 | 性能 | 3s 轮询无 429；`/status` 聚合响应 P95 < 500ms（本地） |
| 7 | 回归 | Python `pytest -q` 绿；前端 9/9；主计划阶段 2 其他域不受影响 |

---

## 七、风险与回退

| # | 风险 | 缓解 / 回退 |
|---|------|--------------|
| 1 | scheduler-svc 内部端点形态与 TUI 对外字段不一致 | 网关适配层承担映射（对外字段以 §2.2 为准），scheduler-svc 不背 TUI 契约；契约测试兜底 |
| 2 | `/logs/*` buffer 统计无法在 TS 复刻 | 已决策：阶段 2 由 legacy-control 承载（§3.3），网关仅代理；若 legacy-control 构建受阻，回退为阶段 2 内嵌 TS 实现 `logs/recent`（文件尾部读取）+ 容错 `buffer_*` 字段（返回 0/空对象，TUI 显示为 0 不崩） |
| 3 | DELETE 响应被 NestJS 默认序列化吃掉（204） | 控制器显式 `return { ... }`，禁止 `@HttpCode(204)`；用例 39 强制非空 JSON |
| 4 | 数值字段被 TS 序列化为字符串（如大数/undefined） | 类型定义 `number \| null` + 用例 40 断言；网关 JSON 序列化用 `JSON.stringify` 默认行为（undefined 字段直接省略——TUI `.get()` 容错为"—"，可接受） |
| 5 | 聚合 `/status` 延迟影响轮询 | 聚合只读内存态/轻接口；若 scheduler `/v1/status` 慢，加 500ms 本地缓存，字段仍以 §2.2 #2 为准 |
| 6 | 从节点角色访问主节点端点被 CORS/鉴权拦截 | 网关不引入新鉴权（现无）；`master-health` 等从节点视角端点按旧语义放行 |

### 7.1 已知技术坑（2026-08-02 T1 排障记录）

> 完整排障链与验证方式见历史 `gateway/README.md`（**网关已随微服务叫停删除，2026-09-14**；原文见 git 历史 `148a0a6` 之前版本）。任何改动 fastify 版本、
> beforeAll 初始化或 404 处理的 PR 必须先跑 `npm run test:tui`。

| # | 坑 | 根因 | 修复 |
|---|----|------|------|
| 1 | `@nestjs/platform-fastify@11.1.28` 硬编码 `fastify@5.10.0`（精确版本），npm 嵌套安装两个 fastify | platform-fastify 的 `dependencies` 是精确版本，无法 dedupe | `package.json` 加 `"overrides": { "fastify": "^5.11.0" }` 统一 |
| 2 | supertest 首个请求崩溃 `Cannot read properties of undefined (reading 'length')`，且 jest `did not exit` | NestJS `app.init()` 不等待 fastify `ready()`；fourOhFour 404 context 的 hooks 在 `preReady` 才初始化，ready 前 `context.preParsing` 为 undefined 而非 null | 测试 `beforeAll` 中 `await (app.getHttpAdapter().getInstance() as any).ready()`（**T2-T6 每个测试文件都必须带**）；生产 `app.listen()` 内部会等 ready，不受影响 |

---

## 八、决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-08-02 | 网关对外端点以 api_server 现有路径/方法/字段为准，§4.2 内部契约不承担对外适配 | TUI 38 调用点审计证明对外面是唯一可信契约；内部端点形态由各服务自定 |
| 2026-08-02 | `/logs/*` 阶段 2 由 legacy-control 承载、网关代理；阶段 3 切 control-svc | `buffer_*` 是进程内内存统计，TS 复刻成本高；代理保持契约不变 |
| 2026-08-02 | 5 项细节（§3.1）以契约测试固化，不做口头约定 | 防止网关实现漂移；用例 39-42 进阶段 2 Go/No-Go |
| 2026-08-02 | TUI 零改动为硬验收（`git diff` 为空） | TUI 是纯 urllib 客户端，改动任何一处都意味着契约已破坏 |
| 2026-08-05 | 保留纯标准库管理 TUI，另建可选依赖聊天前端 | 管理端需要极低部署门槛；聊天端需要异步 SSE、Markdown、长文本滚动和多行输入，不应继续手写完整终端框架 |
| 2026-08-05 | T9 先统一真流式与会话提交契约，再做视觉页面 | 当前 `full` 有完整会话但只发最终事件，`fast` 真流式但语义较轻；客户端不能用重复推理补持久化 |

---

## 九、下一阶段 T9：简化聊天页面（规划）

### 9.1 定位

T9 在现有 7 屏管理能力之外增加第 8 个“对话”入口，交互风格参考 Claude Code 的终端会话，但只吸收适合 QLH 的部分：

- 键盘优先、持续可滚动的对话 transcript。
- 固定在底部的多行输入区，发送后响应原位真流式增长。
- `/` 命令、历史导航、会话新建/恢复和明确的取消操作。
- 顶部/底部状态条展示连接、模型、请求路由和生成状态。
- 终端重绘后保留输入与对话，不因窗口缩放丢状态。

T9 不是 Claude Code 的代码 Agent 复刻。首期不实现 shell 执行、文件编辑、工具权限确认、子 Agent、Git 操作、后台 Bash 或自动读取当前仓库；这些能力既不是 QLH 推理系统的必要条件，也会显著扩大安全面。

官方交互参考：

- [Claude Code Interactive mode](https://code.claude.com/docs/en/interactive-mode)：快捷键、多行输入、命令历史、`Ctrl+R`、transcript 和屏幕重绘。
- [Claude Code Common workflows](https://code.claude.com/docs/en/common-workflows)：恢复历史会话和并行会话工作流。

### 9.2 页面布局

```text
┌ QLH Chat ─ 会话: 调试流水线 ─ 模型: Qwen ─ 连接: master@100.x ┐
│                                                                  │
│ You                                                              │
│   为什么这个请求没有走从节点？                                   │
│                                                                  │
│ Assistant                                                        │
│   正在生成的 Markdown / code block / 普通文本……                  │
│                                                                  │
│   Pipeline 分布式 · master -> worker-01 · 42 tokens · 18.3 tok/s │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│ > 多行输入区                                                     │
│   Enter 发送 · Alt+Enter/Ctrl+J 换行 · Ctrl+C 停止               │
├──────────────────────────────────────────────────────────────────┤
│ route:auto  dist:on  history:on  /help  /sessions  /status       │
└──────────────────────────────────────────────────────────────────┘
```

布局要求：

- Transcript 为主要区域，不把每条消息包成装饰性卡片；角色、正文和每轮指标按稳定行距排列。
- 输入区高度随内容在 1-8 行内增长，超过后内部滚动，不能挤掉状态栏。
- 终端宽度不足时指标换行；最小支持 60×18，低于下限时显示明确提示并允许退回管理 TUI。
- 中文、宽字符、emoji、Markdown code fence 和 ANSI 转义必须正确计算显示宽度。
- 生成期间只更新最后一条 assistant 消息，不整页闪烁，不因 token 到达改变输入焦点。

### 9.3 技术路线

建议采用“双入口、共享客户端”而不是直接重写当前稳定管理端：

| 层 | 方案 | 说明 |
|---|---|---|
| 当前管理入口 | 保留 `src/tui_admin.py` | 纯标准库、7 屏、`--plain` 和单命令模式继续可用 |
| 新聊天入口 | 新建 `src/tui_chat.py`，成熟后拆为 `src/tui/` 包 | 首期命令为 `bjtu chat`；验收完成前不替换 `bjtu` 默认入口 |
| 终端框架 | 首选 Textual | 使用 App/Screen/Widget、异步 worker、Markdown/滚动容器和 TextArea，避免继续手写复杂流式布局 |
| HTTP/SSE | `httpx.AsyncClient` | 处理 POST SSE、连接超时、取消、重连和请求 ID；按行解析现有 `data:` 事件 |
| 共享层 | `TuiApiClient`、命令注册与格式化函数 | 管理/聊天入口共享模型、会话、路由和 metrics 解析，不复制端点字符串 |

引入第三方依赖是有意决策，但必须隔离：

1. 在 `qlh-shell/requirements-tui.txt` 锁定与项目 Python 3.10-3.12、现有 `httpx` 兼容的 Textual 版本。
2. 源码用户使用 `qlh-shell/.venv-tui` 或已有项目虚拟环境；禁止静默写入系统/全局 Python。
3. `bjtu chat` 缺依赖时只显示检测结果和明确命令，可在用户确认后调用 `QLH_SHELL_ROOT/scripts/setup_tui_env.py` 创建 `qlh-shell/.venv-tui`。
4. 安装引导尊重 `HTTPS_PROXY`、`PIP_INDEX_URL` 和用户指定镜像；网络失败保留重试命令，不让启动器循环弹窗。
5. Windows/Linux 正式安装包直接携带 TUI 运行依赖，不要求最终用户首次启动联网安装。
6. 可提供 wheelhouse 离线安装路径；无 Textual 时当前管理 TUI 和 `--plain` 不受影响。

### 9.4 聊天与会话契约

复用现有接口：

| 接口 | T9 用途 |
|---|---|
| `POST /api/chat/stream` | 发送消息与 SSE token/done/error 事件 |
| `POST /api/chat/generations/{id}/cancel` | `Ctrl+C` 或 `/cancel` 取消当前生成 |
| `GET/POST/PUT/DELETE /api/sessions*` | 会话列表、新建、重命名、切换和删除 |
| `GET/DELETE /api/conversations` | 恢复/清空当前会话消息 |
| `GET /api/models/current` | 顶栏模型与引擎 |
| `GET /api/cluster/config/distributed-inference` | 显示集群全局分布式开关，不作为实际参与证据 |

正式实现前必须升级 chat stream 契约：

1. 增加版本化 `interactive` 流式模式，或让 `full` 同时支持逐 token 与完成时事务提交历史；不能先调 `fast` 再重复调用 `full`。
2. 客户端先生成合法 `generation_id` 并随请求发送，因此在收到第一个 token 前也能可靠取消。
3. SSE 至少定义 `start`、`token`、`done`、`error`、`cancelled` 语义；保持对旧纯 `data:` 客户端兼容。
4. `done` 必须包含最终正文、`metrics`、`generation_id`、`request_id`、会话提交结果和可选 followups。
5. 客户端断开后服务端触发取消；迟到 token 不得写入已切换会话。
6. 历史提交必须是 user + assistant 一次事务；取消时保留用户消息与已生成 partial 的策略必须先冻结，不能由客户端猜测。

TUI SSE parser 必须处理 TCP chunk 任意切分、一个 chunk 多事件、UTF-8 多字节拆分、尾部无空行、keepalive、错误事件和服务端提前关闭。

### 9.4.1 interactive 事件契约（T9.0 冻结，2026-08-06）

`streaming_mode=interactive` 已落地（`src/api_server.py` chat_stream；`ChatRequest` 增加 `routing_preference`，两处协议：`api_server.ChatRequest` 与 `inference_service/protocol.ChatRequest` 同步）。事件序列固定为：

```text
start -> token* -> done | error | cancelled
```

| 事件 | 字段 | 语义 |
|---|---|---|
| `start` | `start=true`、`generation_id`、`request_id`、`session_id`、`routing_preference` | 流开始；客户端可据此显示请求级路由意图 |
| `token` | `token=<delta>` | 增量文本，客户端原位 append；不携带累计全文 |
| `done` | `done=true`、`response`（全文）、`thinking_content`、`followups`、`metrics`（含 `generation_id`/`request_id`/`routing_preference`/`distributed_requested`）、`session_id`、`history_committed` | 生成完成；user+assistant 已按一次事务提交（`history_committed=false` 表示持久化跳过/失败，客户端可提示） |
| `error` | `error=<message>`、`request_id` | 失败；不提交历史 |
| `cancelled` | `cancelled=true`、`generation_id`、`request_id`、`session_id`、`partial` | 取消（含首 token 前）；不提交历史 |

规则：

1. `generation_id` 由客户端预生成（`gen_[a-f0-9]{12}` 格式已兼容），首 token 前即可用 `POST /api/chat/generations/{id}/cancel` 取消。
2. 从节点请求 interactive 返回 `error` 并引导连接主节点，不静默降级转发（聊天请求不绕过网关直打本地模型）。
3. 执行路径与 fast 一致：外部路由 → 流水线逐 token → 单机 PyTorch 逐 token → llama.cpp 假流式（整段作为单 token 事件）。
4. `routing_preference` 语义见 §9.5；当前版本把 `distributed_preferred`/`distributed_required` 回显到 `metrics.distributed_requested`，实际调度影响在 T9.5 落地。
5. 兼容性：`full`/`fast` 行为与事件格式不变；旧客户端不感知新字段。

测试证据：`tests/test_chat_interactive.py`（8 用例：事件序列/取消/错误/事务提交/路由偏好/从节点引导/真实 local_store 提交）、`tests/test_tui_sse.py`（13 用例：chunk 切分/UTF-8/多事件/keepalive/残帧）、`tests/test_tui_chat.py`（6 用例：Textual headless fixture 重放与命令）。

### 9.5 本地与分布式推理选择

当前全局 `distributedInference` 开关不足以表达单次请求意图。T9 计划给 `ChatRequest` 增加向后兼容的请求级字段：

```text
routing_preference = auto | local_only | distributed_preferred | distributed_required
```

语义：

| TUI 选择 | 后端行为 |
|---|---|
| `auto` | 沿用集群配置和 scheduler 决策 |
| `local_only` | 本请求只在服务主节点本地完整引擎执行，不临时修改全局分布式开关 |
| `distributed_preferred` | 优先选择合格分布式路径；不可用时允许本地回退并给出原因 |
| `distributed_required` | 没有合格分布式 Provider/Worker 时明确失败，不静默回退 |

“分布式”仍可能对应 PyTorch 层流水线、完整 Worker、任务链或后续其他 Provider。TUI 不根据开关、节点在线数或请求来源推断结果，只读取完成事件 metrics：

- `distributed_requested`
- `distributed_used`
- `execution_mode` / `route`
- `workers_used` / `layer_assignments`
- `actual_providers`
- `fallback` / `fallback_reason`
- `serving_node_id`

显示规则：

- `distributed_used=true`：展示实际执行模式和参与节点，例如 `Pipeline 分布式 · master -> worker-01`。
- 请求了分布式但 `distributed_used=false`：黄色显示 `已请求分布式，实际本地` 和后端原因。
- `fallback=true`：即使最终成功，也必须展示 fallback reason。
- 只有主节点本地执行时显示具体引擎，例如 `PyTorch 本地` 或 `llama.cpp 本地`，不显示模糊的“分布式：否”。
- 从节点 TUI 若连接的是从节点 API，应引导切换到当前主节点 endpoint；聊天请求不绕过网关直打本地模型。

### 9.6 输入、快捷键与命令

基础快捷键：

| 操作 | 默认键 | 说明 |
|---|---|---|
| 发送 | `Enter` | 输入为空时不发送 |
| 换行 | `Alt+Enter` 或 `Ctrl+J` | `Shift+Enter` 仅在终端协议能可靠区分时作为别名 |
| 停止生成 | `Ctrl+C` | 第一次取消当前 generation；空闲时不直接杀后端 |
| 退出聊天页 | `Esc` | 返回管理主菜单，后端保持运行 |
| 新会话 | `Ctrl+N` | 未提交输入先确认 |
| 历史搜索 | `Ctrl+R` | 搜索本地已加载的输入历史；不在每次按键时请求后端 |
| 重绘 | `Ctrl+L` | 修复终端残影，不清会话 |
| 滚动 | `PgUp/PgDn`、鼠标滚轮 | 新 token 到达时仅在用户位于底部才自动跟随 |

聊天输入框复用 `/` 命令体系，并新增：

- `/new`：创建并切换新会话。
- `/sessions`：打开会话选择器。
- `/resume <session_id>`：恢复历史会话。
- `/rename <title>`、`/delete-session`：会话管理。
- `/route auto|local|distributed|required`：设置请求级路由偏好。
- `/cancel`：取消当前 generation。
- `/clear`：清空当前会话，必须确认。
- `/thinking on|off`：控制是否请求/展示 thinking 内容。

现有 `/model`、`/models`、`/switch`、`/quant`、`/engine`、`/dist`、`/status`、`/nodes`、`/help`、`/quit` 继续可用。模型切换和会话切换期间必须先取消当前生成，禁止旧请求结果落入新模型或新会话。

### 9.7 内容渲染边界

首期支持：

- CommonMark 常用子集：段落、标题、列表、引用、行内代码、代码块和链接文本。
- 代码块保留空格与横向滚动，不执行内容。
- 思考内容默认折叠；只有请求 `show_thinking=true` 且后端确实返回时才可展开。
- 大响应使用虚拟化/增量追加，设置单条和会话显示上限；完整历史仍由后端持久化。
- 控制字符、终端 escape 和不可见字符必须转义，模型输出不能注入 ANSI 指令或改变终端标题。

首期不支持图片内联、富媒体、文件拖放、语音输入、工具调用 UI 和任意 HTML 渲染。项目已有 `/api/chat/upload`，待基本聊天稳定后再规划文本附件选择器。

### 9.8 实施步骤

- [x] **T9.0 契约冻结与 PoC** ✅ 2026-08-06
  - 冻结 `routing_preference`、interactive SSE 事件和会话取消/提交语义（§9.4.1）。
  - `src/tui_chat.py`（Textual + httpx）单页 PoC：fixture 重放与真实后端双模式、Enter 发送/Alt+Enter 换行、Ctrl+C 取消、epoch 迟到事件 fencing、metrics 状态行。
  - 验收：`tests/test_chat_interactive.py` 8 + `tests/test_tui_sse.py` 13 + `tests/test_tui_chat.py` 6 全绿；管理 TUI 回归 `tests/test_tui_commands.py` 无回归（245 项相关回归全绿）。
  - 环境：`qlh-shell/requirements-tui.txt`、`qlh-shell/scripts/setup_tui_env.py`、`bjtu chat` / `bjtu.sh chat` 启动路由（缺依赖只提示不污染全局解释器）。

- [x] **T9.1 可选依赖与启动引导** ✅ 2026-08-06（主体完成）
  - `qlh-shell/requirements-tui.txt`（textual==8.2.8 + httpx）、`qlh-shell/scripts/setup_tui_env.py`（创建 `qlh-shell/.venv-tui`、幂等、尊重代理/镜像、`--wheelhouse` 离线安装、失败保留重试命令）。
  - `bjtu chat` / `bjtu.sh chat` 启动路由：缺依赖只显示检测结果与安装命令并 exit 2，不污染全局解释器；管理 TUI 与 `--plain` 不受影响。
  - Windows/Linux 安装包携带依赖的开发接线已在 T9.6 完成；干净环境实测截图仍属于 T9.6-R 发布门。

- [x] **T9.2 聊天页面与命令复用** ✅ 2026-08-06（含终端布局快照验收）
  - transcript（Markdown 增量渲染）、输入区（Enter 发送 / Alt+Enter、Ctrl+J 换行）、状态栏、滚动与第 8 屏入口（`bjtu chat`）。
  - 共享层 `src/tui_shared.py`：API_PATHS 端点常量、`build_interactive_request`、`format_metrics`、`parse_session_line`、命令注册表 `COMMAND_SPECS`/`help_text`、`resolve_route_arg`；纯标准库可导入（管理 TUI 亦可复用）；聊天页不再散落端点字符串。
  - 布局快照验收（2026-08-06）：`scripts/tui_chat_walkthrough.py` 在 80×24 / 120×30 / 60×18（窄终端下限）三尺寸 headless 驱动真实聊天页，18 项断言 × 3 尺寸 = 54/54 通过（布局/中文+emoji/代码块/流式完成/路由命令/帮助/新会话/取消 partial/退出），SVG 截图证据存 `build/tui-chat/`。

- [x] **T9.3 真流式、取消与错误恢复** ✅ 2026-08-06
  - interactive SSE、预生成 generation ID、Ctrl+C 取消（首 token 前后均可）与 epoch fencing（取消/切换会话时迟到事件丢弃）已落地。
  - 错误恢复状态机：`done`/`error`/`cancelled` 为终止事件；网络断开（Read/Connect/Timeout）显示“连接中断”并保留已生成 partial；连接正常关闭但无终止事件（空响应/服务端提前关闭）显示“流意外结束/连接意外结束”；生成失败只结束当前 assistant 占位，不清空 transcript。
  - 修复：Markdown widget 在 mount 完成前 update 会被构造参数覆盖（Textual 8 行为）——消息追加链 async 化（`await transcript.mount(md)` 完成后再 update）；Ctrl+C 在 TextArea 内拦截转发（内部 copy 绑定优先于 App binding）。
  - 验收：`tests/test_tui_chat.py` 17 用例（含空响应/error 保留 partial/断线/取消 partial/输入清空）+ 全量 T9 回归 126 passed + 终端走查 54/54（见 T9.2）。

- [x] **T9.4 会话与历史** ✅ 2026-08-06
  - 新建（POST /api/sessions）、恢复（POST /api/sessions/{id}/activate 返回历史）、重命名（PUT /api/sessions/{id}）、删除（DELETE /api/sessions/{id}）与 `/new` `/resume` `/rename` `/delete-session` 命令落地。
  - 启动时自动恢复最近活跃会话（`active_session_id` + activate 加载历史）。
  - 会话/模型切换先自动取消当前生成（不再拒绝切换）；epoch fencing 丢弃迟到事件，切换后旧流 token 不污染新会话。
  - 修复：parse_session_line 兼容后端 `id`/`session_id` 字段；`_clear_transcript` 全量清空避免重复 widget id。
  - 验收：`tests/test_tui_chat.py` 17 用例（含 new/resume/rename/delete/启动恢复/切换竞态）+ 全量 T9 回归 126 passed；主节点重启与 DB 断开的历史行为依赖真实环境（T9.6-R 发布验收复查）。

- [x] **T9.5 本地/分布式路由与真实指标**（local 部分完成 2026-08-06；分布式真机验收待物理从节点）
  - 请求级 `routing_preference` 接入执行路径（api_server 单体）：`local_only` 强制本地（跳过外部路由/主节点转发/分布式流水线，含 full 与 interactive/fast）；`distributed_required` 无分布式路径时明确失败（interactive/fast 发 error 事件、full 走既有 SSE error 语义），不静默回退；`distributed_preferred` 不可用时本地回退并标注 `fallback=true` + `fallback_reason`。
  - metrics 补全：`distributed_used`（实际走流水线时 true）、`fallback`/`fallback_reason`（请求分布式但本地执行）。
  - 验收：`tests/test_chat_interactive.py` 17 用例（local_only 客户端/主节点、required 失败/放行、preferred 回退 metrics、full 模式 local_only/required）+ 全量相关回归 263 passed。
  - 待办：inference-svc 的 interactive 历史事务提交（engine_host 薄实现 `history_committed=false` 如实上报）；物理从节点可用后验证真实参与节点与 fallback 跨端一致（TUI/Web/任务统计）。

- [ ] **T9.6 打包、回归与默认入口决策**（开发接线、Windows 发布构建、开发机实装和跨卷保留门完成 2026-08-11；外部干净机/Linux 发布门待验）
  - [x] 独立 `QLH Launcher` 建立 GUI/TUI 双入口，`bjtu launcher/ui/tui/update/version` 接线；Linux 将 Bootstrap `qlh-launcher` 与主应用载荷 `qlh-app` 分开。
  - [x] Windows 发布构建门：`requirements-cpu.txt` 锁定 `textual==8.2.8`；专用 CPU/CUDA venv 均完成主应用、控制台伴随程序、清单工具、跨卷保留 helper 和 Inno Setup 全量构建。最终 CPU 清单 `5015` 文件、deep `5016 checked / 0 failed`，安装器 `193205616` bytes（SHA-256 `fb3a4157...2ba02b7`）；CUDA 清单 `4736` 文件、deep `4737 checked / 0 failed`，安装器 `1504200870` bytes（SHA-256 `021a3514...63106f`）。安装器现在只读取已签名发布树，并在安装结束前执行 deep，不再用仓库文件覆盖签名内容或只验证 manifest 签名。
  - [x] Linux 开发接线与快速门：`.deb` 的包内 venv 由同一 `requirements-cpu.txt` 安装 Textual，`bjtu chat` 直接执行 `/opt/qlh-edge-inference/venv/bin/python src/tui_chat.py`；首次进入不安装依赖、不访问网络。`build-deb.sh cpu|cuda --preflight-only` 会拒绝 Windows 互操作命令并检查 Linux 原生 Node.js 18+/npm/git/CMake/C++/make/Python venv/dpkg/signing key。
  - [x] `T9.6-R2-WDEV` Windows 开发机实装门：CPU/CUDA Setup 分别安装到 `%LOCALAPPDATA%` 隔离同卷目录，安装期和独立 deep 均为 0 failed，`bjtu chat --help` 返回 0，冻结 EXE fixture 启动 5 秒无早退；五类哨兵数据完成 CPU/CUDA 卸载保留和重装关联，最后只清理本票数据，既有主节点状态未改。真实后端/TUI 契约回归 `34 passed`，打包/清单定向 `25 passed`。
  - [x] `UP-N6.4W` Windows 跨卷数据保留：外置事务 journal、空间预检、4MiB 流式复制、逐文件大小/SHA-256、整批 staging/commit、提交后源隔离删除和四阶段崩溃恢复已落地；同卷继续走原子改名。冻结 helper 完成真实 `G:`→`C:` retain/reassociate 往返，CPU Inno Setup 完成安装→跨卷卸载保留→同路径重装回迁，安装期 deep `5016/5016`；五类哨兵内容、marker、临时物清零和非用户程序文件边界均通过。全量 Python `1836 passed / 33 skipped`。
  - [ ] `T9.6-R2-EXT` 外部发布安装门：在不含既有 QLH 状态的干净 Windows 环境安装 CPU/CUDA Setup，复核环境变量、新 shell、真实模型后端会话和卸载；Linux 真机或完整 Linux 构建环境生成/安装 `.deb` 后走同一用例。当前 WSL 缺 Linux 原生 `node/cmake/c++/make`，`npm` 仅为不可用的 Windows 互操作路径，预检按预期 fail-closed。开发机隔离安装不替代该门。
  - 2026-08-11 本机预检（不计入外部发布安装门）：默认 `Qwen-1.8B-Chat` 的完整本地 Safetensors 已以 PyTorch INT4 加载到 RTX 4060 Laptop GPU（加载 6.9s，allocated 1788.2MiB）；`interactive + local_only` 实际返回 `start → token → done`，metrics 为 `pytorch / single_streaming / distributed_used=false / history_committed=true`。Textual 页面用同一后端完成一次真实渲染与完成态验证；TUI 定向回归 `126 passed`、三尺寸 fixture 走查 `54/54`。当前 CPU 签名发布树 `dist/QLH-Edge-Inference` 的 deep 为 `5016/5016`，冻结 `QLH-TUI-Chat.exe --help` 返回 0。上述证据不覆盖干净机 Setup、新 shell、环境变量、卸载或 Linux `.deb`；`packaging/dist/QLH-Edge-Inference` 是 7 月旧开发树，不可作为本门发布候选。
  - 保持 `tui_admin.py --plain`、单命令和 7 屏管理契约回归。
  - 经过真实使用窗口后再决定 `bjtu` 默认进入聊天还是继续进入管理菜单；在此之前 `bjtu chat` 显式启动。

### 9.9 测试矩阵

| 层 | 必测项 |
|---|---|
| SSE parser | 任意 chunk 切分、UTF-8、多个事件、done/error/cancelled、尾部残留 |
| UI | 输入焦点、滚动跟随、缩放、CJK、Markdown、超长 code block、ANSI 注入 |
| 会话 | new/resume/rename/delete、切换竞态、重启恢复、迟到 token fencing |
| 推理 | local_only、distributed_preferred、distributed_required、分布式回退 |
| 指标 | execution mode、workers、provider、fallback 与 Web/后端一致 |
| 故障 | 网关 502、SSE 中断、模型未加载、OOM、取消超时、主节点切换 |
| 环境 | 干净 venv、缺 Textual、代理、离线 wheelhouse、Windows/Linux 安装包 |
| 回归 | T1-T8 契约、7 管理屏、`--plain`、单命令、优雅退出 |

### 9.10 T9 发布门

以下全部满足前，聊天页不得成为默认入口：

1. 真流式与历史持久化是一次请求、一次生成、一次事务提交。
2. `Ctrl+C` 能在首 token 前后取消，且不会把迟到结果写入其他会话。
3. local 与 distributed 都经过真实模型验证；分布式参与以 metrics 和任务统计为证据。
4. 请求分布式但实际回退时，页面明确显示原因。
5. TUI 依赖安装不修改全局解释器；正式安装包首次运行不依赖联网下载。
6. 现有管理 TUI 的 T1-T8 契约和低依赖回退入口保持可用。

### 9.11 主要风险

| 风险 | 处理 |
|---|---|
| Textual 与现有手写 ANSI 主循环互相干扰 | 两个进程入口，不在同一事件循环混用 |
| full/fast 两种模式导致流式和历史二选一 | 先实现 interactive 契约，不做双请求补丁 |
| Shift+Enter 在不同终端不可区分 | Alt+Enter/Ctrl+J 为稳定换行键，Shift+Enter 仅作能力别名 |
| 模型输出包含 ANSI 控制序列 | 渲染前过滤/转义，Markdown renderer 禁止原始 HTML/escape |
| 用户看到分布式开关就误以为从节点参与 | 只以完成 metrics 和实际 Worker 列表下结论 |
| 切换会话/模型时旧流继续到达 | generation ID + session/model epoch fencing |
| 新依赖让轻量 TUI 不再可用 | 保留 `tui_admin.py` 和 `--plain`，聊天环境为显式可选组件 |

---

**文档版本**：1.1
**维护者**：QLH 开发团队
**下次复核触发**：T9.0 契约/PoC 结论，或 chat stream 契约发生变化时
