# ISSUE：尾段 worker 的两个可用性问题

> 状态：**已验证**（文中引用的 v2 工具链——`chain_client` / `stage_worker` / `shim/` / `koakuma`——已退场，归档在 `slimarc/prototype`，命令与路径只作历史口径）
> 更新日期：2026-11-04
> 适用范围：`stage_worker.py` 的 tail 与中继段
>
> 本文记两个**互相独立**的现象：一条长连接就能让段对新连接沉默（设计使然，但后果严重），
> 以及两次 `0xc0000005`（已查明由诊断代码造成）。两者机制不同，不能合并结论。

## 1. 一条长连接就能让整段对新连接沉默

`stage_worker` 是单线程 `accept → _session`：一个连接占满 `_session` 期间，新连接只在
`listen(4)` 的 backlog 里排队。内核替它完成三次握手，但没有 accept，所以外部看到的是
「端口 LISTENING、连得上、收不到 HELLO」。

Y700 上的完整复现：

1. `s1` 连上 9106、读到 HELLO 后**不发数据**
2. `s2` 连上 → TCP 成功，但 5 s 内收不到 HELLO
3. 此时 `kill -9` keep_head → 4 s 后 `stage_worker` 仍 state=S、端口仍 LISTEN、`s2` 仍无响应
4. 关掉 `s1` → `stage_worker` 立刻发现子进程已死并退出，supervisor 2 s 内重启，`s3` 拿到 HELLO

这是**设计**，不是崩溃。但后果是实的：**客户端异常退出而 TCP 未关（或进程被挂起）时，
整段会对所有新连接不可用**，而进程看起来完全正常。判断它不需要崩溃、不需要内存压力，
只需要看 `netstat` 上有没有一条久居的 ESTABLISHED。

## 2. 两次 `0xc0000005`：已查明是诊断代码造成的

| | 时刻 | faulthandler 末条栈 | 形态 |
| --- | --- | --- | --- |
| A | 14:57:12 | `base64.py:58 b64encode` | 三段 20 轮第 16 轮 |
| B | 16:04:13 | `protocol.py:346 _recv_into` | 三段长跑第 14 轮 |

两次共同点：崩溃点都在**普通 Python 语句**上、都在 `faulthandler.dump_traceback_later`
的 10 s 周期 dump **之后约 1 秒**、dump 文件都停在半行 `File `。10 s 窗口里连中两次约 1%。

对照实验（关键证据）：

| 组 | 诊断 | 轮数 | 结果 |
| --- | --- | ---: | --- |
| A1 / A5 | faulthandler 周期 dump | 20 + 14 | **2 次 AV** |
| **B1** | **无任何诊断** | **60** | **0 次** |
| A2 / A3 / A4 | faulthandler | 43 | 0 次 |

有 faulthandler 累计 77 轮 2 次 AV，无诊断 60 轮 0 次。

物理论证：崩的是 `stage_worker`，**它不加载 `qlh_keep_head.dll`**（shim 只在
`keep_head_worker` 子进程里），所以 llama/ggml 的越界写不可能破坏它的堆。叠加"紧跟 dump"
与"去掉就不复现"，结论是 `dump_traceback_later` 在 Windows 上跨线程遍历栈时自己触发了 AV。

**教训**：长跑观测不要用 `dump_traceback_later` 的周期 dump。改用外部 `py-spy`（低频，
30 s/次）或 `procdump -ma` 抓完整 dump 后离线分析；`faulthandler.enable()` 只留作 fatal 兜底。

## 3. 已排除的假设（全部实测）

| 假设 | 证据 |
| --- | --- |
| 工作集过大 / 内存墙 | 4 层中继段（约 4 GB）也"崩"，8 层尾段（8.4 GB）反而干净 |
| 上下文打满 | `gen 900`（932/1024）3 轮全过，`n_past` 到 919，`find_slot`/`ASSERT`/`nan` 零命中 |
| 层数与 MTP | 8 层带 MTP 跑 20 轮全过 |
| OOM 或 lmkd | `oom_kill` 为 0；三个 python 的 `oom_score_adj = -1000`；logcat 无对应 Killing |
| 日志写爆磁盘 | relay-records 0.1 MB 级，磁盘剩 89.8 GB |
| 中继段 hidden 通路越界 | 崩的进程不加载 shim，物理上不可能 |

## 4. supervisor 的判据

`poll()` 只看进程退出，抓不到「进程活着但段不可用」。现在有两道判断：

- **心跳**（`_heartbeat_age`）：worker 在 `accept` 超时与 `select` 等数据两处 touch 一个
  文件，30 s 没更新按崩溃处理。能抓「卡在 engine 调用里」。
- **不覆盖** §1 那种「长连接占满 session」：那时 worker 在 `select` 里空转、心跳照常跳，
  段却对所有人沉默。这一类只能靠连接自身超时，或把 worker 改成能并发。

心跳是为「卡在 engine 调用」准备的，而那种僵死在本次调查中**从未复现**。它的价值是预防性
的，不要当成已验证的修复。

## 5. 还没做的

1. **真正僵死的栈**：本次两次异常都是进程被 AV 杀死，不是僵死。要抓僵死，得在 `_session` 的
   `select` / `protocol.read` / `engine.*` / `keep_head._reply` 四处加带时间戳的单行 trace，
   先定位卡在哪个阶段，再针对性 attach
2. **并发化**：让 worker 在服务长连接的同时响应新连接（多线程或 select 双监听），才能根治
   §1。代价是 engine 不是线程安全的，必须串行化
