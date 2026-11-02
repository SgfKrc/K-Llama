# 重写决策：保留什么，抛弃什么

> 状态：**待拍板**
> 更新日期：2026-10-28
>
> 依据：`INVENTORY.md`（232 文件 / 129,337 行全量盘点）+ 本轮实测验证。

---

## 0. 一句话概括

**旧仓库的形态是：一条 62 行的接力循环，被包在 29,752 行控制面里。**

接力本体在 `qlh_core/execution/linear.py`，只有 62 行；`scheduler_pipeline.py` 8,691 行
的主体是模型供给编排（`_handle_layer_config_locked` 948 行、`_push_layer_config_to_clients_locked` 637 行），
且纯接力路径已被降级为探针（`config.py:741-748`，`QLH_RELAY_PROBE_ONLY` 默认 1）。

---

## 1. 必须保留（硬依赖）

| 资产 | 位置 | 行数 | 为什么不可省 |
| --- | --- | ---: | --- |
| keep-head shim（C） | `scripts/model_tools/keep_head_shim/qlh_keep_head.c` | 620 | 裸 llama.cpp 的 `llama_get_embeddings_ith` 返回 `output_norm(H)`，多一次归一化（rel_err=0.0018），拿不到接力需要的原始 hidden |
| llama.cpp 层段补丁 | `android/app/src/main/cpp/llama.cpp`（QLH 分支） | 248 | 提供 `llama_set_embeddings_layer_inp` / `llama_get_embeddings_layer_inp` 等导出槽位 |
| 工件裁切 | `scripts/cut_layers.py` | 690 | 全仓唯一被原型依赖的旧代码；head/middle/tail 三种模式 |
| Python 封装 | `src/llama_keep_head.py` | 871 | 但只用到 6 个方法，可精简到约 260 行 |

**注意**：裁切的真身不在 `src/`。`src/model_supply_assets.py`（2,353 行）里的裁切只占约
200 行，是调 `cut_layers.py`（`model_supply_assets.py:1242-1289`）。

---

## 2. 必须重写（旧实现不可复用）

| 项 | 旧实现 | 问题 |
| --- | --- | --- |
| **跨机传输** | `src/relay_transport.py` | **硬性拒绝非 loopback**：`:535` `raise RelayProtocolError("non_loopback_endpoint_rejected")`。跨机靠外部 SSH 隧道把远端映射回 127.0.0.1（`config.py:731,737`），链路固定开销 297 ms 占 49.7% |
| 协议帧 | `src/tcp_comm.py` 2,840 行 | 可迁移的只有约 280 行帧编解码；`TCPServer` 955 + `TCPClient` 1,017 行是心跳/重连/epoch/重放防护 |

新原型的 `protocol.py`（66 行）已经是对的：4 字节大端长度 + 1 字节类型 + payload，
hidden 走原始 float32。**跨机直连，不用隧道。**

---

## 3. 直接抛弃（连同理由）

| 域 | 文件 | 行数 | 理由 |
| --- | ---: | ---: | --- |
| 调度控制面 | 28 | 29,752 | 占 23%；A3 供给编排与本期无关 |
| 模型供给 | 27 | 14,157 | 手工拷工件即可 |
| 引擎适配 | 12 | 7,066 | 首期只接 llama.cpp |
| qlh_core | 47 | 8,108 | 含 6 个 SQLite 仓储 |
| 其余 F09–F14 | 92 | 56,810 | TUI/安卓/多模态/遥测等 |

### 3.1 过度工程化清单（新原型一律不复制）

| 模式 | 旧仓库位置 | 新原型 |
| --- | --- | --- |
| 两阶段提交 | `scheduler_pipeline.py:1698/3163/3361/3594` | 不做 |
| 代际 generation | `scheduler_pipeline.py:601,620,754,780,790`；`:3019 _assignment_connection_generation` | 不做 |
| 分片租约 | `llama_rpc_contract.py:12 RpcShardLease`、`:34 RpcShardLeaseBook`；`config.py:630-662` | 不做 |
| 七相位状态机 | `qlh_core/contracts/outcome.py:1-27` | 不做 |
| 注册 epoch + 重放防护 | `tcp_comm.py:140,1140,1303`（全文 30+ 处） | 不做 |
| 重试退避 | `scheduler_pipeline.py:3912/3924/3931`，常量 `+5.0s` 在 `:794,2491,3061` | 不做 |
| 恢复栅栏 | `scheduler_pipeline.py:1021/1251/1433`；`pipeline_reshard.py` 975 行 | 不做 |
| 三层套娃 | provider 3 份(1,911) / worker 12 份(9,094) / contract 10 份(3,175) | 一层 |
| 同功能多实现 | scheduler 8 文件 21,016 行；GGUF 头解析 3 套；传输 4 套 | 各一套 |
| 兼容分支 | 全 src 439 处「兼容/legacy」，`scheduler_pipeline.py` 单文件 115 处 | 零 |
| 环境变量 | `config.py` 94 个 `QLH_*`，全 src 180 个 | 9 个 |

---

## 4. 1000 行预算

| 模块 | 预算 | 现状 |
| --- | ---: | --- |
| `config.py` | 60 | 70 ✅ |
| `protocol.py` | 70 | 66 ✅ |
| `topology.py` | 60 | **170** ⚠️ |
| `engine.py` | 260 | 83（但依赖旧仓库 871 行 → 需内联精简） |
| `stage_worker.py` | 120 | 112 ✅ |
| `runner.py` | 130 | 79 ✅ |
| `cli.py` | 60 | 无（分散在 bench/smoke） |
| `selftest.py` | 120 | 有 `check_correct.py` 85 ✅ |
| **合计** | **1,000** | 约 870（不含拓扑超支） |

### 4.1 三个待定的口径问题

**① 行数口径**：`DESIGN.md` §2.1 写「≈1,400 行 Python」，`INVENTORY.md` §5 按 1000 行编。
需统一——建议以 **1000 行为交付口径**，1400 是含工具脚本的上限。

**② `topology.py` 超预算 110 行**：它承担 10 条校验。
盘点建议"砍到 60（固定两段）"，但用户已明确要**三段、配置驱动**。
建议保留 170 行，从别处省。

**③ 工具脚本是否计入**：`bench.py` / `breakdown.py` / `probe_hops.py` / `smoke.py`
共 233 行，建议**移出交付行数**，它们是一次性测量工具。

---

## 5. Rust 版的行数预期

Rust 的控制面不含 `engine.py`（llama.cpp 仍是独立 worker 进程），所以：

| 模块 | 预算 |
| --- | ---: |
| `main.rs` | 80 |
| `topology.rs` | 120 |
| `protocol.rs` | 180 |
| `worker.rs` | 100 |
| `chain.rs` | 150 |
| `tui.rs` | 200 |
| **合计** | **830** |

比 Python 版少，因为 Python 版的 `engine.py`（83 行）实际还要内联进
`llama_keep_head.py` 的精简版（约 260 行）。Rust 版通过子进程调 worker，不需要这部分。

---

## 6. 需要拍板的四件事

1. **行数口径**：1000 还是 1400？
2. **`topology.py` 保留 170 行还是砍到 60 行**（砍就意味着放弃三段配置驱动）？
3. **工具脚本是否计入交付行数**？
4. **`llama_keep_head.py` 精简到 260 行内联**，还是继续 `sys.path.insert` 依赖旧仓库？

第 4 点最重要：**现在的原型在偷偷依赖旧仓库**（`engine.py:19-24` 直接
`sys.path.insert(0, "G:/C/PYT/qlh/src")`）。作为独立分支提交时这不可接受，必须内联。
