# 票 13 产品接力证据门

当前门禁入口为 `scripts/product_relay_gate.py`，真实 Textual 采集入口为 `scripts/run_product_relay_gate.py`。普通 pytest 中的合成样例只验证判定逻辑；它们不构成三机验收。

每次发行验收必须包含 **DL 和 LL 两档**。每档都要求 master 加两个不同远端，按原始顺序完整覆盖 decoder 的连续 proper segment。DL 的首段为 PyTorch，其余为 llama.cpp；LL 的所有段为 llama.cpp。整模装载、任务图整模路径和回退不能通过本门。

## 证据如何判定

每个 bundle 保留以下六类原始附件及其 SHA-256。附件路径必须位于 bundle 目录中，不能引用上级目录或通过符号链接逃逸。

| 附件 | 核对内容 |
| --- | --- |
| `intent` | 公共 current-model-intent 响应，ready、loaded_ready、admitted、完整连续有序 assignment |
| `response` | 真实 SSE done，非空且正确的回答、显式 distributed_used=true / fallback=false、required route、Route-A、精确模型和请求身份 |
| `handoffs` | master 本次日志中的 `Route-A stage completed: ` JSON。逐生成步、逐远端核对顺序、层范围、assignment、工件 SHA、成功结果种类及时间 |
| `topology` | 各节点进程的完整 OS 连接观测、master 与所有远端的双向边；不允许 worker 互连、缺边、权限拒绝或部分读取 |
| `native` | 每节点实际 native loader / 进程映射观测；源码 revision、引擎、本段范围与计划一致，full_model_loaded 明确为 false |
| `ui` | 真实 Textual/Web/Android 的提交、界面回答、请求身份和截图；回答与 SSE 及预期完全一致 |

原有 `Route-A stage handoff:` 在发起远端执行前写入，不能证明完成；新 `stage completed` 只在 provider 返回且结果校验成功后写入，不包含提示词或激活原文。正常 EOS 可以比可见 token 的 steps 多最后一轮，其余缺轮、多轮、重复、乱序均失败。

模型 generation 必须以 JSON 整数保存，不能先经过浮点数再转回整数；它可能超过 JavaScript 安全整数上限。请求的 `route_a:<task_id>` 与模型 `config_id` 是不同身份，门中分别核对。

## 采集和复核

实验台先通过正式模型意图入口准备对应模式，系统自动规划层段。采集器只检查前提和驱动聊天，不会给设备手工指定层段。只有设置 `QLH_RUN_PHYSICAL_TESTS=1` 才允许真实采集。缺少第二个远端时在聊天前失败，不会通过本地回退凑出结果。

外部实验台需在该档运行窗口内采集完整的 topology/native JSON。每份包含 `operation_id`、`config_id`、整数 `generation` 和 Unix 秒 `captured_at`；观察结果不能靠重写时间补齐。`topology.nodes` 中每项包含 node_id、complete、worker_peers 和 raw_evidence；`native.nodes` 中每项包含 node_id、complete、source_revision、source、engine、layer_range、full_model_loaded 和 raw_evidence。raw_evidence 为实际命令／loader 输出的文件引用；source 只能使用门支持的真实观测种类。OS 权限拒绝应保留原输出并令 complete=false。

这些附件来自受信实验台。SHA-256 用于发现缺失、截断或被替换的附件，本身不证明数据来自真实设备。设备观测采集、构建 provenance 和实验台权限仍需由发行流程保证，不能把手工填写的布尔值当作独立实测。

先按采集器 `capture --help` 准备每档文件，再用 `combine --help` 合并。发布门不接受单档代替完整 DL/LL。

```powershell
.venv-test/Scripts/python.exe -B scripts/product_relay_gate.py `
  --bundle build/product-release/bundle.json `
  --require-revision <完整的待发行Git-SHA> `
  --json build/product-release/result.json
```

bundle、run 或 native 节点观测若显式含 `source_dirty`，只能是布尔 `false`；`true`、null、0 或字符串等值一律拒绝。旧 schema 缺省字段仍兼容，但不能通过外层 clean 标记覆盖内部 dirty 事实。

缺 bundle、缺某档、缺附件、错误 SHA、非当前源码 revision、旧请求／代际拼接、全 skip 均不会变成 PASS。历史三机材料可以保留用于追溯；没有本轮新完成记录与同 revision 观测时，不能改写为本轮通过。

## 当前验收范围

2026-10-15 的首轮两机 LL 和 DL 容量等待材料按原结果保留。修复 HELLO 锁竞争和并发收尾后，master／ROG／Y700 的 Qwen3.5-9B 自动三段 LL 与 DL 均完成真实 API 2／6、Textual/SSE 2，正常停止且无 fallback；LL 42／DL 12 条远端完成回执逐项核对，原生层段观测有独立材料。开发证据汇总为 `build/ci-release-followup-20261015/physical-post-fix-final.json`。当前 dirty 开发部署和 Y700 debug APK 不能组成同 revision 的可信发行 bundle；`final_release_gate_passed=false`，最终发行门仍开放。详见 [票 13 续验记录](ci-release-followup-2026-10-15.md)。
