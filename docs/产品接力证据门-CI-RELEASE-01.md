# 票 13 产品接力证据门

> 状态：现行；严格三机复采与最终发行执行待验。
> 更新日期：2026-10-16

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

外部实验台需在该档运行窗口内采集完整的 topology/native JSON。每份包含 `operation_id`、`config_id`、整数 `generation`、Unix 秒 `captured_at`、原始起止时间及一致的 `node_addresses`。地址映射只用于将真实 OS socket 对应到节点，不从计划推导连接边；它可以包含未分配的已知集群节点，以发现 worker 互连。

每个节点的 topology/native 摘要必须引用同一份 SHA-256 绑定的 `product-node-observation` v1 JSON。合同与纯校验位于 `scripts/package_contract/product_observations.py`。门从 raw 重新推导摘要，核对以下事实：

- 原始命令、退出 0、完整 stdout/stderr 与原始起止时间；SSH 传输也须成功且包住远端原时间。
- 同 PID／创建时间的前后身份及完整进程子树；每个进程的子进程、映射与连接都读取完整。权限拒绝、重启、遗漏模型子进程不能通过。
- 同 operation/config/generation 的实际 loader、引擎、proper 范围、decoder 总数、本段工件 SHA 与重算值。llama.cpp 须在实际 loader 进程看到对应工件映射及层数；Windows 的硬链接旧映射名须有同一稳定 PID、完整映射区域与全段字节 SHA 证明，OS 原路径保留。
- PC／master 的运行 PID 绑定启动／部署 manifest 及完整文件 rehash；checkout HEAD 或 CLI revision 只作期望，不作为运行来源。Android 来源必须从稳定 PID 对应的实际安装 APK 抽取，前后路径／APK SHA 与拉取字节一致，不能读取本机构建目录代替。
- DL 额外核对真实选择性材料化的 tensor 名、shape、来源文件和字节账目，拒绝越段 decoder、未归属边界／visual／MTP 张量；空映射或仅有计划层数不能证明没有整模材料化。

摘要中的 complete、source_dirty、full_model_loaded、worker_peers 等逐字段逐类型与推导值核对；所有 raw 原始时间都须在该档 capture 窗口内。旧 tail-only、文字同步声明和手填成功布尔值不能通过。

这些附件来自受信实验台。SHA-256 用于发现缺失、截断或被替换的附件，本身不证明数据来自真实设备。设备观测采集、构建 provenance 和实验台权限仍需由发行流程保证，不能把手工填写的布尔值当作独立实测。

快照生产者为 `scripts/build_product_snapshots.py`。`--revision` 仅指定期望值，`--node-addresses` 指定独立地址映射；`--observations-dir` 导入实验台未改写的逐节点 v1 JSON，并保留原时间。全部节点通过后才输出 topology/native；失败只保存 raw 和 `producer-result.json`，拒绝覆盖已有输出。

live collector 已接入 PID 绑定的 master／PC loader 与启动来源回执、DL tensor 材料化明细、Android prepare／工件回执和完整进程树。接口与启用方式见 [运行端回执与独立采集](ci-release-followup-2026-10-16.md#8-运行端回执与独立采集2026-10-16)。master／PC 需在业务导入前设置 `QLH_PRODUCT_OBSERVATION=1` 后启动，旧进程须更新并重新准备；Android 产品采集使用 `--product --require-process-tree`，保留普通发行 idle 探针。当前已部署三端开发版本，Android 进程枚举兼容和 prepare 回执绑定已通过装载实测。Safetensors 双摘要、Windows 映射字节绑定及三机 DL／LL 开发实跑结果见 [工件身份与复采记录](ci-release-followup-2026-10-16.md#9-工件摘要与-windows-映射身份2026-10-16)。严格来源、实验台时钟和最终发行验收按该记录保留实际判定。

冻结 clean 来源并部署后，先按采集器 `capture --help` 准备每档文件，再用 `combine --help` 合并。发布门要求完整 DL／LL 两档。

```powershell
.venv-test/Scripts/python.exe -B scripts/product_relay_gate.py `
  --bundle build/product-release/bundle.json `
  --require-revision <完整的待发行Git-SHA> `
  --json build/product-release/result.json
```

bundle、run 和各 native 节点观测都必须显式含 `source_dirty=false`；缺字段、`true`、null、0 或字符串等值一律拒绝。未知来源不能作为旧 schema 清洁来源兼容，不能通过外层 clean 标记覆盖内部 dirty 或缺失事实。2026-10-16 发现快照生产者刻意省略该字段会掩盖 Android raw 的 dirty 状态，相关旧 passed=true 结果已撤销严格验收效力，详见 [下午复核](ci-release-followup-2026-10-16.md#6-新提交与三机-bundle-的重新核实2026-10-16-下午)。

缺 bundle、缺某档、缺附件、错误 SHA、非当前源码 revision、旧请求／代际拼接、全 skip 均不会变成 PASS。历史三机材料可以保留用于追溯；没有本轮新完成记录与同 revision 观测时，不能改写为本轮通过。

## 当前验收范围

2026-10-15 的首轮两机 LL 和 DL 容量等待材料按原结果保留。修复 HELLO 锁竞争和并发收尾后，master／ROG／Y700 的 Qwen3.5-9B 自动三段 LL 与 DL 均完成真实 API 2／6、Textual/SSE 2，正常停止且无 fallback；LL 42／DL 12 条远端完成回执逐项核对，原生层段观测有独立材料。开发证据汇总为 `build/ci-release-followup-20261015/physical-post-fix-final.json`。当前 dirty 开发部署和 Y700 debug APK 不能组成同 revision 的可信发行 bundle；`final_release_gate_passed=false`，最终发行门仍开放。详见 [票 13 续验记录](ci-release-followup-2026-10-15.md)。
