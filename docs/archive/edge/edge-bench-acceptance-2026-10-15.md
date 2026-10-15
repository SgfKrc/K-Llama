# EDGE-BENCH-01 实现与验收记录（2026-10-15）

本轮选择先完成票 19，再进入票 13。票 19 给通信和构建优化提供可信的测量基线；票 13 的自动化及发行证据另有缺口，不能把已有两机结果当作三机产品门通过。继续使用当前 `dev` 工作树，保留本轮前的修改和备份。

## 交付及测量口径

`scripts/edge_bench.py` 使用生产 `KeepHeadUpstream`／同源 CPU shim 对真实裁层 GGUF 做前向标定。每个线程／上下文组合启动独立进程，使用 `nextn` 导出本段末层 hidden；中间段接收 hidden，尾段返回 argmax。没有整模 completion、网络或嵌套 worker IPC。原始失败和中间报告保留，只有 `*-closed.json` 是最终验收报告。

每格预填充 32 个位置，warmup 1 次，采样 3 次，每次单独计 16 个单位置 decode。每次先 reset 并完成 prefill，再开始 decode 计时。计时包含 Python 参数转换及 native 段前向，不包含加载、reset、prefill、输入准备、输出哈希、网络和 IPC。报告保存每步耗时、输入／输出摘要、尾段 argmax、实际层数、线程、上下文、构建及资源快照。

输入为固定 token ID 或固定 seed 的 synthetic hidden；这是层段微基准，不是自主生成文本。当前 native graph 对所有角色都执行 final norm／词表投影，head 还包括 embedding。`ms_per_layer` 是实际层数归一化后的摊销成本，含这些固定开销。可与已知问题 #71 的 `c_D` 按相同模型、段角色、输入、构建和上下文比较，不能将其冒充纯 decoder 层成本，也不能直接比较本轮 28 层 head 与 4 层 tail 来判断设备快慢。

## 工件合同与兼容

新增共享的 `src/gguf_metadata_contract.py`，从有界 GGUF 头读取 metadata／tensor directory，不启动 native、不导入 `llama_cpp` 或 `gguf`。供给与标定都以该读取器作为同一校验入口。四项关键声明是 `block_count`、`head_count_kv`、`key_length`、`full_attention_interval`；缺字段不等于匹配，显式 `null` 只表示 GGUF 没有该键。整数、整数数组、布尔、浮点严格区分。

`cut_layers.py` 在 native manifest 的 `gguf_metadata` 中写本段声明；顶层 `block_count`／`nextn_predict_layers` 继续描述源模型，不再与 `kept_block_count` 混淆。新合同覆盖制作、同代复用、跨代重绑定及接收。Android 接收端复用已有流式 reader，给出字段级 missing／typedvalue／mismatch 信息，下载和缓存复用都必须校验。没有新增供给 wire 字段或 hello 能力键。

旧工件的升级只发生在有经过 SHA 校验的源模型权威时：补缺项，已有错误声明必须拒绝。在新 staging／generation 中发布新 native 文件及新供给摘要，旧 ready 目录保持原样。无源权威的旧不完整 receipt 会具名拒绝，不能静默当作通过。 完整且未变化的 native 声明则直接验证并硬链接复用；全量回归曾发现重绑定路径不必要地重写它，已保留原有文件身份／复用字节断言并修复生产逻辑。部署新版本后，旧不完整 receipt 应经 master 重新供给发布新凭据，worker 不能凭历史缓存自行补值。

本轮用实际 9B 的两份旧段验证了上述升级：权重 SHA 不变，原 native manifest 字节不变；源 GGUF 共 33 blocks，其中 1 个 MTP，decoder 总层数为 32。供给结果在 `build/edge-bench-20261015/supply-legacy-upgrade-final.json`，源身份与 metadata 在同目录 `source-metadata.json`。源文件只作头部解析和流式摘要校验，没有进入推理引擎。

## 加载与数值边界

标定要求 native manifest 声明 proper segment，并校验实际 tensor 层号、实际 block count、来源范围、角色及 artifact SHA。拒绝 split GGUF，避免 native 自动加载未核验的其他分片。父子进程绑定同一份 manifest bytes 的 SHA，并回验执行的 cell、角色和范围。Windows 在检查至 `close()` 期间持有只允许共享读的文件句柄，阻止写入／替换；其他平台使用私有副本，副本也必须通过全部检查后才能加载。

shim 新增有限 logits 检查和可查询的 guard／词表能力；NaN／Inf 返回 `-7`，不再可能变成稳定 token 0。工具拒绝缺少这些能力的旧 shim，核对 native/header 的层数和 embedding，并校验 hidden 类型、形状、有限性及 token 范围。新 DLL 在独立目录构建，没有覆盖生产 DLL。完整依赖、头文件、编译命令及 SHA 见 `build/edge-bench-20261015/native/build-evidence.json`。

最初一次 `layer_inp` 边界探针失败后改为生产 `nextn` 入口。另一次本机最后一格在 `LoadLibrary` 阶段遇到 Windows 共享错误，未进入前向；DLL SHA 和写入时间均未变化，占用来源未确定。工具只对这个加载阶段的共享错误做最多 3 次、合计 0.6 秒的有界重试，不重试模型初始化或前向失败；失败报告 `master-9b-final.json` 保留。最终整矩阵重新运行，记录每格实际重试数。

## 两机标定结果

工件均来自已验收的 9B LL 分配：master `[0,28)`，ROG `[28,32)`；全部 native 实例只加载相应层段。每机测 `threads=4,8` 与 `ctx=128,256`，最终结果为：

| device / role | ctx | threads | ms/token | segment tok/s | ms/layer |
|---|---:|---:|---:|---:|---:|
| master head 28/32 | 128 | 4 | 147.570 | 6.776 | 5.270 |
| master head 28/32 | 128 | 8 | 170.991 | 5.848 | 6.107 |
| master head 28/32 | 256 | 4 | 150.571 | 6.641 | 5.378 |
| master head 28/32 | 256 | 8 | 171.244 | 5.840 | 6.116 |
| ROG tail 4/32 | 128 | 4 | 52.935 | 18.891 | 13.234 |
| ROG tail 4/32 | 128 | 8 | 55.337 | 18.071 | 13.834 |
| ROG tail 4/32 | 256 | 4 | 52.962 | 18.881 | 13.241 |
| ROG tail 4/32 | 256 | 8 | 55.567 | 17.996 | 13.892 |

这两台设备在本轮四格中均是 4 线程较快；只登记实测基线，不据此更改全局默认线程或推理开关。已有产品集群处于就绪闲置状态，报告保留其进程和资源快照；不是干净设备专用跑分。后端全量回归和最终 Android JVM 测试在性能采集完成后启动。

最终报告：`build/edge-bench-20261015/master-9b-closed.json`、`rog-9b-closed.json`，各自 `*-cells/` 中保留请求、结果和 native 日志。ROG 使用独立 `C:/Users/ROG/qlh-edge-bench-20261015`，没有改动其正在运行的后端部署。

## 验证与边界

| verification | result | evidence |
|---|---|---|
| Full backend unit / 4 workers / loadscope / seed 1973875962 | 6630 passed, 16 skipped, 0 failures | `full-unit-verified/`, `full-unit-verified-check.json` |
| Supply / cache / cutter / metadata / benchmark / native targeted regressions | 390 passed, 1 skipped | `targeted-final.xml` |
| Full Android JVM | 556 passed, 4 skipped, 0 failures | `android-full-jvm.log`, `android-full-jvm-results.json` |
| Real 9B metadata preflight | complete declaration accepted; wrong key_length / missing head_count_kv rejected before native | `real-artifact-negative-metadata.json` |
| Physical CPU segment matrix | master 4/4 cells, ROG 4/4 cells; stable repeated outputs | `master-9b-closed.json`, `rog-9b-closed.json` |
| Real legacy receipt upgrade | both segments accepted; old ready bytes and weight SHA unchanged | `supply-legacy-upgrade-final.json` |
| Deployment and production state | benchmark source/native SHA match; both original APIs healthy; master full-model loaded=false | `rog-final-deployment-proof.json` |
| Unified documentation checks / whitespace | passed | `doc-checks.json`, git diff --check |

Evidence root: `build/edge-bench-20261015/`. Counts overlap and must not be summed. The failed first full run is preserved separately; the final full run completed with unchanged source hashes and unchanged master identity. `round-changes.patch` is relative to the start of this round, so prior uncommitted work is distinguishable. `final-acceptance.json` binds the final reports and checks.

本轮验收的是可重复的层段标定和供给 metadata 校验，不把它计作新增三机 DL／LL 产品 E2E。Surface、Y700 离线，Android 真机安装与接收仍需设备恢复后补测；其他平台的 native 标定也未实跑。整模单机基线、重启自动拉起继续按用户要求后置。 本轮没有证明聚合容量不可替代；[上一轮容量验收](../runtime/capacity-freshness-rog-acceptance-2026-10-15.md) 已记录该 9B 配置的单节点整模候选包含 master，不能把“使用两个 proper segment”直接等同于“单机装不下”。星状拓扑、master 协调和 DL／LL 连续层段接力均未改变。

## 下一票顺序及票 13 复核

建议按 **19 → 13 → 21a** 推进。22a 可作为独立的依赖治理工作；22d 的 ISA A/B 在本票基线之上进行，22b 等同样先过 22a。21b 的 draft、hybrid 状态回滚和多 token 在飞前置尚未满足，不能提前对 9B 开启。

票 13 当前仍是部分实现，以下缺口必须收敛：

- `.github/workflows/core-tests.yml` 的干净 checkout 没有准备所有测试需要的子模块／外置资源；本地全 unit 通过不等于干净 runner 通过。
- 产品 E2E 尚未同时硬验两个不同远端、连续完整范围、每段真实 handoff、显式 `fallback=false`、星状连接及同一 operation/config/generation。已有 12d 三机历史证据应复用，本机＋ROG 不能替代第二个远端。
- `scripts/summarize_test_channels.py` 对空 runs、缺 JUnit 而 returncode=0 的记录可能返回成功；`physical.ran` 也需要与实际执行绑定。普通合理 skip 可保留，但必验产品档的全 skip 不能算通过。
- `scripts/package_contract/release_artifacts_gate.py` 的 Linux 分支仅检查 docker／runner 存在，Android 分支仅检查源码版本／构建入口，却可能报告发行 smoke 通过；它们还未验证最终产物、安装后就绪、Android native／签名。Windows 当前的版本／清单／启动探针也不足以覆盖完整干净安装。
- 三平台最终产物与严格产品 E2E 尚未接成统一发版判定。`docs/发行物门-RELEASE-ARTIFACTS.md` 中“三平台只有一项缺口”的旧结论不能用来关闭票 13，应在实施该票时修订。

因此票 13 优先修“未执行却显示通过”的判定，再补干净 CI 依赖、严格产品证据校验和实际发行物门。当前可以完成自动化实现；缺设备、干净安装或签名产物的部分继续明确列为未验，不以其他测试替代。
