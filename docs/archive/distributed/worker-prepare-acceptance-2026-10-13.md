# WORKER-PREPARE-01 验收记录（2026-10-13）

当前状态：WORKER-PREPARE-01 已完成。本轮实现、自动化回归及最终源码的 Qwen3.5-9B 三机 DL／LL 推理、真实 public unload 均通过；Surface 模式退出／重入与清理债重放通过。

## 行为与边界

master 仍是唯一协调者，配置、供给和推理均保持星状连接。DL / LL 的连续多段接力保持原设计。分布式运行只装载节点对应的 proper layer segment，任何节点均不得装载整模；磁盘源模型供裁层使用不计作执行器装载。显式 local 模式退出集群，必须先确认独立 worker 执行器实际卸载，再允许本地整模加载。

准备事务使用 fenced `model_prepare`：prepare → 各 worker 验证并真实装载 → master 物化本段 → commit → 全员相同身份的 loaded-ready → 发布执行计划。manifest、库存、容量规划与文件 available 都不能代替实际装载确认。

完整准备身份绑定 operation/config/generation/assignment/node/model/artifact/manifest/source/tokenizer/layer-range。执行 stage 必须携带 committed 的五元组，匹配当前 published plan、registry ready、连接代次及 native commit ACK。PC worker 的准备器与 API 正式执行入口共用同一个 EngineHost，远端首条聊天不得通过 lazy load 补齐准备。master LL 在首条聊天前已通过进程内绑定物化本段；聊天时可创建仅持有同一本段的 keep-head 辅助运行器，不得装载完整源模型。

abort / finalize / release 独立记录持久清理债。release 成功必须在原生运行时真实卸载后返回 loaded=false；失败保留所有权用于重试。finalize 重放可返回 loaded=false，仅证明旧代际清理终局，不能授予执行就绪。断连、退出、取消、旧 ACK、master 重启和旧 schema 债迁移均由身份与代次隔离。

PC 可保留 candidate 与 previous 执行器，准备时立即禁止旧 stage；资源关闭失败不得丢失引用。Android JNI 单槽位替换时先禁止旧代际，abort 后需重新 prepare 恢复；native load 不可中途打断，取消需等待返回后真实清理。模式切换及整模加载等待放在 full-chat 锁外，超时或卸载失败具名拒绝，避免等待线程与执行锁死锁。

## 本轮补齐的边界缺陷

- stage-only 计划绕过准备事务，worker 首次推理才 lazy load。
- 连接换代、materialize 后取消、最终发布期间取消或 native error，可能错误恢复 ready。
- ready 计划失效只清 active plan，其他 worker registry 仍保留执行资格。
- disconnect 后 unload 丢失远端 release 债；普通 release 只清 reservation 未真正关闭执行器。
- master 恢复写清理债时覆盖原模型/assignment；旧 schema 离线清理债迁入新事务后再次恢复失败。
- available supply 的取消遗漏准备事务及 API 建事务前窗口。
- PC 独立执行器与 API stage 缓存实例分叉；local/full load 与异步卸载抢锁。
- Android 大于 2^53 的代际比较、整模 tensor directory 冒充裁剪工件、重连后的终局清理证明。
- DL 只按显存规划大头段，再因源权重 RAM 暂存不够整体拒绝。源层字节约束现进入规划，GPU 准入及事后 RAM 复查均保留。
- 已确认 prepare ACK 被 retry monitor 重新加入重试队列；现按阶段确认集合跳过，未确认节点仍继续退避重发。
- 磁盘暂存估算在逐层切片之外重复计入完整 decoder aggregate，并计入 Safetensors 不写出的视觉／MTP 权重；GGUF 又可能把被裁除的 MTP 块算进 other。现 Safetensors 按裁段边界计数，GGUF 与 builder 共用 tensor 目录选段规则，保留双份暂存余量及 support／header 写前检查。
- LL public unload 漏关 master keep-head 辅助子进程，API 虚报 unloaded；现 engine unload 先关闭并确认该 owner，再清 key 与模型状态，关闭失败保留引用并传播。关闭不再同步读取可能永不返回的协议响应，采用有界 wait / terminate / kill 并回收管道。
- keep-head 初始化响应失败且后续清理也失败时，构造赋值未完成会丢失存活 child。现 KeepHeadCleanupError 携带待清理 owner，master 与 PC worker 保留它、禁止执行并允许真实卸载重试。关闭失败后 stdin 已关闭的重试也已覆盖。
- 相邻 RPC autostart 关闭失败先清引用、打开失败未接管启动中的 worker；现 session / worker 确认退休后才清所有权，初始化清理失败向 engine 移交 session。再次 load／metadata prepare 先检查实际 owner 并退休，失败时拒绝覆盖模型身份；is_loaded=false 不再被当成没有资源。加载失败与 public unload 采用同一关闭顺序：先张量，再 RPC 设备／子进程。此项自动化覆盖，不以三机直连验收宣称原生 RPC 端到端已验证。

## 自动化验证

结果不可跨组合相加，以下均为各次独立运行的通过数。

| 范围 | 结果 | 证据 |
|---|---:|---|
| scheduler / protocol / assignment / execution capacity / API / supply / retirement | 598 passed | build/worker-prepare-final-master-regression.log |
| 最终 API / 冷启动 / 模式 / prepare runtime / 竞态 / retirement | 236 passed | build/worker-prepare-final-api-runtime.log |
| supply / reuse / runtime dtype / host RAM / capacity / coordinator | 276 passed | build/worker-prepare-final-supply-capacity.log |
| PC controller、真实小型 Safetensors、真实子进程及相邻回归 | 111 passed | 子 agent 最终验证；父会话相关文件复跑见上表 |
| master 确定性竞态 | 24 passed | tests/test_worker_prepare_races.py |
| 源 RAM 范围 helper | 75 passed | tests/test_model_supply_host_capacity.py |
| 最后一次已确认 ACK 重试修复与 assignment / retirement | 36 passed | build/worker-prepare-confirmed-retry.log |
| 最终磁盘估算 / 真实裁段 / supply / reuse / host RAM / coordinator | 292 passed，1 skipped | build/worker-prepare-disk-budget-final-regression.log |
| 最终 native owner / 初始化失败 / RPC 关闭 / model / host / API / prepare / races / retirement | 438 passed，1 skipped | build/worker-prepare-owner-failure-final-regression.log |
| 打包资源契约 | 8 passed | build/worker-prepare-package-contract.log |
| Android JVM | 415 passed，0 failure/error/skip | android/app/build/worker-prepare-final-verification.log |
| Android assembleDebug | 成功 | android/app/build/worker-prepare-evidence.json |

前一轮只读 Reasonix review 使用 128 tool rounds / 1800 s，覆盖已确认 ACK 重试、同 SHA 跨 operation 身份、清理 ACK 完整身份和发布资格。随后真实 LL unload 抓到 master 子进程残留；恢复的只读子 agent 窄复核进一步复现初始化清理 owner 丢失及 RPC autostart 关闭失败。上述问题已补修并纳入最终回归。子 agent 最后确认加载／metadata prepare 的 owner 退休门闭环，未发现本条补丁的具体新增问题；复核结果不替代实机验收。
磁盘估算组合的 1 skipped 为本机无法创建符号链接的 test_receive_symlinked_partial_rejected，不计通过。真实 tiny GGUF+MTP 的 head／middle／tail 三种裁段估算与实际目录权重字节一致；coordinator 同时覆盖空间足够与确实不足。

最终 owner 回归的 1 skipped 是需要 QLH_RPC_INTEGRATION=1 与原生 RPC worker 的显式 opt-in 集成用例，本轮未开启，不计通过。初始化和关闭失败均用确定性进程替身验证；keep-head 不回关闭响应另用真实子进程确认有界回收。

Python 集成回归采用 xdist 4、maxfail=0、order seed=47381。Android APK 已通过 adb install -r 更新，保留应用数据和 SAF 授权，USB 未切 MTP。

APK SHA-256：`671483a72e33785dcf8eae5b46569f8b11be0a67b9045a9ec2938db7c3b35f54`。

18:09 经 adb 在设备内读取已安装 base.apk 摘要，与本轮 APK 一致；包版本 0.1.8.3-debug，lastUpdateTime 为 2026-10-13 16:58:36。

## LL 卸载缺陷与故障复核

修复前，operation supply_01c39359d1d24775a7932be13546746d 已通过三机推理，但 public unload 后仍留下 master keep-head PID 24592。Surface／Android 均已退出，cleanup ledger 为空，master API loaded=false 与实际 child 存活不一致。失败证据保留于 build/worker-prepare-llama_cpp-unload-orphan-failure.json。

根因是 LlamaCppEngine.unload 未覆盖独立 keep-head owner；ModelManager 还吞掉引擎关闭异常，替换 upstream 同样可能在关闭失败后丢引用。修复采用“真实退出后才能清状态”，并使初始化失败的清理所有权及再次加载的退休门保持一致。故障测试覆盖关闭失败 → 引用／身份保留 → 重试成功，以及初始化失败 → 清理失败 → owner 移交 → 禁止执行 → 真实回收。

部署前对上述旧版残留做过一次身份核对后的清理，记录见 build/worker-prepare-old-orphan-cleanup.json；手工清理不计作修复验收。新版验收只通过 public unload 关闭执行器，随后直接检查 PC 进程与 Android maps。

## 真机证据

设备：主机 100.90.76.108；Surface 100.100.52.106；Y700 100.99.211.13。模型 Qwen3.5-9B，32 层。工件由供给系统自动安排，未逐台手选层段。

LL 首轮通过，operation `supply_38a253daaabd46eebf479f765f9b1c4a`，config `23f1375283634f56afe389a6c3d9b8e9`，generation `1791884012521155500`。分配主机 [0,20)、Surface [20,24)、Y700 [24,32)。首条聊天前，远端 prepare/commit 实际 loaded ACK 完成；Surface shim 命令和 Y700 maps 对应当次裁剪工件，master descriptor 仅本段层字节为正，current.loaded=false。两个独立会话返回 `2`、`6`，均 distributed_used=true、fallback=false，耗时 17.922 s / 13.016 s。证据：build/worker-prepare-llama_cpp-initial-pass.json。

进入下一轮前 public unload 已清空终局清理债；Y700 maps 中无 GGUF 映射，此前检查已确认真实退出；最终同版的 public unload JSON 在下文保留完整直接设备证据。

卸载收尾修复前的一轮 DL 已通过：operation `supply_0701ca747e094408a4fc1b5c69d7b463`，config `3c1a41501d754455a7d8e96566084cd5`，generation `1791891512820533700`。自动分配主机 CUDA [0,8)、Surface LL [8,12)、Y700 LL [12,32)。两个独立会话返回 `2`、`6`，耗时 5.984 s / 4.156 s，均 distributed_used=true、fallback=false。首条聊天前已确认两 worker 的 prepare／commit 日志精确身份、Surface 进程对应本代 manifest 工件、Android 仅映射本代尾段；主机 segment_materialized=true、current.loaded=false，Safetensors 索引仅包含 decoder 0–7。证据：build/worker-prepare-pytorch-before-native-unload-fix.json。

同 SHA 跨操作也已验证：三个 artifact SHA 与首轮一致，manifest／operation／config／generation 更新，本代再次执行 native prepare／commit。Android 复用 4,045,951,539 bytes，新增下载 0。证据：build/worker-prepare-same-sha-new-operation.json。早期完整下载经正常 master HTTP 完成，无 adb push 模型。

DL 正式 public unload 通过：主机无准备模型／段子进程，Surface 无 shim，Android GGUF maps 为空，cleanup ledger 为空。证据：build/worker-prepare-pytorch-final-unload-real.log；最终同版完整状态另见下文验收 JSON。

Surface 退出模式通过：已发布的 DL 计划下，local 切换约 1.283 s 返回，Surface shim 实际退出、Android maps 为空、master 计划失效。Surface 退出会断开控制连接，master 保留其 release 债；重新 distributed 后该债重放清空，全程未加载本地整模。证据：build/worker-prepare-pytorch-surface-local-retirement.json。

19:03 左右 USB ADB 短暂消失，期间一次卸载检查只能确认 native release 与 PC 进程，未伪记 Android maps 通过；19:24 恢复后，上述最终 DL 与卸载已补齐直接设备证据。

较早的两节点计划被取消，未算作三机通过。Surface 真实可用 RAM 波动时准入可选择较少执行节点，诊断日志记录各节点预算；验收脚本明确要求三台参与。较早 DL 因源 RAM 不足被拒绝，该失败保留于 build/worker-prepare-pytorch-host-budget-rejected.json，未放宽容量值。

## 最终源码三机验收（含加载重试门）

最终部署 master PID 11872，基线 HEAD 87e687462a96853173d9af177551ef6be84c505e，源码未提交。master 与 Surface 的 19 个关键源码文件摘要一致，记录见 build/worker-prepare-final-source.json。Android 保持上述已验证 APK。

| 路径 | master | Surface | Y700 | 两个独立会话 | distributed / fallback | public unload |
|---|---|---|---|---|---|---|
| DL | CUDA [0,8) | LL [8,12) | LL [12,32) | 2 / 6；5.109 s / 3.234 s | true / false；两远端均参与 | 三节点实际退出，清理债为空 |
| LL | LL [0,8) | LL [8,12) | LL [12,32) | 2 / 6；11.125 s / 11.093 s | true / false；两远端均参与 | 三节点实际退出，清理债为空 |

DL：operation supply_e4b1a489ff7449ea9eaec735206d58bd，config b042fe61e2c54a518ca7a9fb0789018f，generation 1791895557130776300。证据：build/worker-prepare-pytorch-acceptance.json、build/worker-prepare-pytorch-unload-retirement.json、build/worker-prepare-dl-final-complete-real.log、build/worker-prepare-dl-final-complete-unload-real.log。

LL：operation supply_5e55331bc1d842b38d0ffca38ad650d4，config 9e1564633f1d4375839b4817e5f9df51，generation 1791895205238890700。证据：build/worker-prepare-llama_cpp-acceptance.json、build/worker-prepare-llama_cpp-unload-retirement.json、build/worker-prepare-ll-final-complete-real.log、build/worker-prepare-ll-final-complete-unload-real.log。

两种路径均由供给系统自动决定连续层段，本轮最终复验采用 strategy=distribute；先前 auto 的 Surface local_cut 路径也通过同一 prepare 事务。首条聊天前，prepare／commit native ACK 的 operation/config/generation/assignment/artifact/range 与 published plan 一致；Surface 实际进程和 Android 私有 current pointer／maps 指向当代工件。master DL 索引只有 decoder 0–7；master LL 的进程内 GGUF 映射与聊天辅助 child 只对应 [0,8) 工件。Windows 可记录原子改名前的暂存映射名，校验仍限定本次 operation/master 下的 segment.gguf，未允许源整模路径。

这两轮远端相同 SHA 文件复用后都再次确认新 manifest 和新代际 native prepare／commit，文件复用没有替代装载 ACK。最终 public unload 后 master 无准备模型及 keep-head child，Surface shim 为空，Android GGUF maps 为空，cleanup ledger 为空。由 API 正常关闭，没有手工杀进程参与成功验收。

## 后续边界

本票验收操作仍通过 master API 的供给与 prepare 入口。任意设备一处选模、统一进度、无预置工件产品流程属于 MODEL-INTENT-UX-01，未用本轮预置来源和 API 脚本代替该票验收。CI 发行门与单一事实源重构决策按既有排票保留。