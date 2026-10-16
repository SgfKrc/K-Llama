# 票 13 续验与阻塞复核（2026-10-16）

> 状态：票 13 开放；开发三机 DL／LL 与观测修复通过，清洁来源复采及可信三平台发行待验。
> 更新日期：2026-10-16

此前 Android 真裁切和开发推理证据保留；新增提交中的三机 bundle 因来源与观测完整性缺口撤销严格通过结论。运行回执、独立校验与发行附件生产链已补齐，软件验证和设备条件见 §8～§9。

本记录复核此前同日的阻塞归因，保留原始证据于 `build/ci-release-followup-20261016/`；新证据独立存于 `build/ticket13-verify-20261016/`，不覆盖原始日志。§1～§5 是上午续验：当时主仓 HEAD `5f63b267c197dcf24af3905d12a33830f01a6ef2`、Android HEAD `07fcba305d5ac5ebd8e9497bab7aafbf6a78d409`，运行含未提交修改。这些改动已于 12:43 的一批提交入库；下午复核起点为干净主仓 `d657ca7f49365d6d50832b57c4fe1a0dde307c60`、Android `76a408a7f23dd1d5b332c9dac93273b8ab17f323`，新增修复与发行检查见 §6～§9。三端来源分别记录，不能仅以 HEAD 相同宣称最终同 revision 签名验收通过。

## 1. 原阻塞的核实结果

### A：ROG 10054 断连尚不能定性为生产重连代码缺陷

原报告声称注册后总在 10–45 秒断开，且断开后同一进程仍存活。重新按 hello accepted 与 master RST 时间关联，四次间隔为 **15.723／131.816／34.563／63.298 秒**；现存历史证据缺少同一时刻 PID／创建时间观测，无法据端口快照证明断开的原 worker 仍存活。

新增真实 TCP 回归覆盖 RST（在 Windows 确认实际 WinError 10054）、EOF、心跳 ACK 丢失，均重新连接、获得 accepted hello 并恢复 ACK；相关 **163 passed**。本轮也直接观察到经 SSH `Start-Process` 启动的 PID 6920 随 SSH 退出消失；仓库启动脚本已说明 Windows SSH job object 的子进程生命周期约束。

ROG 改用**按需、无开机触发器的计划任务** `QLH-ticket13-verify-20261016` 后，worker PID 2268 自 11:09:54 起持续在线，完成首轮三机 LL／DL 推理及 master 重启后的自动重连；后续因 RAM 不足留作 control-only，TCP 指向唯一 master。此为已观察到的部署因素，**不能据此宣称已查明并修复历史每次 RST 的代码根因**，也不应再以历史 A 阻挡本轮已成功的开发验收。

证据：`connection/findings.md`、`connection/verification-summary.json`、`rog-task-launch.json`、`rog-observation.jsonl`；测试 `tests/test_pc_worker_connection_recovery.py`。原始断线日志仍保留。

### B：真正缺陷是失效 legacy 目录导致跳过已登记的新工件

原报告把 `pipeline_local_commit_failed` 归因为 planner 不读取旧工件库。真实供给操作其实已按自动计划生成并登记 `[0,12)`，旧启动环境却保留了不存在的 `QLH_LAYER_ARTIFACT_DIR=G:\C\PYT\qlh\build\ci-release-followup2\artifacts`。`LlamaCppEngine._find_layer_artifact()` 在目录不存在时提前返回，尚未读取供给注册表的 `selected_gguf_path()`，因此看不到新产物。

修复改为**先验证供给选择路径及原生 manifest，再回退 legacy 目录搜索**。仍要求精确范围、角色、源 SHA 与工件 SHA，不允许整模代替分段。旧目录存在／不存在、错范围和篡改摘要都有回归；相关 **279 passed、8 deselected**。同一个真实约 3 GB、`[0,12)` 工件上，HEAD 旧函数因失效目录失败，当前函数仍命中；配对重放见 `local-lookup-before.json`、`local-lookup-after.json`、`replay_lookup_pair.py`。早期验证 helper 曾将修后结果写入 before 文件名，原字节保留于 `local-lookup-mislabelled-postfix-original.json`；原操作工件已退役，因此配对重放使用本轮新操作的同源同范围工件，并明确记录为重新执行旧函数，不冒充原始时间点快照。

PC cutter 能力探测有运行时 fallback，不能根据 `assets.local_cut_available` 在 `src/` 无静态定义就判为没有裁切能力。无需为了这次缺陷改为让 planner 迁就旧目录的预置分段。

## 2. Android 本地裁切、入口与供给修复

原实现确实固定上报 `local_cut=false`、空 sources，并拒绝本地裁切；此前 APK 缺的是执行能力，不只是按钮。

本轮实现：

- Kotlin `GgufSegmentCutter` 流式裁切，256 KiB 载荷缓冲、64 MiB 头部上限；只生成 proper decoder segment、重编号并排除 MTP，不 mmap 完整源、不调用 JNI 模型加载器。
- 独立只读源登记，沿用现有 SAF 授权，不改变推理模型选择、身份或凭据。设置页 **“GGUF 本地裁切源” → 扫描源文件 → 选择完整源 → 填写准确 master 模型 ID → 登记为裁切源**。用户仍从统一模型入口发起自动供给，由 master 选择分段，不逐台选层。
- inventory 的多 GB 摘要工作移到可取消 job，避免阻塞串行收包和心跳 ACK；本地裁切消费自身超时预算。
- Android 返回的 intent operation 接受合法 `auto／local_cut／distribute` strategy，UI 提交仍限定 auto，未知值仍拒绝。原先合法显式 strategy 记录被 UI 错判 `model_intent_contract_invalid` 的问题已修。
- `allow_fallback` 只接受布尔值并默认 false。显式允许时，仅源不可读、布局不支持、自身裁切超时可从零下载同一个 manifest；取消、绑定变化、SHA／tokenizer／身份不符、磁盘不足仍拒绝，原下载校验与发布门保持生效。不会把废弃裁切字节计成下载或断点复用。

Python cutter 生成的 plain／hybrid（含 MTP 与按层数组）／quantized 共 9 组 head／middle／tail golden，Kotlin 结果逐字节一致。完整 JVM **651 tests：647 passed、4 skipped、0 failures、0 errors**；debug／release Kotlin 与 debug native 编译、APK 来源检查通过。回退专项 14 项真实 loopback HTTP 回归全部通过，相关 104 tests 含 4 skipped，是完整 JVM 的子集，不叠加计数。

开发 APK 固定输出为 `android/app/build/outputs/apk/debug/app-debug.apk`；每次安装保留 `adb install -r` 回执、DEX 功能检查、包内 build-info 与设备安装包 SHA。首个含裁切的安装版 SHA `ecba18818c87b0bfd7f9909eacff1c53e210639d98042ce22ed30ff94c545816`；它用于首轮三机与真实裁切测试，尚不含之后的 intent strategy／回退小修。**收口 APK SHA256 为 `6ee2783877cf2844fffc1fe94ac742f3ccc6e1742958bc8d3251481670f70fa6`（25,801,985 bytes）**，约 12:00 已通过 `adb install -r` 保数据安装，设备安装包 SHA 与构建副本一致，裁切源登记前后摘要不变。设置入口在该安装版上实际可见，截图及 XML 为 `android-cut/final-apk/settings-local-cut.png／.xml`。最终安装副本、DEX 检查、安装回执和包内 build-info 在 `android-cut/final-apk/`；各轮复测索引为 `final-physical-development-summary.json`。

包内全仓 source tree 为 `7045ee2d380a123727445b3c359180ed15db7a887d05c34941de8cd1a524c373`，Android component 为 `8ff76015bac04f025504b7f7744e2059ac0df5772fa45f6787ac085d9da7b202`。APK 构建后只新增 Python compiled 退役补丁、Python 测试和文档修改，Android 源码没有再变；最终比对见 `final-source-info.json`、`final-source-comparison.json`。后端部署来源以 `master-deployment.json` 为准，ROG 保留 11:07 的部署来源，**不声称当前全仓 tree、APK 与三端运行来源完全相同**。

### Y700 真实新裁切（不是缓存命中）

11:51 正式调用 `/api/models/unload` 后，Y700 观测到 loaded=false 且模型映射为空。只覆盖安装 instrumentation 测试包，main APK 的 SHA 未变；测试直接调用 main 内实际 cutter，读取已通过真实设置 UI 登记的 SAF 完整源，不接触生产发布目录。

| 项目 | 实际结果 |
| --- | --- |
| 模型／源大小 | `qwen3-5-9b`，5,780,090,176 bytes |
| 源 SHA | `ada47e14d93f7ca03e30bac750c330fd96b81a29fbe3333e0d0dcc5ebc14b82c` |
| 新裁切范围 | `[8,12)`，4 decoder layers，0 MTP |
| 产物大小 | 1,926,059,296 bytes |
| 产物 SHA | `089944ca61e8a801420cc021c9775be03c38da0208d9fc823878a089995fc20d`，与 master／Python 真实工件完全一致 |
| 裁切／总耗时 | 9,569 ms／17,964 ms |
| 源流实际读取 | 14,593,761,806 bytes，多遍流式校验；不代表运行时权重驻留 |
| 采样 Java／native heap | 15,127,760／10,710,144 bytes；采样值不能当作精确 RSS 峰值 |
| 模型加载 | 测前、测后 GGUF／模型 native 映射均为空，未调用 loader |
| 非测试状态 | registry、SAF grants、settings、生产 supply 快照均不变 |

结果 **OK (1 test)**，输出位于新建唯一 cache 目录，明确禁止复用。证据 `android-cut/real-cut-device-result.json`、`real-cut-instrumentation-adb.log`、`real-cut-instrumentation-run.json`。这证明 APK 内实际 cutter 可处理真实 9B；它没有把测试工件发布成生产供给，不能混称为完整产品供给新下载验收。

## 3. 首轮三机自动 LL／DL 开发验收

master／ROG／Y700 同时在线。用户提交 `qwen3-5-9b` 自动模型意图，未手填范围、未降低预算；首轮两模式均自动分为 **master `[0,8)` → ROG `[8,12)` → Y700 `[12,32)`**。

| 模式 | operation | 引擎 | API／产品入口与回执 |
| --- | --- | --- | --- |
| LL | `supply_7f85ee9c493544e9b2bb7871b2ddb457` | 三端 llama.cpp | 独立 API 回答 2／6，真实 Textual／SSE 回答 2；text_stop；42 条完成回执逐项通过 |
| DL | `supply_05a2266bb426433f96aa2a182331196e` | master PyTorch／CUDA，远端 llama.cpp | 独立 API 回答 2／6，真实 Textual／SSE 回答 2；eos_token；12 条完成回执逐项通过 |

每条回执检查 operation、config、**整数 generation**、精确 range／assignment／artifact SHA、hidden／token 种类、次序。均为 distributed_required、两个不同远端、无 fallback。Y700 JNI 实际 20 层、full_model_loaded=false；ROG 子进程只映射 4 层工件；LL master 只映射 8 层工件，DL master 日志证实 `safetensors_key_filtered` 仅 materialize 首 8 层（108 tensors、5,494,026,496 bytes），而非整模加载。实际连接保持星状。进程／native／socket 观测范围与权限缺口保存在原始 JSON，不能把 Android children_complete=false 写成全进程树已完整证明。

首轮 `ll-first/`、`dl-first/` 保存 API、TUI、逐条回执、native/maps/socket 与来源快照；总索引 `physical-development-summary.json`。**两轮 Y700 命中已有尾段缓存，actual_downloaded=0、reused=4,045,951,700 bytes**。虽然策略标为 local_cut，这不能充作本轮真实新裁切证据；新裁切由上节独立真机实验补足。

### 收口 APK：两机自适应补测与正式产品新裁切

收口 APK 安装后 ROG 仍在线，但容量观测显示 RAM 不足：规划诊断时可用 3,965,550,592 bytes，保持当前顺序与约束的最小 4 层候选需要 5,133,143,860 bytes；独立稍后读到可用 4,602,478,592 bytes。规划器从三个候选中自动选择 master／Y700，ROG 留作 control-only。未停止 ROG 其他程序、未降低倍率／预留／安全余量、未手填层段；这是资源变化后的合法自适应，不能当作收口版本的三节点严格门通过。

| 模式 | operation／目录 | 自动范围 | 实际结果 |
| --- | --- | --- | --- |
| LL | `supply_770fea854e6b4d5688dd6f7640088afb`，`ll-final/` | master `[0,12)` → Y700 `[12,32)`；均 llama.cpp | API 回答 2／6，text_stop、无 fallback，14 条完成回执逐项通过；Android 复用 4,045,951,700 bytes |
| DL 首次尝试 | `supply_ba64e68ac2574acf918d6bb533fcddd6`，`dl-final/` | 未准入，无发布计划 | 旧 master 显存残留导致容量等待；已正式取消，不计成功 |
| DL 修复后 | `supply_57a0eb9df63348c88bfe7e3ce65476f3`，`dl-final-after-retire/` | master `[0,8)` PyTorch／CUDA → Y700 `[8,32)` llama.cpp | API 回答 2／6，eos_token、无 fallback，4 条完成回执逐项通过；Android 正式供给新裁切并实际加载 24 层 |

成功 DL 的 config `8c91fc35b1c740b0a8daea67dee4f4c6`、整数 generation `1792124484513344700`。Y700 正式供给在收口 APK 内执行 `building → verifying → registering → native loaded-ready`，新裁权重 **4,554,565,408 bytes**，SHA `77b7d3dc5c94fa09c48de08ccd4e9185c4e978dbab46cb22ad916a6288aa8ac2`；仅从 master 下载 native manifest **2,321 bytes**，总产物 4,554,567,729 bytes，reused／resumed 均为 0。这补齐了独立 instrumentation 之外的真实产品供给、校验、装载和推理闭环。此轮 master 复用既有 8 层 safetensors 段，不能称所有端均无缓存。

证据：两成功目录的 `api-chats.json`、`completed-api-handoffs-verified.json`、`master-operation-api-events.json`；DL 的 `intent-history.jsonl`、`intent-latest.json`、`y700-runtime.json`、`master-runtime-ready.json`。Y700 JNI 报本段 24 层、full_model_loaded=false，native 映射仅该尾段；星状连接、各端观测权限边界仍按原始证据说明。完整源只作磁盘裁切输入，没有让节点 native 加载整模。收口版本未再执行三机 Textual／SSE 严格门，不拼接首轮来源为最终发行验收。

## 4. 复测发现的 compiled 权重退役缺陷

容量拒绝进一步定位到旧 master PID 19056：模型 API 已表示卸载，但约 5 GB 显存仍被活对象持有。强引用链为 `atexit registry → _release_compiled closure → old ModelManager → _compiled_transformer → OptimizedModule._orig_mod → weights`；仅清 `self.model` 后 GC／empty_cache 无法释放。日志确认 DL 启用了 compile，`USE_MONOLITHIC_FORWARD=False` 不会关闭 transformer compiled wrapper。

`ModelManager.unload_model()` 与 `abort_pipeline_materialization()` 现于 GC／CUDA cache 清理前无条件清空 `_compiled_transformer`、`_compiled_layer_loop`；atexit 回调也同步清空 loop。真实 `torch.compile(..., backend="eager")`／weakref 的两个参数化回归修前 2 failed、修后相关 **180 passed**。这是本轮新增验证，和之前 279／163 组合的覆盖不能直接相加当成全量计数。

真实 CUDA 隔离实验只加载 proper `[0,8)` 工件：compiled allocated 为 5,494,034,432 bytes，仅清 model 后数值不变；同进程调用修后 unload，allocated／reserved 都归零。证据 `compiled-retirement-before.xml`、`compiled-retirement-after.xml`、`compiled-retirement-cuda.json／.log`。

随后将补丁部署到 master PID 17648，完成上节成功 DL。在**同一个 PID、相同创建时间**下正式调用 `/api/models/unload`，GPU used **5,585 → 319 MiB，释放 5,266 MiB**，API 返回 success／unloaded=true，pipeline／model 均退场，Y700 native 模型同时卸载。证据 `dl-final-after-retire/public-unload-cuda.json`、`y700-after-unload.json`。服务保持在线，未清身份、源登记或 SAF 授权。

## 5. 最终发行门与尚未独立核实的外部条件

上午续验只独立核实当时仓库内容：`release-acceptance.yml` 消费 `qlh-final-*` 与 `qlh-product-relay-evidence`，要求 Windows／Linux／Android 三组 self-hosted 标签；当时仓库 workflow 未找到生产并上传这两类 artifact 的 build 工作流；下午补齐源码接线，见 §7，尚未执行远端构建。

**GitHub runner 注册、仓库签名变量／密钥是否已配置，本轮未独立查询，不能沿用原报告“runner 未注册／三项均未具备”的断言。** 最终仍需可信签名发行物、来源匹配、实际平台执行与严格证据包；不代持密钥、不用临时签名绕门。用户已允许最终发行物验收后置，本轮的 dirty／debug 运行不冒充最终门通过。

下一步发行工作应围绕实际 artifact 生产上传、外部 runner／签名配置核实和最终产物执行门安排；无需再根据已经被复核推翻的 A／B 归因另立重复修复票。票 13 保持开放；重构票 14 仍按原先行为边界与独立分支裁决处理。

## 6. 新提交与三机 bundle 的重新核实（2026-10-16 下午）

本轮先核对干净工作区与 Git log，确认上午修改已入库：

| 提交 | 内容／核实结论 |
| --- | --- |
| `915d44b7`、`f70865da`、`5e3786fa`、`a45f42c3`、`f8d2f53f` | 工件查找、compiled 退役、真实 TCP 回归、上午证据文档与 Android 裁切接线已提交 |
| `c66f784e` | 两项 Textual 时序测试稳定性修正 |
| `b43b0a33` | TCP 注册等待区分本代中断与注册失败 |
| `0655fe32`、`d657ca7f` | 记录三机 bundle 的旧门通过结果，并将快照生产者实际加入白名单与 Git |

`build/ci-release-followup2/bench-out/` 确实有 DL／LL 的 Textual/SSE、回答 2、完成 handoff 与三节点记录，revision 字段为 `b43b0a339ce899728fdb1cc024e9b649c074b5ae`，自动连续范围 `[0,4) → [4,8) → [8,32)`，DL 2 轮／LL 7 轮接力。**保留这些真实产品运行记录，但撤销“该 bundle 已补齐严格同来源三机验收”的结论。**

当时 `gate.json` 返回 passed=true 的记录保持原样。对同一原件重新执行 HEAD 旧门仍为 true，收紧缺省规则后的门为 false，证据在 `build/ticket13-log-review-20261016/historical-bundle-audit.json`。原因来自原件与生产代码，不是设备是否实际出过答案的推测：

- `bundle`、run 和 native 节点省略 `source_dirty`，旧门把缺省当 false；生产者明确刻意省略这个字段。Android 原始附件中的 build-info 则明确 `source_dirty=true`，省略后的摘要隐藏了这一事实。
- ROG 两档附件均只有 `probe_stdout_tail`／`probe_stderr_tail`，没有已解析的 probe。生产者按单行找 JSON，实际 pretty JSON 未被解析；仍依据 CLI 默认 revision 和文字同步声明填入完整观测。
- `complete=true`、`full_model_loaded=false` 直接固定，`worker_peers` 由计划推入。失败、权限缺口或未观察实际模型时也能得到成功摘要，原始文件存在和 SHA 正确不能补足语义。
- master 只检查监听进程，没有覆盖实际模型子进程，也没有将运行源码绑定到被观察 PID。Android 来源取主机生成目录内的 build-info，没有与设备实际安装 APK 及稳定 PID 绑定；其 raw 还明确 children_complete=false。

历史原件和旧 gate.json 均不修改，不通过补字段、重写时间、人工填写同步说明来重标为通过。快照生产者与严格门必须修复后，用冻结来源重新采集。

## 7. 观测与构建合同修复

来源清洁度不再作为旧 schema 的缺省兼容项。产品 bundle、各模式 run、各 native 节点以及签名发行物的 build-info 均要求显式布尔 `source_dirty=false`；缺字段、null、0、字符串和 dirty 都拒绝。采集器仅在前后检查真实 checkout 后写入清洁事实，合并器只合并已独立通过的输入，不能从外层覆盖内部未知来源。Linux 容器中的安装来源检查同步收紧。缺省字段回归先在原实现得到 7 项失败，再收紧规则；阶段组合 248 passed 已由下表最终结果取代。

新增纯校验模块 `scripts/package_contract/product_observations.py`，由生产者与产品门共用。门读取各节点结构化 raw，重新推导 topology/native 字段，并逐字段、逐类型与摘要比较；同节点两份摘要必须引用同一 raw。核对原始命令的退出码、完整 stdout/stderr、原始观测窗口、稳定 PID／创建时间、完整进程子树、实际连接、已加载引擎／本段／工件摘要，以及运行来源。master 的工件与 intent 核对，远端工件与实际完成回执中的 prepare identity 核对，不能从计划补成功事实。DL 另需同 PID／代际的实际 tensor keys、shape、来源文件与字节账目；空 mmap 不证明选择性材料化。

`scripts/build_product_snapshots.py` 现在完整解析 pretty JSON、保留失败输出与原时间，master 观察监听 owner 及全部模型子进程，Android 从稳定 PID 对应的**实际安装 APK 字节**抽取 build-info 并前后核对路径／摘要。只有全部节点通过同一纯校验才写 topology/native；缺项写具体失败报告，不留下绿色快照，也不覆盖旧证据。既有 tail-only／文字同步声明不能升级成通过材料。

`--observations-dir` 导入实验台实际生成的 v1 原始报告并保留原时间。运行端新回执与 live collector 已接线，接口、软件验证与部署边界见 §8；旧进程缺新回执时仍明确失败。

补齐 `.github/workflows/release-build.yml` 与 `scripts/package_contract/release_build.py`：manual dispatch 的完整 revision 必须等于当前 run 的 `github.sha`；受信实验台已有同 revision DL／LL bundle 先经严格门校验，再逐附件复制并复验。通过后才执行 Windows CPU／Launcher、Linux CPU deb、Android release 的既有真实构建入口。构建前记录 canonical 来源快照，拒绝旧输出；产物经当前信任库、合同、版本和同一 build-info 静态认证后，才在同一 run 上传 `qlh-final-windows`、`qlh-final-linux`、`qlh-final-android`、`qlh-product-relay-evidence`。构建接线不安装／启动产物，平台执行仍由 `release-acceptance.yml` 完成。配置见 [发行物门](发行物门-RELEASE-ARTIFACTS.md#ci-与总门)。远端 dispatch、上传及签名发行构建尚未执行。

观测合同与构建链的阶段验证如下；这些定向离线组不相加作为全仓或物理验收：

| 范围 | 结果 | 证据（`build/ticket13-log-review-20261016/`） |
| --- | --- | --- |
| 产品门、Textual 采集／合并器、发行物门、发布汇总四组 | **282 passed** | `parent-gate-final.xml` |
| 快照生产者与 raw 观测校验 | **42 passed** | `producer-final.xml` |
| 三平台构建工作流、来源／签名／附件与失败诊断 | **58 passed** | `integrated-final.xml`（该组） |

最后将上述六个文件组合以 `-n 4` 运行，**382 passed，0 failed／error／skipped**，见 `integrated-final.xml`。集成复核另补齐发行上传器遗漏 Textual 前后 transcript 的问题：两项定向回归修前失败，修后完整复制并核对其摘要；原失败记录 `transcript-closure-red.xml` 保留。历史 bundle 与六份节点 raw 的 SHA 全部与首次审计一致，新门重放仍拒绝；未改写旧证据。

发行依赖于已有受信签名材料与实际平台执行。只读检查当前环境：Windows CPU 树缺版本／清单／合同／来源文件；旧 Launcher 使用不在当前公开信任库中的 release-20260809 且缺合同／来源；根 dist 无最终 Linux deb 与 Android release APK。Docker 与发行物门的后续实际检查见 §8。gh CLI 和常用认证接口未接入，GitHub runner／签名外部配置未独立核实；本机所需签名接口未配置不等于外部密钥不存在。

票 13 保持开放。关闭顺序为：完成运行端接线验证 → 冻结目标 revision 与受信构建来源 → 新采严格三机 DL／LL → 生产可信三平台最终产物并取得实际执行报告 → 由 `release_acceptance` 总门裁定。星状拓扑、DL／LL 连续 proper segments、分布式禁止整模装载与容量安全预算保持原合同。

## 8. 运行端回执与独立采集（2026-10-16）

`src/product_runtime_receipt.py` 的 `capture_runtime` 读取现有运行 owner：master 核对已发布 pipeline transaction、已提交 candidate 和实际 manager；PC 核对 worker prepare identity、epoch／fence 与 active runtime。操作／配置／代际、实际引擎、proper 范围、工件和 decoder count 均来自当前运行态。观察过程按模型变更的既有锁序执行；缺项、忙碌或前后变化输出具体失败，查询不创建 lazy manager。

`src/model_module.py` 的 `_LayerRangeLoadTracker` 在实际材料化时记录 tensor key、shape、源／目标 element size、绝对 shard path 和字节账目，保存不可变 `LayerMaterializationProof`。卸载与中止清理 proof；公开 metrics 返回独立副本。现有选择性适配器各段均保留固定 final norm，`product_observations._materialization` 与运行回执共用这一固定边界规则，仍逐 tensor 计入字节；decoder 范围、embedding／lm_head 归属及 visual／MTP 排除继续校验。

`src/llama_engine.py` 的 `load_layer_range` 保留 `NativeLayerLoadProof`，将实际裁片路径／SHA／范围与源总层数绑定到加载。恢复源 descriptor 后仍能区分源模型和实际裁片；加载期间文件变化则不给 proof。`capture_runtime` 读取已有 native handle 的实际 decoder count，并覆盖 keep-head 模型子进程的路径、层数与 PID。

`src/runtime_source.py` 在业务模块导入前记录不可变启动来源，默认关闭。`QLH_PRODUCT_OBSERVATION=1` 启用后，`api_server.product_runtime_observation` 通过现有 API 鉴权提供 loopback-only 的 `GET /api/diagnostics/product-runtime?nonce=<32位小写十六进制>`。转发头不参与本地判定。Git checkout 记录实际 HEAD／dirty／文件内容；Gitless 部署使用从实际 checkout 导出的 manifest，并在目标目录完整重算，环境字符串不能指定 revision 或 clean。默认导出要求 clean；显式 `--include-dirty` 仅保存开发来源的真实 dirty 标记，严格产品门继续拒绝。启动来源记录失败只影响可选诊断，不阻断服务启动。源码变动、漏文件、进程身份漂移均拒绝观测。

`scripts/package_contract/pc_product_observer.py` 用两个不同 nonce 的运行回执夹住实际递归进程树和工件读取。它核对 API listener owner、全部子孙的身份／映射／连接；GGUF 独立读取实际字节，Safetensors 对权重与支持成员逐文件重算，分别证明 loader 目录摘要与 supply weight inventory 摘要，保留原加载身份后派生供给身份，无 hash cache 写入。前后运行态或来源改变则保留失败原件。开启 API 鉴权的实验台可在采集进程设置 `QLH_PRODUCT_AUTH_TOKEN`；令牌仅作为 HTTP Authorization header 使用，不进入命令或报告。

Android 的 `LocalInferenceEngine.runtimeReceipt` 将已提交 prepare identity 与实际 native context、裁片范围、总层数、路径和文件身份绑定，产品探针独立重算实际工件。`ReleaseProcessTree.capture` 递归读取所有线程的 children 并前后复核；内核明确返回 ENOENT 时改用同 UID 的 `/proc`／PPID／启动时间前后枚举，保留完整枚举原件并拒绝权限缺口、漏项与竞态，覆盖全部子孙的 maps／连接；自身 socket 通过复制 fd 的 JNI `TCP_INFO` 获取真实 TCP 状态，子孙无权限等情况明确输出 incomplete。`--qlh-product-probe` 使用同 nonce 的单后台任务轮询，预算 120 秒，每次 dumpsys 快速返回；原 `--qlh-release-probe` 保留 4.5 秒 idle／角色检查。主机采集明确传 `--product --require-process-tree`，并从实际安装 APK 字节抽取来源。

设备和发行环境原始检查存于 `build/ticket13-runtime-20261016/`。master 已以 opt-in 观测配置启动；ROG 的旧源码目录经备份后更新 369 个 manifest 文件，逐文件摘要核对成功，按需计划任务重新注册原节点 `rog-5060`，身份和模型状态保留。部署清单实际记录 HEAD `07916c6a280e…`、`source_dirty=true`，开发采集保持该标记。

Y700 的 children 兼容补丁已保数据安装实测：`children-device/idle-product.json` 的 UID／PPID 前后枚举一致，process／maps／connections 均 complete，空载产品探针仍正确拒绝。最终绑定修复 APK SHA `bfbd5434e7c460bcbc4693da9c03cec5805f335a96b7506a6c76078dac0f385b`，25,851,817 bytes；`binding-device/install.json` 保留 `adb install -r` 回执。SAF、身份和应用数据均保留，USB 模式保持原样。ADB 曾短时消失，Windows WinUSB 接口仍在，重启 ADB 服务后恢复。

首次两机／三机 DL 在 Android native 成功后返回 `prepare_native_load_failed`。`ReleaseRuntimeProbe.PreparedMetadata.parse` 将已为 decoder 数量的 `kept_block_count` 再减源模型 `nextn_predict_layers`，形成错误层数。现直接使用 decoder-only 数量，并交叉核对裁片 metadata、实际 native count、加载文件 stamp 与 prepare 身份；`RuntimePrepareFailure` 输出有界阶段／错误码／异常类别／栈位置，保留取消和具体供给错误，日志不含凭据或 manifest 原文。修复前原件保存在 `dl-live/`、`dl-three/`；修复后 `dl-binding/` 三机自动 `[0,4)→[4,8)→[8,32)` 达到 ready，API 回答 2／6、EOS 正常停止、无 fallback，Android loaded product probe 实际通过。

两个 PC 源码更新并重启后，`dl-observed/` 的同一自动三段计划达到 ready，API 回答 2／6、真实 Textual/SSE 回答 2；3 个请求共 12 条完成回执逐项核对通过。原始采集进一步暴露 master 工件摘要合同和 Windows 映射别名问题，修复、独立字节证明及复采见 §9。ROG 原始采集无 collection_errors，但其时钟比 master 采集机约快 1.6 秒，远端窗口超出 host SSH 窗口；本机 W32Time 停止且启动被权限拒绝，ROG 标准 resync 未取得可用时间数据。原始时间、失败结果和严格窗口判据保留。Android 原件成功采集，严格门按实际 `source_dirty=true` 拒绝，未生成 topology／native 绿快照。

摘要合同修复后的 16 文件 Python 集成回归以 `-n 4` 运行，**812 passed、0 failed／error／skipped**，见 `runtime-release-digest-final.xml`；此后映射别名补丁的验证见 §9。它覆盖运行回执、真实 tiny 选择性材料化与独立摘要、API、PC／Android 采集、快照生产、DL／LL 产品门、打包路径和最终发行合同；780、761 与 review 109 是更早的重叠组合，不累加计数。Android 全量 JVM **675 项：671 passed、4 skipped、0 failure／error**，debug／release Kotlin、debug native 与 `assembleDebug` 成功；来源快照在构建漂移时确实拒绝，重跑成功日志为 `build/ticket13-verify-20261016/android-cut/prepare-binding-validation-retry.log`。

打包入口另修复两处真实路径错误：`packaging/packaging/linux/build-deb.sh` 指向实际嵌套 npm 项目、核对输入与 `dist/index.html`，构建与复制使用同一目录；`packaging/packaging/build-launcher.bat` 按脚本目录定位 Inno 输入／输出，拒绝旧 Setup 和成功退出但缺输出。`tests/test_packaging_build_paths.py` 用真实 shell／batch 执行与受控工具验证，14 项新增回归及 13 项既有兼容检查共 **27 passed**，见 `build/ticket13-log-review-20261016/packaging-paths-final.xml`。Linux 测试采用既有禁网只读 Docker 镜像，实际依赖安装和最终发行构建另行执行。

只读检查发现既有候选发行签名私钥路径存在，公开的 release-20260930 信任链验证通过；私钥内容、私公钥匹配及外部仓库配置均未读取或核实。Android 发行 keystore 配置及独立证书 pin 尚未配置；Windows CPU／Launcher 旧产物缺来源或当前可信签名，Linux 最终 deb 与 Android release APK 缺失。现有 debug／dirty 实跑继续按开发证据记录。

Docker Linux daemon 已启动，既有 `qlh-release-gate:bookworm` 镜像的禁网、只读临时容器返回 Python 3.12.15、dpkg 1.21.23、glibc 2.36。`release_artifacts_gate.py` 实际返回 1：Windows CPU 缺版本／清单，Launcher 未通过当前信任库校验；Linux 缺最终 deb，Android 缺 release APK。环境执行记录为 `release-environment-executions.json`，逐项结果为 `release-current-environment.json`。容器环境检查通过不等于完成最终 deb 的安装／运行验收。


## 9. 工件摘要与 Windows 映射身份（2026-10-16）

master 的 Safetensors 加载身份由 `model_sync.compute_model_sha256` 覆盖权重、支持文件和供给 marker；供给身份由 `model_supply_assets._weight_digest` 只覆盖权重 inventory。`pc_product_observer.rehash_artifact` 现独立读取全部成员，分别计算两种摘要，验证原加载身份，再由 `bind_artifact_rehash` 生成对应供给身份。原 endpoint receipts／stdout 保持原样，派生事实另存 `artifact.load_identity` 与 `materialization.loaded_artifact_sha256`。多 shard、支持文件／marker 修改、同 size／mtime 篡改、缓存绕过与物化证明错绑均覆盖。摘要专项 91 passed；真实四 shard tiny master prepare 不传 SHA override，经实际选择性材料化、独立重算和原严格合同验证。

`dl-frozen/master-digest-contract-real.json` 记录真实 `[0,4)`：独立读取 11 个清单成员、4 个权重 shard，共 3,787,061,057 bytes，供给 SHA `73c195fec0cf9eea534c7d67ea61a2cd3cb026b96f63c183bedd056b963ada82`，原加载目录 SHA `c509966d0d8a0b39c6f85abdadcb917531030e24cb0be9d5a62023dd64791726`。实际 55 个张量、decoder 恰为 0–3，材料化 3,764,136,064 bytes；原回执、磁盘工件和历史 raw 保留。

Windows 供给硬链接复用及旧目录回收后，`GetMappedFileNameW` 可继续返回已删除的旧 link 名。ROG 的 DL／LL 新路径具有相同 File ID，分别通过新路径建立只读映射时均复现旧名；native 命令和日志实际指向新路径且报告 4 层。该 64 KiB 复现用于判定路径语义，完整加载身份另由整段映射字节证明。只读调查见 `rog-file-identity-20261016-181303/review.md`，原 raw SHA 前后相同。

新增 `windows_mapped_artifact.capture_windows_mapped_artifact` 对每个旧名对应的完整只读 `MEM_MAPPED` 区域，以最多 2 MiB 缓冲独立执行 `ReadProcessMemory` 和 SHA-256；核对页对齐大小、实际读取总长度、PID／创建时间、前后全部映射清单与 `VirtualQueryEx` 属性。权限拒绝、短读、额外区域、错误内容与映射变化均失败。collector 保留 OS 原路径及命令原件，判定器 `_mapped_artifact_aliases` 仅在完整证明与独立重算的当前 proper GGUF 同时一致时接受绑定；文件名、磁盘 SHA 或旧路径存在性均不足以证明别名。

本机 root／native 的 1,983,370,528 bytes 与 ROG native 的 1,926,059,296 bytes 已完整读取，分别匹配 SHA `35349547f1bbe6eed52cc39c6cfe7c5bcf1f28732ec6300add35f9f5fd662169` 与 `488ec10b5f9ed93f118130cae3fc2c0f11ecf6d15fd0a6f56b375d91c55e0e9d`。独立证据为 `master-mapped-helper-proof.json`、`rog-mapped-bytes-182201/mapped-bytes.json`。它们保存原时间，作为开发诊断附件；旧 raw 未增加证明字段或重标通过。

collector／产品门两文件的映射合同回归 243 passed，见 `mapped-contract-final.xml`。测试日志曾由 autouse fixture 写进同一 `tmp_path`，触发供给精确成员检查；`multi_file_supply` 已将测试工件放入独立 `artifact/` 子目录，日志与工件 inventory 分开。该现象在 `-n 0` 同样复现，隔离后 `-n 4` 通过。


映射 helper 的专用 140 项全部通过，其中 4 项采用 Windows 真实 tiny mmap，覆盖双映射／硬链接、同大小不同字节、别名代替精确 OS 路径和可写映射拒绝。冻结 helper SHA `35f737ca95961352c4112e82668cb9d326d670a5396cdb4d4fcf5d28ab2071f4`。最终将上述范围与运行回执、模型加载、API、采集器、产品门、打包及发行合同合并，以 `-n 4` 运行 **17 文件、984 passed，0 failed／error／skipped，71.14 秒**；记录 `runtime-release-mapped-final.json／.xml／.log`。140、243、812 均与该组合重叠，分别记录用途，不累加为全仓数字。

两个 PC 保状态更新 370 个源码清单文件并重新启动，完整重算的启动 manifest SHA 同为 `0324b4ef747257b0051c230eafe4b3397aea2fe75fc6733e669492050c1466fc`，实际 HEAD `07916c6a280ed57c0eca047fca3fd5bdd35e4e13`、dirty=true。新 PID 为 master 32348、ROG 8428，Y700 继续使用 PID 20324 的已安装 bfbd 调试 APK。旧源码与全部运行日志保留；身份、SAF 和应用数据保留。

Qwen3.5-9B 经正式模型意图入口自动分配 `[0,4) → [4,8) → [8,32)`，保持原容量倍率与安全余量：

| 模式 | operation | 实际引擎与停止 | 产品入口／完成回执 |
| --- | --- | --- | --- |
| DL | `supply_b7d3782eceef49568634bfcc59c98eeb` | master PyTorch，ROG／Y700 llama.cpp；EOS | API 2／6，真实 Textual/SSE 2；12 条回执逐项通过 |
| LL | `supply_b81d2e8fc8374b018d62f6940ba393ed` | 三端 llama.cpp；text_stop | API 2／6，真实 Textual/SSE 2；42 条回执逐项通过 |

两档均 distributed_required、两个不同远端、fallback=false。每端递归进程树、映射、连接观测 complete，前后 PID／创建时间及模型 operation／config／整数 generation 一致；实际连接为 master 与两 worker 双向边，worker 各自仅连接 master。DL 的首段仍为 55 个实际 tensor、4 个 decoder；LL master／ROG 为 4 层，Y700 为 24 层，实际已加载 GGUF 与本段 SHA 独立重算一致。DL 的 ROG 与 LL 的 master／ROG 旧名映射均由正式 collector 记录完整映射字节证明。各端保持 proper segment，未观察到整模加载。两档权重均命中已校验缓存、actual_downloaded_bytes=0；真实新裁切证据仍取 §2，不把缓存复用计为新裁切。

新原件为 `dl-mapped/`、`ll-mapped/` 的 API、Textual/SSE、完成回执与 `product-capture/*-nodes/`；总索引 `physical-runtime-mapped-final-summary.json`，其 scope 明确为开发事实，strict_product_gate_passed=false。六份新 raw 均无 collection_errors；两个 `producer-result.json` 仍正确返回 1：master／Android 为 dirty 来源，ROG 原始远端窗口超出 SSH 时间窗口。未改时间、未写 topology／native 绿快照，历史 bundle SHA 仍为 `1249cdf8bb8a5fe013f1635f582467685ccba57079c44113f2c4ab12db9be817`。临时 logcat 采集已退出，`adb-final-devices.txt` 确认 HA26PYZQ 仍为 device。

票 13 的剩余关闭条件为：冻结并部署 clean 目标来源、建立有效的实验台时间同步后重新取得严格 DL／LL bundle，再构建可信签名的三平台最终产物并完成实际执行总门。当前保留清洁度、时钟和发行物的真实拒绝结果，票 14 尚未启动。

## 10. 「就地重采」的冻结来源尝试与四条硬约束（2026-10-16 晚）

借当前设备窗口（master／ROG／Y700 三端在线）尝试就地重采严格三机 bundle。**冻结树机制本身可用**，但以下四条使「就地重采出通过结果」在本机不可行。记录供 §9 末条「冻结并部署 clean 目标来源」参考。

冻结树做法（已验证）：`git clone --branch dev --single-branch <repo> G:\qlh-frozen` ⇒ 只含 tracked 文件（1087 项、**20 MB**、`git status` 零改动、HEAD 即目标 revision）。master 从该树启动后，`GET /api/diagnostics/product-runtime` 正确报出 `source_revision=c7e085d1…` 且 `runtime.errors=[]` —— 即**路径与 revision 都能对上**，运行时收据机制本身工作正常。

四条硬约束：

1. **`source_dirty` 不能靠改判据解决。** `src/runtime_source.py:123` 的判据是 `bool(git status) or not set(names).issubset(tracked)`，而 `_inventory`（:76-93）用 `os.walk` 收 `src/`＋`scripts/` 下**所有** `.py`（含 git-ignored）。`scripts/` 是白名单制（`.gitignore:187 scripts/*`），磁盘上仍有 32 个一次性实验脚本 ⇒ 从工作树启动**恒为 dirty**。而且 `tests/test_runtime_source.py:153 test_ignored_executable_source_still_dirties_startup_manifest` **明确断言**该行为（配合 :121 注释「Broad ignore rules must never hide executable Python from cleanliness」）⇒ 这是**设计**，不是 bug。
2. **供给源拒绝穿越文件系统链接。** `models/` 用 junction 接入时供给直接失败：`SUPPLY_SOURCE_INVALID: Registered source cannot traverse a filesystem link`（`src/model_supply_sources.py`）⇒ 冻结树里 `models/` 必须是**真实文件**。
3. **真实文件装不下。** `H:\qlh_models` 全量 **215.8 GB**，而可用盘位不足（G 盘剩 34 GB）。按注册表子集复制 9B 目录（10.9 GB）＋顶层有源 gguf 后仍不够。
4. **源是「配置」而非「文件」。** 最终卡在 `SUPPLY_SOURCE_MISSING: No executable pinned or configured download source` —— 它要的是 pinned／configured 下载源；现有供给记录（state 内）大概率绑定**原路径** `G:\C\PYT\qlh\models\…`，与冻结树路径不一致。

⇒ 结论：**「冻结并部署 clean 目标来源」不只是换一个干净 checkout，还要迁移 215.8 GB 模型并重绑源配置／state 路径**，属容量规划与状态迁移。本轮不改判据、不降门、不动现有部署：冻结树与复制物已删除（删除前确认 junction 数为 0，删后复核 `H:\qlh_models` 53 个条目与 639,446,688 字节探针文件完好、G 盘可用空间回到 45 GB），master 已退回工作树运行。
