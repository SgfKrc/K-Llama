# 票 13 续验与阻塞复核（2026-10-16）

> 状态：**现行**（票13产品验收收口；严格来源重采与可信签名发行实际执行为独立后置门，票21a剩余验收单列）
> 更新日期：2026-10-18

此前 Android 真裁切和开发推理证据保留；新增提交中的三机 bundle 因来源与观测完整性缺口撤销严格通过结论。运行回执、独立校验与发行附件生产链已补齐，软件验证、来源阻塞复核及协议修复见 §8～§11，干净构建见 §12，ROG免安装续验及frozen来源合同见 §13；当前终态与重构基线见 §14。

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

发行后续围绕实际artifact生产上传、外部runner／签名配置及最终产物执行门安排；A／B被推翻的归因不重复立票。票13终态及独立重构分支见 §14。

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

票 13 的产品链收口与最终发行执行分别验收，当前范围见 §12。`release_acceptance` 继续按真实签名产物及三平台实际执行报告裁定发布总门。星状拓扑、DL／LL 连续 proper segments、分布式禁止整模装载与容量安全预算保持原合同。

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

该次清洁度、时钟和发行物的原始拒绝结果保留。当前票13终态与后置发行范围见 §14。

## 10. 干净运行来源与模型存储绑定（2026-10-17 复核）

`runtime_source.checkout_manifest` 会将 ignored 的 `src/`、`scripts/` Python 文件计入来源；工作目录中的一次性脚本使来源带 dirty 标记。严格门继续要求真实 clean 来源。干净 checkout 可以导出观测 manifest，Gitless 运行目录由 `_export_manifest` 完整重算全部来源文件，使用原有模型存储和状态配置。

`model_supply_sources._trusted_storage_path` 明确支持应用 `models` 根目录映射：只将该根的后续路径转换为真实存储路径，模型内容内部的链接仍由 `_existing` 和 `_hf_metadata` 拒绝。`ensure_registered_sources` 优先验证已登记本地源；完整有效源存在时不调用下载或转换。注册表支持相对应用目录和绝对路径，见 `model_config.resolve_model_path`。

`build/ticket13-resume-20261017/clean-source-storage-binding.json` 与 `clean-source-full-validation.json` 记录独立验证：从 clean HEAD `638365519d647b6bc626620861b180d8cab3c487` 导出 338 个运行来源文件，manifest SHA `2e7d5689be722bb66d48355bf1931c56eeac9cb469bd4c89e8cfbcbc825ca19f`、source_dirty=false。Gitless 目录的 `models` 根映射至原 `H:\qlh_models`，生产注册表以 SQLite query-only 方式读取，作为供给查找的只读快照。完整调用 `ensure_registered_sources(want_gguf=True, require_cross_format=True)` 成功验证 Qwen3.5-9B 的 Safetensors、原 manifest 和绑定 GGUF；模型复制字节为 0，无下载、转换、权重加载或生产注册表写入。该记录证明存储绑定和现有源合同，完整产品准备及发行仍分别验收。

冻结来源的部署需求是保持目标模型的存储绑定、运行依赖和状态配置。2026-10-17 开发服务和调试 APK 记录 dirty=true；协议修复现已提交至 `37c87c5c03fd7b453f307a96130f77bdab3c0469`，新冻结来源与构建见 §12。

### 已推翻结论

- “clean 来源部署必须迁移 215.8GB 模型、供给拒绝应用 models 根映射、必须配置下载源才能使用已有模型”：被 `_trusted_storage_path`、`ensure_registered_sources` 及上述真实完整验证推翻。此前失败原件保持原样。

## 11. 票 21a 协议回归修复与设备窗口（2026-10-17）

新 HEAD `63836551` 已包含 PC／Android 二进制 hidden 接线。`TaskWorkerControlPlane.receive_on_coordinator` 现仅对声明新能力的 worker 添加可选布尔 `hello_ack.payload.binary_hidden`；缺省或 false 均回到原 JSON 路径，旧 worker 保留原 ACK 字段集。PC 的 `worker_binary_hidden_enabled` 和 Android `SocketTaskWorkerTransport` 将启用状态绑定到当前接受 hello 的 socket；重连重新协商。`QLH_TASK_WORKER_BINARY_HIDDEN=0` 在 master 端令 ACK false，预检、provider 和收发共用协商结果；PC worker 端关闭时完全省略新能力键，以兼容旧 master 的精确白名单。

`task_worker_protocol.validate_message`、provider、执行结果及缓存重放显式传递二进制预算上下文。还原后的内存 base64 表示继续参与原业务校验，帧大小按实际 metadata＋4 字节 body 长度头＋raw 判定，上限为 8 MiB＋1024；普通 JSON 保持原上限。PC／Android 均拒绝未协商二进制、非整数声明长度、长度冲突、超限或截断体，以及 body 与内联字段／分片引用同时存在。`max_hidden_tokens` 的 body 长度头按每帧一次计入。

| 本地范围 | 结果 | 原件 |
| --- | --- | --- |
| 供给、模型意图、运行来源、产品门与 Textual 采集器，8 文件 | 722 passed，0 failed／error／skipped | `build/ticket13-resume-20261017/supply-product-integration.xml` |
| PC 帧层、协议、provider、连接归属及层段相关，17 文件 | 436 passed | `build/ticket13-resume-20261017/protocol-final.xml`（边界补丁前） |
| Android worker JVM | 208 passed，0 failed／error／skipped | `build/ticket13-resume-20261017/android-jvm-summary.json` |
| Android debug Kotlin／ARM64 native／APK | assembleDebug 成功，来源快照 verified=true | `build/ticket13-resume-20261017/android-final-artifacts.json` |

独立复核进一步发现旧接收线程的 ACK 归属及回程完整外层帧预算两个 P1。`TCPClient._recv_loop` 现于收帧后再次核对 socket／注册 epoch／连接代际，回调通过线程本地上下文携带原始归属；`receive_on_worker` 在修改去重或协商状态前校验 owner。scheduler 的 ACK 发布按原 hello 锁序原子验证，并直接读取回调 owner，覆盖当前 client 已替换或暂时为 null 的窗口。`receive-owner-red.xml` 保留修前 4 failed／1 passed，`detached-receive-red.xml` 保留 null 窗口修前失败，`receive-owner-green.xml` 为中间 131 passed。

`binary_hidden_frame_bytes` 使用与 TCP 相同的外层 `json.dumps`（含空格、ASCII 转义和可选控制证书）计算真实长度，回程超限在发送前选择分片。新增回归包括 raw=8,389,080 bytes、ASCII／中文 metadata 的 limit−1／limit／limit+1、证书扩展和完整逐帧重组。追加 null 窗口回归后的最终组合为 **25 文件、1168 passed，0 failed／error／skipped，pytest 总耗时 91.97 秒**；JUnit 原件为 `build/ticket13-resume-20261017/source-final-detached-regression.xml`，总耗时见同名 `.log`。

上述 Python 阶段组合与最终 25 文件组合存在重叠，分别记录用途，不累加为全仓测试数量。APK 为 25,851,393 bytes，SHA `b21cf35c52191912bc19a812ac46a573390c4f98f71dd0bbab9ef81a4a1f381a`，已通过 `adb install -r` 保数据安装；USB 模式、设备身份和 SAF 保持原配置。

设备与供给证据根为 `build/ticket13-resume-20261017/`。Surface 的 Tailscale 已恢复，worker 沿用原节点 `TABLET-2TLUCNU8`，保留原 state、models 与凭据。Surface 已部署与 master 相同的 dirty 冻结运行 manifest：SHA256 `ed433021be9a161ae4c412f7895df4e6bebeb0a435e3b4eefdb57d0f7787e05d`，370 文件，源 HEAD `638365519d647b6bc626620861b180d8cab3c487`，`source_dirty=true`。部署与身份配置见 `surface-frozen-deployment.json`、`surface-identity-settings.json`，冻结来源见 `final-runtime-source-freeze.json`；`three-restored-nodes.json` 记录 master、Surface、Y700 三节点均在线。

Qwen3.5-9B 由正式 `/api/models/intent`、`strategy=auto` 自动规划三段 **master `[0,4)` → Surface `[4,24)` → Y700 `[24,32)`**，DL／LL 均达到 `status=ready`、`loaded_ready=true`。未手填层段，容量倍率和安全余量沿用产品配置；`scripts/run_product_relay_gate.py` 的正式三节点 `preflight` 与真实 `drive_tui` 均执行成功。

| 模式 | operation | 实际引擎／停止 | 产品入口与完成回执 |
| --- | --- | --- | --- |
| DL | `supply_4af7a565275b41989e832b70564cc7fe` | master PyTorch／CUDA，Surface／Y700 llama.cpp；EOS | API 2／6，Textual/SSE 2；12 条完成回执逐项通过 |
| LL | `supply_edfa6155e2c34b8eb8d109cfd11e5019` | 三端 llama.cpp；text_stop | API 2／6，Textual/SSE 2；42 条完成回执逐项通过 |

每档包含两个独立 API session 和一个真实 TUI session，均为 `distributed_required`、两个不同远端、`fallback=false`；操作／配置／整数 generation、阶段顺序、range／assignment／artifact SHA 与实际完成回执一致。原件分别存于 `dl-three-final/`、`ll-three-final/` 的 `api-chats.json`、`tui-development-three-node/`、`completed-handoffs-verified-v2.json` 和 `product-capture/`。总索引 `three-node-final-development-summary.json` 保存六份节点 raw 的 SHA，并核对最终源码仍等于冻结 manifest，既有两机证据 SHA 保持原值。

六份节点 raw 的 `collection_errors` 均为空，递归进程树及其 children／maps／connections 均 complete，前后 PID／创建时间稳定。DL master 实际材料化 55 个张量、3,764,136,064 bytes，decoder 恰为 0–3；LL master 实际加载 4 层，Surface 与 Y700 在两档均实际加载 20／8 层。每端 loaded artifact 的独立重算 SHA 与供给身份一致，观测的 `full_model_loaded=false`；Surface 的旧临时映射名由正式 collector 的完整映射字节证明绑定到当前裁片。实际非 loopback 连接为 master 与两个 worker 的双向边，worker 各自只连接 master。

`master-final.err.log` 记录 Surface 的 20 层、Y700 的 8 层 `layer_weight_bytes` 全为正数，双方 `binary_hidden_negotiated=True`；`binary-wire-final.log` 在两档均记录真实二进制 stage_offer、Surface 返回 stage_result 和 master 再转发 Y700。`ll-three-final/live-capacity.json` 的 `pipeline_layout` 三节点均为 `llama_cpp`，与实际运行回执一致，投影入口为 `Scheduler._attach_pipeline_node_contract`。

DL 的 Surface 执行新本地裁切，`reused_bytes=0`、`actual_downloaded_bytes=0`；Y700 复用此前已验证尾段。LL 复用已验证工件，各端新增下载字节均为 0。容量口径分别保留：DL 的 `single_node_full_model_candidates=[]`，LL 的该字段含 master，因此 LL 记录验证多段执行及整模禁载，单机容量拒绝证据取 DL，见两档 `intent-latest.json`。

两个 `producer-result.json` 均按原严格门返回 1，具体拒绝项为 master／Surface 的 dirty source manifest 和 Android 的真实 `source_dirty=true`。原始时间与失败结果保留，未生成 topology／native 绿色快照；这两档的来源口径为冻结 dirty 开发验收。

为切换三机，原两机 `dl-final/` 操作 `supply_5b9994a78b8c4c4e944ca9831892e9c4` 已通过正式 DELETE 取消；`dl-final/cancellation-for-three-nodes.json` 记录 `status=cancelled`、`reason_code=INTENT_CANCELLED`。两机旧结果继续保留为开发证据。修复前 `ll-before-fix/` 记录正式模型意图自动得到 LL `[0,24) → [24,32)`、Y700 新本地裁切、API 回答 2／6、text_stop、distributed_used=true、fallback=false；此前 900 秒超时请求尚未生成层段计划，记录不足以归因于二进制前向。

上述dirty冻结部署按开发证据记录，票13终态见 §14。票 21a 的 PC／Android 三端新格式接力已有上述证据，32-token 新旧线格式对拍、旧／新能力对端混跑与时延收益仍待验收。重构设计与当前阶段见 §14。


## 12. 干净来源与Windows修复记录（2026-10-18）

普通CI、来源和发行合同检查继续保留；同clean revision的master加两个不同远端严格DL／LL证据转入 §14 的独立发布前门。真实 Textual/SSE、逐段完成回执、独立进程／连接／装载观测、proper segments 与容量预算继续按 `run_product_relay_gate.capture`、`product_relay_gate.validate_bundle` 和 `product_observations.derive_node_observation` 验收。

可信签名 Windows／Linux／Android 最终产物的实际发行执行列为后置的发布前门；Linux 近期实机执行后置。这项范围划分允许产品链先行收口，`release_acceptance` 的签名、来源、产物执行要求及既有失败报告继续保留。Windows 免安装包可用于 Surface 隔离目录验收；需要清理现有仓库时先列 dry-run，保留模型、state、节点身份、凭据及可恢复路径。独立Launcher裁撤职责已纳入[重构设计](单一事实源重构设计-2026-10-18.md#10-裁撤与数据迁移)。

当前先在独立分支完成事实归属与状态机设计，再按重构阶段迁移；19～22已有成果及剩余优化按[新设计依赖表](单一事实源重构设计-2026-10-18.md#11-分期排票与完成门)接入。

干净基线为 `37c87c5c03fd7b453f307a96130f77bdab3c0469`，固定 Android `aaa5938879922143817fd2e3bdb5c866fb78376c` 及其余四个仓库 pin。隔离 detached checkout 的 Git 状态和真实文件 inventory 均经过验证；`runtime_source.main` 导出 338 个运行文件，`source_dirty=false`，canonical manifest SHA `bbc356970e1781ff959bda17faafd952011a60e0f00d84665d5d8cbf2354932d`。新证据根为 `build/ticket13-clean-20261018/`，旧 dirty raw 与原始时间、SHA、失败结果保持不变。

主节点已正式卸载旧模型后切换到 Gitless 新目录，继续使用生产 state 与映射至 `H:\qlh_models` 的应用模型根，模型复制 0 字节。Surface 现有 C 盘约 80.0 GB 可用，暂未执行仓库清理；已准备同 manifest 的新源码归档，原 worker、模型和身份保持原状。远端部署执行被自动审批拒绝，返回 `blocked by policy`，未给出具体原因；干净三机重采待 Surface 完成运行目录切换后执行。来源归档或本机 HEAD 不能替代运行端的 PID 绑定观测。

### 已安装 APK 与三机产品回归

从冻结来源真实编译 Android ARM64 native 和 debug APK，使用既有日常 debug 证书保持数据覆盖安装。设备端 `sha256sum` 与 pull 后重算均为 `fb648c7b7b285a72e448ab352374dd5d797c229f673ab9e3117d3b52a9bf35e3`，APK 25,858,915 bytes，包内 build-info 为 `37c87c5c`、`source_dirty=false`；UID 10324 保持，SAF、节点身份与凭据沿用。首次隔离构建证书不匹配的安装失败原件保留，随后以同一 APK 内容和已安装公开证书匹配的开发签名完成覆盖安装，见 `android-device-final/installation-result.json`、`installed-build-info.json`。签名口径为 debug 开发证书。

正式 `/api/models/intent`、`strategy=auto` 对 Qwen3.5-9B 自动分配 master `[0,4)` → Surface `[4,24)` → Y700 `[24,32)`；两个不同远端与容量倍率、安全余量保持产品配置。主节点和 Y700 来源为 clean `37c87c5c`，Surface 仍为 `63836551`、dirty=true：

| 模式 | operation | 产品入口与完成回执 | 实际装载 |
| --- | --- | --- | --- |
| DL | `supply_d13428369a8345108d9fce61f7030470` | API 2／6，真实 Textual/SSE 2；12 条完成回执逐项通过，EOS | master PyTorch／CUDA；55 个实际张量、3,764,136,064 bytes、decoder 0–3；Surface／Y700 native 20／8 层 |
| LL | `supply_b36de85af7304785b23c29acf6df7b34` | API 2／6，真实 Textual/SSE 2；42 条完成回执逐项通过，text_stop | 三端 llama.cpp，native 4／20／8 层 |

六份节点 raw 的递归进程树、children／maps／connections 均 complete，实际连接 peer 为 master 与两 worker 的双向星状边；各端 loaded=true、full_model_loaded=false，operation／config／整数 generation 与已发布计划一致，proper ranges 与装载引擎逐项一致。三次请求均 distributed_required、fallback=false。DL 的 `single_node_full_model_candidates=[]`；LL 的该字段含 master，单机容量拒绝证据取 DL。总索引 `physical-runtime-current-summary.json` 保存两档入口、逐条回执、六份 raw 与 SHA，scope 为混合来源产品回归。

DL 的 `product-capture-addresses-r1/producer-result.json` 与 LL 的 `product-capture-time-r1/producer-result.json` 均严格拒绝 Surface 的 dirty source manifest，尚未生成 topology／native 绿快照。LL 首次采集还记录远端结束超出 SSH 窗口约 0.049 秒；Surface 实际执行 `w32tm /resync` 后，新采集时间窗检查通过。原失败时间与报告原样保留，时钟查询与校时记录见 `clock-verification.json`。DL 首次地址表类型错误保留，后续按节点对应 IP 列表的正式合同重采。

`final-closure-evidence-check.json` 独立重算旧三机 20 份证据 SHA、冻结来源和主节点部署各 338 个运行文件，六个 checkout 状态保持 clean；旧 clean Windows 失败包和新安装 APK 的 SHA 另行核对。原 dirty／失败证据继续保留其原始语义。

### Windows 免安装包的实际缺陷与修复

冻结 clean `37c87c5c` 的正式 `qlh-edge.spec` 试包已真实构建为 102,382,673 bytes，ZIP SHA `9af62c661dcbafa9aa06e18e1d0ae9518266d0858a2353c2cb09b9bef9768f94`。五个程序入口、491 文件往返及 244 个 x64 PE 静态闭包完成验证；主程序实际首启暴露 `_prompt_tailscale_setup` 未定义 `owner_hwnd`，同时缺少 LL 首段／中段所需的 patched keep-head 组。原包及失败报告位于 `windows-package/`，其状态仍为首启验收失败。

`packaging/packaging/launcher.py::_prompt_tailscale_setup` 现接收可选 owner，`_check_tailscale_requirement` 的两条路径显式传递。无控制台和无 Tailscale 的回归修前 1 failed，修后与已有启动门组合 31 passed；证据 `windows-startup-fix/`。这是当前主程序的首启修复，旧独立 Launcher 的治理顺序保持原计划。

`src/keep_head_shim.py::default_shim_path` 在 frozen 模式解析 `_MEIPASS/keep_head/qlh_keep_head.dll`。`keep_head_resources.collect_keep_head_binaries` 静态验证九 DLL 的 x64、所需补丁导出、组内导入与依赖；`qlh-edge.spec` 将完整私有组直接加入 COLLECT，与 pip `llama_cpp/lib` 分目录保留，避免 Analysis 将同名 ggml 搬到公共目录。缺文件、补丁符号、依赖或错误 ABI 导入具名失败。定位、spec 和供给打包回归 96 passed，两次真实独立进程 native load／backend init／free 成功；最终与启动门联合验证 **6 文件、127 passed**，见 `windows-fixes-integration.xml`。独立只读复核未发现新的可复现 P0/P1，C header ABI 另以 frozen proper 裁片实际加载验收。

上述 Windows 修复仍在工作树，原 clean `37c87c5c` 包与主节点冻结运行来源保持原字节。修后试包使用新的 dirty 来源与独立输出；原包的失败状态保留。

### 干净 checkout 的 Windows 原生生产前置

`scripts/model_tools/build_keep_head_cpu.ps1` 从锁定的 `android/app/src/main/cpp/llama.cpp` 干净 checkout 生成 x64 CPU llama／ggml，再调用既有 `build_keep_head_shim.ps1`；`packaging/packaging/build-cpu.bat` 的 llama_cpp_only 档在 PyInstaller 前执行，原签名门与精确失败退出码保留。脚本要求明确的 UCRT64 gcc／g++／make 和 CMake，支持 `QLH_KEEP_HEAD_TOOLCHAIN_BIN`、`QLH_KEEP_HEAD_CMAKE`；全新目录、固定源码、`GGML_NATIVE=OFF`、AVX512 关闭及 GPU／网络构建关闭均核查实际 cache，已有 cache 或输出具名拒绝，生产组保持原字节。

`native-fresh/summary.json` 记录实际 Windows PowerShell 5.1 构建退出 0、约 62 秒；patched llama.cpp SHA `784cbbbbe03568727b9e02e64efda51d52bc8bbf`、前后 Git clean，九 DLL 通过真实 PE 合同。builder SHA `68c760237b8e1062b9aff329923082e87dd7d66fab4e363ee882498b537c388f` 与成功构建回执一致，完整命令、工具版本、pins、输入和输出摘要分别留证。本机默认工具目录缺 cmake.exe，成功构建使用显式 `-CMake`；缺工具与复用旧目录均退出 1，拒绝检查后的产物摘要保持。新增 16 项合同覆盖路径含空格、缺工具、缓存拒绝、工具与 shim 失败传播。

最终将定位、私有闭包、原生构建、首启、供给打包、来源、签名及工作流合同联合以 `-n 4` 运行：**10 文件、261 passed、4 subtests passed，0 failed／error／skipped，56.30 秒**，原件 `windows-product-build-final.xml／.log`。前述 96、127、118、16 均与该最终组合重叠，分别记录用途。额外旧前端检查得到 118 passed／3 failed，原件 `windows-build-integration.xml`；三个失败均期待已不存在的 `api_server._resolve_frontend_dist`，当前 HEAD 源码与原测试均未改，核实见 `legacy-frontend-test-failure-audit.json`。该旧前端测试边界独立登记，保留失败结果，治理范围随后评估。

### 修后 frozen EXE 的实际验证

`windows-fixed/smoke-report.json` 记录无 Tailscale、无交互 stdin 的真实 frozen EXE 已成功执行弹窗“否”，进入 `--check-only` 的具名未配置模型拒绝（预期退出 1）。此前两次探针的隐藏桌面设置未传入 Win32 启动结构，程序停在 MessageBoxW；失败 JSON 与采样栈保留，新探针通过实际 CreateProcessW／STARTUPINFOW 的 lpDesktop 控制桌面，明确记录一次弹窗选择与进程退出。

同一主 EXE 经 `--keephead-worker` 从包内私有组实际加载 Qwen3.5-9B 的 proper `[0,4)` GGUF，ready 报告 n_layer=4、n_embd=4096，关闭 ACK 后退出 0；前后模型 SHA `35349547f1bbe6eed52cc39c6cfe7c5bcf1f28732ec6300add35f9f5fd662169` 相同，模型复制 0 字节。pip 和 keep_head 两组各在独立进程用包内目录与 System32 搜索，实际模块路径分别归属对应目录。主 EXE SHA `c9aa43f063b3b032abdf57cc84f9ae4853b11108de0aa6c3c233b0aac5e90fc5`。产品自带 frozen Python，首启无需外部 Python／pip；实验控制器仍使用本机 Python。生产占用 UDP 29876，API 就绪与 Surface 上该包的实际执行尚未验收。

修后试包的主程序由 dirty 修复 overlay 重新冻结，伴随程序和该试包的九 DLL 复用来源独立登记；全新九 DLL 的生产验证取上节，两个来源各有真实摘要。试包用途为隔离执行验证，可信发行签名仍按发布前门处理。

修后 ZIP `windows-fixed/QLH-Edge-Inference-0.1.8.3-windows-x64-cpu-37c87c5c-dirty-fixed-unsigned-test.zip` 为 105,941,302 bytes，SHA `69eb2d1db4394f09a1de488d8c6e84d2d380fd020c84d6ae100189497244400e`。502 项逐文件解压往返的大小／SHA 通过，实际已测主 EXE 与包内、解压后字节一致；旧 clean 包及其证据 54 文件摘要保持，见 `result-summary.json`。PyInstaller 公共 `_internal` 的 ggml 别名为 pip 原字节，patched 字节仅留私有目录，核对见 `native-byte-isolation.json`。该 EXE 的四层验证范围为 native context ready／close，forward、middle 工件、完整接力仍分别按产品证据验收。实验进程均退出，隐藏桌面资源已关闭。

### 后置发布前验收

严格来源重采、可信签名实际执行和平台安装的现行边界统一见 §14。上述clean／dirty包及失败、成功原件各自保留。Surface窗口不足由ROG独立免安装与三机产品续验接续，实际结果见 §13。

## 13. ROG 免安装续验与 frozen 来源合同（2026-10-18）

适用范围为 Windows 免安装程序的独立运行环境、同来源 Android 更新、三机产品路径与来源观测；发行签名和发布总门继续按 §12 的后置安排验收。证据根为 `build/ticket13-rog-clean-20261018/`，产品进展索引 `physical-product-progress.json`。

### 同来源产物与 ROG 首启

固定来源为 `5d157b05eab6911328b3edcb87d9188eb7d09e85`，主仓及五个子模块 clean/pinned；`frozen-checkouts.json`、两端产物内的 `build-info.json` 分别记录实际修订。Windows 重新执行五个正式 spec，native llama／ggml／shim 按锁定源码实际编译，四个 GCC runtime DLL 来自指定工具链，九 DLL 摘要均绑定 `windows-package/native-fresh-build-receipt.json`。ZIP 为 129,597,720 bytes，SHA `5d4c7d893765d321858ff714547da2b059ea55f363e3bbc9c00938f2bf966fd1`；主 EXE SHA `753921abd4235123308edd01051faae997ca02b67553efb7644a3f7ed986b516`。559 项解压往返与 281 个 x64 PE 依赖检查通过；worker 文件共享冲突及单文件工具首次加载失败的原件保留，有界复试通过，详见 `windows-package/validation-report.json`。

ROG 使用全新 `C:\Users\ROG\QLH-Portable-20261018-5d157b05` 目录，产品由包内 EXE 和依赖运行，PATH 限定系统及 Tailscale，沿用默认 state、节点 `rog-5060` 和加密凭据。按需计划任务没有重启触发器；PID 20348 实际持有 API 8001 与 TCP 8889，`/api/health` 返回 ok，worker 注册／hello 完成，连接指向唯一 master。旧验收源码目录仍在磁盘，未配置为产品依赖；该口径是隔离程序环境，并非全新操作系统。原件为 `rog-clean5d-deployment.stdout`、`rog-first-start.stdout`、`rog-ready-after-probe-cleanup.json`。

Android 实际执行 `assembleDebug` 和正式 native 构建，APK 25,863,011 bytes，SHA `7d3a3a259f08cb29ee2449d008b9dc36085be96d41f4124fd14ad6afd3f35ec9`。沿用日常 debug 证书覆盖安装，安装字节回拉摘要完全一致，包内来源为 clean `5d157b05`。应用数据、SAF、身份及 USB 模式保持，见 `android-build/result.json` 与 `installed-update-receipt.json`。

### 三机自动 DL／LL

正式 `/api/models/intent`、`strategy=auto` 对 Qwen3.5-9B 形成以下连续 proper segments。容量倍率、预留、安全余量及 master 唯一协调边界保持产品配置。

| 模式 | operation | 自动分配 | API 两问 |
| --- | --- | --- | --- |
| LL | `supply_ba62b17cf4004217a97a3e59cf20d640` | master llama.cpp `[0,20)` → ROG `[20,24)` → Y700 `[24,32)` | 2／6，text_stop，无 fallback |
| DL | `supply_7186cf5a90764bca9a1e40232d34fc32` | master PyTorch/CUDA `[0,8)` → ROG llama.cpp `[8,24)` → Y700 `[24,32)` | 2／6，eos_token，无 fallback |

两档每问均为 distributed_required，实际 workers 为 ROG 与 Y700，operation／config／整数 generation 与 ready 计划一致。LL 选择由独立产品操作创建，已有 ready 操作保留后采集；DL 由记录中的正式自动意图创建。DL 供给中 ROG 采用 distribute，Y700 采用 local_cut 后登记；真实状态原件见 `dl-portable-three-clean5d/intent-history.jsonl`。

`pc_product_observer.observe_pc` 的 master 两档采集无 collection_errors：LL 实际范围 `[0,20)`，DL `[0,8)`，full_model_loaded=false，来源均 clean `5d157b05`。DL immutable materialization 为 108 个张量、5,494,026,496 bytes。Android 两档正式 product probe 的递归进程树 complete，native 均为 `[24,32)` 8 层、full_model_loaded=false。ROG 的 DL 独立 OS 映射观测仅见一个 3,465,964,704-byte GGUF，头信息为 qwen35／16 blocks，完整 SHA `aeb895453613e728baee5fb55e229140e992c7bbdc0a9b6bfd97f161a4b7875f`；产品进程及其 native 子进程均来自包内主 EXE，映射符合 `[8,24)`，见 `dl-clean5d-product-observation/rog-mapping.json`。

一次 PowerShell 只读观测在 JSON 序列化时超时，遗留进程占用约 15 GB 私有内存，导致两机尝试被真实容量门拒绝。控制器以完整命令指纹确认并仅结束自身探针，ROG 可用 RAM 恢复至 10,145,439,744 bytes；产品进程与其他应用保持。拒绝诊断、独立内存观测、定向清理及恢复记录分别保存在 `ll-portable-two-node-clean5d/`、`rog-capacity-independent.json`、`rog-diagnostic-process-cleanup.json` 和 `rog-ready-after-probe-cleanup.json`。

### frozen 来源缺口与修复

clean5d 的 `runtime_source.initialize` 沿用源码 checkout 清单，frozen EXE 缺少源码目录／Git 时捕获失败。ROG 加载成功后，完整 `/api/diagnostics/product-runtime` 明确返回 409、`PRODUCT_OBSERVATION_UNAVAILABLE`、`startup source Git observation unavailable`，见 `rog-clean5d-frozen-source-refusal.json`。产品推理成功与严格来源通过分别记录。

`runtime_frozen_source.export_package_manifest` 从实际完整程序包及已 staged 的 build-info 导出固定 `product-package-observation.json`。`runtime_source.initialize` 的 frozen 分支以真实 PID、创建时间、OS EXE、安装根 cwd 和完整文件大小／SHA 捕获启动事实；`observe_source` 复算同一库存，拒绝文件增减、内容漂移、链接／reparse、非法路径和启动身份变化。排除项仅为代码固定的顶层运行数据、清单自身及后生成的精确 `manifest/install-manifest.json`；其余程序文件仍纳入清单。`build-cpu.bat` 的 llama_cpp_only 档在伴随工具与 pubkeys 完成、provenance stage 之后导出，随后原发行签名步骤覆盖该来源清单。

`product_observations._installed_exe_source` 和 `pc_product_observer.rehash_installed_exe` 增加独立的 installed_exe 合同，绑定 OS EXE／cwd、启动 manifest、完整程序库存、真实 build-info 字节 SHA 及 clean/full revision。独立复核重放曾发现纯门漏验 OS cwd，现已补规范化比较和前后漂移拒绝，原复现已转为明确拒绝。观测元数据用于证明来源字节一致性；可信发行认证继续由既有 release gate 验收。

`api.routes_diagnostics.product_source_observation` 提供空载即可访问的 `/api/diagnostics/product-source`，共用 opt-in、真实 loopback peer、nonce 与现有 API 认证边界，不读取模型 manager；完整 runtime 入口保留模型事实要求。诊断router抽取和OpenAPI基线由`7d3390ad`入库；其余来源修复及安装产物合同随后由`fb870d1d`入库。

最终定向整合 **10 文件、489 passed、3 skipped、47.86 秒**，跳过项均为本机 Windows 符号链接创建权限限制，见 `frozen-source-fix/final-integration/`。独立 observer 的 cwd 补充组合 **633 passed、3 skipped**；CPU 接线及既有 packaging 合同 **71 passed**；API／router 专项 **50 passed**。组合存在重叠，分别记录用途。

### 修复试包的 ROG 实际验证

五项产品补丁复制到独立 `5d157b05` checkout，正式 begin／stage／构建后复核均保留 source_dirty=true。五个 spec 重新冻结，九 DLL 复用上述同 pin、同输入的 clean5d 实编组并单列来源；新 ZIP 129,642,550 bytes，SHA `b5b68a19a14f1a760f2f89972a98c01a276136e05bc0ccd921ce819e1913e05e`，EXE SHA `8c930013333fbdd994876d5e58cb0cf294acb4b51c25e2f6ac2ee286616d11c4`，562 项完整解压往返通过。新 helper／router 的 frozen bytecode、来源清单与程序字节分别核验，原 clean 包保持原样，见 `windows-observation-fixed/zip-ready.json`。

ROG 在独立 `QLH-Source-Observation-20261018` 目录、临时 state、single 模式和 API 8013 启动该真实 EXE。PID 14956 的 product-source 返回 200／installed_exe，560 个程序文件与独立外部 observer 的完整复算一致；PID／创建时间、OS EXE、安装根 cwd、清单和前后来源均绑定。实际观测约 3.66 秒，进程树没有模型映射；完整 product-runtime 因空载继续返回 409。严格 `_source` 明确拒绝 `source_dirty=true`。原件为 `source-only-independent-result.json`；实验实例已定向停止，默认 state 和 clean worker PID 20348 保持，见 `rog-source-only-stop-r2.json`。

### 后置验收项

同clean revision完整严格重采与可信签名最终产物实际执行转入 §14 的独立发布前门。既有试包和观测保留原来源、时间、SHA及严格拒绝结果。

## 14. 票13收口与重构基线（2026-10-18）

票13按实际达到的产品易用性和稳定性范围关闭，结束继续追验。§13记录Windows免安装独立运行、同来源APK更新、自动供给及Qwen3.5-9B三机LL／DL的实际成功；来源修复已由`fb870d1d4ef2d1d2901db1c5e299ac387fc9597b`入库。严格来源重采、可信发行签名与完整三平台实际安装尚未全部验收，其判据和原始失败记录保持。

剩余工作为独立发布前门：

- 从同一clean revision及实际子模块pin构建最终包，重采真实Textual/SSE、逐段完成回执与完整独立节点观测，生成严格DL／LL证据包。
- 可信签名最终产物的认证及实际执行分别验收，debug或dirty试包继续仅按实际用途记录。
- Windows／Android优先安排目标环境实际安装；Linux完整产品执行随发行范围安排。ROG首启为独立程序环境验证，保留既有state及磁盘残留，不描述成全新操作系统。

严格发布门继续由现有工具执行；产品收口不修改旧passed／failed值。后续产品不退化由重构R6承接，设备缺口按实际未执行登记。

独立分支`refactor/ssot-design-20261018`从干净`fb870d1d`建立。功能证据见[当前功能与事实归属审计](当前功能与事实归属审计-2026-10-18.md)，目标设计与阶段见[单一事实源重构设计](单一事实源重构设计-2026-10-18.md)。生产实现迁移和代码裁撤尚未开始。
