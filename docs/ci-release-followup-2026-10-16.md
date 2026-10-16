# 票 13 续验与阻塞复核（2026-10-16）

状态：**工件查找早退和 PyTorch compiled 权重卸载残留已修复；Android 本地裁切已进入并安装收口开发 APK，正式供给真实新裁 9B 尾段并完成推理。首轮三机 LL／DL 开发验收通过，收口 APK 的两机自适应 LL／DL 补测通过；各轮来源和范围分开记录。票 13 的最终同 revision 签名三机发行门继续开放。**

本记录复核此前同日的阻塞归因，保留原始证据于 `build/ci-release-followup-20261016/`；新证据独立存于 `build/ticket13-verify-20261016/`，不覆盖原始日志。当前主仓 HEAD `5f63b267c197dcf24af3905d12a33830f01a6ef2`、Android HEAD `07fcba305d5ac5ebd8e9497bab7aafbf6a78d409`，本轮有未提交修改。三端来源分别记录，不能仅以 HEAD 相同宣称最终同 revision 签名验收通过。

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

本轮只独立核实仓库内容：`release-acceptance.yml` 消费 `qlh-final-*` 与 `qlh-product-relay-evidence`，要求 Windows／Linux／Android 三组 self-hosted 标签；当前仓库 workflow 未找到生产并上传这两类 artifact 的 build 工作流。

**GitHub runner 注册、仓库签名变量／密钥是否已配置，本轮未独立查询，不能沿用原报告“runner 未注册／三项均未具备”的断言。** 最终仍需可信签名发行物、来源匹配、实际平台执行与严格证据包；不代持密钥、不用临时签名绕门。用户已允许最终发行物验收后置，本轮的 dirty／debug 运行不冒充最终门通过。

下一步发行工作应围绕实际 artifact 生产上传、外部 runner／签名配置核实和最终产物执行门安排；无需再根据已经被复核推翻的 A／B 归因另立重复修复票。票 13 保持开放；重构票 14 仍按原先行为边界与独立分支裁决处理。
