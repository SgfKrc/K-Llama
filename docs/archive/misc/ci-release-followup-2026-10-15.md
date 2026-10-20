# 票 13 续验记录（2026-10-15）

状态：修复后 Qwen3.5-9B 三机 DL／LL 的自动供给、真实 API 与 Textual/SSE 开发实测均通过；最终同 revision 签名发行验收仍开放，票 13 尚不能关闭。上轮记录见 [首次验收](ci-release-acceptance-2026-10-15.md)。

## 2026-10-15 续验：Android 探针与三机恢复

本轮证据另存 `build/ci-release-followup-20261015/`。继续当前分支，未提交；未使用 Reasonix。开始基准主仓为 `1c9551bc8339f03c0fa6eb88a887872ba71c9e37`，期间其他会话提交 ignore／Android 指针、README 仓库名及文档归档；归档提交 `ae90e45a` 也调整两份生产文件的文档引用注释。部署参考 HEAD、逐文件摘要与安装包来源保留各自实际值，不将后续 HEAD 直接套作运行身份。运行源码以逐文件摘要记录，不能把 HEAD 变化当作已部署源码的认证证明。

新增 Android nonce／PID 绑定的只读服务探针，以及主机采集与发行门接线。角色与连接状态、native 后端设备、实际已加载 decoder 数及层段范围必须相互一致。进程 fd 只复制检查，不发送或接收网络数据；maps 有界读取，失败明示。Y700 无权读取 proc children 文件时保留 `children_complete=false`，另存系统 ps 原始观测，不伪造完整标记。Gradle 生成真实源码来源并嵌入 APK，发行门拒绝明确 dirty 元数据。

| 续验项目 | 实际结果 |
| --- | --- |
| Python Android／发行／产品门／通道组合 | 341 passed，0 skipped；`release-product-focused.xml`，不含后加的 provenance 及 dirty 产品门测试 |
| 来源生成器最终专项 | 75 passed；包含 ignored／未跟踪源码、assume-unchanged、Git 指针漂移、签名前快照逐字段／类型校验、重复键与非法值拒绝；见 `build/provenance-package-tests/1d4f48f0fe124e5a89a804686ace7ea3/results.xml` |
| Android JVM 全量 | 581 项，577 passed／4 skipped，0 failures／0 errors；53 份真实 JUnit，见 `android-junit-summary.json` |
| Android 构建 | 完整 `assembleDebug`、release Kotlin 编译及后加的签名前来源复核均通过；篡改 `source_dirty` 的实际 package 负向实验被拒；未产生可信 release APK |
| Y700 保数据更新 | 安装／启动成功；同 PID 25112 的 JNI 后端可用，worker READY；未清 SAF 或身份数据 |
| ROG 受控部署 | 194 份 src 与 49 份支持文件完成摘要校验；既有手动 task 启动新 worker PID 25616，独立 SSH 复核存活；身份／凭据仅存摘要，未修改 |
| 构建修复后的首次冻结普通通道 | 6,993 passed／16 skipped（unit 6,961／16，external 32／0）；`channels/final-frozen/manifest.json`，早于后续 hello／TUI 修改，物理／smoke 为 NOT_RUN |
| HELLO／legacy／TUI 最终冻结普通通道 | **7,035 passed／16 skipped**（unit 7,003／16，external 32／0）；`channels/hello-legacy-final/manifest.json`，1,021 份源码／配置前后相同，身份与部署前一致；物理／smoke 仍为 NOT_RUN，真实设备开发实测另列 |
| Windows／Linux 来源打包接线 | 最终 32 passed；`packaging-provenance-final-tests.xml` |
| 量化工具／模型工具专项 | 最终 108 passed／2 skipped；`quantizer-toolchain-final-v3-tests.xml`，两项 skip 为本机目录符号链接权限 |
| 本轮最终发行物门 | 两次实际退出 1；Docker daemon 和预备镜像已可用，Windows 旧包及 Linux／Android 最终产物缺口保留；最后一份为 `final-artifact-retry-result.json` |

Y700 安装包 SHA-256 为 `06821347196791fe61e917c5cadfc1ee654ed30d04a787ca9fbe943e6dd69175`，来源树摘要为 `f4a25d6e61932655bf9ff19be7d1cbd0747a26c136267ed91a81464dd45712db`，`source_dirty=true`。这是真实开发包观测，不作为最终签名发行通过。

首个正式自动意图 `supply_d049299cb4ed4555b5d1f42700f1b829` 发现三个候选，但规划为 master `[0,24)`＋Y700 `[24,32)`，ROG 留在控制面。ROG 本次 inventory 可用 RAM 为 5,224,882,176 字节，小于最小四层段保守预算 5,495,497,319 字节，差 270,615,143 字节。未放宽容量守卫或手动指定层段。供给／校验／原生准备均自动完成，Y700 新探针确认真实 8 层与 `full_model_loaded=false`；此两机结果不足以通过三机产品门。


### 首轮两机开发物理结果与剩余条件

LL 补充 API 请求分别为 `1+1` 与 `3+3`，返回 `2`、`6`；每次 7 个可见 step、7 份已完成远端回执，均为 `finish_reason=stop`／`stop_reason=text_stop`、required route、`distributed_used=true`／`fallback=false`。回答、请求及精确代际保留于 `three-node-ll/supplement-api-chats.json`，原始本次 operation 日志保留于 `three-node-ll/master-operation.log`。这是真实两机补充，未运行 Textual/SSE 三机产品档。

master 进程与全部子进程的映射／连接读取完整。已删名的旧构建临时路径没有重建别名；直接读取完整模型映射内容的 SHA，核对本次 manifest 的 `[0,24)` 裁段工件。Y700 同次 JNI 观测为 `[24,32)`、native decoder 数 8、`full_model_loaded=false`，maps 与本段路径一致，socket 对端仅为 master。ROG 没有本次执行段，不能凭控制连接记作三段接力。

随后通过正式模型意图切换 DL，操作为 `supply_902d2832f54d456da52fa96e945e36f1`。旧 LL 原生状态正确退休；新 inventory 为 master CUDA 7,451,181,056、Y700 RAM 10,835,623,936、ROG RAM 4,595,417,088 字节。没有可行计划，系统保持 `waiting_nodes`。实验台保留观测后用公开 DELETE API 取消等待，未发送 DL 推理。容量不足时该状态仍使用泛化的“等待更多节点”说明，是后续 UX 复核项；本轮没有把它改报模型或分段已就绪。

ROG 盘点未找到可确认属于旧测试的原生孤儿，旧 PID 1480／29132 已不存在；未清理用户应用。全机进程扫描有 22 项权限／读取不全，保留 partial 标记；worker 及子进程自身的映射／连接读取完整。通用只读 runtime probe 与调用说明留在 `rog-deployment/runtime-probe-readme.md`，供释放资源后的 LL／DL 验收复用。

Y700 切档后新 nonce／PID 探针确认模型卸载；后续回读已安装 APK SHA 与原部署包一致。adb 曾短暂掉线，采集器有界重试恢复，没有切 USB 文件传输模式。物理状态汇总见 `physical-followup-status.json`，明确 `passed=false`；该诊断材料不是发行产品 bundle。

本次首轮未形成三段。后续 ROG 可用容量上升后的自动三段结果见下一节；每次准入仍使用当次 live inventory，不以注册画像推算。可信 release 签名包、稳定来源、最终 Windows／Linux 产物及成功 CI run 仍是最终发行前提。


### 后续真实三机 LL、Textual/SSE 与 DL 容量等待

后续 `supply_422b9673f1704963804584acad2b48c1` 自动规划并准备 master `[0,20)` → ROG `[20,24)` → Y700 `[24,32)`，三段均由 llama.cpp 执行。没有手调层段、放宽容量预算或加载整模来准备主节点。config 为 `d4178ca2b0984efe9c3303e8a8148446`，generation 为整数 `1792052406970269400`。

两次正式 API 请求分别得到 `1+1 → 2`、`3+3 → 6`；真实 Textual 当前画面的输入经 SSE 得到 `2`，保留输入前后 transcript、原始 SSE 和 `screen.svg`。三次请求均为 required route、`distributed_used=true`、`fallback=false`，以 `text_stop` 正常停止；每次 7 步、14 条完成远端回执。`three-node-retry2-ll/completed-handoffs-verified.json` 逐项核对共 42 条回执的 operation/config/整数 generation、层段、assignment、工件 SHA、结果类型与先后顺序；不是仅凭答案或节点在线状态判定接力。

这次真实 TUI 首次暴露采集器错误：App 的默认 screen 已隐藏，`app.query_one` 仍可能查询默认 DOM。采集器现从 `app.screen` 读取输入、状态和 transcript，新增真实双 screen 测试，防止把隐藏旧答案当本次输出。定向 `39 passed`，修后真实 TUI 的 `tui-active-screen/diagnostic-result.json` 为 passed。

同窗 master 与 ROG 的进程树、模型映射和连接观测完整，均只有一个匹配本次 manifest 的正确层段映射；master 的删名临时路径通过完整映射内容 SHA 核对。该次 Y700 在线参与回执，但 adb 再次不可用：设备列表为空，mDNS 广播的 `192.168.111.238:38879` 拒绝连接（10061）。没有切 USB 传输、重置授权、配对或修改 adbd。早先 JNI 的 8 层／`full_model_loaded=false` 不能代替此次新鲜观测，因此不能据本次材料宣称三机 native／拓扑最终门通过。该次原始核查为 `adb-post-hello-recheck-v2.json`；稍后 USB adb 自然恢复后的新观测见末节。

DL 重试 `supply_b673c6c9b5c949e482ceb5fa436e2cfd` 的 inventory 为 master CUDA 7,451,181,056、Y700 RAM 9,561,690,112、ROG RAM 6,716,305,408 字节；未形成容量可行计划，保留等待观测后公开取消。没有发送 DL 推理，也没有把 LL 成功当作 DL 通过。

该次 DL 的具体约束另经只读离线重放确认，材料在 `dl-capacity-explanation/report.md` 与 `analysis.json`。master CUDA 最多接八层；在现有 master → ROG → Y700 顺序下，ROG `[8,16)` 要求 6,749,906,125 字节，比该次 inventory 多 33,600,717 字节；Y700 `[16,32)` 要求 9,328,794,829 字节，可以通过。master `[0,8)` 另需 host staging 至少 7,129,705,012 字节，但当次精确 host inventory 未归档，不能宣称只释放约 32 MiB 就一定现场成功。生产纯 planner 的门槛／门槛减 1 字节边界已验证；没有修改预算或把离线放行当真实加载成功。`waiting_nodes` 是容量拒绝后自动扩池的外显状态，不能把“worker 尚未加入”写成此操作已证实的根因。

只读预算复算另发现保守多计：GGUF descriptor 的 `other` 包含 150,980,608 字节 decoder 外 MTP，实际 builder 排除它，planner 仍将其计入边界保留，四层 LL 门槛因倍率及余量多出 362,353,459 字节。这不是每段重复计入整模。见 `ticket13-readonly-budget-mtp-finding-160150.json`；本轮未凭算术下调内存守卫，后续需以 builder 同源预算和 native 峰值证据修复。

### HELLO 锁竞争与并发收尾修复

首次三段供给 `supply_d370b8e9af1c495cbe937f6d4ed5a22b` 的工件均 available，但 prepare 返回 `SUPPLY_WORKER_DISCONNECTED`。ROG 日志显示 hello ACK 已及时收到，随后心跳 ACK 连续缺失三次并重连；master 的多次 hello 回调收尾延迟后集中返回。供给 GC admission 持有 full-chat／stage／native 锁，原 hello 同步收尾可能等待 stage，把接收线程留在回调里。GC 是该次具体持锁者的高置信推断，现场没有完整历史线程栈；ACK、延迟返回及重连则有直接日志。

修复保留接收路径的身份校验、hello ACK 和 registration fence，后续 provider 准入、配置发布、cleanup／recovery／opt-out 复用既有 layer-config retry monitor；每节点一项、全局受 max_nodes 限制，stage 使用 nonblocking 获取。GC admission 的真实锁竞争回归证明 ACK／fence／后续 heartbeat 不等待 GC。

独立复核发现并修复两个异步窗口：同连接新 hello 更新 snapshot 但尚未入队时，旧 changed 意图必须保留交接；旧 opt-out 等待配置锁后，必须在真实副作用边界持既有 owner 锁重新核验身份和角色。相关组合 190 passed／hello 专项 19 passed，证据 `hello-fix/p1-targeted-final.xml`、`p1-regressions-second.xml`。独立复核进一步指出 legacy 空 candidate 事务的锁外清理可能误卸载后继层段。已通过既有执行锁串行旧清理与 legacy materialization，双方获锁后核验 host／事务／阶段身份，原生操作不持配置或注册锁；增加 5 项确定性交错，最终相关组合 195 passed，scheduler 受影响选择 36 passed（组合有重叠，不相加）。独立复核 `review-legacy-final.md` 确认原复现的后继保持 loaded、该项 P1 闭环；闭环前未部署。原报告及最终复核分别保留为 `hello-fix/review.md` 和 `review-final.md`，不覆盖发现历史。

中间普通通道 `post-hello-frozen` 为 unit 6,988 passed／16 skipped／2 failed，external 32 passed；两失败来自真实卸载测试 fixture 未提供新增收尾所需 refresh 锁，补真实锁后该文件 22 passed。后续 `post-hello-p1-frozen` 为 unit 6,997 passed／16 skipped／1 failed、external 32 passed：取消终局不等待 provider 线程退出，原测试在 runner join 后过早读取 fake 记录。已用显式事件控制两阶段，保留终局、释放与真实取消记录断言；相关 33 passed，目标真实重复 100 次通过，见 `parallel-cancel-fix/REPORT.md`。生产取消合同未改。最终 `hello-legacy-final` 为 7,035 passed／16 skipped，前后冻结一致；中间失败证据保留。

### 构建来源与量化工具闭环

Android `preBuild` 从真实主仓和固定子模块生成来源，所有 APK variant 在编译／native／合并后、package 动作执行前复核原快照。`--verify` 只读比较全部字段及类型，不能通过重新生成覆盖构建前来源。实际负向实验只翻转 `source_dirty`，保持摘要不变，package 仍拒绝且原快照未被改写；正常 `assembleDebug` 成功且嵌入元数据逐字节匹配。证据在 `build/provenance-package-verification/`。此后新构建没有覆盖 Y700 原有安装，不能将新 APK 摘要当作设备已安装身份。

Windows CPU／Launcher 和 Linux 构建入口也已接入构建前独立快照与签清单前复核；来源变化、已有签名树、已有 build-info 和无 Git 导出目录均拒绝重新贴身份。开发来源 dirty 如实保留，最终发行门拒绝 dirty，不生成临时密钥绕过门。

实际执行 `llama-quantize` 构建还发现并修复三处缺陷：原脚本把上游基线误当实际补丁 commit；Windows 选择 GCC 后可能被默认 Visual Studio 生成器忽略且默认构建 Debug；复用构建目录时可能优先选中另一配置留下的旧文件。现在分别记录上游与真实 source revision／补丁摘要，构建前后重核来源；独立查找 SDK Ninja，Windows GNU 缺 Ninja 明确失败；核对 CMake 实际生成的 C／CXX compiler 文件，显式构建 Release，并按实际单／多配置唯一选择产物。

最终源码真实构建成功，证据为 `quantizer-native-verified-build.json`；工具链 Release／Ninja／GNU 15.2.0，实际源码为固定 `784cbbbbe03568727b9e02e64efda51d52bc8bbf`，上游基线仍为 `47e1de77aa0f06bf73cfd8c5281d95979f89fcbe`，vendor 未修改且自身 clean。help、Q4_K_M、Windows 运行时依赖及托管包摘要校验通过；可执行文件 SHA-256 为 `2ac1976162555d99c8df679d038c71bd27870dc9ba035704d2384e3da80b2971`。只读复核发现的旧产物误选已回归闭环。本次没有执行真实 MSVC 或 Linux 量化工具编译，也没有替换既有生产托管包。

最初 `channels/final-core` 也通过，但构建脚本在其运行期间变化，因此另起 `final-frozen`，不合并成同版冻结证明。后者前后 1,017 份源码／配置 SHA 完全一致，节点身份与部署前摘要一致；见 `verification-freeze-final-before.json`／`verification-freeze-final-after.json`。期间外部会话提交 `8204bead` 仅修订归档文档链接，Git HEAD 变化如实保留，不能把本地脏工作树或部署字节等同于该 HEAD 的认证发行。

Linux 镜像首次两次构建因 Debian HTTP 502 失败。后用 build 目录内独立 HTTPS 配方预备 `qlh-release-gate:bookworm` 成功，再以禁网容器确认 Python／dpkg／必需库；见 `docker-release-image-prepared.json`。这是环境前置解除；最终 `.deb` 仍缺失，不能算安装／启动验收。

### 修复部署后的三机续验（17:10 起）

受控公开 unload 后，master 从 PID 27152 重启为 21696；真实节点身份摘要未变。重启加载 195 份 src Python 的摘要与最终测试冻结完全一致，`QLH_INFER_*` 无额外覆盖。ROG PID 25616、Y700 PID 25112 继续使用各自已记录的部署来源，不能因此宣称三端同 revision 的签名发行物。两 worker 完成新 accepted／healthy hello；供给期间超过两分钟的 13 次观测全部健康且心跳前进，ROG 能力更新 hello 也被正确处理，见 `hello-fix/post-deploy-heartbeat-result.json`。

新 LL operation `supply_a0e5346411b44298b719bd815a0ca70f`，config `4d20a74a91ae4549baa445469cd13267`，整数 generation `1792055745037321200`，自动计划仍为 `[0,20)` → `[20,24)` → `[24,32)`，工件 available 后三节点 prepare／commit／finalize 完成。真实 Textual/SSE 得 `2`，正式 API 得 `2`、`6`，均正常 text_stop、无 fallback；三请求共 42 条完成回执精确验证通过。证据在 `three-node-post-hello-ll/`，其中 `completed-handoffs-verified-v2.json` 同步更新已恢复的 Android 观测范围；先前 checker 遗留的“无新鲜 adb”说明已由 v2 更正，原材料保留。

Y700 USB adb 随后自然恢复，未切 USB 模式或重置授权。当前 APK SHA 与部署包相同；新 nonce／PID 探针确认分布式 worker READY、JNI 八层 `[24,32)`、`full_model_loaded=false`，maps 为本段、socket 对端仅 master。master／ROG 同窗进程映射与连接完整、均一个匹配本次工件的 proper segment。Y700 应用进程 children 读取仍受权限限制，保留 `children_complete=false` 并另存系统 ps 原始记录，不把它改成完整标志。

随后通过正式意图切换 DL：`supply_991cf2808039491f8bd1b9d5453b51bd`，config `d899d985abc24652a11465349ebd0dd7`，整数 generation `1792056278284219200`。本次实时容量通过原守卫，自动规划 PyTorch CUDA master `[0,4)`、llama.cpp ROG `[4,12)`、llama.cpp Y700 `[12,32)`，三端供给及 prepare／commit／finalize 完成。旧 LL 由公开切档流程退休，没有加载整模或手调区间。

ROG 实际下载 2,434,677,366 字节，Y700 实际下载 4,045,951,700 字节，均 `reused_bytes=0`；master 本地构建新 Safetensors 头段。因此本次 DL 包含新层段的真实传输与校验。真实 Textual/SSE 回答 `2`，正式 API 回答 `2`、`6`；均 required route、分布式、无 fallback，EOS 正常停止。每次 1 个可见 step，含停止轮的 4 条远端完成回执；三请求共 12 条的 operation/config/整数 generation、range/assignment/artifact、结果类型与顺序全部通过。

DL 原生加载证据分别在 `three-node-post-hello-dl/`：

- master `master-selective-cuda-evidence.json`：工件 manifest 所有文件完整 SHA 匹配；Safetensors header 只含 decoder 0–3，共 55 个张量。真实 loader 为 `safetensors_key_filtered`，source 3,764,137,024 字节、物化 3,764,136,064 字节，CUDA peak delta 3,764,140,544 字节；CUDA 权重已脱离源文件映射，不能把 OS 未保留 Safetensors mmap 误写为没有加载模型。
- ROG 同窗 OS probe 完整，一个正确 `[4,12)` GGUF 映射，内容摘要与当前 operation 相符。
- Y700 新 nonce／PID probe 为二十层 `[12,32)`、`full_model_loaded=false`，一个当前尾段映射；另以只读 run-as 核对 mapped payload 完整 SHA、manifest 的 operation/node/range 均一致。socket 对端仅 master，应用 children 的权限限制仍保留。

首次 master OS 采集脚本只处理单文件路径，把 DL 目录的 manifest 查到了上一层，故 `dl-post-hello-orchestration.json` 如实保留 false。已用目录感知的只读脚本补采 `master-runtime-after-api.json` 并补上述真实 CUDA／header／文件摘要观测；没有修改历史失败或将这次脚本错误解释为推理失败。最终证据综合另存 `physical-post-fix-final.json`，`development_relay_checks_passed=true`、`final_release_gate_passed=false`。

收尾健康观测 `final-runtime-health.json` 确认 master PID 21696、ROG/Y700 accepted 且 healthy、三节点 online、DL ready、所有段均为 proper segment；1,021 份源码／配置及真实身份未变化。服务保留在该 DL 就绪状态。没有运行显式整模基线，不宣称这次证明了整模单机绝不可能运行或特定性能提升。

### 后续容量诊断与 MTP 预算修复

用户安排先处理两项开发修复，最终发行物验收等待真实设备窗口。代码、真实源预算复算与隔离原生层段观测见 [专项记录](容量诊断与MTP预算修复-2026-10-15.md)。本轮未部署生产三机，不替换上文各次真实验收的来源或历史结论。

### 仍待排期／外部验收

| 项目 | 本轮结论与下一步 |
| --- | --- |
| 同来源最终发行 | 当前 dirty 开发树、ROG 已部署源码和 Y700 debug 包保留各自来源。需固定 clean 主仓／子模块与最终签名产物，同 revision 重跑严格 DL／LL bundle。 |
| Windows | CPU 最终 version／安装签清单缺失；旧 Launcher key 不受当前 trust。按正式信任链重建并验安装／启动，不能借临时 key 绕过。 |
| Linux | Docker daemon 与预备镜像已可用；最终签名 `.deb` 及安装／首启验收仍缺。 |
| Android | 当前 debug JNI／adb 开发验收通过；可信 release APK、最终升级与来源／签名验收仍缺。 |
| CI | 后续容量／MTP 修复最终普通通道 unit 7,284 passed／16 skipped、external 32 passed；物理 pytest／smoke 是 NOT_RUN。另存的真实设备开发实测不能冒充一次成功的最终 CI release run。 |
| 准入说明 UX | 后续已修：保留真实容量原因，API／TUI／Android 展示各节点 RAM／显存／host staging 候选缺口，区分容量等待与实际加入；详见 [容量诊断与 MTP 修复](容量诊断与MTP预算修复-2026-10-15.md)。 |
| 预算同源性 | 后续已修：descriptor 与 builder 共用 GGUF 张量选择，排除明确声明的 decoder 外 MTP，保留未知非块张量。真实源最低四层预算减少 362,353,459 字节，倍率／余量不变；独立四层 native 峰值观测通过，范围与限制见上述修复记录。 |
| 用户后置事项 | 重启自动拉起、显式本地整模基线仍后置；单机退出集群／本地模式的设计不因本轮改变。 |


