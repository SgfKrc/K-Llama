# MODEL-INTENT-UX-01 产品流程验收（2026-10-13）

状态：2026-10-14 已完成票 12d。三机 DL／LL、Web／Textual／Android、分段装载和退出／重入验收通过；以下保留排查过程及失败记录，最终结论见文末。

## 目标与边界

任意 PC／Android 启动软件后，使用统一入口发现并绑定 master，选择主节点目录中的模型。master 沿既有供给与 native prepare 事务自动规划、传输、校验、装载和提交；用户无需复制 host 或逐台选择层段。星状拓扑、master 唯一协调以及 DL／LL 连续多段接力保持原有设计。

分布式模式任何节点不得装载整模。磁盘上的源整模可用于裁切；显式切换 local 会退出推理集群，真实卸载后才可加载整模基线。重新进入 distributed 必须真实卸载本地整模。

## 实现

- `src/model_intent.py` 记录用户所选供给 operation 及 prepare invocation，并投影既有供给 owner 和 native transaction；不另建库存、分配或 loaded-ready 事实状态。
- `/api/models/intent` 接收模型 ID、engine、quantization 和自动策略，不接收用户本地模型路径、手工层段或 master host。`catalog/current/{operation_id}/connect` 提供统一目录、恢复、进度、取消和发现入口。
- PC worker 依据当前已注册 TCP 连接和 bootstrap identity，将白名单意图请求转交 master。来源、鉴权和 master control fence 均沿用既有边界；不制造或提升凭据。
- operation 沿用 `supply_*` ID，config/generation 由 native 事务投影。仅同代际 active plan 已提交且分布式开关有效才报告 `loaded_ready=true`。推理 metrics 增加相同 model operation/config/generation。
- Web 模型页、Textual 主入口和 Android 选模面板均使用统一目录及 operation 进度。聊天需真实 ready，分布式请求明确发送 `distributed_required`；local 请求明确发送 `local_only`。
- 未启动 worker 的选模可进入 `waiting_nodes`，取消、模式退出、连接替换与迟到 callback 仍使用原 owner 的门。

## 物理设备与无预置工件条件

验收三节点：本机 CUDA master（API 8000/TCP 8888）、Surface CPU worker（API 8001/TCP 8889）、Y700 Android worker（USB ADB，Tailnet）。模型 `qwen3-5-9b`，32 decoder layers，hidden 4096，源及 tokenizer identity 沿既有供给契约。

- Y700 旧自动供给目录可逆移入 `files/model-intent-backup-20261013/supply`，新 `files/models/supply` 为空。保留 app data、SAF 授权和旧工件；不经 adb 传模型，不切 MTP。
- Surface 旧 received 库存逐项移入隔离目录，新 received 为空。本轮需重新裁切或实际接收本段，不能将旧准备状态作新验收。
- 2026-10-13 经用户提示检查 Surface C 盘：旧验收缓存占用约 46.1 GiB。逐 manifest 核实 proper segments、无 native 引用、无 reparse 后，仅删除 21 份可再生旧 `segment.gguf`；保留清单和模型源文件。
- 释放 49,135,889,120 bytes，C 盘可用从 42,414,247,936 bytes 增至 91,550,195,712 bytes（约 39.5 → 85.3 GiB）。清理路径严格限制在验收隔离库存。
- Surface ScheduledTask 实际运行 runtime 内 `launch.ps1`。本轮为已批准的三机验收增加明确 `/32` 来源配置，保留生产默认来源边界；修改其它 build 脚本不会影响该任务。

缓存审计证据：`build/intent-surface-cache-inventory.json`、`build/intent-surface-cache-cleanup.json`。

## 实际产品验收记录（待收口）

PC Web 访问实际 Surface worker API，由 worker 转交 master。通过模型页选择 9B 和 PyTorch，未指定节点或层段，提交 operation `supply_faa05ff3ee3c48d7be9d179fb2e1123d`。master 自动分配：CUDA `[0,4)`、Surface `[4,8)`、Y700 `[8,32)`。

Surface 自动就地裁切本段。Y700 从空供给目录真实下载 951,058,432 bytes，reused/resumed 均为 0。22:55:49 Surface 连续丢失三次 heartbeat ACK 后发生连接替换，本次 operation 安全失败 `SUPPLY_WORKER_DISCONNECTED`；未将供给失败算作推理成功。通过产品重试按钮提交第二次 operation `supply_6c8131823b16454593fd80f36527e7af`。第二次 operation 已证明跨操作身份核对后的断点恢复 951,844,864 bytes，继续记录新增下载字节。

Android 和实际 Textual 主入口（访问 Surface 真实 API，无替身）恢复同一 operation，并在未 ready 时阻止聊天。Textual 截图与 API 调用记录在 `tui-live-progress.svg/json`。真实聊天、native maps、退群／重入和 LL 操作待完成。

产品证据目录：`build/model-intent-20261013/`。UI 操作脚本只驱动实际控件，无 HTTP route 替身或预置 fixture。native 证据脚本只读取已由 UI 提交且 ready 的 operation，不代替产品选模。

## 自动化验证（持续更新）

- 后端 582 项组合：580 passed、2 failed。失败在单文件也复现，确认是卸载测试 fake 缺少现有契约；補齐后 unload/cold-start/intent/gateway/node-mode 六文件 193 passed。原 582 项组合不计为全部通过，最终修复后再跑适当组合。
- PC Web：此前主线 79 Playwright 通过；聊天请求路由与思考补充相关 31/31 通过，typecheck/build/diff check 通过。
- Android：此前 442 JVM tests 通过，debug APK 与 androidTest APK 构建通过。真实安装保留数据；新增进度前置、名称/ID 搜索与未下载源模型允许提交，两 false 思考参数，定向 JVM 79 passed，APK/androidTest APK 构建完成，待实机更新验证。
- Textual：213 passed、10 skipped。跳过的真模型／物理设备 opt-in 不计为 passed。

## 只读审计与修复收口

复核发现 intent 写盘异常、创建失败、取消重试、退群后的 prepare 迟到回调、prepare deadline 与 task finally 六项边界，需要复用供给/native owner 修复。另发现新设备若仅靠本轮临时 `/32` 环境配置才能访问 path-free intent，仍不满足启动即用；正在收口为只允许现有受信来源或完成 bootstrap 且当前真实 TCP 注册 peer 的意图入口，原一般模型路径/注册/下载管理仍默认 loopback。上述修复均需自动回归及后续真机重测。

Web/Textual 聊天完成状态已从真实 response metrics 展示 model operation/config/generation，避免当前选模变化改写旧回答的模型身份。新增 Web 8 项、Textual formatter 与 UI fixture 41 项通过；fixture 不替代三机验收。

## 剩余验收门

- 完成新工件供给、同代际 native prepare/commit、真实两个远端 handoff 的 DL／LL 聊天，确认答案、`distributed_used=true`、`fallback=false` 与 operation/config/generation 一致。
- 核对 Android `/proc/maps`、Surface native child 参数与 master proper crop descriptor／权重键／maps，无节点整模装载。
- 实测 local 退出与 distributed 重入，确认旧 UI／后台迟到请求不能改写选择，并核对 native cleanup 债。
- 更新最终回归结果、物理设备证据、限制与排票状态。本轮不提交代码、不另建分支。
## 2026-10-13 23:45 后续核验

- 后端修复后组合回归：735 passed（26.43s）；生命周期定向组合：675 passed、1 skipped（Windows symlink 权限）；intent source gate 定向组合：336 passed。Textual 补充组合此前为 121 passed；跨设备 current 同步仍在补测。
- 第二轮 DL 在产品取消时 Android 已接收 1,845,231,616 bytes：本轮真实新增 893,386,752 bytes，跨操作恢复 951,844,864 bytes。取消用于部署新 APK，保留有效 partial，未进入 native prepare 或推理。
- 来源门证据修正：此前移除进程变量后，仓库 `.env` 自动补回 `100.64.0.0/10,fd7a:115c:a1e0::/48`，该轮成功访问不能证明新增 registered-peer 校验。现验收启动器显式指定不匹配三机的 `192.0.2.0/32`，检查真实 master 进程环境后，Surface intent/current=200、普通 models/current=403；Y700 实际 UI 刷新 catalog/current=200。固定 path-free intent 使用真实注册 peer；未修改 `.env` 或生产来源门。
- Surface 清理后又生成本轮有效层段，23:40 实测可用 87,881,768,960 bytes（81.8 GiB）。缓存清理保留源 GGUF、本轮工件与清单。
- Android Receiver 拆分定向 Contract 42、Worker 17、Intent 17，共 76 JVM tests 通过；debug APK 已保留数据安装，SHA-256 `c1fd8425739741592d4421765d3f8d3a26702e95ce9b00ae4aaf51a6588f4608`。`receive` DEX 18,368→9,072 code units，HTTP helper 940，传输 helper 2,554；5 秒聚合耗时诊断已接线。尚无实机提速结论。
- 新 APK 实机冷启动又发现 HTTP catalog 早于 TCP 注册：23:44:06 来源门正确拒绝，23:44:09 注册、23:44:12 hello ACK，但初始目录失败不启动 poll，界面不会自动恢复。该缺陷正在补上同绑定、同 epoch 的目录重读；不能用人工刷新替代启动体验验收。

来源门证据：`master-restricted-source-environment.json`、`surface-restricted-source-gate.json`、`android-restricted-source-gate.json`；原 broad trust 进程环境作为失败证据保留。
### 最后部署与第三轮 DL

Textual current 同步补丁完成：22 定向 tests；TUI 全组 229 passed、10 skipped（物理设备／真实模型 opt-in）。Surface 已部署 runtime bundle SHA-256 `b9892d88ce5bdd099c99f746f99c5ad975d2e7632ce56fc2be1ed58fffaee2a9`。

Android 整组 457 JVM tests 通过；补充冷启动恢复 21 定向 tests 通过，新增 4 条竞态测试。最终 debug APK SHA-256 `cdef0fbab07ffaa761064e174c0a0835cc0c71b978d96d541ef9714e92bdfb5b`，已保留数据安装。在不点刷新、不切页面的初始聊天页，首次来源拒绝后自动恢复 cancelled current；XML 和请求日志已留证。Android 的当前操作进度移到模型目录前，并支持名称／ID 搜索。

第三轮从真实 Web `重试此操作` 提交 `supply_4fc17d1de11e48b0985849965ce58919`，没有指定 node_ids 或层区间，仍自动分配 master `[0,4)`、Surface `[4,8)`、Android `[8,32)`。Android 未点击刷新便显示该新 operation；真实 Textual 只读入口也恢复同一 operation 并在未 ready 时阻止聊天。当前供给／推理结果继续记录于 `dl-final-ui.json/log`，不能在准备及推理通过前计为验收成功。
## 2026-10-14 持续验收与收口

第三轮下载保持稳定，Android 恢复 1,845,231,616 bytes 的有效 prefix 后继续网络下载；Surface 已供给本段。接收耗时诊断窗口中约 95% 为 HTTP read，live-check、写盘／hash、checkpoint、progress 占比小，拆分后尚无显著提速的证据。直连地址由本机 Tailscale ping 确认；未减少 live fences 或摘要校验。

真实 Android 进度在目录之前可见，尚未准备时 `/proc/maps` 无 GGUF。新增 `downloading` 中文标签。

检查真实聊天调用又发现 `ChatRepository` 仍使用 `distributed_preferred`；修为 `distributed_required`。新增两个直接调用仓库的 HTTP 契约测试，核对成功响应的 operation/config/generation 持久化及拒绝后不加载本地模型。修后 API 契约组 54 passed；最终 Android 全组 **463 passed，0 skipped**，debug APK 与 androidTest APK 构建通过。APK SHA-256 `19e5edbfaa1461df9f006c9452fb79b69597a2fcef98b870d86a8610e7b11143`，待第三轮 DL 完成后安装，避免中断传输。

只读复核另确认两个 P1，正在有界修复并复用现有 owner：

1. 同 node_id 的 TCP session 原地替换，旧 native assignment 已被执行门拒绝，但 intent ready 投影此前未核对连接代次，可能让 UI 暂时放行聊天。
2. ready DL 直接切 LL 时，旧 Android 段要到新 prepare 才卸载；供给在更早 inventory/planning 使用旧占用下的 available，存在提前容量拒绝、重试仍不释放旧段的路径。需在新供给前沿原卸载／cleanup ACK 清理旧 runtime；不虚报释放内存。

上述第二项是静态复核确认的条件性路径，不能声称当前三机已实际触发。准备、聊天、LL 与模式退出／重入仍须真实完成后再关闭票面。
### 三段 native 就绪后暴露的真实聊天问题

第三轮 Android 层段总计 4,554,567,568 bytes，断点恢复 1,845,231,616 bytes，本轮实际新增网络下载 2,709,335,952 bytes。2026-10-14 00:43 三机 native prepare／commit 全部就绪，config `005c0a0ce8f54da68b3b94fba2e8e2b5`、精确 generation `1791909775815263900`。只读 native 证据显示 master 权重键仅 0–3，Surface 与 Android 均映射本段；保留为 `native-pytorch-third-selection.json`。这些证明准备与分段，不代表聊天通过。

第一次真实 Surface Web 聊天因 worker 使用 interactive 模式被明确拒绝。Web／Textual 已依据真实 binding 的 `is_master=false` 选择 full 模式，保持 `distributed_required`。Web 相关 24 Playwright tests、typecheck 通过；Textual 定向 139 tests 通过。第二次真实聊天暴露锁循环：worker 在转发等待 master 时持有 full-chat 模型锁，master 回调同一 worker 的层段执行又需该锁。旧请求取消并且远端 stage 退出的日志已保留。后端两入口对 required worker 转发解除该锁；跨线程回调、失败不执行本地模型、本地执行仍串行，以及冷启动／deadline／判停组合 **74 passed**。首轮新增测试的四个断言未适配既有 503、SSE error(done=true) 与追问二次执行契约，修正后通过；旧失败日志保留。

两项 P1 收束：native-ready 投影复用 worker 原执行门检查连接代次；新供给前沿原 full lock、transition 和身份重查完成卸载，等待 identity-bound cleanup ACK，再重新检查 mode／role 开始 inventory。迟到卸载不能覆盖新 native／local 选择。最终 intent 定向 **128 passed**，此前 owner 组合 **1260 passed、1 skipped**（Windows symlink 权限）；后者对应本轮 routes_chat 锁修复前状态，不能作为锁修复后的整组结果。

2026-10-14 01:13 部署 runtime bundle SHA-256 `879be68fcbe902f5e0db29887367fa84d9a3fd429f84e515e405ded5d14d9d54`，master PID 25324；Android 最后 APK `19e5edbfaa1461df9f006c9452fb79b69597a2fcef98b870d86a8610e7b11143` 已 install-r 成功、保留应用数据与 SAF。ADB 曾短暂断开，恢复后安装完成；没有切换 USB 模式。

通过真实 Web 重试按钮提交 `supply_a3eba8f2837d44c2b76860ce6abd219a`，Android 初始聊天页自动显示该新 operation 的清点进度。仍需 DL／LL 真实回答、远端 handoff 及退出／重入证据才关闭本票。UI 脚本保留 HTTP 中 generation 的原始十进制字符串，避免将 JavaScript 超安全整数后的舍入值当作精确代次证据。
### 默认客户端入口继续核验

`dl-handshake-ready` 已由真实 Surface Web 控件完成两题：2／6 均正确，`distributed_used=true`、`fallback=false`，两个 worker、operation/config/generation 均匹配。对应 master workflow 为 `wf_4c9bc17ae13f`、`wf_df32f63e9002`，真实 remote handoff 日志各包含 Surface 与 Y700；native 分段证据为 `native-pytorch-dl-handshake-ready.json`。本轮 Android 完整段被复用，实际新增网络下载 0。

下一步默认客户端实测没有计为成功：Textual default `auto` 未按 distributed owner mode 收紧，虽选择了 full stream，仍触发旧锁等待，60 秒后 SSE 客户端取消，`wf_e8b3fd43e22a` 停止；绑定 mode 决定请求路由的修复正在补测。Android 实际发送按钮在 01:31:24 被通用 API 来源门拒绝 403，错误和界面均保留于 `android-dl-chat.log` 与 `android-dl-chat-result.xml`。

供给冷启动新缺口：首次选模在 Android TCP 注册但 hello 扫描尚未完成时发出 inventory，消息进入 hello 等待阶段，后续连接替换，旧 operation 空等超时。startup 门现复用原 control dispatch／pending 与真实 TCP epoch；inventory wait 也重查原连接身份。定向 66 passed，组合 1206 passed、1 skipped（Windows symlink 权限）。补丁尚待统一部署，后续不能靠人工等 hello 绕过该问题作为最终冷启动验收。

转发锁旁路的角色变化边界也已修复：通过旁路的请求带执行约束 `forward_only`，开始执行时角色不再 client 就具名拒绝；guard 后的角色变化仍只能调用原 forwarding owner，拒绝后不进入 master pipeline／native 追问。相关组合 **82 passed**。已认证 peer 的 required text chat 来源门新增精确请求白名单，保留 Bearer、原控制证书与同连接代次复查；路径、auto/local/preferred、任务图、外部／图片请求和一般模型管理都不借该来源例外放行。既有来源／intent／聊天组合 **213 passed**，专门新增的 API 边界组在收口。
### 最终部署与实测轮次

registered required text-chat 边界专项 **93 passed**，相关组合 **340 passed**；literal false 使用对象身份判断，数值 0 不得伪装为 external=false。Textual owner mode 请求路由修复定向 **147 passed**；修后真实默认 Textual 输入并回车，`dl-tui-route-fixed` 返回 2，required/full、两 remote 与精确 generation 通过。

最后 runtime bundle SHA-256 `d890f4ba7d5057d957cae231e0bfe19735ed4be3596631fc8904e2257c75cb14`，master PID 4212，真实 source CIDR 仍为 `192.0.2.0/32`，已记录 `master-final-source-environment.json`。Android 默认 Gson 改为 LONG_OR_DOUBLE，保留 constructor 注入。API 定向 62、完整 JVM **466 passed，0 skipped**，APK／androidTest 构建通过；最终 APK SHA-256 `8681943d35bcf9b41ebdc1445264d4c05dcc1d5e1cf3e2bb23a0a69f4e508f84` 已保留数据安装。

安装 APK 后过早提交 `supply_c96d4cb5f308402ca250bbfb75b930b6`：提交在 02:00:02，Android TCP 注册在 02:00:09，hello ACK 在 02:00:22，节点尚未注册时本次快照只有 master／Surface。这轮安全失败 `pipeline_distributed_capacity_insufficient`，没有 prepare、没有加载整模。取消脚本在失败之后找不到取消按钮而超时，不能记作成功取消；亦不能用这轮证明 hello 等待门成功。三台已加入后真实 Web 重试提交 `supply_c45001a7afba48729c69c5780632b65d`，当前继续实际产品验收，未关闭票面。
## 2026-10-14 02:34 收束记录（仍在验收）

真实 Android DL 聊天已返回 `2`，持久化 Room 记录确认 `distributed_used=true`、`fallback=false`，两个远端 worker 均参与；operation `supply_c45001a7afba48729c69c5780632b65d`、config `94eaaff5848c4463b10fcdeb26757b94`、精确 generation `1791914962549556600` 均一致。对应 workflow `wf_999a71dcc7b4` 的 Surface 和 Y700 实际 handoff 已留存。证据：`dl-final-android-persisted-chat.json`、`dl-final-android-result.xml/png`、`native-pytorch-dl-final-before-retirement.json`、`master-dl-final-before-retirement.err.log`。只读证据脚本不再复制 SQLite SHM，每次使用独立 DB/WAL 克隆并显式关闭连接；此前克隆脚本的 Windows 文件锁错误不记为产品失败或聊天成功。

Web 较长 full 聊天仍被约 10 秒首部预算取消：失败 workflow `wf_18a8bac95580` 保留于 `dl-final-three-joined-ui.json/log`。修复 full SSE 在等待上下文事务锁及推理之前立即发送合法 `: processing` 注释，使代理能及时发送首部；不伪造 token 或完成事件。真实阻塞执行首块测试及来源、冷启动、deadline、判停组合 **176 passed**。Web 另补流式正文的用户取消与 reader 释放；超出安全整数的身份字段在 JSON 边界保留原十进制，wire 仍为 number，产品进度和回答均显示精确代际。新增边界测试 **7 passed**、typecheck 通过；相关 47 项首轮 46 passed、1 项已有滚轮测试差 1 像素，单项重复 3/3 通过，未把首轮称为全部通过。

启动集合修复：仅对自动节点选择且允许等待的操作，在 source 检查结束及发布计划前重新读取原节点/注册/hello owners；新增 worker 先入 operation 再读取 inventory，保持已捕获会话不可替换。容量不足仅在原 deadline 内等待新增节点；building 以后不改层段。新增 Android 如需 GGUF，沿原 source owner 补来源且复核 provenance。定向 **95 passed**；相关组合 **1186 passed、1 skipped**（Windows symlink 权限）。

Surface 实际本地退出使旧 intent 立即 `loaded_ready=false`，原因 `pipeline_worker_opted_out`；IPv6 公共状态接口与 native 子进程核对确认已真实卸载。退出后推理 TCP 断开，master 保留 identity-bound release ledger，重入时按原 owner 重放并已收齐 ACK。证据：`surface-dl-withdrawal-native-local.json`。首次验收脚本误按 POST 过滤实际 PUT，及随后的 API 超时均保留为失败，不伪装退出/重入全流程通过。

切本地后 IPv4 API 卡住的 py-spy 证据定位到 `list_gguf_models` 在 async handler 内同步读取大 GGUF 计算摘要，阻塞事件循环；健康监控告警只是断开后的现象。将该目录 handler 改为 FastAPI 同步线程池执行，新增大文件读取被阻塞时状态接口仍响应的真实 ASGI 测试；相关组合 **133 passed**。未修改模式/HA owners 来掩盖此问题。新 Surface runtime bundle SHA-256 `8d080f9a71f98413648ec89fac311213436e2f6f3be86fed5a5a2d8db7ce1e9e` 已部署，master PID 24656；master 重启前调用公共卸载接口。Surface 实测剩余 78,758,821,888 bytes，约 **73.4 GiB**，保留源 GGUF 及本轮工件。

最终 DL/LL 产品入口和退出/重入验收继续，票 12d 尚未关闭。本轮未 commit、未另建分支。
## 2026-10-14 03:15 三机接力与入口记录

最终 DL 操作 `supply_90638ba719bd49309de1ee5a96b8073e`，config `9a2eb68c8c664f1fbfa8c8721c816d35`、精确 generation `1791916805306663300`：master PyTorch CUDA `[0,4)`，Surface llama.cpp `[4,8)`，Y700 llama.cpp `[8,32)`。真实 Web 两题 2／6、默认 Textual 2、Android 实际聊天 2 均通过；各回答的 operation/config/generation 精确一致，required/full、两个远端、distributed=true/fallback=false 及实际 master workflow handoff 均已留证。证据为 `dl-product-verified-ui.json`、`dl-product-final-tui.json`、`dl-product-verified-android-persisted-chat.json` 及各自 handoffs；native 证据为 `native-pytorch-dl-product-verified.json`。原 `dl-product-final-ui` 的显示 guard 超时仍保留，不把失败 runner 当作通过。

DL→LL 前已核对三端卸载、master ledger 清空，证据 `dl-to-ll-native-retired-before-plan.json`。第一次 Android 真实 LL 选模因当时 Surface live available RAM 仅 4,486,152,192 bytes，保守容量规划只有 master `[0,8)` 与 Y700 `[8,32)`，不计三机通过。该轮暴露 inventory 候选被错误投影为 ready，已修为仅投影 admitted plan 的 assignment 参与者；新增 preparing/ready 回归，组合 130 passed。不调整安全预算、不终止用户程序。Surface RAM 后来回升，再次 inventory 为 8,819,511,296 bytes；其变化原因尚未确认。

真实 Android“重试模型准备”提交当前 LL 操作 `supply_7bb4952892f1464f89ae32c70bd60b2d`，config `46bd27f6792341908f4de53cea97745f`、精确 generation `1791918091977630100`，三段为 master `[0,4)`、Surface `[4,8)`、Y700 `[8,32)`。三机 prepare/commit 已就绪，native 证明三端仅装载 proper crop；本段供给工件合计（含 manifest）为 master 1,983,372,464 bytes、Surface 1,926,061,236 bytes、Y700 4,554,567,568 bytes，见 `native-llama_cpp-ll-product-final.json`。实际默认 Textual 和 Android 聊天均返回 2；响应／Room 持久化身份、required/full、两个远端、distributed=true/fallback=false、实际 handoff 已通过，证据 `ll-product-final-tui.json`、`ll-product-final-android-persisted-chat.json`、`android-ll-product-final-chat.xml/png` 与对应 handoffs。

本轮 LL 的首题 Web API/SSE 已返回正确 2，49.084s，判停命中 `<|im_end|>`，两个远端及原始 HTTP 十进制身份均正确；但实际 UI 显示 guard 未通过。保留 `ll-android-product-verified-ui.json/log` 与 failure.png；正在复现历史加载与流式占位符覆盖的竞态。不能将该 runner 或完整 LL Web 入口计为成功。两端退出／重入仍待完成。

最后后端入口／owner／TUI 20 文件组合为 **912 passed**（46.60s），见 `final-product-owner-regression.log`；full SSE 首块及提前关闭补充后相关组合为 **177 passed**。Android **466 JVM passed** 的最后 APK仍有效，本段未再改 Android 产品代码。补池／SSE 的只读复核未发现新增具体 P0/P1。

Surface 磁盘清理证据 `build/intent-surface-cache-cleanup.json`：此前明确隔离的 21 份可再生旧分段共释放 49,135,889,120 bytes（约 45.8 GiB），39.5→85.3 GiB；源 GGUF、清单、有效 partial 保留。03:14 重新盘点，C 盘剩余 74,063,269,888 bytes（69.0 GiB），当前供给目录 12.56 GiB、源模型目录 24.21 GiB、隔离旧库存剩余 0.32 GiB。各目录存在父子包含关系，大小不相加。见 `surface-disk-final-before-modes.json`。当前 master PID32968，Surface runtime SHA-256 `8d080f9a71f98413648ec89fac311213436e2f6f3be86fed5a5a2d8db7ce1e9e`；Surface worker投影由新 master owner 转发取得，未为一致 SHA 重启当前 native。

本票继续验收，未 commit、未另开分支。
### 03:24 实际模式切换边界

Surface 通过实际 Web“退出集群，使用本地推理”→“集群分布式”完成模式切换，两次真实 PUT 均成功。退出后 Surface native shim 进程消失，Android maps 同步无 GGUF；master intent `loaded_ready=false`，重入后 cleanup ledger 为空，四次后续 current 轮询没有复活旧 ready。证据 `surface-ll-mode-final.json/log`、local/rejoined 截图及 `surface-ll-mode-withdrawn-verified-native-cleanup.json`、`surface-ll-mode-rejoined-verified-native-cleanup.json`。master 可保留已失效计划的本段；不能以 `pipeline_prepared=true` 推断全局 loaded-ready 或整模，初次辅助脚本多加的该断言不符合实际契约，失败保留并移除。退出成员及远端旧执行段确已卸载。

Android 实际 Settings“切换”对话框选择“本地推理”，随后选择“分布式协作”，两个本机模式边界均留存 XML/png。本地页自动显示“尚未加载模型”，未点击本地加载，maps 无 GGUF；重入后 maps 仍无 GGUF、ledger 空、旧 intent 连续四次不 ready。证据 `android-mode-select-local-before/after.xml`、`android-mode-local-final.png`、`android-mode-select-distributed-before/after.xml`、`android-mode-rejoined-final.png` 及两份 `android-mode-*-final-native-cleanup.json`。Android 的操作发生在 Surface 退出已令旧段卸载之后，不能声称 Android 本轮独立完成“从已加载本段退出”的完整场景；Surface 退出的全局旧段清理已单独证明。

远端卸载后的 Android 状态卡仍显示旧元数据名称，自己切本地后刷新为 unloaded；正在复核远端 worker 生命周期到设备状态 snapshot 的刷新路径，不能把旧卡片当作真实整模加载证据。当前通过实际 Surface Web 新选 llama.cpp 9B 提交 `supply_afbcba3e75094963836fbe719f39cc94`，不预选节点或层区间，等待三机重新准备，以完成修复后的 Web LL 两题。
### Web 历史／流式响应竞态的复现与修复

延迟初始 history 与超过十秒的真实 SSE test server 明确复现：修复前 `LL partial` 已进入 DOM，迟到 history 用 16 条旧 DL 消息替换 turns，后续 SSE done 仍只能 patch 已消失的 replyId，当前回答／精确代际均不可见。新增统一 `historyBusy = loading || refreshing`，供 send、发送按钮、Enter、表单提交及清空／删除共同使用；保留用户草稿并提示历史完成后发送。会话 key／history cleanup／SSE abort 原保护经测试仍有效，没有扩展新的状态源。

新增五项回归覆盖初始 history、主动刷新、失败后释放发送门、弃用会话的迟到 history、切会话时旧 SSE。修后 18 条消息及 `模型代次：1791918091977630100` 均在 DOM；四文件相关组合 **22 passed**（59.3s），typecheck、JS 语法、diff check 与父任务最终 `npm run build` 通过。前后 JSON／截图保存在前端 `tests/test-results/chat-history-race-before` 与 `chat-history-race-final`，不替代当前三机 Web LL 验收。
### 03:37 Web LL 闭环与 Android 最小显示修复

真实 Surface Web 新选 llama.cpp 后自动完成三段供给／native prepare／commit，操作 `supply_afbcba3e75094963836fbe719f39cc94`、config `60d2659959ab41d591c5a15704264635`、精确 generation `1791919906890923900`。修复后的真实 Web 2／6 均通过，页面中回答及精确代际均存在；两个 remote、required/full、distributed=true/fallback=false 及 master workflow handoff 已核对。证据 `ll-web-final-ui.json/log`、两题截图、`ll-web-final-handoffs.json`、`native-llama_cpp-ll-web-final.json`、`master-ll-web-final.err.log`。原 Web guard 失败保留。

Android 设备卡旧值定位为远端 layer retirement 后未刷新 runtime snapshot。最小修在原 InferenceService 层段加载／释放 owner 完成后发布仅含刷新代次的通知，ViewModel 重新调用原 `getRuntimeStatus()`，拒绝读取期间已失效的旧 snapshot；不按 intent ready 伪造 loaded。卡片同时展示 engine 已有 `loadedLayerRange`，明确“已加载层段 [8,32)”和“已加载整模”。JNI 名称来自 `llama_model_desc()`；Qwen35 的 native 类型判定将裁切为 24 层、embedding width 4096 的 crop 描述为 2B，该描述不是源参数量或整模装载的证据，源与裁切范围仍由 verified manifest 核对。

新增十项 JVM 用例覆盖远端释放、失败后实际 loaded 保持、迟到 snapshot、旧 generation 及标签。定向 **49 passed**、完整 **476 passed，0 skipped**；父任务读取 46 份 XML独立计数相同，见 `android-final-jvm-count.json`。默认 debug APK 与 androidTest APK 构建通过；父任务最初误用已移除的 full flavor task，构建在任务解析阶段失败，失败日志保留，未修改构建配置。最终 APK SHA-256 `330faf807cdab9436711ca55293d45def184deb67588174bdb055d9c35b29608` 已 install-r，保留 data／SAF。安装前公共卸载 owner 收齐 cleanup ACK、master runtime 和 Android maps 无旧段；该过程留于 master 日志。辅助脚本的固定输出随后被最终清理捕获覆盖，不将其当作独立的安装前快照，最终卸载证据另存 `final-cluster-native-cleanup.json`。

Android 重启后实际聊天页自动恢复 `INTENT_PLAN_RETIRED`，从实际“重试模型准备”按钮提交 `supply_056cee99ea8846a1a69b0f117f8fda8c`。新 APK 状态卡已在未装载时显示“尚未加载模型”；当前等待自动供给以完成其层段装载和远端卸载刷新实测，未关闭票面。

TCP 证据 `star-topology-final.json`：master OS 仅两条 API worker control TCP8888 Established，来源为 Surface 与 Y700；Surface OS 仅连 master control TCP8888，无 worker→worker 连接。Android `/proc/<pid>/net/tcp*` 被系统拒绝读取，保留原错误，不把该本机 socket table 称为读取成功；Android 边由 master 接受的真实 Established 连接及各题 handoff 证明。
### 03:52 最后 APK 参与的三机证明与 TUI 预算修复

最终 APK 参与 `supply_056cee99ea8846a1a69b0f117f8fda8c`：config `ea4826d881df46e681d1e6b08e467f81`、精确 generation `1791920668886047000`。真实 Web 两题 2／6、DOM 精确代际与远端 handoff 均通过，见 `ll-final-apk-ui.json/log`、`ll-final-apk-handoffs.json`。native proper crop 仍为 `[0,4)`、`[4,8)`、`[8,32)`，见 `native-llama_cpp-final-apk.json`。Android Settings 持续停留设备状态页，未点刷新，在 native 加载后自动显示“已加载层段 [8,32)：qwen35 2B Q4_K - Medium”，截图/XML为 `android-final-status-native-loaded.png` 和 `android-final-status-native-loading-before.xml`。

这轮追加默认 Textual 聊天因历史增长到 594 个 prompt token，LL full 前向超过其 60 秒 socket read timeout，客户端取消真实请求；master 与远端 cancel ACK 已确认停止，失败保留于 `ll-final-apk-tui.json/log`，不能计通过。产品修复 full 默认读预算为有限 120 秒，与 Android read budget 一致；interactive 仍 60 秒，显式 caller override 保留。预算及 TUI 相关五文件 **142 passed**，见 `tui-final-read-budget-regression.log`，正在真实重测。此项修复属于客户端等待 full 结果的体验边界，不降低 worker deadline／来源／native 准备门。

Surface 最后在 manifest、当前 receipt、无 reparse 与 native 引用核对后，仅清理八份先前操作的旧本段 `segment.gguf`，保留最终当前段、源模型、清单与 partial。新增释放 15,408,474,368 bytes（14.35 GiB），C 盘剩余 84,416,057,344 bytes（78.62 GiB），见 `surface-cache-final-cleanup.json`。此前释放 45.76 GiB，加上本轮新增累计约 60.1 GiB；剩余空间因后续生成有效工件变化，不应以累计释放直接推算。
## 2026-10-14 03:56 最终验收结论

**票 12d MODEL-INTENT-UX-01 完成。** 三机 qwen3-5-9b 的 DL 与 LL 真实产品入口、自动规划／供给／prepare／commit、正确回答、两个实际远端 handoff、精确 operation/config/generation 及无节点整模装载已闭环。星状拓扑、master 唯一协调和连续多段接力保持；没有人工给各设备选择层段。本轮不 commit、不另开分支。

| 最终证据 | 操作／结果 | 文件（位于 build/model-intent-20261013） |
| --- | --- | --- |
| DL，三入口 | supply_90638ba719bd49309de1ee5a96b8073e；Web 2／6、Textual 2、Android 2 | dl-product-verified-ui.json、dl-product-final-tui.json、dl-product-verified-android-persisted-chat.json 与各 handoffs |
| DL，native | master PyTorch CUDA [0,4)、Surface LL [4,8)、Y700 LL [8,32) | native-pytorch-dl-product-verified.json |
| 最终 APK LL，三入口 | supply_056cee99ea8846a1a69b0f117f8fda8c；Web 2／6、默认 Textual 2、Android 2 | ll-final-apk-ui.json、ll-final-apk-tui-fixed-tui.json、ll-final-apk-android-persisted-chat.json 与各 handoffs |
| 最终 APK LL，native | master LL [0,4)、Surface LL [4,8)、Y700 LL [8,32) | native-llama_cpp-final-apk.json、master-ll-final-all-entries.err.log |
| Android 实际回答 | 最终 APK 的真实输入／发送、Room 持久化、精确 long 代际 | android-final-apk-chat-result.xml/png、ll-final-apk-android-persisted-chat.json |
| 远端卸载自动刷新 | Android 不刷新／不切页，Surface退出后 loaded layer→unloaded | android-final-before-remote-retire-before.xml、android-final-after-remote-retire-before.xml/png |
| 退出／重入 | 真实 Surface local/distributed、Android local/distributed；maps清空、ACK空、旧ready不复活 | surface-final-apk-remote-retire.json、surface-final-apk-withdrawn-native-cleanup.json、surface-final-apk-rejoined-native-cleanup.json、android-mode-*-final-native-cleanup.json |
| 星状连接 | master接收两worker，Surface仅连接master | star-topology-final.json |
| 空间清理 | 旧隔离21段45.76GiB；新增旧8段14.35GiB；清理后78.62GiB／最后盘点82.15GiB | ../intent-surface-cache-cleanup.json、surface-cache-final-cleanup.json |

最终 LL 三入口使用相同 config `ea4826d881df46e681d1e6b08e467f81` 和 generation `1791920668886047000`，每题响应均 distributed=true、fallback=false、workers为 Surface/Y700。修复后的默认 Textual 在相同 594-token 长历史下实际耗时 **87.602 秒**，正确返回2并命中 `<|im_end|>`；因此120秒full读取预算的实测覆盖了之前60秒失败点，未靠缩短prompt或扩大验收脚本预算过关。

最终自动化按代码范围分别记录：后端 owner／入口／TUI 组合912 passed；最后TUI预算补丁相关142 passed；Android完整JVM476 passed（0 skipped）、debug及androidTest APK构建通过；Web history／identity／路由四文件22 passed，另identity/cancel专项7 passed，typecheck/build/diff检查通过。各组合可能有重叠，不相加为独立用例总数；物理设备opt-in及Windows symlink skip不计passed。最终APK SHA-256 `330faf807cdab9436711ca55293d45def184deb67588174bdb055d9c35b29608`。

验收范围及后续项：本轮物理机器已有可工作的软件／依赖／bootstrap信任，已经以空供给目录及真实网络新增下载、跨操作断点恢复证明自动供给，不等同于全新Windows发行物安装；干净Windows及Android发行验证沿用户裁决留待票13。缓存复用后仍需分钟级裁切／摘要校验，LL短答首前向为几十秒且随历史增加；本票证明功能／生命周期，不给出单机吞吐或容量倍数收益结论。显式local切换和卸载门实测通过，但本轮没有加载整模采集单机基线。native对裁切模型的“2B”类型描述仍保留，UI现明确层段范围；如需源模型友好名称可后续优化，不能因此改变真实工件身份。

最后Surface已重入distributed，Android保持distributed；旧操作因退出验收而失效，旧ready不复活。服务器与worker继续在线；结束前沿公共unload owner释放剩余master本段，避免验收占用内存。用户后续可从产品“重试模型准备”开始新操作。
### 最后状态与证据索引

2026-10-14 03:59：沿公共unload owner释放剩余master本段，ledger空，master loaded=false／pipeline_prepared=false，Android maps无GGUF，Surface无native shim且仍为distributed。证据 `final-cluster-native-cleanup.json`、`surface-final-state.json`。Surface最后C盘剩余 **88,204,156,928 bytes（82.15 GiB）**（准确字节以现场JSON为准）；与03:49清理后78.62GiB是两个不同时间点，不能据此额外声称又清理了文件。Surface runtime仅同步最新TUI transport文件并核对SHA，以便本机启动TUI也使用120秒full读取预算，不为此重启API。总体通过索引 `final-acceptance-index.json` 已读取并断言六份DL／LL三入口记录为passed、两个worker与身份一致，以及最终cleanup及476JVM结果；原失败证据保留。