# 票 16–18 修复与验收记录（2026-10-14）

本轮保持星状拓扑、master 唯一协调与 DL／LL 连续多段接力。分布式各节点只加载本段；显式退出集群的 local 模式沿原卸载门保留。当前工作在既有分支，未新建重构分支、未提交。

状态：以下三机验收是 2026-10-15 早先的历史结果，完整 unit 为 6340 passed、16 skipped。此后 Surface 下线、Y700 没电，旧三机意图已退休，不能作为当前 ready。2026-10-15 本轮已修复 LL master 引擎展示及换拓扑时的离线债务／恢复身份退休衔接，并以本机＋ROG 自动两段 LL 完成两条 API 问答和前后裁段映射核验；详见 [LL 布局与两节点验收](pipeline-layout-rog-acceptance-2026-10-15.md)。Windows 重启自动拉起、单机整模基线按用户要求后置；9B 首次在供给准入被拒绝，RAM 自然回升后的重试完成供给，但 prepare 仍使用旧 3.0 GiB 画像预算而被拒绝；两次失败均保留。

以下按执行顺序保留中间失败、修复及曾通过后失效的记录；末尾“最终验收收口”记录当时三机状态，当前结论以上述两节点补充记录为准。历史failed／中断operation与误分类采集文件均保留，不改写为当前ready。

## 现场与注册表问题

Surface 重启后没有 worker；2026-10-14 17:06 沿已核对的 `QLHModelSupply20261012` 计划任务恢复，重新注册到 master。C 盘可用 88,512,106,496 bytes，约 82.4 GiB。Y700 的 ADB serial 为 HA26PYZQ，保持原 app data／SAF，不切 MTP、不经 ADB 传模型。

本轮实际主节点使用用户真实 SQLite，位于 `%LOCALAPPDATA%/QLH-Edge-Inference/state/qlh-control.sqlite3`，不是此前 `H:/qlh_acceptance/model-supply-master-state` 的验收库。初始 `master_identity_v1.bound_at=2026-10-14T08:26:17.865689Z`，记录为本机五个真实 MAC；此前假 MAC 已由上一会话纠正，本轮没有自动清身份或放宽身份验证。

真实当前换模操作 `supply_236a4fcd87a24fef9ebd3787b28e3b6a` 选择 qwen3-0.6b/PyTorch 后失败，码为 `SUPPLY_SOURCE_METADATA_INVALID`。精确根因是 v1 双格式注册把 HF 目录外的明确 GGUF 写成 `external/qwen3-0.6b-q8_0.gguf`，供给校验却将它解释为 HF 目录内文件。修复使用登记的显式路径绑定，保留原 manifest 身份，并重算现场文件 SHA；不猜邻近文件、不删旧清单。

另有两项一致性问题：模型目录／普通加载优先用内置配置，而供给独自应用 SQLite 的路径／格式覆盖；以及磁盘格式可用被当成层接力运行时可用。当前共享运行时集合不含 qwen3，不能把 Qwen3 映射成 Qwen2；旧 q35-artifacts 指向 [0,4) 裁段，也不能作为完整供给源。分布式目录及选择应在退休旧 native 之前给出具名限制，本地资产列表仍可以表示文件存在。

H:/qlh_models 根目录另有六份对应 GGUF 已缺失的旧 qwen25 cut/head/mid manifest 侧车。尚无证据表明它们是本次 qwen3-0.6b 失败成因，已记录但没有批量删除，也没有遍历或删除模型目录内 junction。

现场证据：`build/tickets16-18-20261014/real-identity-before.json`、`real-registry-before.json`、`intent-before.json`、`nodes-before.json`、`registry-live-audit.json`。这些包含本机私有路径，未加入公共产品响应。

## 已有验证

注册表解析与供给校验定向：151 passed，含 legacy 双格式 external 显式绑定、缺失／同尺寸篡改／未登记拒绝、mixed 转换到独立目标保留原 GGUF 和原清单、第二次调用复用证明。该结果是同进程重复调用测试，不能冒充跨进程重启或三机二次零下载。

票16子 agent 的监听／隔离组合：153 passed；身份定向12 passed。会话 SQLite fallback 沿 `fb1fdab1` 保留，新增 teardown 后迟到后台线程验证；监听 `--backend-host` 与客户端 `--host` 分离，来源 CIDR 边界保持。

上述补丁在共享树回滚事故后重建，最终重建后的回归结果须另记，不以先前结果直接关闭票。

## 执行工具事故

完整 unit 通道开始于 17:19 左右，使用4 workers／loadscope；其间共享根工作树的未提交修改于17:32:33被批量回滚，HEAD仍为7d21c651，Android嵌套仓库修改保留。该次完整测试已停止，状态为中断／无效，不能计为通过。

已确认是 Reasonix 工具的并发隔离与回滚设计缺陷：一次写模式调用中，桥接层返回 `write touched paths outside allowedPaths: android, qlh.py; rollback completed`，随后对共享树 changed paths执行 git restore。调用启动时根仓干净，执行期间其他并发修改没有在原 snapshot中，回滚因此误伤。父没有提交、reset或restore用户代码。该写通道已经停用；后续只使用原生编辑子 agent，分配不重叠写集，逐块备份源码补丁。用户已关闭 Reasonix 写通道；本轮后续恢复的是 Codex 原生子 agent。只读通道未观察到此类写入。

只读复核另发现两个需要收口的身份边界：已登记 manifest 被删或被重新自签后不能降级为未登记；旧转换任务不能无条件覆盖期间的新登记。正在补登记摘要绑定及 SQLite 原子条件更新回归。

## 待闭环

- 票16：重建后回归；完整单测前后真实 identity不变；无临时 /32 的发现、bootstrap、注册实测。
- 票17：Android真实连接与进程状态分离、PC持续失败后的重试 owner；断连接、心跳超时、控制面重启实测及真实状态。
- 票18：Python与Android跨operation／restart内容复用、新operation manifest身份绑定；local_cut默认优先与具名fallback；真实native／待prepare引用保护下有界GC；三机二次新增下载0与磁盘量级。
- 注册表附加修复：不可降级的已登记身份、转换提交CAS、分布式选择前具名拒绝不支持架构／裁段。
- 三机：最终源码／APK部署后，用Qwen3.5-9B分别跑DL与LL，记录三个proper crop、实际handoff、回答与精确operation/config/generation。不加载整模，不靠手选层段。
## 本次续跑的新增现场证据

2026-10-14 18:08 续跑时本机 master 进程已退出，Surface 两个 Python 进程与 Y700 app 仍在。真实身份逐字未变，native 生命周期为 rejected 且 cleanup ledger 为空；未清身份，随后用真实 state 目录重启 master，`QLH_MODEL_API_TRUSTED_CIDRS` 显式为空。两台真实 worker 均自动重新注册；Android 在 TCP 尚未注册期间的 intent 请求得到403，登记 ACK 后恢复200，未放宽来源边界。

18:23 在 Surface 的独立临时 state／node_config／SQLite 中启动全新客户端，沿真实 Tailnet discovery → `/api/bootstrap/info` 主身份确认 → 加密 first-connect → TCP HMAC REGISTER，全链通过。没有复制真实 worker 凭据或改动其状态。原容量为3，首次探针作为第四节点按既有容量规则被拒绝；临时增加一个名额后通过，finally 已恢复容量3。探针自行断线；高风险删除 API 要求实名登录并返回401，未绕过该边界，临时内存节点将在部署重启后核实消失。证据为 `build/tickets16-18-20261014/fresh-client-bootstrap.json` 与 `fresh-client-probe-actual.log`。

Y700 部署前实际 app PID 为20137，读取该 PID 的 maps 确认无 GGUF 映射；当前 supply 目录4,452,215 KiB。另有历史验收备份约60 GiB，未将这些备份算成本轮缓存新增，也未删除。证据为 `physical-predeploy.json`。

source identity 与 SQLite CAS 最终定向109 passed；中断的旧219项组合没有退出状态，不能计通过。
## 19:30 续跑发现与修复

最终部署后的第一轮完整 unit（4 workers、loadscope、seed 1973875962、maxfail=0）为 **25 failed、6184 passed、16 skipped**，耗时 702.30 s。失败集中在 master 节点管理与 HTTP 队列的角色前提。该次真实 `master_identity_v1` 逐字未变，源码 hash 未变；失败证据保留在 `build/tickets16-18-20261014/full-unit/` 与 `full-unit-check.json`。不能把这次结果记为通过，也不能据此将全部失败归咎于 xdist 本身。

隔离检查发现 runner 给四个 worker 共用一份 node_config/state，而旧 fixture 仅按用例隔离 SQLite；`node_config.apply_runtime_config` 还会同步 scheduler 等已导入模块的角色别名，这些别名不属于原 config 快照。原生子 agent 在测试层隔离每个 worker、每个用例的状态目录、node_config、凭据及 journal 的落点，并复原已加载 facade 别名；依赖 master 的管理测试明确声明自己的角色前提，生产角色检查保持原义。首个定向组合 **216 passed、1 deselected**，最终随机并行与全量结果另记。

现场断线验收已有：全新 Surface 探针完成验证发现、加密首连和 HMAC REGISTER；主进程暂停 55.219 s 触发 Android `heartbeat_timeout` 与真实退避状态，恢复后两台 worker 重新注册；Surface 的最终 TCPClient 物理探针在主动关闭 socket 后自动重连（这是独立探针，不冒充真实 worker 的 socket）。主节点重启后两台真实 worker 也重新 REGISTER。但该轮重启检查只验证了节点注册，未确认 hello，故不能宣称 worker 模型供给可用。

19:10 从真实 Android 界面选择 Qwen3.5-9B / PyTorch、自动策略、不指定节点或层段，operation `supply_dbac01f9fd054f86a72df7ca724517df` 失败。主日志证据为 `operation.json.tmp → operation.json` 的 Windows WinError 5，发生在 waiting_nodes 更新；同时 Surface 的新源码日志显示，首次 REGISTER 后于 18:45:49 发 hello、收 accepted ACK，而 18:46:17 底层自动重连只完成 REGISTER，此后没有重新发 hello。因此三个节点在线的表象不能替代两台 worker 的当前注册握手。

本轮新增修复使 TCP 自动重连通知上层刷新 hello，并在耗时能力检查结束时校验当前 client 和 registration epoch；旧代际的检查结果不进入新连接。定向 **24 passed**，相邻 TCP／worker adapter／本地模式组合 **200 passed**。实机等待器改为同时核对当前 REGISTER、accepted hello、hello 时间晚于注册时间及健康心跳。Windows 状态持久化另外使用唯一临时文件、有界 sharing-error 重试和原 owner 锁，真实文件锁与永久失败回归正在完成；尚无证据锁定实机上具体哪个 reader/scanner 持锁。

Python 供给最终相邻回归 **388 passed、1 skipped**；Android 初轮完整 JVM **509 passed、1 skipped**（共510），最终引擎可用性 UX 相关 **97 passed**。最后 APK 已 install-r，保留 app data／SAF。真实界面验证 qwen3 的不支持引擎选择按钮禁用，并显示具名限制；Qwen3.5-9B 同时支持 PyTorch／llama.cpp，默认引擎保持原偏好。上述计数不能代替后续修复后的全量与三机验收。
## 20:05 续跑结果

最终 Python 源码包共318文件，SHA256 `b32d8c7c8184ee7b310db304e93707e46e820c10dc88ce5c40aee80669215811`，Surface 逐文件 hash 验证完成。19:38 主节点真实重启后，Surface 与 Y700 均自动完成新的 REGISTER、accepted hello ACK 和健康心跳；hello 时间均晚于该节点本次注册时间。证据 `final-deploy-handshake.json`。修复前 Surface 只有 REGISTER 的反例保留在 `surface-before-final-deploy.err.log`。

Windows 持久化最终组合 **200 passed**，包含新增26项与7项真实共享锁／只读权限测试；仅 Windows WinError 5/32/33 最多8次替换尝试、累计退避1.27 s。永久失败仍抛，取消与新选择的次序在同一 owner 锁内保持。证据与补丁为 `state-json-atomic-final-*`。

测试角色隔离的最终定向 **273 passed**、随机四 worker **292 passed**。原失败 seed 的完整重跑为 **1 failed、6252 passed、16 skipped**，1044.20 s；真实 identity 未变，changed_python_paths为空。剩余失败 `test_state_dir_override_is_user_configurable` 把隔离fixture设定的 journal 文件名误当作重新导入初始化结果，现改为独立解释器、临时state/config/credentials/SQLite验证真实生成路径；增量定向23项、原seed四worker组合89项通过。下一完整运行在 `full-unit-closed/`，不能提前记通过。

19:39 Android界面选择的 operation `supply_b836ede079ec4451a50feeddaa956c5a` 因 `SUPPLY_CONVERSION_PROOF_INVALID` 具名拒绝。发现 qwen3-5-9b 的真实SQLite登记仍绑定旧 `model.safetensors.index.json`：71,138 bytes／SHA `b66c8ee6…`，磁盘当前文件为71,140 bytes／SHA `e4eb2704…`；磁盘manifest与转换证明相互一致、四份权重分片摘要相同。没有把自动选择降级为信任新manifest，而是进行本轮显式数据修复：完整验证HF全部声明文件、GGUF目标与cross-format证明，确保验证前后文件stat未变，以完整旧登记行做CAS同步到已验证manifest。保留旧行、逐文件差异、新行与验证记录 `q35-registry-reconciliation.json`；模型、manifest与provenance文件都没有改动。

随后由Android实际界面重新选择Qwen3.5-9B / PyTorch、auto、不指定node_ids，operation `supply_1412536ba0104eea95445f1546abb000` 自动规划 `[0,4) → [4,20) → [20,32)`，三台 proper segment，single_node_full_model_candidates为空。Surface执行默认local_cut并完成available，actual_downloaded_bytes=0、具名 `SUPPLY_LOCAL_CUT_PREFERRED`；Y700无完整源，具名 `SUPPLY_LOCAL_SOURCE_UNAVAILABLE` 后下载tail。20:05仍在传输，native及问答尚未验收。

Android原有分阶段性能日志显示时间主要消耗在网络read。独立的8MiB合成数据探针通过网络发送并在Android `/dev/null` 丢弃，Tailnet约624,852 B/s、LAN约1,402,503 B/s；不是ADB模型传输。证据 `android-network-throughput.json`。当前原生接收保留断点，未为验收清数据、切MTP或改SAF。缓存前置盘点 `cache-before-final.json`：master供给根约5 KiB（尚无段），Surface约1.93 GB，Android4,452,215 KiB。后续磁盘增量需与这些基线比较，不把历史验收备份算成新缓存。
## 重连握手只读复核追加项

20:13原生只读复核用事件控制复现了两个P1窗口：能力检查之后、hello创建之前发生同TCPClient注册换代，旧结果可能经默认socket发送到新连接；以及旧socket的发送失败在新hello已ACK之后才执行，无条件disconnect清掉新coordinator状态。初步的无锁epoch再检查只覆盖能力检查期间换代，不足以关闭这两个窗口。已按registration与hello owner补短同步区、捕获socket发送、异常按精确owner条件清理，并对能力检查失败尚未创建hello的情况保留他人状态。

因此 `full-unit-closed` 在约一半处主动停止，仅记录为中断，不计最终验收；`full-unit-closed-interruption.json`保存了仅终止测试runner及其子进程的核对。真实master和供给继续运行。握手owner补丁、确定性交错回归及最终相邻/完整结果另记，不能沿用前一个200项结果声称新同步实现已验收。

20:22读取Android实际 `.supply-resume.json`，持久前缀1,342,177,280 bytes的SHA与master对应源段实时前缀SHA相同：`afa2462a64a4f8436065e556190a21c68ba0fc2fda1d06509cde95b3c8c6172b`。证据 `android-dl-live-resume-checkpoint.json` 和 `android-dl-checkpoint-prefix-verified.json`。这证明当前已落盘断点与本次工件一致，尚不能替代重启后重新校验并续传的验收。
## 20:55 续跑与最终握手部署

原生子 agent 的 hello owner 修复最终相邻组合为 **206 passed**，独立 TCP seam 为 **16 passed**。最终源码包318文件，SHA256 `63f7af7350c77fea9066427ff22ca9229c63a14204dd2c0b0d38351d8ef42d1e`；20:42 在 Surface 逐文件校验部署，并重启真实 master。两台真实 worker 自动完成新 REGISTER、accepted hello ACK 和健康心跳，证据 `hello-owner-deploy-handshake.json`。当前已恢复的原生子 agent 只读复核 GC 与 native 引用保护。

为部署上述握手修复，旧 DL operation `supply_1412536ba0104eea95445f1546abb000` 于20:40沿真实 intent 取消接口退出；保留了 Android 的断点与原 owner，未清数据或 SAF。该次主动维护取消不计 native 验收通过。证据 `dl-cancel-for-hello-owner-deploy.json`。

随后再次由 Android 实际界面选择 Qwen3.5-9B / PyTorch / auto，operation `supply_94a54df2248a4f4db72a1769e7fe4334` 仍自动规划 `[0,4) → [4,20) → [20,32)`。PC 头段复用3,787,061,057 bytes、Surface 中段复用3,465,966,743 bytes，两者 actual_downloaded_bytes均为0。Android 在新 manifest 下实时复核并接续旧 operation 的2,272,264,192 bytes，resumed_bytes如实上报；20:55累计已到约2.55 GB。旧取消 receive 和新 receive 的 inode不同、链接数均为1，证明断点复制后旧 owner仍独立保留；这些未完成目录不冒充完成缓存，后续磁盘量级需把保留断点单列。

新的完整 unit 运行 `full-unit-owner-final` 尚未结束；日志已出现失败，不能计通过。源码在这轮运行期间冻结，结果落盘后再按失败用例处理。三机 native、精确回答、LL和最终GC仍待收口。
## 21:38 DL通过、LL换模暴露的重连入口冲突

DL operation `supply_94a54df2248a4f4db72a1769e7fe4334` 于21:10完成三机 native prepare/commit及两次精确问答。Android最终 `actual_downloaded_bytes=742397779`、`resumed_bytes=2272264192`，总计3014661971。三端加载范围分别为 `[0,4)` / `[4,20)` / `[20,32)`；本机 safetensors index 只含0–3层，Surface真实shim命令和Android实际PID maps均指向本段 `segment.gguf`。两请求分别返回 `2`、`6`，distributed_used=true、fallback=false、workers_used含两台远端，operation/config/generation逐项一致。证据 `dl-owner-final.json`，不以单纯REGISTER或磁盘文件可用冒充native通过。

随后通过同一Android界面选LL，operation `supply_8b576270c52b4d9d88c5623fe915b0da`。Surface和Y700均完成新manifest绑定，`actual_downloaded_bytes=0`，内容分别复用3465966743与3014661971 bytes；master新建本地GGUF头段。21:16 Surface出现heartbeat ACK超时，底层TCPClient自动重连与scheduler周期重连同时各起一连接，真实日志显示重复REGISTER相互替换。本次prepare以INTENT_PREPARE_FAILED退出，未跑问答，不能计LL通过。证据 `ll-owner-final.json`、`ll-owner-surface-reconnect.log`；仍需查清初始ACK超时，不能把后续恢复注册当成本次模型ready。

为消除重试owner竞争，scheduler自动入口现复用同目标TCPClient的完整重试意图，包括EOF交接、退避和retry_exhausted；显式重试或新endpoint在新建前停止旧intent。对应41项定向通过，与partial/intent合并135项通过。意外prepare异常新增master端带operation的异常日志，产品响应不泄露本地路径。

GC只读复核确认：旧取消`.receive-*`的checkpoint被复制到新operation后原目录没有回收入口，反复取消会累积独立磁盘副本。因此新增Python退休checkpoint清理：仅当receiver/donor执行互斥可取得、原manifest/journal与实时字节一致、同node/source/tokenizer/range/role/proof的完整副本覆盖全部成员时，有限扫描/哈希/删除；活动owner、pending选择、未知native引用、未完成、变更、链接或未知文件均保留。新增25项正反回归通过；相邻供给356 passed、1 skipped（末尾owner失效闸补充后另有135项合并通过）。Android对应实现与物理回收仍在进行。旧partial的保留不冒称已完成GC。

上一完整unit最终 **4 failed、6254 passed、23 skipped，823.08s**；真实身份逐字未变、changed_python_paths为空。原生子agent修复两文件：真实socket/subprocess重试测试改为握手控制，排除0.3s进程启动时序假设；llama多序列构造使用隔离假绑定，真实GGUF数值测试标记real_model并显式opt-in，保留数值断言及绑定失败。定向93 passed、1 skipped、8 deselected；四worker原seed125 passed、1 skipped；直接无opt-in22 passed、8 skipped。最终全量须在生产补丁冻结后重跑，不从这些定向结果推定全量通过。

## 2026-10-15 首次完整通过记录（之后断线失效）

本轮沿用原工作树与分支，没有提交、reset、restore、清Android数据或SAF；ADB只用于界面操作和证据读取，未传模型权重。网络恢复后继续验收，不复用失败的模型意图。

### Surface恢复与旧启动配置

恢复后曾创建 `supply_362fd503323c46c681ef386c9f60a43a`，失败为 `SUPPLY_WORKER_TIMEOUT`。根因是旧计划任务 `qlh_worker` 启动基础仓库 `C:/Users/surface/Documents/LEDS_BJTU`，该部署缺少 `src/model_supply_worker.py`；它能REGISTER/HELLO，却不能返回model-supply inventory结果。重新启动已有正确任务 `QLHModelSupply20261012` 后，实际后端来自 `build/model-supply-20261012-runtime`，广告的本轮段为 `[4,20)`，16个层字节数全部为正，HELLO accepted且健康。

旧 `qlh_worker` 已导出任务XML备份后禁用，正确任务保持Running，没有为本次验收再次重启已ready的三机。证据为 `surface-task-audit-20261015.json`、`surface-legacy-worker-disabled-20261015.json`。早先 `handshake-resumed-20261015.json` 广告的是旧 `[4,24)`，不能当作本轮正确握手证据。

现有正确任务仍无登录/开机触发器、无失败重启策略，执行时限为12小时。进程内断线重连已经验收，但这不等于设备重启后自动拉起；启动器和部署事实源的统一仍需后续治理，不记作已解决。

### DL与LL真实接力

DL沿用2026-10-14 21:10成功证据 `dl-owner-final.json`：

- operation：`supply_94a54df2248a4f4db72a1769e7fe4334`
- config：`53b8819809ca4c5691a637dc6d8c6695`
- generation：`1791983385972546700`
- master `[0,4)` 为PyTorch裁段；Surface `[4,20)` 与Y700 `[20,32)` 为llama.cpp段。
- 两次回答为 `2`、`6`；均distributed_used=true、fallback=false，远端层段及operation/config/generation一致。

LL由Android实际界面选择 `qwen3-5-9b / llama.cpp / auto` 产生新意图，未手选设备层区间；系统自动供应、prepare、commit并达到三节点ready：

- operation：`supply_b4327735ba434691b5f6a10ff7c4a9ce`
- config：`b2f5286f4c7843599b300316ba657b4e`
- generation：`1791995884221568200`
- master `[0,4)`、Surface `[4,20)`、Y700 `[20,32)`，每个节点都是proper segment。
- prepare/commit原生回执同时绑定operation、config、generation、assignment、artifact SHA和精确层范围；ready/prepared节点集合均为两台实际worker，cleanup ledger为空。
- 独立API两问均为 `2`、`6`；随后Android真实聊天界面再次两问，也为 `2`、`6`。从SQLite回读的两条Android回执均为android_http/android-d83bf4d0/distributed，distributed_used=true、fallback=false、layer_segments=`[[4,20],[20,32]]`、claimed_layers=`[4,32]`，且完整模型意图身份一致；停止原因为text_stop，匹配ChatML结束序列。

LL正式组合证据为 `build/tickets16-18-20261014/ll-owner-closed-20261015.json`。其中包括供应事件、UI聊天前后的三机快照、API与Android持久化回执、磁盘统计和证据SHA。原 `ll-owner-runtime-20261015.json` 保留failed：其失败在旧采集器只读 `logs/api-model-supply.err.log` 的断言，实际本轮日志为 `logs/api-model-supply-timeout180.err.log`；不能把采集失败文件直接改为通过。

### 分段加载的运行时证据

`ll-runtime-20261015-before-ui.json` 与 `ll-runtime-20261015-after-ui.json` 均通过严格检查。master API为loaded=false、segment_materialized=true，descriptor只有0–3层字节非零；master进程及全部子进程中所有GGUF映射均与当前 `[0,4)` 工件为同一物理文件。Surface后端和shim的模型映射均与当前 `[4,20)` 工件为同一物理文件，普通整模状态loaded=false。Android实际包进程的 `/proc/<pid>/maps` 只出现当前manifest绑定的 `[20,32)` GGUF。

Windows映射路径会显示同一文件的旧硬链接名称。master旧路径与本次路径的文件ID均为 `5910974512296655`，Surface相应文件ID为 `13792273859237368`；两端os.path.samefile=true。验收以文件身份和当前manifest/代际绑定为准，未删除仍被native映射的旧别名。首轮Android空映射结果来自ADB shell传参，直接run-as cat `/proc/<pid>/maps` 后得到真实尾段映射，原始maps另存。

这些证据支持本次LL供应后的准备屏障及真实聊天期间各节点仅使用本段；不会仅凭loaded=false、REGISTER或一个父进程快照推定全部native状态。

### 去重与磁盘收敛

本次LL三节点actual_downloaded_bytes均为0。master复用1,983,372,464 bytes、Surface复用3,465,966,743 bytes、Android复用3,014,661,971 bytes，并为新operation重新绑定manifest；Android进度中的downloaded_bytes是完成逻辑字节，不能当作本轮网络新增下载。

`cache-ll-runtime-20261015-final.json`：

| 节点 | 最终实物占用 | 核验结论 |
|---|---:|---|
| master供给根 | 8,464,049,526 bytes（按文件ID去重） | 当前三个GGUF段及少量元数据；头段旧/新operation别名共用一份权重 |
| Surface runtime供给根 | 5,392,066,634 bytes（按文件ID去重） | 原基线约1.926GB + 一份本轮约3.466GB中段；多operation目录不增加权重副本 |
| Android供给根 | 2,946,919 KiB（约2.81GiB） | 一份本轮尾段、current指针；已无旧`.receive-*`或重复segment目录 |

Android在DL ready时仍保留的退休checkpoint已由产品GC收敛，未通过ADB手删来制造结果。历史验收备份及供给根外完整源不计入本轮缓存增量。仍被native引用的别名、未知或未完成工件按保守规则保留；没有宣称所有历史目录均已清空。

### 最终测试与身份保护

2026-10-15第一轮冻结源码的完整unit为 **2 failed、6323 passed、16 skipped，330.01s**，真实master身份不变、changed_python_paths为空。剩余失败均在 `tests/test_llama_ngram_spec.py`：该单元契约直接依赖真实llama-cpp-python导入，遇到同worker此前的模块模拟后出现partially initialized/circular import，并使注入检查降级。

仅修这份测试：每个用例使用独立、自动恢复的llama_cpp/llama_speculative模拟；保留配置参数注入、调用方draft不被覆盖、缺类降级断言，定向 **5 passed**。未改n-gram生产逻辑，也未把测试失败转为skip。

最终按原seed `1973875962`、4 workers、loadscope完整重跑，结果 **6325 passed、16 skipped，313.36s**，退出码0；真实 `master_identity_v1` 逐字不变、changed_python_paths为空。证据为 `full-unit-closed-20261015-check.json`、`full-unit-closed-20261015-console.log` 及同名目录中的JUnit XML。此前4失败、中断或其他旧运行不作为最终通过证据。

### 仍需区分的容量与展示口径

DL供给计划的single_node_full_model_candidates为空，支持该档位下的组合容量准入价值。LL Q4_K_M供给计划仍列 `master` 为可能整模候选；runtime capacity为空候选是基于已经裁成 `[0,4)` 的服务descriptor，不能反推源整模在单机上绝对装不下。因此本次LL证明自动分段接力及没有整模推理加载，尚未证明LL在这个量化档位的聚合容量不可替代性，也没有采集手动退出集群后的单机整模基线。

LL runtime pipeline_layout目前把master标成pytorch，与本轮GGUF头段及llama.cpp加载/shim证据不符。该展示字段不作为引擎执行判据；其事实源一致性需后续修正，未以此把本次LL混报为DL。

## 2026-10-15 网络恢复后追加缺陷

用户确认断线来自本机网络环境变更、自行登出了局域网。网络恢复后Surface重新REGISTER/HELLO正常，Android却在HELLO_ACK前收到旧config的MODEL_PREPARE release，报unexpected_message_type，最终retry_exhausted。旧operation `supply_b4327735ba434691b5f6a10ff7c4a9ce` 因pipeline_worker_disconnected进入failed，不再作为当前ready证据。master的cleanup ledger仍保留Android旧config的release。缺陷在REGISTER后清理重放先于HELLO屏障，且公共MODEL_PREPARE发送入口没有握手约束；票17重新打开，修复与重新验收进行中。证据：`runtime-after-final-unit-20261015.json`、`post-acceptance-disconnect-master-20261015.log`、对应Surface/Android日志。

## 2026-10-15 HELLO与清理边界补充修复（已完成复验）

本机网络退出局域网由用户确认。恢复时的Android循环断开另有生产缺陷：REGISTER后旧模型release先于HELLO_ACK发出。master现将首次prepare、commit、重试与终局清理统一置于当前连接的HELLO屏障后；ACK发送失败或拒绝不能放行native命令。清理债务保留到匹配native ACK，不以“尝试发送”清账，旧operation不复活。

进一步只读复核补齐了跨连接代际的三类竞态：check/send之间替换socket；旧HELLO写入新连接snapshot或解除pending；迟到REGISTER回调重设新连接pending。TaskWorkerControlPlane在同一张连接表中保存generation、connected和pending，状态写入在同一把锁内校验代际；transport固定向已检查的连接发送。authenticated getter的confirmed与epoch来自同一个ClientConn，避免两次查询混接新旧连接。原生子agent Gibbs仅只读复核，未写文件或真机操作。

完整unit最新冻结版本：**6340 passed、16 skipped，316.89s**，真实master身份逐字不变，changed_python_paths为空。证据：`full-unit-cleanup-hello-verified-20261015-check.json`及console/JUnit。早期2/4项失败均保留：轻量测试宿主补真实控制面成员，供给/握手fixture使用正确注册顺序与递增代际；未跳过断言。实测Windows WinError 5还暴露manifest固定.part发布没有文件共享锁重试，现复用atomic_json的独立临时文件、fsync和有限重试，保留原manifest排序、缩进及换行字节；新增manifest回归通过。

最终部署master PID 4960，日志`logs/api-cleanup-hello-verified-20261015.err.log`。两台worker再次accepted HELLO，旧candidate的两条release债务均确认清空；身份与部署源码SHA见`cleanup-hello-verified-deployment-hashes-20261015.json`，握手见`cleanup-hello-verified-handshake-20261015.json`。Android保留数据、凭据和SAF，未切MTP或以ADB传权重。

供给缓存复用在Windows上映射显示旧硬链接名字，旧别名不存在时不能伪造samefile。采集器直接ReadProcessMemory读取整个权重映射计算SHA256，只回传摘要：Surface实读3,465,964,704 bytes，SHA `488088cdaa22766b33dcf5d5b67e6680b9ab6658911d41d0c87cbab1eac1acde`与当前[4,20) manifest及weights文件条目一致。保留path_exists=false、same_file_as_current=false和file_id=null，以mapped_content_matches_current单独表示内容绑定，不恢复或创建别名制造结果。

最终版本第一次界面选择因动态布局位移实际选择auto；master materialize的是PyTorch裁段，应归为DL，不能凭自动化脚本标签记LL。operation `supply_c2674655002c4042bfe817054592dfdc` / config `0bd8ed6f6f5b4a4cb0eaa4a81748273e` / generation `1792001370401449300`完成三机native barrier、API两问2/6和Android界面第一问2；均distributed_used=true、fallback=false。`dl-runtime-20261015-verified-dl-before-ui.json` checks_passed=true，master index只含0–3层、无GGUF native子进程；Surface/Y700为中段/尾段。第二条界面操作误按返回退出Activity，Android worker断开，旧DL意图正确以pipeline_worker_disconnected失效；双方随后已自动重连。相关原始失败与误分类文件保留，当前ready须由新意图确认。

下一次LL选择使用界面`✓ llama.cpp`反馈及后端intent.engine=llama_cpp双重确认；新operation已完成供给、native prepare/commit、两条API及两条Android界面问答，并通过问答前后严格运行时复验，详见以下最终记录。

## 2026-10-15 最终验收收口

最终源码部署后，当前Qwen3.5-9B LL意图为 `supply_8fa1619945e94efcb77e7fdd3586b1df`，config为 `2e3ef27afdbe4ee084df6bf8fbc17fe1`，generation为 `1792002475427263900`。选择来自Android真实界面，engine=llama_cpp、strategy=auto、node_ids为空；未手动给设备指定层段。系统自动完成inventory、内容复用、三机native prepare/commit及ready。自动分段保持master `[0,4)`、Surface `[4,20)`、Y700 `[20,32)`，每段由master协调。

| 最终版本执行路径 | API问答 | Android真实界面问答 | 实际加载核验 |
|---|---|---|---|
| DL：PyTorch头段 + llama.cpp中段／尾段 | `2`、`6`，均通过 | `2`通过；第二条因自动化返回退出Activity未完成 | master safetensors index只有0–3层；Surface／Y700为各自中段／尾段 |
| LL：三段llama.cpp | `2`、`6`，均通过 | `2`、`6`，均通过 | 问答前后进程及子进程映射均绑定当前本段工件 |

每条已通过的问答均核对distributed_used=true、fallback=false、finish_reason=stop、两台远端worker、远端区间 `[[4,20],[20,32]]`、claimed_layers `[4,32]`，以及精确model_operation_id／model_config_id／model_generation。Android回执来自真实SQLite session_messages，request_origin=android_http、request_origin_node_id=android-d83bf4d0，session_id=1；LL两条message_id分别为4308、4310。回执不从界面文字推断，也不混用旧意图。

真正LL的 `ll-runtime-20261015-verified-ll-explicit-before-ui.json` 与 `ll-runtime-20261015-verified-ll-explicit-after-ui.json` 均为checks_passed=true。master普通整模状态loaded=false、segment_materialized=true、descriptor仅0–3层字节非零；master及其全部子进程、Surface后端及native子进程的所有模型映射绑定各自当前manifest，Y700实际进程maps仅含当前尾段。Windows缺失的旧硬链接别名保留原始缺失事实，改以整个映射内容SHA与当前工件一致作为单独证明，未创建旧别名。该结论覆盖已采集的native准备及问答前后运行区间，不外推未观测的整个生命周期。

最终健康快照 `cleanup-hello-verified-final-health-20261015.json` 确认master PID 4960在运行，两台worker均connected／accepted／healthy、HELLO时间晚于本次REGISTER；当前LL意图ready、loaded_ready=true、native生命周期ready，config／generation一致，cleanup_ledger为空。部署后原有两条release债务由当前握手后的native ACK清账，旧operation未复活。

最终完整unit仍为 **6340 passed、16 skipped，316.89s**，退出码0；采用4 workers、loadscope及seed 1973875962。最终健康采集再次对全量测试冻结记录中的712个Python文件逐一计算SHA，python_drift_since_final_unit为空，真实master身份与最终单测记录逐字一致。源码及测试此后没有变更，不重复运行已经通过的同一轮测试。

本次LL三个节点actual_downloaded_bytes均为0；master／Surface／Y700分别复用1,983,372,464／3,465,966,743／3,014,661,971 bytes，并绑定本次operation的新manifest。Android的downloaded_bytes是完成逻辑字节，不能算作本轮新增网络下载。最终盘点 `cache-ll-explicit-final-20261015.json`：

| 节点供给根 | 最终实际占用 | 说明 |
|---|---:|---|
| master | 12,251,138,548 bytes，按文件ID去重 | 当前三个GGUF段；另保留上一轮DL的头段safetensors及少量元数据 |
| Surface | 5,392,066,634 bytes，按文件ID去重 | 原基线约1.926GB及一份本轮中段；逻辑别名合计8,858,033,377 bytes不作为物理副本 |
| Y700 | 2,946,919 KiB，约2.81GiB | 一份当前尾段与current指针；供给根无旧.receive目录 |

Surface C盘当前可用84,904,071,168 bytes，约79.07GiB。历史验收备份和供给根外完整源不计入本次缓存；未为取得盘点结果通过ADB手删模型或清app data／凭据／SAF。

正式组合证据为 `build/tickets16-18-20261014/tickets16-18-closed-20261015.json`，含最终单测、精确意图身份、DL与LL各自实际通过的问答范围、三机运行时、最终健康和缓存，以及31份证据／采集脚本／JUnit的SHA256索引。其status为passed_with_recorded_acceptance_limits，限制逐项保留：LL Q4_K_M供给计划仍列master为可能整模候选，不能从裁段后的runtime空候选推导单机绝对装不下；没有采集显式退出集群后的整模单机基线。Surface正确任务仍无开机／登录触发器与失败重启策略，执行时限12小时；进程内重连通过不代表设备重启自动拉起。LL pipeline_layout主节点引擎仍投影为pytorch，与intent／GGUF／native运行时不一致，暂不作为执行判据。最终版本DL第二条Android界面问答没有补记通过，导航断开后failed意图仍保留。

排票计划中的票16、17、18已同步本轮状态；保留并行会话写入的票19–22。工作仍在dev分支，未提交、未新建分支、未reset／restore；原生子agent只读复核已完成并关闭。


## 2026-10-15 本机＋ROG 后续修复

LL `pipeline_layout` 的 master PyTorch 误投影现已修复，assignment 绑定本次准入引擎，候选不能重标 active。设备数由三台变为两台后，旧离线 release 债务完整保留并按连接状态阻塞；显式成功退休旧 recovery 目标以持久化记录解除旧模型限制，不伪造 native ACK。新两节点 `qwenseek-2b` LL 已自动供给 `[0,20)`／`[20,24)`，两问正文2／6、distributed=true、fallback=false，问答前后只加载本段的映射核验通过。9B 首次因 ROG CPU RAM 预算不足在供给准入被拒绝，RAM 回升后的第二次重试虽完成供给，prepare 却复用旧 3.0 GiB 画像而拒绝；尚无两节点 9B 成功记录，2B 结果不替代容量价值证明。旧三机 operation 不再是当前 ready。最终冻结版本完整 unit **6373 passed、16 skipped，415.13s**，真实 identity 不变、Python 无漂移；RPC 测试泄露的底层 llama_cpp stub 已由 fixture 正确回收。完整说明、最终回归及限制见 [本轮验收](pipeline-layout-rog-acceptance-2026-10-15.md)。
