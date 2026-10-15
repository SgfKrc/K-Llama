# LL 布局引擎与两节点接力验收（2026-10-15）

本轮修复 LL `pipeline_layout` 将 master 误投影为 PyTorch 的问题，保持星状拓扑、master 唯一协调、DL／LL 连续层段接力。用户指定的 Windows 重启自动拉起和显式退出集群后的单机整模基线后置。Y700 没电、Surface 下线，本轮只操作本机与 ROG。

## 修复及事实来源

原布局从硬件 `device_info.backend_id` 推断本机引擎，缺省 PyTorch；容量计划适配器又忽略 assignment 上已确定的 engine，造成真实 llama.cpp／GGUF 执行被投影成 DL，连带制造假的跨引擎边界。

- `scheduler._pipeline_node_metadata()` 通过 lazy-safe `backend_id_for(self._host)` 读取活跃运行时，不为展示实例化 lazy manager。
- 新计划将同一次容量准入记录的 engine 绑定到返回的 assignment；缓存和事务计划保持已绑定身份，原输入不就地改写。
- `pipeline_node_contract.pipeline_layout_from_capacity_plan()` 按精确选中 GGUF 工件、assignment engine、节点 metadata 的顺序确定身份；显式未知引擎拒绝。引擎与工件种类一致，不复用旧 metadata 的完整 artifact/digest。
- 只读复核指出候选 LL 不能重标仍在服务的 active DL。最终实现保留这条边界：prepared／materialized／committed 候选不改变旧 active，fresh candidate 的计划从它自身的准入记录绑定引擎。真实 DL 的跨引擎边界继续保留。

## 真实换拓扑暴露的退休缺陷

重启后，旧三机 LL 的两条 release 债务正确保留，但意图协调器把整个账本当作新拓扑的全局阻塞条件。现改为仅 master 和当前连接节点的债务参与退休闸门。离线债务不删、不伪 ACK；节点重连后重新阻塞，仍须精确 identity／generation／cleanup ID 的 native ACK。

随后自动两节点供给已完成，新模型的 prepare 却被旧 9B recovery identity 拒绝：成功卸载清空了 live transaction，却留下旧恢复目标。修复在成功本地模型变更的 owner 边界内持久化 `phase=invalidated, published=false` 的退休记录，保留全部 cleanup ledger；仅持久化成功才解除恢复栅栏。持久化失败、无效／不可读恢复状态保持阻断，本地模型变更失败不退休旧身份。

意图入口现在也把 pending recovery 纳入退休目标及等待期间的身份核对；仅有旧恢复状态、没有 live transaction 时也走真实 unload。等待模型锁时 recovery 的 config／generation／phase 改变，旧请求返回 `INTENT_RETIRE_SUPERSEDED`，不得卸载新模型。

只读复核进一步验证恢复校验失败与普通 cleanup ACK 的组合：`_pipeline_recovery_validated` 与 write I/O 状态分开，loader 只有全部校验通过才允许持久化；无效／不可读记录禁止 partial ledger 覆写原 raw。合法 ACK 可在内存结清已知债务，但不得擦除尚未合法恢复的义务；成功换模、restore／finish／retire 均不能绕过。回归以原持久化字节逐字核对，并经二次重启确认仍 fail-closed。此前在该发现前启动的 `full-unit-closed` 已主动中断，单独保留中断原因，不计为通过。

## ROG 环境与部署

SSH 为 `ROG@100.116.17.12`。隔离运行目录 `C:/Users/ROG/qlh-runtime-20261015`；旧项目 `L:/新建文件夹/LEDS_BJTU` 保留。经产品 encrypted first-connect 加入 `100.90.76.108`，未复制集群 secret 或整模。运行时源文件及 9 个 native DLL 从本机受检构建部署，SHA 对应证据保留。ROG 缺失的 `diskcache` 已补装为 5.6.3，`llama_cpp` 0.3.35 导入复验通过。

ROG 为 Ryzen 5 9600X、6 核／12 线程、15.151 GiB RAM、RTX 5060 8GB；初探 C／L 盘分别约 145／617 GiB 空闲。native 当前为 CPU 构建，不能把显存算入 CPU LL 容量。未关闭用户应用或降低容量安全预算。运行任务没有开机／登录 trigger，自动拉起未在本轮验收。

ROG 的早期重复启动写下过陈旧 `worker.pid`；最终采集以实际 8001 LISTEN 的进程、运行目录和全部子进程识别当前 worker，并保留 PID 文件不一致事实。2B 映射验收时实际 worker PID 为 23860，master PID 为 23932；部署最终 recovery 校验保护后 master 为 24700。PID 仅是本轮采集事实，不作为长期身份。

## 9B 的真实拒绝与边界

`qwen3-5-9b`／LL、`strategy=auto, node_ids=[]` 创建操作 `supply_3794c1ea1c5949d3a62af1a37e59a4da`，进入 inventory 后拒绝，码为 `pipeline_distributed_workers_unavailable`。不是未握手或缺少可供给能力：实测 source header 和日志预算可复现其容量排除。

| ROG CPU LL 预算项 | Bytes |
|---|---:|
| 操作时可用 RAM | 3,375,964,160 |
| 固定 reserve | 536,870,912 |
| GGUF non-block 权重 | 1,557,481,472 |
| non-block × runtime multiplier 2.0 × safety 1.2 + reserve | 4,274,826,445 |
| 最小对齐尾段 `[28,32)` 总需求 | 5,633,044,276 |
| 相对当时可用 RAM 的缺口 | 2,257,080,116（约 2.10 GiB） |

ROG 在 decoder 分配前已被容量归一化排除，只剩 master，所以报“分布式 worker 不可用”。原 API 失败消息过于概括；精确 `excluded_nodes`、逐层／分量字节、alignment=4 和复现计划保存在 `two-node-9b-budget-diagnosis.json`。首次拒绝未传输或加载 9B 权重；后续重试见下。本轮未把分页文件或空闲显存算作 CPU RAM。

2B 验收结束并经产品接口卸载后，ROG 自然可用 RAM 升至 5,720,571,904 bytes；未关闭用户应用。再次提交相同 9B 自动意图，operation 为 `supply_193480fc9c5343129715becbd2c452b2`，供给计划准入并自动得到 master `[0,28)`、ROG `[28,32)`。两段工件分别为 5,063,183,538 与 1,983,372,529 bytes；ROG 完成尾段下载并上报全正的逐层字节，accepted hello 后进入 prepare。

但 2026-10-15 11:18:04 的 prepare 容量复核返回 `pipeline_capacity_single_node_insufficient`，意图最终为 `INTENT_PREPARE_FAILED`。没有 native ready，也没有进行本次 9B 问答。不能将传输完成记为接力通过。只读复核已定位预算事实源不同：供给 inventory 读取远端实时 `psutil`，prepare 对远端使用注册画像的 `device_info.ram.available_gb`，仅本机重新采样；ROG 的画像仍为 3.0 GiB。心跳补报复用缓存画像，不能替代实时容量。证据为 `two-node-9b-retry-latest.json`、`two-node-9b-retry-runtime-capacity.json` 及对应 master 日志。该次 9B 开始前的 2B 已退休，9B 失败也不是 ready；随后使用最终代码新建的 2B 当前意图见下一节。

| 第二次 9B 的 ROG 尾段 `[28,32)` | Supply | Runtime prepare |
|---|---:|---:|
| 容量 bytes | 5,720,571,904（inventory 实测） | 3,221,225,472（缓存画像） |
| 段需求 bytes | 5,633,044,276 | 3,897,580,544 |
| 预算余量／缺口 bytes | +87,527,628 | −676,355,072 |

runtime 尾段按角色计 decoder 565,924,096、final norm 16,384、lm_head 834,355,200 bytes，再乘 runtime multiplier 2.0 和 safety 1.2，最后加 512 MiB reserve；不计 tail 的 embedding。源 `other=150,980,608` 属于本次裁段排除的 MTP 张量，runtime 为 0。供给／运行时组件口径也有差异，但本次 runtime 段需求更低；拒绝来自缓存预算，未发现磁盘回收统计被重复扣作 RAM。只读纯求解器按原逐层字节与缓存画像复现同一拒绝码和总预算，换回供给预算则同一约束分段准入。未采集失败瞬间远端真实 RAM，不能断言其当时真实不足。

**后续修复要求（本轮未修改生产路径）：** prepare 使用带采样时间、注册代际和过期策略的当前远端运行时预算，供给／准备明确同一预算来源；拒绝日志保存逐节点预算和角色需求。不得通过手改画像、降低安全系数或删除 range 约束取得通过。另外 `CodedHTTPException.error_code` 被 intent 的 `.code` fallback 丢失，需保留原结构化错误码。修后须重跑自动两节点 9B native、问答及 proper-segment 映射。完整复核见 `two-node-9b-retry-budget-source-review.json`。

本机 `qwen3-5-2b` 的旧完整 GGUF 侧车又存在 tokenizer 摘要与 GGUF 头不一致：声明 `5f9e4d…`，实读 `a5c186…`，故 descriptor fail-closed，产品暂给 `SUPPLY_FULL_SOURCE_REQUIRED`。旧资产和失败操作保留，未重写侧车。该现场元数据及错误分类仍需后续处理；本轮转用另一份已登记的 `qwenseek-2b` 完成拓扑验收。

## 两节点 LL 实测

在最终 recovery 校验保护部署后，重新执行本轮验收；最终操作来自产品意图接口：`model_id=qwenseek-2b, engine=llama_cpp, strategy=auto, node_ids=[]`。未手选设备层段，未迁就旧 Surface／Y700 的三段工件。

- operation：`supply_1a74bf597ea64d5ea230a9b4e470d51c`
- config：`a82b3045bfbd47f08c097c322f210e01`
- generation：`1792034859979145400`

| 节点 | 自动分段 | 布局引擎／工件 | 本次新增下载 |
|---|---|---|---:|
| master | `[0,20)`，head | `llama_cpp`／`gguf_segment` | 0，复用既已构建头段 |
| rog-5060 | `[20,24)`，tail | `llama_cpp`／`gguf_segment` | 0，复用上一失败操作已供给的尾段 |

首次供给 ROG 确实经 master 下载约 580 MB 尾段，随后新 operation 重新绑定 manifest 复用，不把逻辑完成字节误算成本轮新增网络下载。两节点区间连续完整覆盖 `[0,24)`，每段都小于整模；layout 均无 `cross_engine`／`handoff_at` 假边界。

native prepare／commit 收据均匹配本次 operation、config、generation、assignment、artifact 及范围。问答前后实读双方进程和全部子进程的模型映射：ROG 通过文件 ID／samefile 绑定当前尾段；master 的 Windows 映射显示已消失的旧构建别名，保留 `path_exists=false, same_file_as_current=false`，另对整个映射内容计算 SHA，严格等于本次头段 manifest 的权重 SHA。未创建旧别名伪造 samefile。

master API 的普通整模状态为 `loaded=false`，`pipeline_prepared=true, segment_materialized=true`；descriptor 仅 0–19 层字节非零。native 生命周期 ready、当前参与 worker 只有 ROG。旧 Surface／Y700 两条 release 债务仍在账本，未被当作当前两节点阻塞债务。

| 请求 | 最终正文 | 生成 token | finish reason | 分布式证据 |
|---|---|---:|---|---|
| What is 1+1? Reply with only 2. | `2` | 49 | stop | ROG 唯一远端、fallback=false |
| What is 3+3? Reply with only 6. | `6` | 7 | stop | ROG 唯一远端、fallback=false |

首轮 16 token 请求只生成了该模型的推理前缀，未得到最终答案，原失败证据保留。最终以 256 token 预算重新创建会话，两问均正常停止；不是把截断回答计为通过。

**验收范围：** 本轮证明两节点自动供给、native 层段接力及 LL 布局事实一致。2B 供给计划明确列 master、ROG 均可能单机装下，因此不作为 9B 聚合容量不可替代性的证明，也不是单机整模基线。本轮未进行新的 DL 真机测量，历史三机 DL／LL 结果仍按原验收记录的当时范围保留，不声称旧三机当前在线／ready。

## 回归与证据

布局最终相邻组合 **592 passed，48.97s**，见 `layout-corrected.xml`。离线债务回归修前 3 failed、修后 3 passed，相邻 **168 passed**。恢复退休 owner 与真实意图 callback 组合 **69 passed**，覆盖二次重启、持久化失败、本地变更失败和等待期间新代际。

布局及首批离线债务修复后的完整 unit：**6359 passed，16 skipped，421.59s**；真实 master identity 逐字未变，冻结 Python 文件无漂移。此轮早于随后实测发现的 recovery 退休修复，保留为中间基线。

恢复退休最终生产源码的第一轮完整 unit 为 **6 failed、6352 passed、16 skipped、15 errors，457.34s**。6 项失败来自轻量 `_Worker` 测试宿主遗漏新恢复校验成员，补齐宿主后包含布局／恢复退休的定向组合 **60 passed**；15 项错误来自既存 RPC 测试直接注入 `sys.modules["llama_cpp.llama_cpp"]`，而 teardown 仅复原父包，后续真实导入误复用残留 stub。四处注入已使用 `monkeypatch.setitem` 由同一 fixture 回收，不跳过模板断言，也不全局清模块缓存。最小复现修前 **1 passed、15 errors**；修后完整 RPC→模板组合 **45 passed、1 skipped**，真实 binding 预加载后的对象身份亦复原；7 文件原 seed 串行乱序 **117 passed、1 skipped**。该失败运行真实 master identity 逐字不变、冻结 Python 无漂移；保留 `full-unit-validated-failure-triage.json` 及原 JUnit。最终完整复跑结果另记，不能以此前通过结果替代。

最终冻结版本完整 unit 已按原 seed `1973875962`、4 workers、loadscope 重跑：**6373 passed、16 skipped，415.13s**，退出码 0。真实 master identity 逐字不变，全部受检 Python 文件 `changed_python_paths=[]`；六个生产文件 SHA 与本次实机部署记录逐一相同。证据为 `full-unit-final-verified-check.json`、`full-unit-final-verified-console.log` 和同名目录 JUnit。失败／中断轮次保留，不充作最终通过。最终组合验收与 SHA 索引为 `pipeline-layout-rog-final-acceptance.json`；它明确保留两节点 9B 未通过、2B 不替代容量价值及单机基线、未新增 DL 真机结果等限制。

全部现场证据位于 `build/pipeline-layout-rog-20261015/`；验收主证据包括 `two-node-final-ll-runtime-before-chat.json`、`two-node-final-ll-runtime-after-chat.json`、`two-node-final-ll-chats.json`、`lifecycle-after-auto-retirement.json`、`two-node-9b-budget-diagnosis.json`、部署 SHA、JUnit 与当前源文件摘要。失败／中断操作不复活，证据保留。当前仍在 dev，未提交、未新建分支、未 reset／restore，排票计划的并行票19–22保留。

## 2026-10-15 后续：实时预算与两机 9B 已闭环

本文件“9B 未通过／后续修复要求”描述的是本轮结束时的历史现场。后续已修复 prepare 读取缓存画像、供给工件 hello 竞态、迟到 hello 越过 deadline 及 intent 原错误码丢失。最终源码重新经产品意图入口自动得到 master `[0,28)`、ROG `[28,32)`，Qwen3.5-9B native prepare／commit、两问正常停止及前后所有进程本段映射均通过，无整模加载。最终定向520 passed，完整 unit 6500 passed, 16 skipped, 426.05s，真实身份和源码无漂移。新的范围、完整证据和容量价值限制见 [实时预算与9B验收](../runtime/capacity-freshness-rog-acceptance-2026-10-15.md)。本轮历史记录与失败证据保留。
