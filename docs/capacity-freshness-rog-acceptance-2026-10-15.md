# 供给到 prepare 容量事实源与工件握手验收（2026-10-15）

本轮已修复上轮 9B 自动供给完成、prepare 却读取旧画像的问题，并用本机＋ROG 完成最终源码的 **Qwen3.5-9B 自动供给、LL native 接力、两问和问答前后本段映射核验**。保持 master 唯一协调、星状拓扑和 DL／LL 连续层段设计。Surface／Y700 未操作；重启自动拉起和显式退出集群后的单机整模基线后置。

## 修复及边界

- 新 `model_supply_capacity.py` 通过现有认证 `MODEL_SUPPLY inventory` 通道采集一次 prepare 所需的当前预算；每次新建 wire query ID，绑定 parent operation、连接时间、认证注册代际、prepare owner、取消状态及原供给计划。既有 PC／Android 协议字段不变。
- 请求空 `model_id` 仅查询资源，不重新计算数 GB 源权重摘要。PC 将 RAM 采样放在源检查之后，避免把哈希之前的旧采样当作当前值。
- 工件 available 回执可能早于异步 capability hello。只读复核用真实探针和 solver 确证：旧 `[20,24)` hello 配合新 `[28,32)` 供给会误拒。故增加最多 30 秒的本次工件握手等待，按工件 SHA、source SHA、tokenizer、精确范围、model 关联、engine／format 核对 accepted hello。等待释放 condition；取消、换 owner、重连立即拒绝，超时 `SUPPLY_RUNTIME_CAPABILITY_TIMEOUT`。
- 工件握手完成才发资源查询。master 单调时钟从查询派发起计最多 30 秒；在实际求解入口再次验证 owner、代际、有效期及当前 capability，传入 records 本身也必须匹配本次工件，防止读到旧 snapshot 后 hello 刚更新的竞态。记录 provenance，不改设备画像、不另建全局容量缓存。
- llama.cpp 使用实时 RAM；PyTorch 使用其选定 CPU／CUDA runtime 预算，不把空闲显存计入 CPU RAM。reserve 和既有安全系数保留，范围与工件约束不放松。
- 直接 prepare 当前 supplied path 也进入同一 guard／预算屏障。预算异常清理候选并保留原 active；transport OSError 转具名 409。`ModelIntentCoordinator` 保留 `CodedHTTPException.error_code`、header／结构化 detail 的错误码，原始异常字符串不公开为 reason。

## 最终两机 9B 实测

产品请求为 `model_id=qwen3-5-9b, engine=llama_cpp, strategy=auto, node_ids=[], timeout_seconds=600`。全程未手选节点、区间或强改画像。

- operation：`supply_fe9955dfee144ecaa3f738a8ccae74ae`
- config：`40b914bfc35b471bbfa8a0706031e307`
- generation：`1792037852144859800`
- master PID：11476；ROG 实际 8001 listener PID：29132。PID 仅表示本次采集，不作为永久身份。

| 节点 | 自动层段 | 布局 | 当前预算 bytes | runtime 需求 bytes | 本次新增下载 |
|---|---|---|---:|---:|---:|
| master | `[0,28)` head | llama_cpp／gguf_segment | 18,106,728,448 | 10,659,739,444 | 0 |
| rog-5060 | `[28,32)` tail | llama_cpp／gguf_segment | 5,993,144,320 | 3,897,580,544 | 0 |

两段分别复用约 5,063,183,538 和 1,983,372,529 bytes 工件；新 operation 重绑 manifest，不把逻辑完成字节当新增网络下载。区间连续覆盖 `[0,32)`，每节点只承载 proper segment，无 `cross_engine`／`handoff_at` 假边界。

prepare 的两个预算都来自 `inventory.ram.available_bytes_live`。同次 query 为 `supply_1e4eab405b1e4f8d9b8e55e6293294ee`，ROG 认证注册代际 1；master 请求时间 1792037846.6254508，ROG 回包 1792037846.6396065，master 本地采样返回 1792037846.706899。请求级证据记录在公共 capacity plan，未回写缓存画像；当时 ROG 画像仍为 5.3 GiB，预算没有改用它。

native prepare／commit 收据匹配本次 operation、config、generation、assignment、artifact SHA 和精确层段。普通整模状态 `loaded=false`，流水线 `pipeline_prepared=true, segment_materialized=true`；master descriptor 仅 0–27 层非零。

问答前后均检查两端主进程及所有子进程的 GGUF／Safetensors 映射，只出现本次对应段。Windows 映射指向已消失的旧构建别名时，保留 `path_exists=false, same_file_as_current=false`，另对整个映射内容计算 SHA 并匹配当前 manifest；未创建旧别名伪造 samefile。

| 请求 | 正文 | 生成 token | 停止 | 分布式引擎时间 |
|---|---|---:|---|---:|
| What is 1+1? Reply with only 2. | `2` | 7 | stop／text_stop | 7.597 秒 |
| What is 3+3? Reply with only 6. | `6` | 7 | stop／text_stop | 5.821 秒 |

两问均 `temperature=0, distributed_required, max_new_tokens=256`；实际生成正常停止，不是截断。`distributed_used=true, fallback=false`，远端仅 `rog-5060`、区间 `[28,32)`；metrics 的 model operation／config／generation 与 ready 身份一致。终止命中分词拼出的 `<|im_end|>`，证明此前文本停止链仍在真实 LL 执行中生效。表中时间是响应 metrics 的引擎耗时，不作为端到端吞吐基准。

## 回归与中间现场

首批定向 206 passed；扩大组合 488 passed，42.45 秒。只读复核确认工件 hello 竞态后主动中断此前完整 unit，不计为通过。新增 32 项握手边界首轮 **108 passed、1 failed**，确证“第 31 秒才到的正确 hello 可越过 30 秒期限”；生产已调整先检查 deadline，再接受能力。最终组合 **520 passed，48.30 秒**，包含真实 scheduler、候选清理、原错误码、旧回包、注册代际、取消／过期、Android 既有库存格式和工件握手边界。

最终冻结版本完整 unit：**6500 passed, 16 skipped, 426.05s**。真实 master identity 逐字不变，全部受检 Python SHA 无漂移；最终部署八个生产文件 SHA 与受检当前源码一致。

第一轮新意图 `supply_01ba20954ce94b1096fa2521bfe5a126` 的 ROG inventory 为 5,160,759,296 bytes，低于既有尾段供给需求 5,633,044,276 bytes；等待期间因部署握手修复主动取消，没有 native 加载。未降低安全预算取得通过。

第二轮 `supply_614a9ca37fa4454a8f13858fda85526b` 已 ready，且完成问答前映射核验；随后因补齐迟到 hello 的超时顺序边界，正常退休并重新部署。此轮不代替最终源码验收。最终第三轮重新从产品意图入口完成，证据如上。失败、中断和取消记录均保留。

ROG worker 采样文件 SHA 为 `de58e8af4e81f8350cd35369a0222ad6e32c02b4fb60709f468c44655390072d`。SSH 直接启动的临时 PID 13768 没有留存，之后使用既有无重启触发器的手动测试任务成功启动，未操作用户进程。

## 结论范围与后续

本轮证明两节点随在线拓扑自动选段、供给、native LL 接力以及 9B 全程按段执行。Surface／Y700 两条历史 release 债务仍保留，不伪造离线 ACK，也不阻塞当前两节点。

本轮没有新增 DL 真机结果，历史三机 DL／LL 结果仍按 [票16–18记录](tickets16-18-acceptance-2026-10-14.md) 的原范围保留。自动拉起和单机整模基线仍按用户裁决后置。供给 metadata 仍列 master 为 `single_node_full_model_candidates=['master']`，因此不能把本次 9B 成功宣称为“聚合容量不可替代”或实测单机装不下；当前通过的是两端均未加载整模的 9B 层段接力。

易用性上，本轮只提交一次模型意图即可等待系统完成，复用工件无新增传输；完整自动流程约 175.51 秒。源校验与缓存回收仍占据准备阶段，尚非性能标定，不把这两次短问答当作票19–22的吞吐结论。

## 证据

证据根 `build/capacity-freshness-rog-20261015/`：

- `two-node-9b-ll-runtime-before-chat.json`、`two-node-9b-ll-runtime-after-chat.json`：生命周期、当前布局、实时预算、每个进程模型映射、native 收据和身份。
- `two-node-9b-ll-chats.json`：两条请求、完整响应与 metrics。
- `master-verified-deployment.json`、`rog-worker-deployment.json`：部署过程与 SHA。
- `targeted-handshake-final.xml`、`full-unit-closed-check.json` 与同名目录 JUnit：最终定向／完整回归和冻结比对。
- `capacity-freshness-final-acceptance.json`：最终组合断言与工件 SHA 索引。
- `round-changes.patch`：相对本轮 `before/` 的增量，不是 Git HEAD 的所有未提交改动。

保持 dev，未提交、未重开分支，保留已有修改与排票19–22。
