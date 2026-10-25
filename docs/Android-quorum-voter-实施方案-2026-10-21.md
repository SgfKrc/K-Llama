# Android 端 Quorum Voter 实施与验收记录

> 状态：**部分实施**
> 更新日期：2026-10-25（状态复核：仍现行——§6 第 54–55 行自述「Android voter 服务尚未配置，Tailnet `100.99.211.13:8765` 当前拒绝连接」「PC→Android quorum 选举和真实投票尚未联调」，与「部分实施」一致；#107 仍在「已知问题记录」未闭环区「待实网验收」）
> 适用范围：Android Y700 作为 QLH 自动选举 voter 的实现、配置与验收边界。协议语义以 `src/cluster_quorum.py`、`src/cluster_control_contract.py` 和 `src/cluster_auto_role_runtime.py` 为准。

## 1. 拓扑合同

`QuorumPolicy.election_allowed()` 要求 voter 总数至少为 3，门槛独立于多数票计算。Android 提供第三个 voter 后仍保持 worker/client 角色，不参与 leader 竞选。自动选举与静态 master 是互斥控制面，切换规则由 `src/cluster_fence.py:163-176` 执行。

Android voter 当前要求 API 33+。实现入口为 `android/app/src/main/java/com/qlh/inference/service/QuorumVoterService.kt`，服务通过前台 Service 托管 HTTP listener，绑定 `0.0.0.0:8765`。

## 2. RPC 合同

RPC 路径为 `POST /api/cluster/quorum/voter`，认证头为 `X-QLH-Quorum-Secret`。Android 从已引导的 cluster credential 读取共享 secret；PC peer 的 `QLH_CONTROL_VOTER_RPC_SECRET` 或 `QLH_CLUSTER_SECRET` 必须与之相同。`QuorumVoterRpcServer` 使用常量时间比较，限制 header 16 KiB、body 1 MiB、socket timeout 3 秒，并拒绝 `Transfer-Encoding` 请求。

| 操作 | Android 响应 |
|---|---|
| `reserve_term` | `{"ok":true,"term":...}` |
| `prepare` | `{"ok":true}` |
| `sign_vote` | `{"ok":true,"signature":...}` |
| `commit_certificate` | `{"ok":true}` |
| `snapshot` | `{"ok":true,"snapshot":...}` |
| `leader_certificate` | `{"ok":true,"certificate":null,"tcp_port":0}` |

协议实现位于 `control/QuorumVoterContract.kt`、`data/QuorumVoterLedger.kt` 和 `network/QuorumVoterRpcServer.kt`。Python `HTTPRemoteVoter` 按 `QUORUM_ERROR_CODES` 保留 voter 返回的稳定错误码。

## 3. 耐久状态与签名

`QuorumEd25519` 使用 Bouncy Castle 1.86 的 RFC 8032 轻量 API。Y700 Android 15 不提供可用的导出型软件 Ed25519 key-generation provider；轻量 API 直接处理 32-byte seed、32-byte 公钥和 64-byte 签名，不注册或选择 JCA provider。raw seed 仍由 Android Keystore AES-256-GCM key 加密后存入独立 SharedPreferences；Keystore key 与密文不参与 Android backup/device transfer。签名所需 seed 只在内存短暂解封，调用方和 voter 生命周期结束时清零。设置页提供密钥生成、公钥复制、voter set 配置和服务状态。

`QuorumVoterLedger` 使用独立 SQLite 文件，文件名由 voter ID 的 SHA-256 派生；启用 WAL、`synchronous=FULL` 与 5 秒 `busy_timeout`。metadata 绑定 schema、cluster、voter 和 voter-set epoch。term promise、votes、active certificate 与 recovery floors 在事务内持久化。

签名载荷按 Python 合同排序键、无空格、ASCII escape 编码。JVM golden-vector 测试同时覆盖空水位和非空水位的 canonical payload 与 Ed25519 签名。

## 4. 水位边界

Android voter 保留 `recovery_floor_schema=2`，对本机已保存的 floor 执行单调前缀校验。当前 Android 没有可证明 PC workflow commit watermark 的耐久 journal replica，因此 `QuorumVoterLedger.signVote()` 对任何非空 `commit_watermark` 返回 `quorum_watermark_unavailable`。这项拒签是协议安全边界，不得通过跳过 floor 检查或放宽错误处理绕开。

因此，设备验收配置必须产生空水位提案。若集群启用 journal watermark，Android voter 的 journal 复制与经验证的恢复导入仍需实现，完成前该类配置不能通过自动选举验收。

## 5. 运行配置

在 Android 设置页“自动选举投票节点”中生成本机 key，将公钥加入三节点共同的 voter set；voter set 至少包含三个身份，Android 的 cluster ID 与 bootstrap cluster 一致。保存配置时，应用校验 voter ID、公钥与本机 Keystore key 匹配后启动 `QuorumVoterService`。停用或配置变更会先关闭旧 listener，再校验并启动新配置；新配置无效时服务保持停止。

PC 的 `QLH_CONTROL_VOTER_SET` 使用同一 voter set，`QLH_CONTROL_VOTER_PEERS` 为远端 voters 配置 Android Tailnet URL，例如 `http://<Y700-Tailscale-IP>:8765`。候选主机使用 `QLH_NODE_ROLE=auto`；不要同时配置静态 master voter fence。

## 6. 验收状态

- Android JVM 全量单测 968 项通过；`QuorumVoterContractTest` 覆盖 Python golden vectors、base64url 规范、voter-set 精确字段及随机密钥签验。
- Y700（TB321FU，Android 15/API 35）voter instrumentation 4/4 通过：Keystore key 重开、SQLite vote/certificate 重开、HTTP secret 认证和带水位提案 fail-closed。
- 修复 APK 已安装启动；真机确认 arm64 JNI 库已加载、TaskWorker 前台服务已启动，Android worker 重新在线。
- Python `tests/test_cluster_auto_role_watermark.py`、`tests/test_worker_prepare_coordinator.py`、`tests/test_worker_prepare_handshake.py` 与 `tests/test_pipeline_retirement.py` 共 58 项通过。
- Android voter 服务尚未配置，Tailnet `100.99.211.13:8765` 当前拒绝连接；后续需配置共同 voter set 并验证 PC 到设备的认证 RPC。
- 当前 PC control-plane 为静态 master、自动选举关闭；PC→Android quorum 选举和真实投票尚未联调。跨机控制面门通过前不开始三机推理实测。非空 watermark 仍不在支持范围内。
