# Android 端 quorum voter 实施方案

> 状态：**方案待评审**（未实现）
> 更新日期：2026-10-21
> 适用范围：把 Android（Y700）接入 QLH 自动选举的 voter 集合，使三机拓扑在只有两台 PC 可用时仍能满足自动选举的 voter 数量下限。协议语义与安全边界以 `src/cluster_quorum.py`、`src/cluster_control_contract.py` 为准。

## 1. 缺口

自动选举对 voter 数量有硬下限：

```python
# src/cluster_quorum.py:89-91
def election_allowed(self, voter_count: int) -> bool:
    """Require a third configured voter for automatic quorum operation."""
    return voter_count >= 3
```

该判据**独立于** `quorum_size`（`len(voters)//2 + 1`）。因此 2 个 voter 永远无法进入 leader，实测表现为：

```
event=auto_role_start accepted=False state=read_only runtime_role=client reason=quorum_requires_witness
```

而可跑 Python 后端的设备只有本机与 Surface 两台：ROG（`desktop-imj12e1`）长期离线，且与 Surface 不能同时在线。Y700 虽在线，但 Android 侧没有 voter 能力 —— voter RPC 端点是 `api_server` 的 HTTP 路由（`src/api/routes_cluster.py:1304`），Android 生产代码中 `ServerSocket`／HTTP server／`BouncyCastle`／`Ed25519` 全部零命中。

结论：**在只有两台 PC 的前提下，自动选举被 voter 数量下限锁死，唯一出口是让第三台设备（Y700）能投票。**

## 2. 契约（必须逐字节对齐的部分）

### 2.1 RPC 形状

| 项 | 值 | 证据 |
| --- | --- | --- |
| 路径 | `POST /api/cluster/quorum/voter` | `src/api/routes_cluster.py:1304` |
| 认证头 | `X-QLH-Quorum-Secret` | `src/cluster_auto_role_runtime.py:45` |
| 请求体 | JSON object，含 `op` | `src/cluster_auto_role_runtime.py:472-499` |
| 成功响应 | `{"ok": true, ...}` | 同上 |
| 失败响应 | `{"ok": false, "code": "<稳定错误码>"}` | `src/cluster_auto_role_runtime.py:499-506` |

六个 op 的返回形状：

| op | 成功响应 |
| --- | --- |
| `reserve_term` | `{"ok": true, "term": <int>}` |
| `prepare` | `{"ok": true}` |
| `sign_vote` | `{"ok": true, "signature": {...}}` |
| `commit_certificate` | `{"ok": true}` |
| `snapshot` | `{"ok": true, "snapshot": {...}}` |
| `leader_certificate` | `{"ok": true, "certificate": null, "tcp_port": 0}` |

`leader_certificate` 对纯 voter 返回 `certificate: null` 是合法的：Python 侧 `observe_leader` 会跳过 `None`。

### 2.2 签名与规范化

| 项 | 约束 | 证据 |
| --- | --- | --- |
| 算法 | `SIGNATURE_ALGORITHM = "ed25519"`，字面值固定 | `src/cluster_control_contract.py:43` |
| 私钥 | 32 字节 raw seed，base64url **无 padding** | `src/cluster_control_contract.py:99-121` |
| 公钥 | 32 字节 raw，base64url 无 padding | 同上 |
| 签名 | 64 字节，base64url 无 padding | 同上 |
| 签名载荷 | `{schema_version, document_type, cluster_id, voter_set_epoch, term, leader_id, lease_id, expires_at_ms}`；`commit_watermark` **仅在非空时**加入 | `src/cluster_control_contract.py:376-394` |
| 规范化 | `json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))` | `src/cluster_control_contract.py:90-96` |

**规范化是最高风险项**：`sort_keys` + 无空格分隔符 + `ensure_ascii` 三者必须逐字节复刻。Gson 不保证键序与数字格式，必须自写序列化器并用 Python 侧生成的 golden vectors 逐字节比对。

### 2.3 时钟语义

RPC **忽略请求体里的 `now_ms`**，每个操作都用接收端本地时钟重新采样：

```python
# src/cluster_auto_role_runtime.py:476-491
# The wire timestamp is untrusted. Lease exclusion and expiry use
# a fresh receiver-local sample for every voter operation.
```

`lease_ms` 默认 10000、最小 1000（`src/cluster_auto_role_runtime.py:537-539`）。RPC 超时预算为 `min(timeout, lease_ms / (8000 * peer_count))`（同文件 `:544`），三 voter 时约数百毫秒 ⇒ Android 端必须低延迟响应。

## 3. Android 侧现状

### 3.1 可复用资产

| 能力 | 资产 | 路径 |
| --- | --- | --- |
| HTTP 客户端 | OkHttp 4.12.0 + Gson 2.11.0，含 Bearer 注入与错误体解析 | `android/app/src/main/java/com/qlh/inference/network/ApiClient.kt` |
| JSON | Gson（全仓统一，非 kotlinx.serialization／Moshi） | `android/gradle/libs.versions.toml` |
| 加密存储先例 | AndroidKeyStore + AES-GCM + SharedPreferences 密文 | `android/app/src/main/java/com/qlh/inference/security/ClusterCredentialStore.kt`、同目录 `AuthTokenStore.kt` |
| 集群密钥 | cluster secret 已加密持久化，可作 voter RPC secret | 同上 `ClusterCredentialStore.kt` |
| 前台 Service 模板 | `specialUse` + `stopWithTask=false` + `START_STICKY` + 重启恢复 | `android/app/src/main/java/com/qlh/inference/service/AndroidPresenceService.kt`、`worker/TaskWorkerService.kt` |
| 原始 TCP + HMAC | `Socket` + `Mac`，可作为无 HTTP 依赖的备选通道参考 | `android/app/src/main/java/com/qlh/inference/worker/TaskWorkerClient.kt` |
| 本地库 | Room 2.6.1（`AppDatabase`） | `android/app/src/main/java/com/qlh/inference/data/AppDatabase.kt` |
| 控制帧准入 | 内存态证书校验，`validate` 未接线时抛 `control_fence_unavailable` | `android/app/src/main/java/com/qlh/inference/worker/TcpControlFence.kt` |

构建参数：`compileSdk = 36`、`minSdk = 26`、`targetSdk = 34`（`android/app/build.gradle.kts`）。Y700 实测 Android 15 / API 35 / `arm64-v8a`。

### 3.2 缺失能力

1. **HTTP server**：生产代码零命中，需新增。
2. **Ed25519**：无 `BouncyCastle`／`Conscrypt` 依赖。
3. **voter ledger**：无 `quorum_*` 表；Room 未建模。
4. **canonical JSON 序列化器**：无 `sort_keys`／无空格等价实现。
5. **RPC 分发与常量时间 secret 比较**。
6. **voter 配置项**：`SettingsDataStore` 无 `voter_id`／`voter_set`／`peers`／`lease_ms`／私钥。
7. **voter 前台 Service**。

## 4. 选型

### 4.1 HTTP server

| 方案 | APK 增量 | 取舍 |
| --- | --- | --- |
| **NanoHTTPD** | 约 100 KB | 单路由 POST JSON 天然契合，自带 HTTP 解析与线程池；推荐 |
| 手写 `ServerSocket` | 0 | 零依赖、完全可控，契合本仓边界风格；但需自实现报文解析、body 上限、超时、并发 |
| Ktor server | 2–3 MB | 与现有栈不符（无 ktor），体积与启动成本高；不推荐 |

只服务一个 `POST` 路由，NanoHTTPD 与手写 ServerSocket 均可行。前者少写代码，后者零新增依赖 —— 若本仓倾向最小依赖面，选手写。

### 4.2 Ed25519

`minSdk = 26`，而平台 Ed25519 自 API 33 起由 Conscrypt 提供。两条路：

- **运行时门控（推荐）**：`Build.VERSION.SDK_INT >= 33` 时用平台 Ed25519，低版本显式禁用 voter 并给出具名错误。Y700 是 API 35，当前三机形态可直接用，且不增 APK 体积。
- **引入 BouncyCastle**：全版本可用，代价是 APK 增量与新的供应链面。

需实测确认：平台 Ed25519 是否接受 **raw 32 字节 seed**（`EdECPrivateKeySpec`），还是要求 PKCS#8 包装。若不接受 raw，则需包装转换或回退 BouncyCastle —— 这是本方案唯一的阻塞性未知。

## 5. 契约对齐清单

Kotlin 侧须复刻 `src/cluster_quorum.py:142-517` 的状态机语义。时间窗统一由接收端本地时钟采样。

| Python 方法 | Kotlin 必须实现的要点 |
| --- | --- |
| `_initialize` | 四表：`quorum_metadata`、`quorum_term_state`（单行）、`quorum_recovery_floors`、`quorum_votes`（主键 `epoch,term,voter`）；`journal_mode=WAL`、`synchronous=FULL`、`busy_timeout`；metadata 与配置冲突 ⇒ `quorum_ledger_corrupt` |
| `reserve_term` | leader 非空否则 `quorum_invalid_identity`；`active_expires > now` ⇒ `quorum_active_lease`；`term = max(promised_term, active_term) + 1` |
| `prepare` | `term <= 0` ⇒ `quorum_term_stale`；active lease 且 `term > active_term` ⇒ `quorum_active_lease`；`term < promised` ⇒ `quorum_term_stale`；`term == promised` 且 leader 冲突 ⇒ `quorum_term_conflict` |
| `_save_floor` | 水位单调：`last_sequence` 回退忽略；同 sequence 不同 digest ⇒ `quorum_watermark_unheld`；`committed` 粘性 |
| `_require_preserved_floor` | marker 必须为 `2` 否则 `quorum_watermark_unavailable`；提议水位须 `≥` 本地 floor 且同 seq digest 一致，否则 `quorum_watermark_unheld` |
| `sign_vote` | epoch／cluster 不符、`expires <= now`、promised 未匹配 ⇒ `quorum_vote_rejected`；`_require_preserved_floor` + `_require_accepted_lease`；重复投票同 `(epoch,term,voter)` leader/lease 不同 ⇒ `quorum_duplicate_vote`；expiry 回退 ⇒ `quorum_expiry_regressed`；签名后写 vote 并 `save_floor(accepted)` |
| `commit_certificate` | 校验证书含自身签名；本地 durable vote 须逐字段匹配否则 `quorum_vote_rejected`；active lease 冲突 ⇒ `quorum_active_lease`；`term < active_term` ⇒ `quorum_term_stale`；更新 active 并 `save_floor(committed)` |
| `snapshot` | 返回 promised／active 的 term、leader、lease、expiry、digest |
| `import_recovery_manifest` | 一次性迁移；marker 非 `needs_import` ⇒ `quorum_migration_not_required`；legacy vote 未被 manifest 覆盖 ⇒ `quorum_watermark_unavailable` |

错误码集合（须逐字对齐）：`quorum_unavailable`、`quorum_vote_rejected`、`quorum_active_lease`、`quorum_term_stale`、`quorum_duplicate_vote`、`quorum_expiry_regressed`、`quorum_watermark_unheld`、`quorum_watermark_unavailable`、`quorum_requires_witness`。

## 6. 分阶段实施

### 阶段 0：最小可投票闭环

新增：

- `security/QuorumVoterKeyStore.kt` —— 复用 `ClusterCredentialStore` 模式，Keystore AES-GCM 包裹 raw Ed25519 seed
- `crypto/QuorumEd25519.kt` —— 平台 Ed25519 或 BouncyCastle + canonical JSON
- `data/QuorumVoterLedger.kt` —— term 状态、votes、`reserve_term`／`prepare`／`sign_vote`／`snapshot`
- `network/QuorumVoterRpcServer.kt` —— 单路由 + 常量时间 secret 比较
- `service/QuorumVoterService.kt` —— 前台 Service 承载 server

修改：`data/SettingsDataStore.kt`（voter 配置项）、`AndroidManifest.xml`（新 service）、`gradle/libs.versions.toml` + `app/build.gradle.kts`（按 §4.2 选型）。

出口判据：PC 主节点对 Android 完成 `reserve_term` → `prepare` → `sign_vote`，取得可通过 `validate_certificate` 的签名。

### 阶段 1：完整语义

扩展 `QuorumVoterLedger` 的 `commit_certificate`、`recovery_floors`／`committed_floors`、`_require_preserved_floor`、`import_recovery_manifest`；RPC 增加 `commit_certificate`、`leader_certificate`。

出口判据：三机出证、续租成功、进程重启后 term 与 vote 保持；水位不足时 fail-closed。

### 阶段 2：生产化

WAL/`synchronous=FULL`、metadata 冲突处理、body 上限、超时、崩溃恢复、电池豁免引导、可观测日志。

出口判据：Y700 长稳运行，进程被回收后 `START_STICKY` 能恢复端口与状态。

## 7. 测试策略

Python 侧既有对照：`tests/test_cluster_quorum.py` 覆盖三 voter 出证、`quorum_requires_witness` 下限、分区至多一证书、term 持久与不可复用、同 term 单 leader、过期证书可被更高 term 替代、ledger 重开保状态、单签名不可写、commit 需本地 durable vote；`tests/test_cluster_auto_role_runtime.py` 覆盖 RPC 的时钟语义（未来时钟不可绕过活跃 lease、过去时钟不可复活过期证明）。

Android 侧须新增：

- **JVM 单测**：`QuorumVoterLedgerTest`（term 单调、`prepare` term 冲突、重复投票、水位前缀、commit 需本地 vote、重开保状态）、`QuorumRpcDispatchTest`（6 op 形状 + 错误码映射 + 忽略请求体 `now_ms` + 常量时间比较）、`QuorumCanonicalJsonTest`（**golden vectors**：用 Python 生成的 `signing_bytes` 与 Kotlin 输出逐字节比对，含 `commit_watermark` 空／非空两种）
- **Instrumented**：Keystore 私钥存取、`QuorumVoterService` 生命周期与端口存活、真实 HTTP 往返
- **端到端**：PC 主节点 `QuorumCollector` + `QLH_CONTROL_VOTER_PEERS` 指向 Y700，验证三机出证

## 8. 边界：Android 只投票、不竞选

可行，且是推荐形态。`install_auto_role_controller` 仅在 `QLH_NODE_ROLE=auto` 时挂载 `AutoRoleController`（`src/cluster_auto_role_runtime.py:629-632`）；远程对端只被当作 `HTTPRemoteVoter` 调用（`src/cluster_auto_role_runtime.py:545-554`），不需要对端运行 controller。

因此 Android 只需实现 §2.1 的六个 op，role 保持 worker／client，不参与 leader 竞争。这正是把 voter 与 candidate 分离的正确形态，也避免了同机第二实例与 master 争 term 的问题。

配置侧要求：`QLH_CONTROL_VOTER_SET` 须包含 Android 的 voter 身份与公钥；两台 PC 的 `QLH_CONTROL_VOTER_PEERS` 加入 Android endpoint；voter 总数须 ≥ 3（`src/cluster_quorum.py:89`），且 `peers + 1 >= quorum_size`（`src/cluster_auto_role_runtime.py:553`）。

## 9. 风险

| 风险 | 等级 | 说明与对策 |
| --- | --- | --- |
| 平台 Ed25519 是否接受 raw seed | **高** | 唯一阻塞性未知；先做最小探针实测，不接受则包装 PKCS#8 或回退 BouncyCastle |
| canonical JSON 逐字节一致 | **高** | Gson 不保证键序与数字格式；必须自写序列化器 + Python golden vectors 验证 |
| Keystore 不导出私钥 vs 需 raw 签名 | **高** | 用 Keystore AES-GCM 包裹 raw seed，签名时解密到内存（复用 `ClusterCredentialStore` 模式） |
| 后台存活 | 中 | Doze／厂商省电会杀进程；前台 Service + 电池豁免（参照 `AndroidPresenceService`） |
| RPC 超时预算 | 中 | 三 voter 时预算约数百毫秒，server 必须低延迟 |
| 时钟漂移 | 中 | lease 判定用接收端本地时钟；Y700 与 PC 墙钟漂移需 NTP 或容忍窗 |
| SQLite 并发与耐久 | 中 | voter ledger 用独立 SQLite 文件（对应 Python 的 `quorum-<voter_id>.sqlite3`），不与 Room 混用 |
| secret 一致性 | 低 | RPC secret 可复用 cluster secret，但两端须一致 |

## 10. 与其他能力的关系

voter 配置与静态 master 互斥：一旦 `QLH_CONTROL_VOTER_SET` 存在，`ControlFence.from_environment` 强制 `required = True`（`src/cluster_fence.py:163-176`），静态 master 没有 controller 证书，所有 control write 会被拒（实测表现为 `fenced control message has no current certificate`）。

因此两种拓扑必须显式切换，不能靠"配了不用"共存：自动选举需 `QLH_NODE_ROLE=auto` + 完整 voter 配置；静态 master／client 需移除 voter set。且 `QLH_NODE_ROLE=auto` 只能经进程环境变量设置 —— 写 `.env` 会被 `node_config.json` 的 role 覆盖（`src/qlh_core/adapters/settings.py:315` 的合并顺序为 `.env` → node_config defaults → `os.environ`）。
