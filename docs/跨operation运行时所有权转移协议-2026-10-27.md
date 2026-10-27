# 跨 operation 的运行时所有权转移协议

> 状态：**设计稿（未实现）**
> 更新日期：2026-10-27
>
> 适用范围：worker 侧推理运行时（`llama_model` / `llama_context`）在 operation 更替时的
> 复用判定与所有权交接。处理 #127 的「unload → reselect 后仍需重载 native」一段耗时。
> 本文只定义协议与不变量，不含实现代码；引擎参数、量化配方、段切分不在本文范围。

---

## 1. 现状与问题

### 1.1 复用键把两类身份压进同一个元组

`src/inference_service/engine_host.py` 的层段上游复用判据：

```python
binding = (executor_id, prepare_identity, dict(identity), tuple(span), model_path, version)
if (binding != getattr(self, "_layer_upstream_binding", None)
        or self._layer_upstream is None or not self._layer_upstream.is_available):
    ...校验 manifest、digest、version...
if (... and binding == self._layer_upstream_binding):
    return self._layer_upstream
self._layer_upstream.close()
```

而 `prepare_identity` 的字段集合（`src/worker_prepare_contract.py` 的 `IDENTITY_FIELDS`）
同时包含两类语义：

| 类别 | 字段 | 语义 |
| --- | --- | --- |
| 工件身份 | `artifact_sha256`、`manifest_sha256`、`model_sha256`、`source_model_id`、`source_model_sha256`、`tokenizer_sha256`、`layer_range`、`total_layers`、`hidden_size`、`segment_mode` | 描述「这份权重是什么」，决定运行时**能否**被复用 |
| owner 身份 | `operation_id`、`assignment_id`、`generation`、`config_id`、`node_id` | 描述「谁在用它」，决定运行时**归谁** |

两者被压进同一个 `binding` 元组，于是 owner 身份一变，`binding` 就不等，必然走到
`close()` 并重新 `llama_model_load_from_file`。

### 1.2 实测代价

master#cpu 段的工件 `[8,24)` 为 2046.9 MiB。重载路径包含：全量读工件、建立
`llama_model`、repack 到约 1264 MiB 的匿名缓冲、创建 `llama_context` 与 KV。
unload → reselect 后的末段耗时实测落在 15–21 s 区间。

同一份工件在 operation 之间**没有任何字节变化**（`artifact_sha256` 与 `manifest_sha256`
相同，`file_stamp` 相同），因此这段重载在正确性上不需要发生。

---

## 2. 只比工件摘要为什么不行

把复用判据放宽为「`artifact_sha256` 相等即复用」会同时破坏两件事。

### 2.1 所有权失去唯一持有者

当前一个运行时实例只有一份，由最近一次 prepare 的 operation 隐式持有。
若新 operation 凭工件摘要直接取用，则同一实例出现两个声明持有者：

- 旧 operation 的 unload 路径仍认为它拥有该实例，会调用 `close()`；
- 新 operation 已在用它执行前向。

结果是「一方正在读、另一方关掉底层句柄」的竞态。`llama_context` 与映射区被释放后，
前向调用会落到已失效指针上，属于内存安全级错误，不是性能问题。

### 2.2 清理围栏被绕过

清理围栏的作用是：operation 结束后，阻止仍持有旧身份的迟到请求影响新状态。
围栏的判断依据是 owner 身份（`operation_id` + `generation`）。若新 operation 未经过
转移就直接取得运行时，围栏视其为**未授权持有者**：

- 新 operation 的合法调用会被围栏拒绝（身份不在白名单）；
- 或者围栏被整体放宽而失去作用，等同于取消该保护。

两条路都不可接受。安全复用的前提是**持有者身份本身发生一次被记录的转移**，
而不是让第二个持有者出现。

---

## 3. 协议定义

### 3.1 身份拆分

`prepare_identity` 在参与复用判定前，先按语义拆成两个不可变子结构：

```
artifact_identity := {
    artifact_sha256, manifest_sha256, model_sha256,
    source_model_id, source_model_sha256, tokenizer_sha256,
    layer_range, total_layers, hidden_size, segment_mode,
    artifact_path, artifact_version,          # file_stamp 派生
    executor_id,
}
owner_identity := {
    operation_id, assignment_id, generation, config_id, node_id,
}
```

- `artifact_identity` 决定运行时**内容**是否等价；
- `owner_identity` 决定运行时**归属**。

复用判据改为只比较 `artifact_identity`；`owner_identity` 的变化由转移协议处理。

### 3.2 运行时注册表

每个 worker 维护一张运行时表，每项至少包含：

| 字段 | 含义 |
| --- | --- |
| `runtime_id` | 实例标识，进程内唯一且不复用 |
| `artifact_identity` | 创建该实例时的工件身份 |
| `owner` | 当前 owner 身份 |
| `generation` | owner 代数，创建时为 1，每次转移 +1 |
| `refcount` | 活跃引用数（持有者 + 在飞调用） |
| `state` | `live` / `retiring` / `closed` |

### 3.3 转移操作

```
transfer(from_owner, to_owner, artifact_identity, expected_generation) -> Receipt | Error
```

前置条件（全部在注册表锁内检查）：

1. 存在满足 `artifact_identity` 等价的 `state == live` 实例；
2. 该实例的 `owner` 等于 `from_owner`；
3. 该实例的 `generation` 等于 `expected_generation`（CAS，防并发双重转移）。

原子步骤：

1. 旧 owner 的围栏条目标记为 `superseded`，记录 `superseded_by = to_owner`；
2. 实例 `owner := to_owner`；
3. 实例 `generation += 1`；
4. 实例 `refcount += 1`（新 owner 持有一份引用）；
5. 返回 `Receipt{runtime_id, generation, artifact_identity}`。

后置条件：

- 新 owner 可直接用该实例执行前向，**不触发**模型加载；
- 旧 owner 的任何 `close()` 调用因 `generation` 不匹配而被拒绝，返回
  `stale_owner`，不释放底层句柄；
- 旧 owner 的引用在其调用结束时递减；`refcount` 归零才进入 `closed`。

### 3.4 拒绝转移的条件

以下情形**不转移**，按现有语义走「关闭并重建」：

- `artifact_identity` 不等价（工件摘要、层区间、段类型、版本任一不同）；
- 实例处于 `retiring` 或 `closed`；
- `generation` 不匹配（并发转移或原地重建中）；
- 目标 owner 的 `load_spec` 与实例创建时不一致（`load_spec_digest` 不同）。

---

## 4. 不变量

| 编号 | 不变量 | 保障手段 |
| --- | --- | --- |
| I1 | 任一时点每个运行时实例恰好一个 owner | 转移在注册表锁内完成，`owner` 单字段替换 |
| I2 | `generation` 严格单调递增 | CAS 前置条件；转移与创建都 +1 |
| I3 | 旧 owner 的清理不释放仍被新 owner 持有的实例 | `refcount` 归零才 `closed` |
| I4 | 围栏不会长期拒绝合法 owner | `superseded` 条目在新 owner 首次成功调用后回收 |
| I5 | 转移不改变运行时内容 | 转移前后 `runtime_id`、`artifact_identity` 不变 |

I3 与 I5 是本次设计的核心：**转移只改归属，不改内容**。任何需要重读权重的情形都应
落到 3.4 的拒绝分支，而不是在转移路径上做部分刷新。

---

## 5. 与现有实现的映射

| 现有位置 | 现状 | 需要的变化 |
| --- | --- | --- |
| `engine_host.py` 的 `binding` 元组 | 含完整 `prepare_identity` | 拆为 `artifact_identity` + `owner_identity`，复用只比前者 |
| 同处的 `close()` 分支 | `binding` 不等即关 | 改为「工件不等才关；工件相等但 owner 不同则走转移」 |
| 清理围栏 | 按 owner 身份判定，无转移概念 | 增加 `superseded` 状态与 `superseded_by` 记录 |
| `worker_prepare_contract.py` 的 `IDENTITY_FIELDS` | 单一集合 | 增补两个子集的显式划分，供复用与围栏共用 |
| 未加载状态 `loaded=False` | 与是否持有运行时无强对应 | 由注册表 `owner` 与 `refcount` 表达真实持有关系 |

`loaded=False` 这处需要单独说明：pipeline 模式下 `model_host.model_loaded` 恒为 False
（`api_server.py` 的 pipeline 分支），该字段表达的是「整模是否由本进程加载」，与
「段运行时是否存活」不是同一件事。转移协议应以注册表为准，不复用该标志。

---

## 6. 验收方法

1. **单元级**：构造两次 prepare（工件相同、`operation_id` 不同），断言第二次不调用
   模型加载入口，且返回的 `runtime_id` 与第一次相同、`generation` 递增。
2. **竞态级**：并发发起两次转移到不同 owner，断言只有一次成功，另一次得到
   `generation_mismatch`。
3. **围栏级**：转移后由旧 owner 发起 `close()`，断言返回 `stale_owner` 且实例仍为
   `live`、可执行前向。
4. **负控**：工件摘要不同、层区间不同、`load_spec` 不同三种情形，断言均走重建路径。
5. **收益口径**：分别测「cold load / 同 operation 复用 / 跨 operation 转移后首次前向」
   三段耗时，与现有 15–21 s 对照。三段的计时口径应独立，不合并成总时长。

---

## 7. 未决

- 转移的**发起方**：由新 operation 主动请求，还是由注册表在 prepare 时自动匹配。
  前者需要新增一个调用点，后者需要注册表感知 prepare 的生命周期。
- `config_id` 归入 owner 还是工件：它由配置快照派生，工件相同而配置不同的情形是否
  允许复用，取决于配置是否进入前向语义。当前归入 owner。
- 多 worker 一致性：本协议是单进程内的所有权模型；跨 worker 的同类问题（同一工件在
  两台 worker 上的持有）不在本文范围。
