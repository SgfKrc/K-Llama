# 阶段化 graph 切换的接口设计
> 面向 v3 的 prefill/decode 设备分配 · 基于现有 sched 与 op_offload · 产出最小改动面

> 状态：**方案**
> 更新日期：2026-11-14
> 适用范围：v3 阶段切换的实现者。只覆盖单进程内的设备分配；分段链只给边界（§6 第 5 条），跨进程协议不在本文。
> 待定项：见 §6 的 6 条，其中第 1 条（`ggml-vulkan-buffers.cpp:382` 的 import 条件）决定本文全部收益的前提是否成立。

证据分级：**[代码]** 读自 `H:\llamacpp-c` 的 `slimarc/v2.2`（`d43316a0d`），行号以该树为准；**[文档]** 转述项目文档口径；**[推断]** 未实测，算式写在旁边。

## 1 现状：为什么现在做不到

### 1.1 一个 sched，两种图，缺的是「形状 → 设备」

[代码] `llama_context` 只有一个 `sched`（`llama-context.h:707`），backend 列表在构造时由模型的设备集合固定（`llama-context.cpp:564-584`），`sched_new()` 在 `:794` 一次建好。

[代码] 但 sched 已经服务两种图：`sched_reserve()` 先 reserve pp 图（`:869`，`n_tokens = min(n_ctx, n_ubatch)`）、再 reserve tg 图（`:901`，`n_tokens = n_seqs`）、最后再 reserve 一次 pp（`:926`）。启动日志把两者并列打印（`:955-961`：「graph (pp bs=512, tg bs=1): nodes = … / …」）。

**「同一 sched 支持 prefill 与 decode 两种形状」已经成立。缺的不是形状，是「这个形状该给谁算」。** 今天这个映射由两处隐式决定：权重 buffer 的归属（`ggml-backend.cpp:1384`）与 `op_offload` 的 per-op 阈值（`:1387`）。

[文档] 先排除一个同名不同义的混淆：v2.2 的「prefill/decode 分离」指**同一 batch 内一个序列在 prefill、另一个在 decode**（`V22-MULTISEQ.md` §1.3），不是按阶段分配设备。本文谈的是后者。

### 1.2 op_offload 是单向逃逸，被 buft 二分卡死

[代码] `ggml-backend.cpp:1387` 的三个合取项，缺一不进：

```c
if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer))
```

1. `op_offload` 打开；2. 权重当前落在最低优先级 backend（约定是 CPU，`:1363`）；3. 权重 buffer 是 host 类型。

第 3 项是硬门。`ggml_backend_buffer_is_host` 查 buft 的 `is_host`（`:221-227`），而 Vulkan 的 device buft 把 `is_host` 设为 `NULL`（`ggml-vulkan-buffers.cpp:36`）。所以权重在 `Vulkan0` 时**这段代码不执行**——不是判为不该 offload，是分支不可达。方向只有一个：host → GPU。

### 1.3 「数据不动」在 buft 二分下不成立

[代码] 两侧只认自己那一边：

| backend | `supports_buft` | 位置 |
|---|---|---|
| CPU | `is_host(buft) \|\| extra` | `ggml-cpu.cpp:490-493` |
| Vulkan | `get_name == 自己的 device buft 名` | `ggml-vulkan.cpp:16628-16636` |

Vulkan 连**自己的** host buft 都不认：`Vulkan_Host` 的 `get_name` 是 `ggml_backend_vk_host_buffer_type_name`（`:13696`），比较对象是 device buft 的 name（`:13648`），两个函数指针不同 → `false`。

[代码] 后果在 `split_graph` 里：`src` 与 split 的 backend 不同且 buft 不受支持 → 建 copy（`ggml-backend.cpp:1792-1818`），`copy_input` 每个 split 拷一次（`:2097-2147`）。UMA 上「同一块物理内存」在 ggml 眼里是两块，**每次切换都要拷**。

[代码] KV cache 走同一条规则，且归属与 sched 无关：`llama-kv-cache.cpp:214-221`，`offload_kqv` 为真时 `buft = dev_buffer_type(model.dev_layer(il))`，否则 CPU buffer。`llama-graph.cpp:2697` 的注释把后果写死了：「the device of the cache must compute the matmul, else the scheduler copies the cache to another backend」。

### 1.4 零拷贝的门不是 import，是权重归属（2026-11-14 修正）

> **本节原先把门认错了。** E0 实测：改 `ggml-vulkan-buffers.cpp:382` 的 `!multi_device` **零差异**
> （heap 峰值 1.32 → 1.320 GiB），import 分支**一次都没执行**。原因：Iris Xe 上 `ggml_vk_host_malloc`
> 走非 portable 分支，分配时就把 `views[dev0]` 填好了，所以那条 import 分支**永远不可达**——
> `multi_device` 在不在都一样，它是**死代码路径**。

[代码] 真正的门是 `src/llama-model-loader.cpp:1472`：

```cpp
if (buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev) &&
        (use_mmap || ggml_backend_dev_type(buft_dev) == GGML_BACKEND_DEVICE_TYPE_IGPU)) {
    buft = ggml_backend_dev_buffer_type(cpu_dev);   // iGPU 上无条件回退
}
```

权重本来要进 `Vulkan_Host`，被这行强制退回 CPU buft。日志证据：
`tensor 'token_embd.weight' (q4_K) (and 426 others) cannot be used with preferred buffer type Vulkan_Host, using CPU instead`。

[实测] **临时打开 1472**（配 `-lm none`）：

| 配置 | pp512 | heap 峰值 |
|---|---|---|
| 改前（权重 CPU buft） | 74.04 | 1.32 GiB |
| 1472 打开（9B） | **89.15 ± 4.77（+20%）** | 5.61 GiB |
| **1472 打开（35B-A3B Q2_K）** | **pp64 21.78（改前 13.95，+56%）** | **12.84 GiB（< 15.82，装得下）** |

> **[2026-11-15 修正] 上表在 HEAD 上不复现，而且它的「改前」是病态基线。** `e3fa282d5` 之后实测：
> `host` 92.829 vs `auto` 93.184（9B，**噪声**）、`device` 93.856（+1.1%）；35B `host` 23.806 vs
> `auto` 23.677（**噪声**）、`device` pp512 71.701（+2.8%）。原因是 HEAD 自然配置下只有
> **545 MiB（9B）/ 515 MiB（35B）** 的张量首选 host buft（device buft 已接受 90–96%），策略最多只能挪这点。
>
> **E0′ 的「改前」是「全部权重掉进 CPU buft」的病态状态**——用 `-ot '.*=CPU' -ngl 99` 在 HEAD 上重建，
> 得 pp512 86.97 / tg128 **1.186**（对上 E0′ 的 74.04 / 1.57）；那个 1.x t/s 是「每 token 把 5.28 GiB
> 权重搬上设备」的签名，不是 decode 该有的数。而且 **HEAD 的 93.18 已经高于 E0′ 改动后的 89.15**——
> `e3fa282d5` 已经通过 UMA 读回 / MMVQ / op_offload 64 / host buffer 系统内存把收益从另一条路拿到了。
>
> **所以「权重放 Vulkan_Host」不是性能杠杆。** 它作为**策略**仍然交付（`QLH_WEIGHT_BUFT=auto|host|device`，
> `c3315ba48`），因为「谁决定权重归属」应该显式、可回退、非法值报错；但它不加速。

**所以 v3 的岔路口不是「import 开不开」，是「权重放哪」**：放 `Vulkan_Host` → 权重一份、
CPU 与 iGPU 都能读，代价是整份计入 device heap（35B 仍装得下，余 3 GiB）。放 CPU buft →
保持今天的行为。**但这条岔路口不产生速度差异**——它只决定谁有最终发言权。

### 1.5 v2.2 的阈值是算术强度的离散化

[文档] `V3-IGPU-ROLE.md` §7 称阈值 64 为「退化版」——语义上成立（per-op 而非 per-stage），物理上需修正。

[代码] 判据是 `ggml_vk_get_op_batch_size(op) >= op_offload_min_batch_size`（`ggml-vulkan.cpp:16639-16643`），m 取 `MUL_MAT` 的 `ne[1]` / `MUL_MAT_ID`·`ROPE` 的 `ne[2]` / 其余 `ggml_nrows`；默认 iGPU 64、dGPU 32，`GGML_OP_OFFLOAD_MIN_BATCH` 可覆盖。而 `V3-IGPU-ROLE.md` §2 的平衡点是 111 FLOP/byte、折合 **m ≈ 31**——**64 是这个物理判据的保守离散化**（`:5241-5244`：pp32 核显慢 26%、pp64 快 28%）。

**推论：「阶段 = 全给 GPU」在物理上是错的**——prefill 的 tail chunk（m < 32）按算术强度该留 CPU。阶段只能做**准入**，不能做**指派**。这决定了 §3 的形态与 §2 里 D 的取舍。

## 2 四种切法

| | 切换粒度 | 切换代价 | KV 怎么共享 | 权重归属 | 改动面 |
|---|---|---|---|---|---|
| **A 双 sched** | 换指针 | 零分配，但 2× compute buffer | **伪问题**：KV 属于 memory 模块，两个 sched 都能引用 | 不变（model 持有） | 400–700 行（推断） |
| **B 每阶段 reserve** | 每 ubatch | `synchronize` 全部 backend + `gallocr_reserve_n` | 不涉及 | 不变 | ~30 行，但每 token 一次 |
| **C 阶段门控** | 每 ubatch | 一次字段写 + 一次 gallocr 布局重算 | 不涉及 | 不变 | 70–120 行 |
| **D 钉全图** | 每 ubatch | 遍历 n_nodes 调 setter | 不涉及 | 不变 | ~40 行，纯 llama 侧 |

**A 的「KV 怎么共享」是伪问题**：[代码] KV buffer 由 memory 模块持有（`llama-kv-cache.cpp:214-221`），不属于任何 sched，两个 sched 都能通过 `backend_from_buffer`（`ggml-backend.cpp:1305-1325`）解析到它。A 的真代价是两份 compute buffer、两套 `gf_res_prev`（`:827`，换 sched 即失效）、`resolve_fused_ops` 各判一次、`moe_cache` 绑 `backend_ptrs`（`:625`）、`graph_params.sched` 跟着换。

**A 也不解决 §1.3**：权重 buffer 属于 model，不属于 sched，两个 sched 不能让同一份权重「两边都可读」。**B 出局**：`sched_reserve`（`:2430`）先 `synchronize`（`:2434`）再 `gallocr_reserve_n`，若参数变（`can_reuse` 含 `op_offload`）还会 `sched_new` 重建、丢掉全部 buffer——每 token 一次不可接受。

**D 是现成 API**：`ggml_backend_sched_set_tensor_backend`（`:2555-2562`），`split_graph` 的 pass 1 不覆盖用户赋值，`llama-graph.cpp:2716`、`:3221` 已在用。但它是**指派**：钉死全部 node 会让 prefill tail 违反 §1.5，且钉到不支持的 backend 直接崩。只适合当过渡验证。

## 3 推荐方案与理由（E0/E1 之后修订）

**E1 已证明 C 的性能增量是零，E0 已证明真正的杠杆是权重归属。所以顺序反过来：先做「权重放 `Vulkan_Host`」的策略化，C 留作「可声明」的第二层。**

判据（按权重排序）：

1. **切换频率**：阶段边界是每 ubatch ≤ 1 次（continuous batching 下 prefill chunk 与 decode 交错）。只有零分配切换可行 → B 出局。
2. **reserve 代价**：C 复用 `sched_reserve` 的既有预留（pp/tg 都预留过），不新增 malloc。阶段变化使 `backend_ids_changed`（`ggml-backend.cpp:2032-2046`）→ `gallocr_reserve_n`（`:2078`），但 `ggml-alloc.c:895-930` 只在 `new_chunk_size > cur_chunk_size` 时 malloc，否则只重算布局。**[推断]** 成本 = O(n_nodes) 布局重算 + 一次全 backend 同步，量级 10²–10³ µs；未实测，见 §7 G3。
3. **KV**：四种切法都不改 KV 归属（§1.3）。真约束是：**decode 用 CPU ⇒ KV 必须在 CPU 可读池**（`-ngl 0` 或 `--no-kv-offload`）。KV 在 `Vulkan0` 时 CPU 后端 `supports_buft` 为假，每 token 要拷整个 KV——**代价是 12%（§6 第 6 条已重算），不是「走不通」**。
4. **权重**：四种切法都不改权重归属。真约束是 §1.4 的岔路口，**与切法正交，且它才是性能杠杆**。
5. **改动面**：C ~63 行 < D ~40 行 < A 400–700 行。D 更小，但语义是错的（§1.5）。

**C 的价值是「可声明、可观测、可回退」，不是性能**——今天这个行为由硬件默认值与 env 隐式决定，`qlh-split-defaults.h:20` 还要在分段链上把它关掉（`server-context.cpp:1330-1340`），日志里只有一行 debug。

**所以 v3 的第一刀是 §1.4 的权重归属策略**：`llama-model-loader.cpp:1472` 改成可声明（`auto` / `host` / `device`），默认行为不变，收益 9B +20%、35B +56%。它不改 sched、不改协议、不改分段链，是「可声明」最基础的落点。

## 4 接口草案

```c
// ggml/include/ggml-backend.h
enum ggml_backend_sched_stage {
    GGML_SCHED_STAGE_ANY     = 0,  // 不做门控，行为与今天相同（默认）
    GGML_SCHED_STAGE_PREFILL = 1,  // 允许 offload 到高优先级 backend
    GGML_SCHED_STAGE_DECODE  = 2,  // 禁止 offload，全部按 buffer 归属
};

// 只影响 op_offload 分支，不改变 per-op 的 m 阈值
GGML_API void ggml_backend_sched_set_stage(ggml_backend_sched_t sched, enum ggml_backend_sched_stage stage);
GGML_API enum ggml_backend_sched_stage ggml_backend_sched_get_stage(ggml_backend_sched_t sched);

// 可选：阶段 → backend 准入掩码。为「decode 用独显」留口，v3 不用
GGML_API void ggml_backend_sched_set_stage_mask(ggml_backend_sched_t sched,
        enum ggml_backend_sched_stage stage, const bool * allowed);
```

sched 结构加一个字段（`ggml-backend.cpp:1239` 附近，紧挨 `op_offload`），`:1387` 变成：

```c
if (sched->op_offload && sched->stage != GGML_SCHED_STAGE_DECODE &&
    src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
```

llama 侧（`llama-context.h` / `llama-context.cpp`）：

```c
enum llama_stage { LLAMA_STAGE_PREFILL, LLAMA_STAGE_DECODE, LLAMA_STAGE_MIXED };

// n_tokens > n_seqs 即有 prefill 行；投机验证也是多行（k+1），会判成 PREFILL，
// 但那时 m = k+1 < 64，阈值仍把它挡在 CPU —— 误判的代价是零（见 §3 判据 5）
static llama_stage ubatch_stage(const llama_ubatch & ubatch) {
    if (ubatch.n_tokens > ubatch.n_seqs) {
        return ubatch.n_seq_tokens > 1 && ubatch.equal_seqs() ? LLAMA_STAGE_PREFILL : LLAMA_STAGE_MIXED;
    }
    return LLAMA_STAGE_DECODE;
}

void llama_context::sched_set_stage(llama_stage stage) {
    const auto s = stage == LLAMA_STAGE_DECODE ? GGML_SCHED_STAGE_DECODE : GGML_SCHED_STAGE_PREFILL;
    if (s != sched_stage_cur) {                    // 只在变化时写，避免每 ubatch 触发 gallocr
        ggml_backend_sched_set_stage(sched.get(), s);
        sched_stage_cur = s;
    }
}
```

状态机只有三态，转移在 `process_ubatch` 入口（`llama-context.cpp:1924`，`alloc_graph` 的 `:1992` 之前）：

```
          ┌──────────┐  n_tokens == n_seqs   ┌──────────┐
   start ─▶ PREFILL  │─────────────────────▶│ DECODE   │
          │          │◀─────────────────────│          │
          └──────────┘  有 prefill 行        └──────────┘
                ▲                                 │
                └──────── MIXED（按 PREFILL 准入）─┘
```

**MIXED 按 PREFILL 处理**（允许 offload），因为单 graph 内无法按行分离，保守取「不降低 prefill 收益」；代价是 decode 那几行陪跑，与今天相同（`V3-IGPU-ROLE.md` §7 的 +51% 实测就是这件事）。

## 5 改动面

| 文件 | 位置 | 动作 | 行数 |
|---|---|---|---|
| `ggml/include/ggml-backend.h` | `:353` 附近 | 加 `enum` 与三个 API 声明 | +10 |
| `ggml/src/ggml-backend.cpp` | `:1239` | sched 加 `stage` 字段 | +2 |
| 同上 | `:1387` | 条件加 `stage != DECODE` | +1 |
| 同上 | `:2365` | `sched_new` 初始化 `stage = ANY` | +1 |
| 同上 | `:2562` 后 | 三个 setter/getter 实现 | +20 |
| `src/llama-context.h` | `:707-711` | 加 `sched_stage_cur`；`sched_params` **不加** stage（避免 `can_reuse` 触发 sched 重建） | +4 |
| `src/llama-context.cpp` | `:1924-1944` | `process_ubatch` 入口判定并 `sched_set_stage` | +12 |
| 同上 | `:4610-4639` | `graph_compute` 的阶段日志（可选，一行 INFO） | +5 |
| `tools/server/server-context.cpp` | `:1330-1340` | 分段链默认关闭的日志里补一句「阶段门控已启用」 | +4 |
| `src/qlh-split-defaults.h` | `:20` 后 | `QLH_SPLIT_DEFAULT_NO_OP_OFFLOAD` 的注释补阶段门控说明 | +4 |

合计 **约 63 行新增**（推断 ±20）。不动 `llama-kv-cache.cpp`、不动协议、不动 `split-stage`。

## 6 风险与未解问题

1. **「数据不动」尚未成立**（最高优先级）。§1.4 的 import 通道在单 iGPU 上被 `multi_device` 关掉；启用后权重计入 device heap（`#31538` 的 +276 MiB 记账是实测的），9B Q4 的 5 GB 装得下 15.82 GiB，35B-A3B 的 15.82 GiB 卡在边界。**在 E0 之前，本文的一切收益都以「每 split 拷贝」为前提。**
2. **阶段判据的语义**：`n_tokens > n_seqs` 把投机验证（m = k+1）判成 PREFILL。因为 m < 64 阈值仍挡，代价为零；但若将来调低阈值到 32，k ≥ 31 的投机就会误上 GPU。
3. **混合批无法在单 graph 内分离**：一个 ubatch 里 prefill 行与 decode 行共用同一个 `MUL_MAT`，设备只能选一个。要分离必须拆成两个 graph 各 alloc/compute，那是另一个量级的改动（v3.1）。
4. **`gallocr_reserve_n` 的实际成本未实测**：`ggml-alloc.c:895-930` 的 realloc 条件是逐 chunk 比大小，若 prefill 的 GPU buffer 需求大于 decode，来回切会反复 reserve（不 malloc，但每次同步全部 backend）。
5. **分段链**：head 与 stage 是两个进程、两个 `llama_context`、两个 sched，各自按本进程的 ubatch 判定阶段，**不需要协议字段**（同一帧的 `n_tokens` 两端相同）。但若 stage 在无 iGPU 的机器上，它的 PREFILL 准入等于全 CPU，两端日志的 stage 相同而设备不同——排查时要看每端的 split 数，不能只看 stage。
6. **`offload_kqv` 与阶段门控冲突，但代价被高估了**：KV 在 `Vulkan0` 时 decode 阶段被门控到 CPU，会每 token 拷整个 KV。按 `-c 4096`、`head_count_kv=2`、`key_length=value_length=256`、40 层算，每 token 的 KV 是 **2 KB × 4096 × 40 ≈ 336 MB**，按 20 GB/s 是 **17 ms**，对照 decode 的 140 ms 是 **+12%**——不是原文说的「数 GB / 数十至数百 ms」（那是按 `n_ctx = 262144` 算的，实际不会开那么大）。**所以这不是「物理上走不通」，是 12% 的代价**，长 prompt 场景下 prefill 的 2.51× 远大于它。启动校验仍值得做，但判据应改成「按 context 长度估算拷贝占比，超过阈值才警告」。

## 7 验证方案

**E1（零改动，先做）**：`-ngl 0 --op-offload` 的端到端基线。判据：pp512 / tg128 与 `T(N,M) = N/pp + M/tg` 的线性模型误差 < 5%，且 pp 接近核显全量、tg 接近 CPU 全量（两个数见 `V3-IGPU-ROLE.md` §3：93.78 / 7.26 t/s）。不成立就说明「阶段混合」今天不可用，C 的排期要提前。

**E0（一行改动 + 实测）**：临时去掉 `ggml-vulkan-buffers.cpp:382` 的 `!multi_device` 条件，测三件事：GPU 读 import 内存的带宽（对照 24.8 GB/s 的 device buffer）、CPU 读同一块的带宽（对照 `V3-IGPU-ROLE.md` §6.3 的 12.3 GB/s）、`heapUsage` 的增量（对照 276 MiB 的口径）。E0 通过才谈「数据不动」。

**G1–G7（C 落地后；编号与版本号 v3 无关）**：

| # | 判据 | 怎么判 |
|---|---|---|
| G1 | 门控不改变正确性 | `temp 0` 固定 prompt/seed，门控关/开两跑输出逐字节相同。**基线必须同配置**（跨配置比对不成立） |
| G2 | 阶段可观测且确实生效 | 每 ubatch 一行 INFO：`stage`、`n_tokens/n_seqs`、`n_splits`。prefill 的 split 含 GPU、decode 的 `n_splits == 1`，且无 `#Vulkan0#` 命名的 copy 张量 |
| G3 | 切换成本 | 给 `sched_graph_realloc` 加计时，`gallocr_reserve_n` < 1 ms/次；否则改为阶段变化时置 `sched_need_reserve` 的慢路径 |
| G4 | 前置条件被拦住 | `--offload-kqv` + 阶段门控时启动即拒或强 warn，消息点名 `-ngl 0` / `--no-kv-offload` |
