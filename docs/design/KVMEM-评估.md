# KVMem × 分段链：适配评估与硬前提验证

> 状态：**已集成**（adapter 三条修复 + 集成分支 + 四条验收，2026-11-12）
> 更新日期：2026-11-12
> 适用范围：评估把 [KVMem](https://github.com/kvmem/kvmem) 的 KV 虚拟化接进本项目的分段链是否可行、
> 成本多大、值不值得投入。读者是决定要不要开这条线的维护者。
> 相关：[`ARCHITECTURE.md`](ARCHITECTURE.md)（分段链的 memory 层）、[`../limits/MEMORY-LIMITS.md`](../limits/MEMORY-LIMITS.md)

## 1. 结论

**能，且已经做完**：adapter 修了三处、集成分支 rebase 到新主线、四条验收全过。见 §11 的收尾记录。

接入点是**嵌套关系而非竞争关系**：KVMem 在 `llama_model::create_memory` 换掉整个 memory 实现，
我们的 `llama_memory_split` 在 `llama_context::split_attach*` 把该 memory 包成 head。`llama_memory_split`
对 head 只走 `llama_memory_i` 接口（全文件无 `llama_kv_cache` / `dynamic_cast` 之类具体类型假设），
所以「head 换成 kvmem」不需要改 split 的接口。

```
llama_memory_split
  ├── head  = llama_memory_kvmem   ← KVMem 的 hook 在这里生效
  └── tails = 各 stage 的 memory
```

## 2. 硬前提验证（2026-11-12，纯 CPU `-ngl 0`）

问题：**qwen35moe 35B-A3B 能不能在 llama.cpp KVMem 的 hybrid adapter 上跑？** KVMem 的 README 支持表
把 qwen35moe 归给 PolyStrata 后端，llama.cpp 后端只列 Qwen3.5 Dense / 3.6–3.8 27B / Bonsai 27B——
**此前零资格记录**。

四件事全过：

| 项 | 结果 | 证据 |
| --- | --- | --- |
| 构造 `llama_memory_kvmem_hybrid` | ✅ | `KVMem hybrid (attn=slot-pool recr=stock) n_rs_seq=0`；`KVMem slot-pool cells=40960 slots=1280 block_tokens=32 budget=32768 gen_reserve=8192` |
| decode 逐 token | ✅ | 6.2–9.8 tok/s（首次 3.69 含预热） |
| 长 prompt（1943 token prefill） | ✅ | 56.51 tok/s，不崩 |
| **生成超 `gen_reserve`**（9000 > 8192） | ✅ | 完整生成 9000 token，无 `no free GPU slot`、无 error |

第 4 条是 `architecture.md` 与 README 的矛盾点（v1 说「一次生成不能超过 gen_reserve」，0.19 说改成了
可复用工作区），实测在纯 CPU hybrid 下确实没有触发——**agent 的长 thinking 在 KVMem 下能跑**。

## 3. 一个路线图必须修正的事实

**`llama-server` 没有 `--kvmem` 参数。** KVMem 的 `backends/patches/llamacpp-kvmem.patch` 只加 adapter，
不改 `common/arg.cpp`，KVMem 默认 `enabled=false`。官方入口是 kvmem 仓库自己的
**`llama-kvmem-server`**（`tools/llama-kvmem-server.cpp`）。

所以「集成到我们的 `llama-server`」不是加个参数，而是**把 KVMem 的 CLI/驱动层接进我们的 fork**。
这是成本估算里最容易被低估的一块。

## 4. patch 合并的实际规模

`backends/patches/llamacpp-kvmem.patch`（50 文件 / 125 KB）打在 `09338b85e` 的 LF 干净树上：

- **42 个文件干净应用**
- **8 个文件各 1 个 hunk 失败**，全部手工合：
  - 我们 fork 导致 3 处：`common/speculative.cpp`（MTP 共享判据，fork 已自带等价实现，**hunk 全跳过**但需补 2 个成员声明）、`src/llama-context.{cpp,h}`（**只是 include 区**）
  - 上游 4 天漂移 5 处：`ggml/src/ggml-backend.cpp`（sched 结构）、`src/CMakeLists.txt`（link 位置）、`src/models/delta-net-base.cpp`、`qwen3.cpp`、`qwen35.cpp`
- **核心 hook 干净通过**：`src/llama-model.cpp` 的 `create_memory` hunk
- 额外需要 `backends/patches/llamacpp-checkpoint-policy.patch`（`llama-kvmem-server` 的 driver 依赖它的 `common_speculative_state_size`）

**CRLF 陷阱**：在 `H:\llamacpp-c` 的工作树（CRLF 检出）上 `git apply --check` 会 46/50 失败，那是
`git apply` 不做行尾规范化造成的假象，不是代码冲突。必须在 `git archive HEAD` 的 LF 副本上做。

## 5. 未验证的两块边界

| 边界 | 为什么重要 |
| --- | --- |
| **GPU offload 路径** | 全程 `-ngl 0`（35B 权重 21.11 GiB > 8 GB 显存，最稳跑法）。本项目场景是**段级 offload**——head 的几层上卡，KVMem 的 slot-pool 在 GPU 常驻时能否工作，未测 |
| **MTP draft** | 全程 `spec=none`。本项目投机依赖 MTP，而 KVMem 的 `use_gdn_replay` 判据只认 dense `LLM_ARCH_QWEN35`，qwen35moe 不满足——这是下一个风险点 |

## 6. 语义冲突（即使硬前提过了也要解决）

| 冲突 | 后果 |
| --- | --- |
| **checkpoint 缺历史** | KVMem 的 `state_write` 只委托 `kv_->state_write`（GPU 池），不含 host store；而 split 的 chain checkpoint 依赖 head 的 state 覆盖整条序列 |
| **server 的 `pos_min`/`pos_max` 有洞** | KVMem 池里 pos 有洞（sink + recent + 召回块），而 `server-context.cpp` 用 `seq_pos_min/max` 做复用与截断判断 |
| **跨机 query span 不同步** | KVMem 的 retrieval query 是最后 user span（默认 512 token），只有 head 的 server 知道；远程 tail 拿不到就退化 recency |
| **判据要重建** | 现有正确性判据是「与整模逐 token 一致」，而 KVMem 是**近似选块**，byte-identical 判据必须换掉 |

## 7. 若投入，三步走（每步有放弃条件）

1. **接 CLI + 本地链**（1–2 天）：把 KVMem 的驱动层接进 `llama-server`，跑 head+tail 同进程，验证
   split 的 `seq_rm`/`pos_min` 在「池里 pos 有洞」下还准不准、checkpoint 还有没有意义。**不过就停**。
2. **部分 GPU offload**（3–5 天）：`-ngl 4/8` 下验证 slot-pool。不过就退回纯 CPU 版。
3. **MTP + KVMem**（3–5 天）：不过就接受「KVMem 只用于非投机场景」。

总计 **1–2 周**（不含 35B 调优）。

## 8. 收益（来源均为 27B dense，不是我们的 35B-A3B）

- 27B IQ3_S / RTX 5060 Ti 16 GB：原生 `ctx=53248` → KVMem 逻辑 `ctx=131072`，decode −3.05%
  （36.78 vs 37.94 tok/s）——`kvmem/docs/llamacpp-v070-integration.md`
- 27B IQ4_XS / 7900 XT 20 GB：`-c 262144`，prompt 43,150 token，峰值 15,961 MiB，decode 24.79 tok/s；
  retrieval 聚合 43.1 s，stage_in 243 块 / 7776 token——`kvmem/docs/llamacpp-v090-integration.md`

**不能把 27B 的数字直接套到 35B 上**：35B 的权重（head 15.69 GiB）比 27B 大，KV 预算更小。

## 9. 验收判据（2026-11-12 修订）

**不能用「与整模逐字相等」当判据**。KVMem 是**近似选块**：`budget < prompt` 时 GPU 只驻留
`budget` 那部分行，被 stage_out 的 token **不参与 attention**，所以「答案逐字等于全量」**物理上
做不到**——要求它等于要求 KVMem 不省显存。

实测证据（2B，1802 token prompt，`temperature: 0`）：

| 对照 | 结果 |
| --- | --- |
| `budget 1024`（驱逐）vs `8192`（不驱逐） | **DIFFERENT** |
| `budget 2048`（不驱逐）vs `8192` | **EQUAL** |
| `budget 1024` 跑两次 | **EQUAL**（确定性） |

`1024` 那次的 mask 是 `keep=1002 holes=1 keep_pos=[0,1801]`——只有 1002/1802 行参与 attention。

### 用什么判据

**「关键信息保留」测试**：开头埋一个 secret code + 中间 1200 token 填充 + 结尾提问（共 1653 token），
`budget 1024` 下（`keep=1013 holes=1`、驱逐 2 次）答案与 `budget 8192` **EQUAL**。它才是在测
「驱逐有没有让模型看到错的前缀」。

三条判据，按顺序：

1. **能跑**：`budget < prompt` 时 HTTP 200（不是 500）
2. **保留的 KV 是对的**：secret-code 测试答案与不驱逐时一致
3. **确定性**：同配置跑两次输出一致

**不要**再用「逐字等于全量」——它会永远失败，且失败不说明有问题。

## 10. 复现入口

验证副本在 `.qlh_tmp/kvmem-build/`（未入库，约 2.4 GB）：

```
kvmem-root/build/bin/llama-kvmem-server.exe \
  -m H:\qlh_models\qwen3-6-35b-a3b-mtp-official\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -c 100000 --kvmem --kvmem-budget 32768 --kvmem-gen-reserve 8192 \
  --host 127.0.0.1 --port 9199 -ngl 0
```

证据日志：`kvmem-build/EVIDENCE.md`、`server.log`、`server2.log`。

## 11. 收尾记录（2026-11-12）

### adapter 的三条修复

都在 `SgfKrc/kvmem` 的 `fix/seq-pos-min-max` 分支（HEAD `461b05c`），修的是三个不同层次的问题：

| commit | 问题 | 层次 |
| --- | --- | --- |
| `f04053a` | prompt cache 完全失效、checkpoint 从不创建、有个 `GGML_ABORT` 悬着 | **契约**（`seq_pos_min/max` 只报原生 kv，而工作集在 store 里） |
| `723d424` | 双 context 下 head 的 capture 全落到 tail 上，一触发驱逐就 500 | **绑定**（`kvmem_current_execution().memory` 是进程级单例，被后建的 tail 覆盖） |
| `461b05c` | MTP follower 只在构造时 bind 一次 | **绑定**（同一类问题，另一半） |

第三条顺手加了 `KVMEM_TRACE` 门控的诊断——**没有它，「实例一致」这件事没法证明**。

### 集成分支

`SgfKrc/llama.cpp` 的 **`kvmem/integration`**（`3f930466d`，父 `4acf03c8b`），基于新上游 `09338b85e`。
rebase 时 7 处冲突全部「合并而非二选一」——那些是上游 4 天漂移，不是 KVMem 的语义错误。

**一处硬性适配**：上游把 `gf_res_prev` / `gf_res_decode` 从值 vector 换成 `std::array<llm_graph_result_ptr, N>`
（unique_ptr），KVMem 原来的 `res.reset()` 变成**置空指针**，紧接着的 `res->set_max_nodes()` 解引用空指针
→ **启动即 SIGSEGV，连 `--no-kvmem` 都崩**。修复是 `if (res) res->reset();`——reset 指向的对象，
不是指针本身。

### 验收（2B qwen35，本地双段 `0:12+12:24`，`-np 1`，`-ngl 0`）

| # | 项 | 结果 |
| --- | --- | --- |
| ① | MTP + KVMem（`budget 8192`） | 200，`draft acceptance = 0.86207 (50/58)`；head/tail 各 2 个 context、MTP follower 1 个 |
| ② | 驱逐（`budget 1024` + 3638 token，pool = 3072） | 200，`need_offload=1` ×3、`KVMEM_PRESSURE` ×4 |
| ③ | **secret-code 保留** | 200，`content: 'XK-7734-QP'`——驱逐后仍答对 |
| ④ | 确定性 | 两次输出 **byte-identical**，sha256 相同 |

**对照**：不带 `--kvmem` vs `--no-kvmem` 逐字节相同（主线没坏）；**KVMem（budget 8192，无驱逐）的
输出也是同一个 sha256** → 无压力时 KVMem 对输出完全透明。

### 两个必须跟着走的坑

1. **`-lv 3` 下 KVMem 核心日志不可见**：这个 fork 的 server 在 `-lv 3` 下过滤 llama.cpp 核心 INFO，
   只有 `srv`/`cmn` 前缀可见。验收必须 `-lv 10` 或 `KVMEM_TRACE=1`——否则会重演「以为没生效」的误判。
2. **adapter 是 UNITY_BUILD 编进 llama 的**：ninja 的 depfile **不跟踪**这些外部源——改完 adapter 直接
   `cmake --build` 会说 nothing to do、二进制不变。必须先
   `touch build-kvmem2/src/CMakeFiles/llama.dir/Unity/unity_11_cxx.cxx` 再编。

### 明确不做的

- **多序列**：实测 `-np 2` 端到端是单序列的 **0.81×**（纯 decode 重叠有 1.12×，但并发 prefill 只有
  0.70×，而真实请求的 prefill 会同时到达）。见 `ARCHITECTURE.md` §11。
- **异步流水线**：单序列 token 间的依赖链是硬的，可重叠比例 = 0；prefill 块间在当前 24/8 切分下
  `T_pipe = 34.5 s` > 串行 32.1 s。
- **本地链 `-np > 1` + KVMem**：`kvmem_current_execution()` 是进程级单例，多 slot 会再次错位
  （`server_context::spec` 是单例，`server_slot` 只持指针，当前靠「一进程一个 draft context」救）。

### 未验证

- **35B + KVMem**：全程 2B。35B 的硬前提（`qwen35moe` 能构造 hybrid adapter）在纯 CPU `-ngl 0` 下
  验证过，但**没在 35B 上跑过驱逐与 secret-code**。
- **remote tail + MTP**：MTP 在 head 侧、tail 是独立进程的情形没测。
- **GPU offload**：全程 `-ngl 0`。KVMem 的 slot-pool 在 GPU 常驻时能否工作未验证。
