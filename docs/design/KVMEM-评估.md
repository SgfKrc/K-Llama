# KVMem × 分段链：适配评估与硬前提验证

> 状态：**已探明**（硬前提通过，完整方案未投入）
> 更新日期：2026-11-12
> 适用范围：评估把 [KVMem](https://github.com/kvmem/kvmem) 的 KV 虚拟化接进本项目的分段链是否可行、
> 成本多大、值不值得投入。读者是决定要不要开这条线的维护者。
> 相关：[`ARCHITECTURE.md`](ARCHITECTURE.md)（分段链的 memory 层）、[`../limits/MEMORY-LIMITS.md`](../limits/MEMORY-LIMITS.md)

## 1. 结论

**有条件能，硬前提已验证通过，完整方案未投入。**

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

## 9. 复现入口

验证副本在 `.qlh_tmp/kvmem-build/`（未入库，约 2.4 GB）：

```
kvmem-root/build/bin/llama-kvmem-server.exe \
  -m H:\qlh_models\qwen3-6-35b-a3b-mtp-official\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -c 100000 --kvmem --kvmem-budget 32768 --kvmem-gen-reserve 8192 \
  --host 127.0.0.1 --port 9199 -ngl 0
```

证据日志：`kvmem-build/EVIDENCE.md`、`server.log`、`server2.log`。
