# DeepSeek 与社区 CPU 推理成果调研：能否参考 / 复用 / 当引擎后端（2026-10-14）

> 状态：**调研结论**
> 触发问题（用户）：「DeepSeek 因为缺卡，在 CPU 推理上有巨多优化和创新，成果大量开源，有没有什么能参考的/复用的/作为推理引擎后端的」
> 方法：两名只读子 agent（DeepSeek 官方成果 / 社区 CPU 引擎），前者用官方一手来源（deepseek-ai 全仓列表、各仓库 README、open-infra-index、模型卡、ISCA 论文），后者先消化本仓已有三份调研再查外部；结论凡带出处者均实查。

---

## 0. 三问三答

| 问题 | 结论 |
|---|---|
| **DeepSeek 官方有能当引擎后端的吗？** | ❌ **没有**。开源周那批（FlashMLA/DeepEP/DeepGEMM/DualPipe/3FS/profile-data/EPLB 及 2026 增补）**除 3FS（CPU/host 存储）与 EPLB（纯 Python 算法）外全是 GPU/NPU 向**；且 **4060（Ada/SM89）硬件不兼容**（FlashMLA 要 SM100/SM103、DeepGEMM 要 SM90/SM100、DeepEP 要 Hopper+ 与 NVLink/RDMA）。官方 44 个仓库/开源周清单/各代模型卡部署章节**都没有面向 CPU 推理的成果** |
| **社区有能满足「层段硬指标」的引擎吗？** | ❌ **没有**。2026-09-15 那句「无可直接集成者」**复核后依然有效，且上游侧在 2026-10 反而更差**。唯一能满足的仍是 **llama.cpp + 本仓补丁** |
| **那有什么能借的吗？** | ✅ **有，但不在"引擎"层，在 kernel / 量化 / 放置算法层**：ik_llama.cpp 的 IQK 与低比特 GEMM、KT 的 MoE 专家调度思想、EPLB/LPLB 的放置算法、以及"通信不占计算资源"的做法 |

---

## 1. 三条前提纠正（会改变搜索方向，先说）

### 1.1 `--cpu-moe` 不是 KTransformers 的选项

- `--cpu-moe` / `-cmoe`、`--n-cpu-moe` / `-ncmoe` 是 **llama.cpp 原生 CLI**（`android/app/src/main/cpp/llama.cpp/common/arg.cpp:2459,2466`；`--spec-draft-cpu-moe` 在 `:3675,3682`）。
- **QLH 已在用**：`src/llama_server_process.py:71` 按 `ServerSpec.cpu_moe` 追加 `--cpu-moe`；`src/inference_path_selector.py:12-13` 记录了本仓实测（35B-A3B `-ngl 99 --cpu-moe` = 15.94 tok/s = 纯 CPU 的 2.66×）。
- **KT 的真实机理**是「HF 权重 + YAML `optimize_rules` 的 match/replace 算子注入」，逐算子指定 `prefill_device`/`generate_device`，配合 kt-kernel 的 CPU MoE 算子与 SGLang 集成参数（`--kt-method`/`--kt-weight-path`/`--kt-num-gpu-experts`/…）。
- ⇒ 所以这条线**不是"待引入"，而是"已在整模服务路径上用着，但没接进层段 worker"**（层段走绑定，绑定不暴露该开关）。

### 1.2 硬指标比通常描述的更窄，也更乐观

- QLH 要的**不是"逐层 hidden"**，而是 **"段边界 hidden"** = 段末层输出（`output_norm` 之前）：`src/llama_keep_head.py:1-36` 明确两种模式都走 `layer_inp` 的 `lid == n_layer` 槽位；补丁 `scripts/model_tools/patches/llama-cpp-layer-forward-api.patch` + 9 个架构登记，未登记架构 decode 时 `GGML_ASSERT` ⇒ **fail-closed，不会静默给错值**。
- **"部分层 GGUF"这一半不是障碍**：`scripts/cut_layers.py:9-18` 的裁层语义是保留 `blk.K..blk.(N-1)` 并**重命名为** `blk.0..`、`<arch>.block_count` 改为 `N-K`，非 blk 张量原样保留 ⇒ **裁层工件就是标准 GGUF**，任何 GGUF 引擎都能当普通小模型加载。（`first_local_layer_maps_to` 只存在于 **manifest JSON**，不在 GGUF KV 里。）

⇒ **硬指标缺口 100% 在"导出段末 hidden"这一半。**

### 1.3 `GGML_NATIVE=OFF` 是**刻意选择**，不是疏忽（这是决定性权衡）

`docs/跨框架接力数值差异机理-2026-10-05.md:167` 记载：**同架构 + 同二进制 + `GGML_NATIVE=OFF` ⇒ 0/7168 差异；x86_64↔ARM64 ⇒ 7168/7168 全不同**，根因是**归约按 ISA 宽度分 lane**。

⇒ **换 ISA 必然换加法结合顺序** ⇒ 任何"上 AVX512 / 换 fork / 换 kernel"的提速方案**都要重新过「逐 token argmax 一致」门**；而该门在本仓的定位是"margin 相关的概率事件"，不是可证明性质。

---

## 2. DeepSeek 官方成果逐个定性（子 agent A）

| 项目 | 是什么 | 设备向 | 对「层流水线 + CPU 边缘段 + 跨机 TCP」的参考价值 |
|---|---|---|---|
| FlashMLA | MLA 的高效 kernel | **GPU**（要 SM100/SM103） | ❌ 与 4060（SM89）不兼容 |
| DeepEP | MoE 专家并行的 all-to-all 通信库 | **GPU**（Hopper+ / NVLink / RDMA） | ❌ 硬件不兼容；其"用互连换容量"的思路与"按层切段"正交 |
| DeepGEMM | FP8 GEMM 库 | **GPU**（SM90/SM100） | ❌ |
| DualPipe | 双向流水线 + 计算-通信重叠 | GPU 向 | ⚠️ **机制不能搬**（见 §4） |
| **3FS** | 分布式文件系统 | **CPU/host 侧** | ✅ **与装备/供给（supply）有关**，可借鉴其设计 |
| profile-data | 推理 trace 数据 | GPU 向 | ⚠️ 仅供参考 |
| **EPLB / LPLB** | 专家/负载均衡**放置算法** | **纯 Python，CPU 可跑** | ✅✅ **对"层段分配"有直接参考**（正是 `relay_cut_objective.py` 在做的事） |
| Engram / DeepSpec / TileKernels / DeepJIT / DeepSelect | 2026 增补 | GPU 向（Engram 提及 GPU **或 CPU** 上存储） | ⚠️ Engram 是唯一提到 host-memory offload 的 |

### 技术报告四项对 CPU offload 的启示

- **MLA / CSA2**：「**先压缩再跨机**」（KV 压到 890 B/token 量级）⇒ 启示是**减少跨机传输的对象大小**。
- **FP8**：官方**连通信都低调低精度**；但其加速前提是 tensor core ⇒ 对 CPU 段不直接适用。
- **专家并行（EP）**：官方选择**用互连换容量** ⇒ 可借的是 **EPLB/LPLB 的放置算法 + node-limited routing**（**CPU 上就能跑**）。
- **Prefill-Decode 分离**：按负载画像分工 ⇒ 与"按深度切段"是**正交**的两个维度。

### ⚠️ 关键一问：**没有**

官方 44 个仓库、开源周清单、各代模型卡部署章节**都没有面向 CPU 推理的成果**。唯一官方 host-memory offload 表述来自 **Engram**；最接近的"CPU 用途"是**训练侧 EMA in CPU** 与**存储侧 3FS**。

---

## 3. 社区候选逐个 + 可复用性矩阵（子 agent B）

**硬指标 = 能加载「只含部分层的 GGUF」+ 能导出「段边界 hidden」。**

| 候选 | CPU-first | MoE | ①能当层段引擎 | ②工件兼容 | ③需重训/转模型 | ④CPU 预期收益 | ⑤成本/风险 | ⑥许可 |
|---|---|---|---|---|---|---|---|---|
| **llama.cpp（本体，现行）** | ✅ | ✅ | ✅ **唯一已验证**（p ip 绑定 + 自建补丁 DLL + keep-head shim） | ✅ 本体 | ❌ | 有，但不在"换引擎"，在**构建档位/线程亲和** | 已有；升级需复核 patch hunk | MIT |
| **ik_llama.cpp** | ✅ 最强 CPU 路线之一 | ✅ | ⚠️ **现状不能**：仓库**无 `src/llama-ext.h`**（raw 404）、README 自述「last synced with upstream in **August 2024**」⇒ 无本仓依赖的 `layer_inp` 槽位；要当引擎**必须移植 QLH 补丁** | ⚠️ 部分（裁层 GGUF 应能加载，但**段末 hidden 需补丁**） | ❌（仅换引擎）；但需重验 GGUF 互认 | **潜在最高**（IQK/row-interleaved/AVX-VNNI/CPU FA） | 高（该 fork 已大改 ggml/llama 结构 + 数值判据重跑） | MIT |
| **KTransformers / kt-kernel** | ✅（CPU 算子 + GPU 注意力） | ✅ 核心场景 | ❌ **天生不符**：不是"层段前向引擎"，是 **MoE 专家算子 + SGLang/PyTorch 集成**；无 GGUF 层段前向、无 hidden 导出通道 | ❌（CPU 侧权重是 HF/专用 INT4-8 目录） | ✅ 需转 CPU 专家权重 | 对 MoE 大模型有（AMX/AVX512 档；**本机 x86 无 AMX**）；对 dense ≤2B 主线≈0 | 高（Linux-only、Py3.10-3.12、引入 runtime 依赖） | Apache-2.0 |
| **PowerInfer** | 混合 | ❌ | ❌ 天生不符（自有格式 + predictor） | ❌ | ✅✅ **需换 ReLU/ReGLU 稀疏模型** | 对特定 ReLU 模型高；对 Qwen 系≈0 | 很高 | MIT |
| **T-MAC** | ✅（ARM/Intel） | 未见 | ❌（kernel 库） | ❌ | ✅ | 低比特高；**对本仓 Q4_K_M 无对应 kernel** | 高（TVM + Python 3.8 + 自建 fork） | MIT |
| **BitNet.cpp** | ✅ | ❌ | ❌（只吃三值模型） | ❌ | ✅✅ 需三值模型 | 三值档高，与 Q4_K_M 不可比 | 中高 | MIT |
| **IPEX-LLM** | ✅ | 部分 | ❌ **已归档**（README 明写不维护、有 known security issues） | — | — | — | 不建议 | 未核实 |
| **vLLM（CPU）** | 是（服务化） | ✅ | ❌ 天生不符（**token in/token out**，不接收 hidden 边界） | ❌（CPU 量化只列 AWQ/GPTQ/INT8 W8A8，**不覆盖 GGUF k-quant**） | ✅ | 服务化吞吐向；batch=1 不是强项 | 高（违反 Edge 去 torch 纪律） | Apache-2.0 |
| **SGLang** | 非 CPU-first | ✅ | ❌ 同上 | ❌ | ✅ | 同上 | 高 | Apache-2.0 |
| **distributed-llama** | ✅ | ✅ | ❌（自有 `.m/.t` 格式 + **TP 同步**） | ❌ | ✅✅ | 跨机 CPU 聚合有意义，但**节点数必须 2ⁿ、上限=KV heads** | 高（另一套集群体系） | MIT |
| **llama.cpp RPC** | ✅ | ✅ | ⚠️ **不是层接力**（tensor-split / 设备后端语义，`-ngl`/`--tensor-split` ≠ 连续 `layer_range`） | ⚠️ 与 GGUF 兼容但**无法在 RPC 边界导出 hidden** | ❌ | 容量聚合（把弱机当 backend），非单段提速 | 中（已有 `ggml-rpc-server.exe` + `src/llama_rpc_device.py`） | MIT |

### 硬指标裁决

**能做「加载部分层 GGUF + 导出段边界 hidden」的**：
1. **llama.cpp + 本仓补丁** —— 唯一**已被本仓实测证明**（Qwen3.5-9B 两段/三段 32/32、D→L 与 L→L 均逐 token 一致）。
2. 理论上任何"能在各 decoder block 末端挂钩子并暴露 C ABI"的 GGUF 引擎 —— 即**必须自研/fork**。

**天生不符（结构上做不到，不是"还没做"）**：KTransformers/kt-kernel（算子注入层）、PowerInfer / BitNet.cpp / T-MAC（模型域或 kernel 域特化）、vLLM / SGLang（服务语义）、distributed-llama（自有格式 + TP）、llama.cpp RPC（backend 语义）。

---

## 4. DualPipe 能否搬到跨机 TCP 层流水线：**不能**（A 组）

约束：**decode 是 batch=1 单序列、无 micro-batch**；实测每段 2 次跨机往返、网络税占整步约 39%。

- **缺 micro-batch**（DualPipe 靠双向 micro-batch 对填 bubble）
- **缺 backward 的第二路**（它是训练向的双向流水）
- **参数 2×**（双份权重）
- **RTT 在关键路径**（batch=1 下无法被计算掩盖）
- **TCP 传输需要 CPU 参与，且与 CPU 段的计算争核**

官方推理侧实为 **dual-microbatch + 5-stage + RDMA 零 SM**，与"2-3 台异构机 + TCP"不是同一形态。

**可借的只有三点**（这三点本仓恰好都在做或已做）：
1. **"用独立工作流填 bubble"** ⇒ 对应 **γ 投机方案**（本仓已实测 5.43×）。
2. **"通信不占计算资源"**（独立 I/O 线程 + 二进制帧）⇒ 对应 **票 21a**；本仓已有 `QLHR` 二进制帧先例。
3. **"不增加往返次数前提下的 chunk 化"**。

---

## 5. 行动建议

| 优先级 | 动作 | 依据 |
|---|---|---|
| **P0** | **构建档位 + 线程/亲和性 A/B**（层段用的 `libllama` 建库提示是 `-DGGML_NATIVE=OFF` ⇒ AVX2 基线；Surface 的 i5-1035G4 架构支持 AVX-512/VNNI 但二进制没启用；主节点 i7-13620H 的 AVX512 被硬件屏蔽） | 纯配置/构建；但 ⚠️ **改 ISA ⇒ 必须重跑逐 token 判据**（§1.3） |
| P0 | **线程数与亲和性**：worker 默认 `n_threads=8` 而 Surface 是 4C8T；跨核域 + 超订是抖动根因 | 零代码 |
| P1 | 把 **ik_llama.cpp 当作「kernel 与量化的对照来源」**（而非引擎替换）来做收益量级判断 | IQK/低比特 GEMM 是唯一有实质 CPU kernel 增益的候选 |
| P1 | **借 EPLB/LPLB 的放置算法**（CPU 可跑）到层段分配 | 与 `relay_cut_objective.py` 联合 |
| P2 | 关注上游 **`#27112`**（public hidden-state API）作为唯一的"补丁减负"观察点 | 现状 **open + stale**；`#27073` 已由作者关闭、分支已删 ⇒ 不能当依赖 |
| ❌ | 不要为"CPU 跑大模型"另找引擎 | 没有任何候选同时满足层段硬指标 |

---

## 6. 未核实（据实标注）

1. **ik_llama.cpp 的补丁移植成本**：只核实了「`src/llama-ext.h` 不存在（raw 404）」与「README 自述 2024-08 同步上游」；**没有**逐文件确认它是否在别处有等价的 layer-input 捕获，也没试编译。
2. **T-MAC / distributed-llama 2026 年是否仍维护**：README 最新条目分别是 2025-05 / 2025-09 —— **"未查到 2026 条目"不等于"已停更"**。
3. **KTransformers `kt-kernel/` 是否有独立 LICENSE**：只实读了仓库根（Apache-2.0）。
4. **vLLM / SGLang 的"逐层"hidden 能力**：只确认它们不接收 `LAYER_FORWARD`（来自本仓既有判定），`return_hidden_states` 类能力的**逐层性未核实**。
5. **star 数 / 版本号 / 性能数字**：本轮**一律未取**（GitHub API 返回 rate limit），"活跃度"以 README news 条目、issue/PR 状态、仓库页计数为证。
6. ISCA'25 论文 §6.2「CPU Bottlenecks」正文未读到（HTML 抓取截断，只有目录标题）。
