# ExLlamaV3 作为 GPU 段引擎 —— 候选评估

> 状态：**调研结论，未实施**（只读了代码，没有跑过 v3 的任何一次推理）
> 更新日期：2026-11-02
>
> 对象：[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3)（v2 已归档，不讨论）
> 本地克隆：`H:\_ref\exllamav3`（`git clone --depth 1`）
>
> 适用范围：是否值得把 head 段的推理引擎换成 v3。**不含** CPU 段的任何变更——CPU 段仍由
> llama.cpp + `LLAMA_LAYER_RANGE` 承担，见 `docs/ARCHITECTURE.md`。

---

## 0. 结论

**列为将来候选，不在 v2 做。** 触发条件只有一个：

> 单机单进程的 v3（EXL3 + `-mcl`）在这台机器的 head 段上，decode 明显快于 llama.cpp CUDA 的同一段。

跑不出这条，跨机拆分要付的 22 ms 往返（单程 11 ms，实测）就补不回来，整条路不用走。
这个判据必须在**改 v3 之前**验证，顺序反了就是白干一周。

三条已核实的硬障碍（§3）：无 layer_range、EXL3 只能从 HF safetensors 转、8 GB 显存跑 32 层必须配
`-mcl`。其中只有第一条是真正的开发工作。

**为什么不是"现在就做"**：这条路 v1 走过，就是被砍掉的 F12（`docs/INVENTORY.md:27`，37 文件 /
27,825 行 / 21.5%，理由"用户明确不接跨框架、不引入 PyTorch"）。v2 至今在 `docs/PROGRESS.md` §5
挂着"跨框架接力（llama.cpp ↔ torch）| 未做"。重开这条路要先解除那条约束，否则它会以同样的方式
第二次被否。

---

## 1. 与 v1 F12 的关系

v1 的跨框架接力（GPU 段 PyTorch / CPU 段 llama.cpp）不是技术不可行，是产品决策放弃的：

| 证据 | 位置 |
| --- | --- |
| F12 多模态/跨框架/PyTorch = 27,825 行（21.5%），处置**抛弃** | `docs/INVENTORY.md:27` |
| 理由："用户明确不接跨框架、不引入 PyTorch。这是整仓最大单块" | `docs/INVENTORY.md:107` |
| SLIM-ARC 设计范围明写"不含 PyTorch" | `docs/DESIGN.md:8`、§2.2 |
| "跨框架接力（llama.cpp ↔ torch）| 未做" | `docs/PROGRESS.md:116` |

所以本评估的隐含前提是：**"不引入 PyTorch"这条约束解除**。建议将来单开仓库时把它写进 README
第一段，作为显式前提，而不是留成一个会被反复重新讨论的隐性否决。

---

## 2. 已核实的架构覆盖（读代码，非 README）

v3 README 的架构表里有 `Qwen3NextForCausalLM` / `Qwen3_5MoeForConditionalGeneration` / `KimiLinear`
/ `NemotronH` / `OlmoHybrid`，即 SSM/GDN 混合架构是一条产品线。逐项核对本项目模型
（`qwen35moe`，41 层 = 40 + 1 MTP）：

| 本项目 GGUF 元数据 | v3 配置键 | 位置 |
| --- | --- | --- |
| `expert_count = 256` | `num_experts` | `architecture/qwen3_5.py:211` |
| `expert_used_count = 8` | `num_experts_per_tok` | `:212` |
| `expert_feed_forward_length = 512` | `moe_intermediate_size` | `:210` |
| `expert_shared_feed_forward_length = 512` | `shared_expert_intermediate_size`（默认 512） | `:213` |
| `full_attention_interval = 4` | `(idx+1) % interval == 0` 判层类型 | `:35-38` |
| SSM/GDN 层 | `GatedDeltaNet` + `GDNState` | `:331` |
| `nextn_predict_layers = 1` | `mtp_num_hidden_layers` → `Qwen3_5MTPModel` | `:126`、`qwen3_5_mtp.py` |
| MTP 头 enorm/hnorm/eh_proj | `pre_fc_norm_hidden` / `pre_fc_norm_embedding` / `fc`，`native_draft_len=1` | `qwen3_5_mtp.py:29-44` |

**连 MTP 头都是原生支持的**（不是外挂 draft model），与 `bailing_hybrid_mtp` / `deepseek_v41_mtp` /
`nemotronh_mtp` 同属一条产品线。

其它已核实能力：

| 能力 | 状态 | 位置 |
| --- | --- | --- |
| MoE 专家放 CPU | 支持，且是完整实现（worker 进程 + pinned 共享内存 job ring + 绑核 + 预留 host 核） | `model/moe_cpu_host.py`（128 KB，`MoeCpuTuning` 集中调优） |
| 中间层状态导出 | 支持（draft model tap 中间层用），可复用来取第 32 层输出 | `model/model_ls.py` 的 `export_state_layers` |
| CUDA graph 捕获 decode 块 | 支持，attention/GDN/MLA 各有对应开关 | `doc/env_vars.md` 的 `EXL3_BC_ATTN` / `EXL3_BC_GDN` / `EXL3_BC_MLA` |
| 多卡层切分 | 支持（含 autosplit：跑 measuring forward 真实测 transient，OOM 就退到下一张卡） | `model/model_ls.py` 的 `_load_autosplit` |

CUDA graph 那一条对本项目有直接意义：Surface 侧 Vulkan 的 perf 日志里 `CPY ×136 = 13 ms/token`
是纯 launch/拷贝开销（见 `docs/MEASUREMENTS.md`），v3 的 graph capture 正是治这个的。

---

## 3. 硬障碍

### 3.1 没有 layer_range（唯一的真活）

全树 `grep -rn "layer_range\|first_layer\|last_layer"` **零命中**。`Qwen3_5MoeConfig` 会 assert 层数，
`_load_autosplit` 与 `forward_ls` 都假设 `modules` 是全量。

要做的是：放宽各 arch 的层数断言、截断 module list、处理 `logits_output` 标记（截到第 32 层就不该
再算 lm_head）、cache 按层初始化。估 **200–400 行，但调试成本远高于行数**——这是改在别人快速演进
的代码库上的补丁。

### 3.2 CPU 段只能是 llama.cpp（这条反而印证了现有架构）

`moe_cpu_host.py` 的 CPU worker **只做被路由到的专家的 GEMM**，attention / GDN / norm 一概不在 CPU
上跑。也就是说 v3 自己当不了 CPU 段——CPU 段仍必须是 llama.cpp。这恰好就是 v1 F12 的形态。

### 3.3 格式：EXL3 需要 HF safetensors 输入

`convert.py -i <in_dir>`（`conversion/convert_model.py:77`）配 `loader/safetensors_alt.py` 的
`safe_open`/`save_file`，**没有 GGUF 入口**。手上的 21.87 GiB GGUF 用不了，得先有原始 HF 权重。
量化本身需要 CUDA 卡，70B 级要数小时。

### 3.4 8 GB 显存

32 层 3 bpw ≈ 10+ GB，8 GB 装不下，必须配 `-mcl`（部分专家落本机内存）。这不是障碍，是配置约束。

### 3.5 CUDA-only —— 在本架构下**不构成障碍**

v3 只在 head 的 NVIDIA 上跑，永远不需要认识 Intel 核显；Surface 继续跑 llama.cpp CPU。
这一条在本评估里要明确写下来，避免被误当成否决理由。

---

## 4. 目标形态

```
head（本机 CUDA 8 GB）                tail（Surface，Intel CPU）
v3 + EXL3（0:32 层）  ──hidden 8 KB──▶  llama.cpp + LLAMA_LAYER_RANGE（32:40）
                       ◀──token────────
```

右侧已全部就绪（`LLAMA_LAYER_RANGE`、shim 的 hidden 注入、worker serve loop、拓扑、链客户端）。
缺的只有左侧。协议对齐可参照 tail 现有的 hidden 入参（dtype/shape 在 `proto_responses.py` 与
`stage_worker.py` 的 `forward_hidden` 分支）。

---

## 5. 第一步验证（不改 v3，预期半天）

1. 取 HF 权重（`Qwen3.6-35B-A3B` 原版；只有 GGUF 时先解决权重来源）
2. `python convert.py -i <hf_dir> -o <exl3_dir> -b 3.0`
3. `python examples/chat.py -m <exl3_dir> -mcl 32`（MoE 专家放 CPU）
4. 测 decode tok/s，与 llama.cpp CUDA 的同一段对比

**判据**：必须赢出明显一截（至少覆盖 22 ms 跨机往返）。赢不了就停在这里，不要动 layer_range。

---

## 6. 风险

- **补丁腐化**：v3 上游架构表持续快速扩张（`architecture/` 已 87 个文件），layer_range 补丁每次
  跟版都要重调。这正是 v1 F12 那 27,825 行腐化的同一路径——把补丁维护在别人的快速演进代码库上。
- **缓解建议**：将来单开仓库时，把"跨机 hidden 协议 + 层切分"放在自己这边，v3 只作为**可替换的
  GPU 后端**，不要把自己的逻辑写进 v3 内部。
- **约束复发**：若"不引入 PyTorch"未被正式解除，这条会第二次被否。

---

## 7. 与 NInfer 的比较判据

本评估**没有调研 NInfer**（只在 `docs/DESIGN.md:8` 见过一句"NInfer 为并行评估支线"）。将来两者
并列时，建议统一用下面四条打分，否则不可比：

1. 本项目架构（`qwen35moe` + GDN + MTP 头）是否原生支持
2. 能否只加载并前向一个层段（跨机拆分的前提）
3. CPU 段是否自足（能否不依赖第二个引擎独立跑完一段）
4. 量化格式的输入来源（能否从现有权重得到，需要多少额外 GPU 时间）

---

## 8. 复现本次调研

```powershell
git clone --depth 1 https://github.com/turboderp-org/exllamav3 H:\_ref\exllamav3

# 架构覆盖
ls H:\_ref\exllamav3\exllamav3\architecture\ | grep -iE "qwen|mtp"

# layer_range 是否存在（预期零命中）
grep -rn "layer_range|first_layer|last_layer" --include=*.py H:\_ref\exllamav3\exllamav3\

# CUDA 依赖面
grep -rl "torch.cuda" --include=*.py H:\_ref\exllamav3\exllamav3\ | wc -l    # 48

# 转换器输入格式
grep -n "safetensors|add_argument" H:\_ref\exllamav3\exllamav3\conversion\convert_model.py
```
