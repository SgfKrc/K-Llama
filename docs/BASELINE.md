# 主仓「层段推理」核心链路基线框定

> 盘点对象：`G:\C\PYT\qlh`（只读，未改任何文件）。
> 目的：为 Rust 控制面 + llama.cpp worker 的精简原型框定**必须移植**的最小面。
> 标注：**[实测]** = 本轮跑过命令/读过真实工件；**[代码]** = 仅从源码推断。
> 所有行号以本次盘点时的文件为准。

---

## 1. 工件裁切（`scripts/cut_layers.py`）

### 1.1 三种模式 [代码] `cut_layers.py:137-141`

| 模式 | CLI | 张量处理 | 重编号 |
| --- | --- | --- | --- |
| `head` | `--keep-head N` | 保留 `blk.0..N-1` | **不重命名** |
| `middle` | `--k K --end E` | 保留 `blk.K..E-1` | `blk.K.* -> blk.0.*` |
| `tail` | `--k K` | 保留 `blk.K..n_layer-1` | 同上 |

模型写入三处：`<arch>.block_count` 改写为段层数（`cut_layers.py:706-707`）、层张量按模式筛选+重编号（`cut_layers.py:530-566`）、按层数组型 KV 按同一口径切片（`cut_layers.py:446-489`）。

### 1.2 硬约束（全部 fail-closed，分两档）

1. `0 < K < n_layer`，`K < end <= n_layer`，且 **`K`、`end`、`end-K` 都必须是 `<arch>.full_attention_interval` 的整数倍**（`cut_layers.py:144-186`）。原因：llama.cpp **不读** HF `layer_types`，而是按（重编号后的）本地层号对 interval 取模推导（`llama.cpp/src/models/qwen35.cpp:21-28`、`qwen35moe.cpp:24-28`）。
2. `--hf-config` 给出时，额外逐位校验源 `layer_types` 是否与上述取模规律一致（`cut_layers.py:189-240`）。
3. 保留区间内每层的**必需张量**按层类型断言（linear 层 12 个 / full 层 11 个后缀，`cut_layers.py:245-261`、`:264-324`），缺任一即拒绝。仅对 `qwen35` 生效（`:261`）。
4. MTP（nextn）层若不在保留区间 ⇒ `<arch>.nextn_predict_layers` 归 0（`cut_layers.py:710-734`）；否则 llama.cpp 会把最后一个普通层当 MTP 层、报 `missing tensor 'blk.x.nextn.eh_proj.weight'`。
5. 非尾段的 `qwen35`/`qwen35moe` **丢弃 `output.weight`**（`gguf_tensor_selection.py:27-33` + `cut_layers.py:540-544`）；其它张量（含 `token_embd`/`output_norm`）一律原样保留（`cut_layers.py:564-565`）。

**[实测]** 源件 `models/qwen3-5-9b-gguf/qwen3-5-9b-local-Q4_K_M.gguf`：`qwen35`，`block_count=33`、`n_layer=32`（33−nextn 1）、`full_attention_interval=4`、`head_count_kv=4`、`key_length=256`、`n_embd=4096`、442 张量。
**[实测]** `--keep-head 6` ⇒ `FAIL: --keep-head N=6 不是 full_attention_interval(4) 的整数倍`（退出码 1，但**先打印了 plan 再用 FAIL 收尾**，`:686-691`）。
**[实测]** dry-run 张量数：`--keep-head 4`⇒55、`--keep-head 8`⇒108、`--k 8 --end 24`⇒214、`--k 4 --end 24`⇒267。

### 1.3 产出契约：manifest

`_manifest`（`cut_layers.py:327-435`）字段分两类，**这个区分必须移植，否则下游会读错层数**：

* **描述源模型**：`block_count`、`n_layer`、`nextn_predict_layers`、`hidden_size`、`source_model_sha256`、`tokenizer_sha256`、`source_model_id`。
* **描述本段**：`source_layer_range`（半开区间，源口径）、`kept_block_count`、`trim_layers`、`first_local_layer_maps_to`、`tensors_kept`/`tensors_dropped`、`artifact_sha256`、`artifact_contains`（`token_embd`/`output_norm`/`output_weight`/`can_serve_tail`，`cut_layers.py:396-411`）、`gguf_metadata`（窗口投影，`gguf_metadata_contract.py:236-252`）。

`mode=head/middle` 时 `contract` 段显式标 `{"skipped": ...}`（`cut_layers.py:412-415`）——`RelayTrimPlan` 只描述 tail 语义。`contract_valid` 只是汇总打印的 `trim_plan_valid`（`cut_layers.py:762`），**不是独立校验位**。

**[实测]** 存量工件 `q35-9b-local-head4.gguf`（1.98 GB）：GGUF 内 `block_count=4`、`nextn=0`、56 张量；而**同目录 manifest 顶层写 `block_count: 33`、`nextn_predict_layers: 1`** —— 印证上面"源口径 vs 段口径"的分离。同一实测还显示该工件**仍含 `output.weight`**（新代码会丢），说明它是 #147 之前的裁切口径产物。

### 1.4 下游怎么找工件：只认 manifest，不猜文件名 [代码]

`LlamaCppEngine._find_layer_artifact`（`llama_engine.py:701-782`）在目录里枚举 `*.manifest.json`，要求 `source_layer_range == [start,end)` **且** `source_model_sha256` 等于当前整模摘要 **且** 工件实算 sha256 等于 `artifact_sha256`；命中多于 1 个直接报错拒绝静默选择。零依赖读端：`relay_segment_info.read_artifact_manifest`（`relay_segment_info.py:539-584`）。

---

## 2. 层段接力：一个 token 怎么走完

### 2.1 发起方与调用栈 [代码]

`api_server.py:3381-3410`（master、`distributed`、`FORWARD_LAYERS` 三项同时成立）→ `scheduler.run_pipeline_safe`（`scheduler_pipeline.py:8305`）→ `scheduler.py:583-584 build_requests` → `qlh_core/bootstrap/execution.py:194` → **`qlh_core/execution/linear.py:33-42` 是真正的主循环**：

```
for step in range(max_tokens):
    hidden = ports.prefix(plan.segments[0], ...)      # 本机首段，吃 token 吐 hidden
    for segment in plan.segments[1:]:                  # 逐段**串行**
        receipt = ports.stage(segment, identity, hidden, record)
        hidden = accept(identity, receipt)
    token = tail_token(hidden) if last.head else head(hidden)
```

**是纯串行**，不是流水线并行。发起者是 master 的 execution application（`ExecutionApplication`，`qlh_core/bootstrap/execution.py:54`），各段是远端 task worker（或本机附加执行器）。

### 2.2 段间数据格式（两条互不相通的通道）

**通道 A：Route-A v3 协议（生产路径）** [代码]
* 载荷：`root_input.hidden_f32` = **raw little-endian f32 的 base64**（`scheduler_pipeline.py:2913`，归一化在 `_hidden_to_raw_f32`，`:265-306` —— 恒转 f32，**不做 f16**）。
* 回程：`output.hidden_out_f32` + `output.hidden_out_sha256`（`engine_host.py:3526-3531`），接收端强校验长度 `n_tokens*n_embd*4` 与 sha256（`scheduler_pipeline.py:348-369`）。
* 段字段：`layer_range`、`handoff_at`、`hidden_sha256`、`hidden_spec{n_tokens,n_embd,dtype}`，可选 `middle_channel`（`{extract_hidden, keep_head_layer_out}`）、`seq_ids`、`positions`、`prepare_identity`、`token_output`（`task_worker_protocol.py:102-132`）。
* 帧上限 `MAX_MESSAGE_BYTES = 8 MiB`（`task_worker_protocol.py:29`），预算 = 8 MiB − 256 KiB 保留（`:45`、`:1724-1734`）。超预算三条放宽路径：`binary_hidden`（原始字节体，省 33% base64 膨胀，`:1567-1597`）、`stage_chunked_input`（分片，64×1 MiB，`:53-55`）、否则 dispatch 前就拒。

**[实测]** 用模块函数算：`n_embd=2048` 时 f32 每 token 10,924 B ⇒ 单帧最多 **744 token**（base64）/ 991 token（binary）。`n_embd=4096`（9B）⇒ 每 token 21,848 B ⇒ **372 token**。

**通道 B：relay TCP（实验/设备链）** [代码]
* 帧头 `struct("!4sBBHIIQ")` = magic `QLHR` + version 1 + kind + flags + sequence + n_tokens + payload_size（`relay_transport.py:37-40`、`:54`、`:396-404`）。
* `flags` 低 3 位 = hidden 压缩档（`none/f16/int8_block128/int4_block128`，`relay_hidden_quant.py:40`）；旧对端见非零 flags 直接拒 ⇒ 天然 fail-closed（`relay_transport.py:44-47`）。
* 帧类型：`HIDDEN=1`、`TOKEN=2`、`CLOSE=3`、`ERROR=4`、`HIDDEN_SEQ=5`、`TOKENS=6`、`TOKEN_SEQ=7`（`relay_transport.py:121-144`）。
* `HIDDEN_SEQ` payload = `<n_tokens:u32><n_meta_bytes:u32><meta JSON><hidden>`，meta 只允许 `seq_ids`/`positions`/`n_seq_id`，且**必须同时给 `seq_ids` 与 `positions`**（`relay_transport.py:245-290`）。
* 三种段角色：`forward_tokens`（token→hidden，上游）/`forward_hidden`（hidden→hidden，中段）/`forward_hidden_to_token`（hidden→token，末段）（`relay_segment_client.py:235-309`）。

**回答"是否 float16"**：**默认不是**。两条通道的基准都是 f32 LE；f16 只是 relay 通道的可选压缩档，生产 Route-A 路径根本没有 f16 选项（`_hidden_to_raw_f32` 恒 f32）。

### 2.3 hidden 的注入与导出

**导出（上游）**：`llama_set_embeddings_layer_inp(ctx, lid, true)` 打开槽位 → decode → `llama_get_embeddings_layer_inp(ctx, lid)` 取稠密 `[n_tokens, n_embd]` f32（`llama.h:1058-1063`、`qlh_keep_head.c:192-196`、`:210-218`）。
关键：**`lid == n_layer` 是"末层输出（`output_norm` 之前）"的槽位**，由补丁把数组扩到 `n_layer+1`（`llama-context.cpp:80-81`、断言 `:1157`），并由各架构 graph 登记 `res->t_layer_inp[n_layer] = cur;`（实测已登记 9 个架构：`gemma4.cpp:408`、`llama.cpp:234`、`openai-moe.cpp:160`、`qwen2.cpp:152`、`qwen3.cpp:148`、`qwen35.cpp:213`、`qwen35moe.cpp:235`、`qwen3moe.cpp:168`、`qwen3next.cpp:177`）。

**注入（下游）**：走**官方** `llama_batch.embd` 通道（`qlh_keep_head.c:373-381`、`llama_engine.py:2741`），不需要补丁。
**M-RoPE 陷阱（必移植）**：embd 通道下位置按 **planar** 读取，要求 `n_tokens × n_pos_per_embd` 个 `pos`，而 `llama_batch_init(n_tokens,...)` 只分配 `n_tokens` 个 ⇒ 必须额外分配并把同一位置广播到各分量（`qlh_keep_head.c:262-303`；`llama_engine.py:2730-2740`）。遗漏症状：Qwen3.5（IMROPE，n_pos_per_embd=4）接力首步即分叉。

**`layer_inp` vs `nextn` 两种模式**（shim `qlh_keep_head.c:13-17`、`:182-196`）：

| 模式 | shim mode | 槽位 | 配套工件 | 是否跑满全部层 | 适用 |
| --- | --- | --- | --- | --- | --- |
| `nextn` | 0 | `lid = n_layer` | **head 裁层件**（`--keep-head K`） | 只跑 K 层 | **生产上游/中段** |
| `layer_inp` | 1 | `lid = cut_layer`（必须 `< n_layer`，`:154-159`） | 整模件 + `cut_layer=K` | 跑满全部层 | 仅数值对照/整模取中间层 |

两者输出**逐位相同**（都是同一条 `layer_inp` 通道，只差槽位）。2026-09-23 起不再用 `llama_set_embeddings_nextn`：各架构 `t_h_nextn` 挂点不同（qwen2 在 `output_norm` 前、qwen35 在**后**），拿后者当上游会多一次 RMSNorm ⇒ 首步分叉（`patch:47-53`、`:141-163`）。

**产出角色三件套**（`llama_keep_head.py:640-892`）：`forward_tokens_to_hidden`（token→hidden）、`forward_hidden_to_hidden`（hidden→hidden）、`forward_hidden_to_token`（hidden→argmax）/`forward_hidden_to_tokens`（多位置 argmax）。

### 2.4 KV cache 分段管理

* **每段各自持有一份 KV**：每段是独立 `llama_context`，只为自己那几层维护 KV；不存在"拆同一份 KV"（`qlh_keep_head.c:139`、`:628-631`）。
* `n_ctx` 按 `n_seq_max` **均分**：`ctx_per_seq = n_ctx // n_seq_max`（`llama_keep_head.py:108-117`）。配错会把单序列预算压到 `n_ctx/8`，长 decode 中途报 `llama_decode rc=1` → 上游只看到 `runner_failed`（与"模型算错"难区分）。
* 清空 = `llama_memory_clear(mem, true)`（`qlh_keep_head.c:622-632`）。`LlamaCppEngine.reset_kv_cache()` 是 **stateless no-op**，**不能**用来清层段 KV（`llama_engine.py:2758-2768`）。
* 同一进程服务多连接时，每次新会话必须 reset（`relay_mid_service.py:654-660`），否则位置冲突。

### 2.5 各段 `n_past` 如何保持一致

两条路径**机制不同**，这是本盘点最容易被设计文档掩盖的地方：

* **Route-A（生产）**：`positions` 是唯一权威。调用方每步算出 `start = 0 if step==0 else prompt_length + step - 1`，再 `positions = range(start, start+len(hidden))`，**同一份 positions 传给每一段**（`qlh_core/bootstrap/execution.py:554-556`、`:369`、`:386`）。`n_past` 只作为 `root_input.pos_base` 传（`scheduler_pipeline.py:2915`），执行侧 `n_past = root_input["pos_base"]`（`engine_host.py:3481`）；shim 侧 `positions != NULL` 时 `n_past` 被完全忽略（`qlh_keep_head.c:249-260`）。
* **relay（实验/设备链）**：各段自己维护 `self._pos` 累加（`relay_mid_service.py:96-97`、`:353`），跨连接 reset。
* **隐式回落的唯一合法情形**：`positions == NULL` ⇒ `pos[j*n_tokens+i] = n_past + i`（`qlh_keep_head.c:246-248`、`:297`）。用错会得到 `tokens ... have inconsistent sequence positions`。
* **prefill 起始位置为 0 时必须 reset**，否则会撞上上一请求残留位置（`engine_host.py:3539-3549`）。

---

## 3. 设计文档与代码不一致（显式指出）

| # | 文档说法 | 代码事实 | 影响 |
| --- | --- | --- | --- |
| 1 | `H:\slimarc-proto\DESIGN.md:80` 图示 "hidden (BF16, 4 KiB)" | wire 是 **f32 LE**（`relay_transport.py:39-40`；`scheduler_pipeline.py:265-306`）；`n_embd=2048` ⇒ **8 KiB/token**，`n_embd=4096` ⇒ 16 KiB | 带宽/延迟预算算错一倍；f16 需自己加，不是现成默认 |
| 2 | `DESIGN.md:216` 把 `qlh_kh_forward_embd_tokens` 标为"hidden 进、hidden 出（中段）★本次用" | 该函数返回 `int32_t * out_tokens`（**argmax 列表**），是末段多位置出 token 用的（`qlh_keep_head.c:472-476`、`:553-557`）；中段应用 `qlh_kh_forward_embd` / `_seq`（`:363-393`） | 按文档接会得到 token 而非 hidden |
| 3 | `DESIGN.md:209-223` "hidden 通道已存在，不必新写"、`include/llama.h:1058` "只有导出接口，无注入接口" | 注入确实靠官方 `llama_batch.embd`；但**导出**依赖 QLH 补丁（`llama.h:1058-1063` 的声明、`llama-context.cpp:80-81` 的 `n_layer+1` 槽位、9 个架构的 `t_layer_inp[n_layer]` 登记） | "不必碰 llama.cpp"不成立：**必须用带该补丁的 fork**，否则上游出处只能拿到 `output_norm(H)`（实测 cos 0.999998 但首步分叉） |
| 4 | `docs/已知问题记录.md:2279` "shim 把 `layer_inp` 限制为 `cut_layer < n_layer`（`qlh_keep_head.c:111`）" | 该检查今天在 `qlh_keep_head.c:154-159`（行号已漂移）；同段还说"qwen3* 挂点在之后，代码显式拒用" | 后半句已过时：该判据 2026-10-07 删除（`llama_engine.py:2485-2497`） |
| 5 | 存量工件 manifest 与当前裁切口径 | 新代码非尾段丢 `output.weight`（`gguf_tensor_selection.py:30-33`）；**[实测]** `q35-9b-local-head4.gguf` / `mid4-24.gguf` **仍含 `output.weight`** | 旧段件比新口径多约 0.8 GB/段；预算按"旧工件实际大小"记，勿按新口径推算 |
| 6 | `DESIGN.md:64` "≈1,400 行 Python" | 本链路的**真实**依赖面见 §4（单 `llama_engine.py` 就 146 KB / 2954 行，`scheduler_pipeline.py` 456 KB） | 1,400 行只能覆盖"串行两段 + 裁剪协议"，不能覆盖现有合同的校验面 |

---

## 4. 最少可重现集

### 4.1 必须移植

| # | 接口/协议 | 证据 | 为什么不能简化 |
| --- | --- | --- | --- |
| M1 | **GGUF 头读写 + 层张量筛选/重编号 + KV 窗口投影** | `gguf_metadata_contract.py:51-170`、`:236-252`；`cut_layers.py:530-566`、`:446-489` | 裁层是整条链路的入口；重编号错了下游 `blk.0` 就接不上 |
| M2 | **切点约束 `% full_attention_interval == 0`** | `cut_layers.py:144-186`；`qwen35.cpp:21-28` | 违反 = 工件加载必炸；这是 hybrid 架构的硬门 |
| M3 | **MTP 层归属：裁掉即把 `nextn_predict_layers` 归 0** | `cut_layers.py:710-734` | 不归零 ⇒ llama.cpp 要求最后普通层带 `nextn.*` ⇒ 整模加载失败 |
| M4 | **manifest 双口径**（源 `block_count`/`nextn` vs 段 `kept_block_count`/`source_layer_range`/`artifact_contains`） | `cut_layers.py:327-435`；`relay_segment_info.py:539-584` | 混用会导致"段声明 33 层却只有 4 层"的静默错配 |
| M5 | **hidden 导出通道：`layer_inp` 的 `lid == n_layer` 槽位** | `llama.h:1058-1063`、`llama-context.cpp:80-81`、`qlh_keep_head.c:192-218` | 唯一能拿到"末层输出、norm 之前"的方式；embeddings 通道给的是 norm 后 |
| M6 | **hidden 注入通道：`llama_batch.embd` + M-RoPE planar 位置广播** | `qlh_keep_head.c:281-303`、`:373-381` | 不做 planar ⇒ 4 分量模型位置越界读取，首步分叉且不报错 |
| M7 | **wire 契约：raw f32 LE `[n_tokens, n_embd]` + sha256 校验 + dtype/形状校验** | `scheduler_pipeline.py:265-306`、`:348-369`；`relay_transport.py:37-40`、`:263-322` | 尺寸/摘要校验是"错段/串话"的唯一廉价防线 |
| M8 | **positions 是唯一权威、n_past 仅作回落** | `qlh_core/bootstrap/execution.py:554-556`、`:386`；`engine_host.py:3481`；`qlh_keep_head.c:249-260` | 各段 n_past 要一致，靠传同一份 positions；靠自增必然漂移 |
| M9 | **每段 KV 独立 + 起始位置为 0 时必 reset** | `qlh_keep_head.c:622-632`；`engine_host.py:3539-3549`；`llama_keep_head.py:108-117` | 不清 ⇒ 位置冲突；`n_ctx` 不均分 ⇒ 长 decode 假性失败 |
| M10 | **接受判据：逐 token argmax 严格相等**（cosine/bitwise 只作诊断） | `relay_contract.py:30-35`、`:255-305` | 放宽到 cosine 会把"位置越界"这类真 bug 当成精度噪声放过 |

### 4.2 实现细节可简化

| # | 项 | 现实现 | 可简化成 |
| --- | --- | --- | --- |
| S1 | 传输分片/二进制体 | `stage_chunked_input` + `binary_hidden` + 回程分片（`task_worker_protocol.py:53-55`、`:1567-1597`） | 单帧 + 固定 `n_tokens` 上限（prefill 分块推进即可） |
| S2 | hidden 压缩档 | `f16/int8_block128/int4_block128`（`relay_hidden_quant.py:40-171`） | 只留 `none`（f32）；实测只有 f16/int8 在 512 步内不改 token，int4 单step就失败 |
| S3 | 多序列 `seq_ids`/`HIDDEN_SEQ` | `qlh_keep_head.c:363-393`、`:472-558` | 单序列 `seq_id=0`；先把串行链路做对 |
| S4 | `skip_hidden_logits` 优化 | `qlh_keep_head.c:597-617` | 虽实测 1.13–1.20×，但可后置（旧 shim 无此符号） |
| S5 | manifest 摘要链/供给状态机 | `model_supply_assets.py:1322-1490`（生成后逐张量比对源字节） | 留 `artifact_sha256` + `source_layer_range` + `n_layer` 三项即可定位 |
| S6 | `nextn` 模式与 `layer_inp` 模式并存 | shim `mode 0/1` | 只留 `nextn`（配 head 工件）；整模 + cut_layer 仅用于对拍 |
| S7 | 租约/取消/回执/`prepare_identity` | `task_worker_protocol.py:183-217` | 一条连接内串行请求，不需要租约 |
| S8 | 六大架构白名单/`middle_channel` 双通道 | `task_worker_protocol.py:132` | 只留 `keep_head_layer_out` |

### 4.3 `-ncmoe N` 的语义（供段内放置参考）

`-ncmoe N` = 前 N 层的 MoE 专家留 CPU，其余层的专家交 GPU；实现是对每层追加一条 `tensor_buft_overrides`：pattern = `blk\.<i>\.ffn_(up|down|gate|gate_up)_(ch|)exps`，buft = `ggml_backend_cpu_buffer_type()`（`common/arg.cpp:2465-2479`，pattern 由 `common.h:1054-1062` 生成）。这是 **CLI/`common` 层**能力，shim 路径用不上（shim 硬编码 `n_gpu_layers = 0`，`qlh_keep_head.c:101`）；Rust 版若自建 worker 参数，应直接产出同样的 override 列表而不是拼字符串。

---

## 5. 最少必要概念清单（9 项）

1. **裁层 = 换工件**：llama.cpp 没有"按层加载"能力，段 = 一份独立 GGUF，靠 `block_count` 与张量重编号表达。
2. **切点必须是 `full_attention_interval` 的整数倍**，因为层类型是 llama.cpp 按本地层号取模重推的，不读 HF `layer_types`。
3. **MTP/nextn 层是普通层序列之外的最后一层**，裁掉它就必须把 `nextn_predict_layers` 归 0。
4. **接力的分界是残差流、`output_norm` 之前**，用 `lid == n_layer` 的 `layer_inp` 槽位取；用 embeddings 通道会多一次 RMSNorm。
5. **hidden 的 wire 形式是 raw f32 LE `[n_tokens, n_embd]`**（默认不压缩），配 sha256 + 形状校验。
6. **注入靠官方 `llama_batch.embd`，但 M-RoPE 模型要求位置按 planar 广播成 4 份**。
7. **positions 是跨段唯一权威**，`n_past` 只是它的缺省回落值；每段都必须收到同一份 positions。
8. **每段各持一份独立 KV**，`n_ctx` 被 `n_seq_max` 均分，起始位置为 0 的批次前必须清记忆。
9. **验收只看逐 token argmax 是否全等**，cosine/bitwise 只是诊断，不得作为准入依据。

---

## 6. 给 Rust 原型的最小落地顺序（建议）

1. GGUF 头读写 + `--keep-head N` / `--k K --end E` 裁剪（含 interval 倍数、MTP 归零、manifest 双口径）。
2. 单个 llama.cpp worker 进程：加载段件、`layer_inp(lid=n_layer)` 导出、`embd` 注入（含 planar 位置）、reset。
3. 单机两段串行：positions 外生、逐 token argmax 与整模对拍。
4. 协议只有 `hello/forward/result/reset/bye` + 固定 f32 载荷；单帧 token 上限按 `n_embd` 反算（`8MiB - 256KiB` / `base64(n_embd*4)`）。
5. 之后才考虑分片、f16、多序列、`skip_hidden_logits`。

> 参考量级：本链路真正"必须理解"的源码约 `cut_layers.py` 690 行 + shim 620 行 + `relay_contract.py` 457 行 + `relay_transport.py` 1247 行 + `relay_hidden_quant.py` 195 行 + `llama_keep_head.py` 871 行 + `gguf_metadata_contract.py` 223 行 + `gguf_tensor_selection.py` 109 行 + `linear.py` 62 行 ≈ **4.5k 行**；其中按 §4.2 简化后，Rust 侧核心估计 **1.2–1.8k 行**（与 DESIGN.md 的 1,220 行同量级，但需额外承担 M1/M4/M6 三块）。
