# 补丁细节：第四代机制、device-buffer 修复与上游重合

> 状态：**现行**
> 更新日期：2026-11-11
> 适用范围：`docs/deploy/PATCHES.md` 的展开——第四代逐行机制表、六处语义、device-buffer 崩溃的完整修复
> 与组 11、与上游 193 个提交的重合明细、两次对齐的历史记录、工具层参数校验。补丁构成与代际差异
> 归 [`PATCHES.md`](PATCHES.md)；边界代价归 [`SEGMENTED-LIMITS.md`](../limits/SEGMENTED-LIMITS.md)。

## 1. 第四代机制表

| 机制 | 位置 | 要点 |
| --- | --- | --- |
| 层范围进参数 | `include/llama.h` 的 `llama_model_params.split_layer_lo/hi` | 同一进程才能加载多个层范围；`LLAMA_LAYER_RANGE` 降级为兜底 |
| hidden 传递 | `llama_context::decode_split` | 走标准 embedding 路径（`embeddings_pre_norm` + `get_embeddings_ith`），不走 `nextn` 通道 |
| relay 段省 GET_ROWS | `src/llama-graph.cpp` 的 `build_inp_embd` | 没有 `tok_embd` 时不建；那一行被 `ggml_build_forward_order` 强制入图，只看分支选择不够 |
| 聚合 memory | `src/llama-memory-split.{h,cpp}` | 本地段走 `llama_memory_t`、远程段走协议；`seq_pos_*` 问链首，`can_shift` 恒 false |
| 远程段 | `src/llama-split-{proto,net,remote}.*`、`tools/split-stage` | 定长协议 **v9**（v3 定长头 + v4 `OP_HELLO` + v5 候选通道 + v7 验证行 + v8 远程 MTP draft + v9 tap）；socket 层两端共用一份；`--host` 绑地址，默认 loopback |
| 远程段进聚合 | `llama_context::split_attach_remote` | 装 `llama_memory_split`，远程指针借给视图；`seq_rm` / `clear` 广播到协议 |
| server 入口 | `common/arg.cpp` 的 `--split-tail-range` / `--split-remote` / `--split-rs-seq` | `--split-remote` 可重复、按链序；与 `--split-tail-range` 互斥 |
| 位置下发 | `decode_split` 的 `split_expand_pos` | 每个 token 的 4 个 M-RoPE section 全量下发，`req.n_pos` 让两端对齐 stride |
| 角色显式声明 | `split-stage --role relay\|tail` | 用源层数校验：relay 要求 `hi < n_layer_src`，tail 要求 `hi == n_layer_src` |
| checkpoint 覆盖全链 | `state_seq_*` 走 `get_memory()`；`llama_memory_split::state_read()` | 远程段先恢复、链首最后；每段带 `pos_max`，落后于快照的一律拒绝 |
| 候选回传（v5） | `common/sampling.cpp` 的 `common_sampler_candidates_check()`、`server-context.cpp` 的 `plan_remote_candidates()` | 普通 decode 让 tail 跑 `logit_bias → penalties → top_n_sigma → top_k` 后只回候选、不回整行 logits；不等价时按白名单回退整行 |
| 验证通道（v7） | `DECODE_FLAG_VERIFY_ROWS`（`src/llama-split-proto.h`）、`request_remote_verify_rows()` / `plan_remote_verify_rows()` | 验证帧的每行回一个 int32 argmax，一帧 k+1 行是 4(k+1) B（单帧 993,312 → 48 B）；`MAX_VERIFY_ROWS` 两端共享；要求 `off == 0`（帧内行号 = batch 绝对下标），不满足就 `spec_blocked` 整个请求关投机 |
| 远程 MTP draft（v8） | `OP_MTP_DRAFT`、`DECODE_FLAG_NEXTN_ROWS` / `RSP_FLAG_NEXTN_ROWS`、`common/speculative.cpp` 的 `is_remote` | tail 内部跑 k 步递归（每步输入是上一步的输出），回 k 个 id；`stage_id.has_nextn` 占 `_pad` 且不参与身份比较；tail 没有 NextN 时 server 摘掉 draft-mtp 并 WARN，不拦启动 |
| grammar / rbudget 兼容 | `common_sampler_grammar_rejects()` / `common_sampler_rbudget_rejects()`、`common_reasoning_budget_rejects()`、`skip_draft_once` | draft 不看这两个采样器，接受率从无 grammar 2.76 掉到 1.46–1.73 token/帧：head 对每个候选做单 token 自检（只掩码不推进），被拒就整帧回退（撤 KV + prompt + sampler）；预算耗尽只收紧不放宽 |
| 远程 layer_inp tap（v9） | `DECODE_FLAG_LAYER_INP` / `RSP_FLAG_LAYER_INP`、`request_remote_layer_inp()` / `plan_remote_layer_inp()`、tail 侧 `layer_tap_out` | 复用 decode 回复：请求体末尾追加 `layer_inp_req{n_taps,n_embd}` + `n_taps` 个 int32 源层号（不是 stage 本地层号，同一份请求能发到链上每一段），回复体追加 `n_taps` 个 id + `n_rows × n_taps × n_embd` 个 f32；`rsp_header._pad` 一个字节改成 `n_taps`（尺寸仍 32，前四段解析与 v8 逐字相同）；`MAX_LAYER_INP_TAPS = 16`；tap 缺失只让本请求停投机，不 send_error |
| device-buffer 可用性 | `src/llama-context.cpp` 的 `supports_embd_row()`、`common/speculative.cpp` 的每帧判定 | 谓词同时回答「buffer 存在」与「宽度 == `n_embd_inp_enc()`」；`decode()` 准入、`graph_params()` 的 `layer_inp_src`、batch 构造、每帧投机判定问同一句 |
| 两个接线缺口（v9 同期） | `server-context.cpp` 的 `ctx_tail_remote` 置位、`split-stage` 的三条回包分支 | ① `ctx_tail_remote` 原来只对 MTP 置位，DFlash/DSpark 走 `has_draft`，tap 被解析到 head → `GGML_ASSERT(lid <= n_layer())`；② plain 帧原来 nextn 优先、verify/candidate 才是 tap 优先，而一帧只能声明一个 `n_embd` 宽度，prefill 帧回了 nextn、把 tap 吞掉（接受率 0.111）。两处修完，远程链 DFlash/DSpark 零损失 |
| split-stage 的线程 / ubatch | `tools/split-stage` 的 `-t/--threads`、`-tb/--threads-batch`、`-ub/--ubatch-size` | `GGML_DEFAULT_N_THREADS = 4` 且 `n_threads_batch` 也是 4、不会自动跟随；`-t` 同时设两者（head 的 `-t` 不过网），`-ub` 是测量旋钮。Surface 上不投机 +3.5%、MTP +7% |
| 链死自愈 | `llama_memory_split::try_rebuild()` | 位置 0 的请求 → 逐个 `reconnect()`（含重握手）→ 整链 `clear()`；失败保持 dead + 5 s 节流 |
| MTP 投机在本地链上 | `server-context.cpp` 的 `ctx_tail` / `ctx_tail_remote` | draft context 建在 tail 的模型上——head 段 nextn 为 0，建在 head 上会启动失败；DFlash/DSpark 的 tap 也必须解析到 `ctx_tail_remote` |
| DFlash / DSpark 的 tap 跨两段 | `common/speculative.cpp` 的 `tap_ctx` / `tap_lid` | 源层号按 head/tail 拆成（context, 局部层号）；本地链的 tail tap 进程内读、不加往返，远程链走 v9 |
| tail 的 logits 缓冲按角色 | `common/common.cpp` 的 `n_outputs_max` | head 每 token 一行 hidden；tail 只为实际输出行（1 + k）分配 |
| socket 超时 | `src/llama-split-net.h` 的 `set_timeout` | 每 send/recv 120 s，只兜整体停摆；stage 侧不设，靠对端关闭 |
| 单独层范围的拒绝 | `common_init_from_params()` | 范围结束早于末层、又没有后续段时，加载后立即报错；范围不从 0 开始且 `split_range_is_stage` 为假时同样拒绝 |
| 段级 offload | `src/llama-model.cpp` 的 `load_tensors()`：`i_gpu_start` 按 `layer_range_active()` 分叉；`split-stage` 的 `-ngl`（默认 0，不是 `-1`） | 语义、两条边界与实测见 [`SEGMENT-OFFLOAD.md`](../measure/SEGMENT-OFFLOAD.md) §2–§3 |
| MoE 缓存 | `split-stage` 的 `--moe-cache-mib` / `--moe-cache-cpu-share`，走上游 `llama_moe_cache` | `-ngl` / `-ncmoe` / `--moe-cache-mib` 缺一个就退化，启用条件写在 `src/llama-moe-cache.cpp` 的启用循环里；命中率扫描与三种花法见 [`SEGMENT-OFFLOAD.md`](../measure/SEGMENT-OFFLOAD.md) §5 |
| 显存预算 | `src/qlh-vram-budget.h`（625 行）、`src/qlh-split-defaults.h`（20 行）、`server-context.cpp` 与 `split-stage/main.cpp` 两处接入 | 加载前预检 + ctx 建好后按 nvidia-smi 前后差后检；WDDM 会静默 over-commit，所以必须用 nvidia-smi，查不到就跳过并说明；后检永不退出、预检默认只告警（`QLH_VRAM_STRICT=1` 才拒），`QLH_VRAM_DISABLE=1` 总开关。公式与验收见 [`SEGMENT-OFFLOAD.md`](../measure/SEGMENT-OFFLOAD.md) §10 |

## 2. 六处语义

- **`seq_rm` 失败不再静默**（2026-11-08）：`common_memory::seq_rm` / `common_context_seq_rm` 从
  `void` → `bool`，五个调用点按场景降级（全量重算 / 结束响应 / 请求失败），不再 `GGML_ABORT`。
  上游 hybrid 短路（先删 recurrent，失败就 `return false`，attention KV 一行不删；server 默认
  `n_rs_seq=0` 下 partial 恒不成立），上游有 open PR #28007 与 issue #31656。两条见
  [`SEGMENTED-LIMITS.md`](../limits/SEGMENTED-LIMITS.md) §2.9。
- **`--split-rs-seq` 不再是正确性必需**：早先 `n_rs_seq = 0` 的段会让截断式 `seq_rm` 失败，而
  `common_context_seq_rm` 把 false 当致命错误（远程链此前没有 `remote_failure` 出口）；现在
  checkpoint 覆盖全链，默认 0 是可服务的配置。这个参数只在投机时要抬高下限（一轮回滚 `n_max`
  个位置）。`split-stage` 仍强制 `--rs-seq N`：server 的 `common_context_can_seq_rm()` 会回退到
  探测 `seq_rm`，0 快照的段会让探测失败（远端拒绝，链被标死），理由写在 `--rs-seq` 必填检查里。
- **失败不区分「对端拒绝」与「没连上」**：两者对链是同一件事——位置要分叉了。`decode` 的传输
  失败降级成 `-3` 让请求失败；`seq_rm` / `clear` 失败都写 `remote_failure` 把链标死，
  `common_context_seq_rm()` 对已死的远程链只记日志并 return false，此后 `decode_split()` 在算任何
  东西之前就查这个状态并返 `-3`——请求失败，进程活着。
- **链死能自愈**（`llama_memory_split::try_rebuild()`）：下一个从位置 0 开始的 batch 逐个
  `reconnect()` → 整链 `clear()` → 恢复数据通路超时；失败保持 dead 并做 5 s 节流。闸门是结构性
  的：重建后唯一一致的状态是「所有段都空」，空链只能从 0 写。重连必须重新握手（`OP_HELLO`），
  接错的 stage 让重建失败，而不是被当成链好了。
- **checkpoint 覆盖全链之后**，3103 token 历史的第二轮从 19.85 s 降到 0.71 s（数据见
  [`SPLIT-MEASUREMENTS.md`](../measure/SPLIT-MEASUREMENTS.md) §1；那一节记了两个数字各自的口径）。
- **v8 的 per-frame 槽位顺序**：`split_remote_reset()` 清掉所有 per-frame 槽位，包括新加的
  h_nextn 行。`store_nextn` 必须在三个回答分支（候选 / 验证行 / 整行）之后执行——原来先存行、后
  存 logits，行被 `store_logits()` 的 reset 抹掉，stage 回 2048 floats、head 收到 0 行，draft 静默
  消失。**v9 的「一帧只带一种 payload」**：decode 回复只能声明一个 `n_embd` 宽度，nextn 行
  （8 KB）与 tap 段（prefill 时 8 MB）不能同发；三条回包分支必须共用同一优先级，而且 tap 优先、
  nextn 兜底（见 [`SEGMENTED-LIMITS.md`](../limits/SEGMENTED-LIMITS.md) §4）。

## 3. device-buffer 在 split 链下的崩溃与修复

**现象**：`llama-server` + 分段链 + `-md <DFlash 草稿>`，跨机和 loopback 都崩，整模不崩：

```
src/models/dflash.cpp:456（修前 440）
GGML_ASSERT(g.layer_inp_src != nullptr && g.layer_inp_src->ne[0] == n_embd_inp) failed
```

这是 rebase 引入的回归：上游 `2cbca389f dflash: copy hidden features via device copies` 之后， DFlash 特征从 host 的
`inp.embd` 改成优先读 device 侧的 `layer_inp_src`。**根因是 buffer 窄了**： head（`0:32`）只装了 8 路 extract layer 里的
6 路（层 2/7/12/17/23/28），device buffer = 6 × 2048 = 12288 列，而 dflash 的 `fc` 期望 8 × 2048 = 16384
列，断言挂在第二个合取项。插桩输出：

```
[dbg] dflash: embd-row batch, layer_inp_src=0000026FB4028060 ne0=12288 n_embd_inp=16384
```

**根因**：`supports_embd_row()` 回答「buffer 存不存在、读得到吗」，而使用者以为它回答「装的是不是我的
encoder 输入」；上游没有分段链，隐含「列数 = target_layers 数量」这个前提，从未被检查。反证：如果真是 null，`decode()`
会更早返回 -1，根本到不了图构建。

**修法（单一事实源）**：① `src/llama-context.cpp:1528` 的 `supports_embd_row()` 加宽度检查 `src->ne[0] ==
model.hparams.n_embd_inp_enc()`——链下 12288 ≠ 16384 → 假，整模仍真；这是唯一判定入口（`decode()` 准入、`graph_params()`
的 `layer_inp_src`、batch 构造三处问同一句）。 ② `common/speculative.cpp:1439` 每帧重问一次（`use_dev_features &&
llama_supports_embd_row(ctx_dft)`），因为构造期答案可能变陈旧（buffer 被重分配 / 分配失败）。 ③
`src/models/dflash.cpp:456` 断言原样保留，只在前面加自解释诊断（直写 stderr，因为 common 的 logger 是**异步**的，紧贴
abort 的 `LLAMA_LOG_ERROR` 会被丢——实测过），并注明「行号会随上方块移动、始终指这条断言」。

**为什么不选「构造时判 `!is_split_chain`」**：它把链的形状硬编码进 drafter，谓词本身仍然是错的，任何别的
`llama_batch_ext_add_embd_row()` 使用者照样撞断言；而且会误杀一个合法情形——链的 head 覆盖全部 extract layer（如 head
`0:39` + tail `39:40`）时 device 路径其实可用，宽度条件自动允许
它。**这一条是代码推出来的，未实测**。宽度条件才是真不变量：「这个 buffer 装的正好是我 fc 要的
全部层，且顺序一致」；顺序由 `build_layer_inp_dev` 升序 + 构造期 ascending 检查保证。

### 组 11：44 → 52 项

`verify/verify_dflash_dev_path.sh` 并进 `verify/verify.sh`，8 项：整模 device-buffer 路径仍然启用
（宽度谓词没有误杀整模）、整模请求 200 / 112 字符、链按预期回退到 host copies、回退原因写明（head 只装了 8 路里的 6
路）、链输出与整模逐字节相同、没有 `GGML_ASSERT`、请求后进程仍存活、该次请求真的投机了（日志里有 draft
acceptance）。

**为什么必须加**：组 9/10 跑的是 `tests/dflash-taps.cpp`，它只 link `llama`、不 link
`llama-common`——从不调 `llama_set_embeddings_layer_inp_dev`、batch 从不 `llama_batch_ext_add_embd_row`、不可能实例化
`common_speculative_impl_draft_dflash`，所以组 9/10 验的是 tap 通道（host-copies 路径），正好是这条 bug
没弄坏的那一半；整模的 device 路径同样零覆盖。

**并组时的一处误配**：`verify/verify.sh` 会 `export QLH_SPLIT_MODEL`（组 1-8 的 2B
模型），组 11 脚本最初用 `${QLH_SPLIT_MODEL:-...}` 当 target 默认值 → 拿 2B 配 DFlash → `GGML_ASSERT(lid <=
n_layer())`。修法：组 11 只认 `QLH_DFDEV_MODEL` / `QLH_DFDEV_DRAFT`， `verify/verify.sh` 显式传自己的 `MODEL_DFLASH` /
`DFLASH_DRAFT`。两个加固：`wait_health` 带 pid（进程死了立刻返回，原来会干等满 150×2 s）；脚本默认 target 改成和
`verify/verify.sh` 的 `MODEL_DFLASH` 同一文件。

## 4. 与上游 193 个提交的重合明细

| | 数字 |
| --- | --- |
| 我们的改动（相对旧基线 `e0f142d`） | 68 文件 / +18086 −504 |
| 上游 `e0f142d..f477f795c` | **193 个提交 / 659 文件** |
| 两边都碰过的文件 | **35 个** |

把改动 `git apply --3way` 打到 `f477f795c`：新文件无冲突 **22**（分段链核心文件全是新建：
`llama-split-*`、`llama-memory-split.*`、`tools/split-stage`、`tests/split-*`）、干净应用 **26**、冲突 **12 个文件 / 29
个块**（全在 `llama-context.cpp`、`llama-graph.cpp`、`server-context.cpp` 这一层）。rebase 后剩 55 文件 / +18953
−240；17 个消失文件里 16 个与上游逐字节相同。采用的上游提交：

| 上游 | 内容 | 在我们这里的表现 |
| --- | --- | --- |
| `aa588a346` | drop Windows argv reconstruction | diff 里没有 argv 重建代码 |
| `fb5997274` | per-layer rotation（`self_k_rot` → 按层数组，构造期建张量、直接上传） | 「两次对齐」抄回的等价适配整块消失：`llama-kv-cache*` / `deepseek*` 等 12 个文件与上游逐字节相同 |
| `ca8daad19` | `llm_attn_mask` 去掉 `cnv` | 同上；`dflash.cpp` 因诊断行重新进入变更集 |
| `1b6935b22` | `flash_attn_ext_rows` | diff 里已无这个符号 |
| `6023692b1` | per-model `build_rope` / kv shift 修复 | diff 里无 `build_rope` |
| `b74298c10` + `040ce199a` | conv state store / MTP conv 快照 | 不再自己存 conv 状态；`delta-net-base.cpp` 只剩上游代码 |
| `ab3768cdc` | 拒绝超出「最后一个 ubatch」的快照回退 → `cell.rs_max` | 换掉了我们的 `shifts`；`seq_rm` 判定改成按 `rs_max` 分层 |
| `e0f96a1a1` | `model` 变成 `llm_graph_context` 的成员 | graph 接线按成员访问 |
| `2cbca389f` | dflash 设备侧拷贝 | DFlash tap 的取数跟上游走设备侧拷贝，本仓只加「跨段拼接」 |
| `e95426f16` | `moe_cache_cpu_share` | diff 里无这个符号——专家缓存的内存账完全走上游 |

**删掉 `build_rs_shift_snapshots`**：原做法让 recurrent 快照按 ubatch 往前挪，使跨 ubatch 的回退也能读到；上游
`ab3768cdc` 拒绝超出本 ubatch 快照的回退，并用 `split_equal()` 的 `n_keep_tail` 保证整帧回退落在同一 ubatch
内。两套语义互不相容（shift 要求旧槽位保持旧内容，上游按本 ubatch 行号写），留着会让人以为 shift
在补上游的洞，所以整块删掉。**rebase 暴露的真回归**：删 shift 换用 `rs_max` 后，整帧放弃（删 `n_draft + 1`
行）永远被拒，而 `frame_needs_ckpt` 在 `n_draft <= n_max` 内恒假，整帧回退落到 `seq_rm` → 500。这个 bug 只有
`verify/verify_grammar_spec.sh` / `verify/verify_rbudget_spec.sh` 能发现（组 8/9/10 不走这两条路径）；修法是三个名字各司其职，见
[`SEGMENTED-LIMITS.md`](../limits/SEGMENTED-LIMITS.md) §0.3。

**两处上游基线漂移**。其一：`verify/verify_rbudget_spec.sh` 判据 3 要求「不投机时链 == 整模逐字节」，纯上游 `f477f795c` 输出
`...with Paris as the city name...`（296 B），rebase 后本仓与纯上游逐字节相同，旧上游 `ec25cedda` 时期是 `...with the
city parameter set to "Paris"...`（295 B），rebase 后
本仓链与旧上游链逐字节相同——差异全部来自上游改了整模的数值路径。判据因此改成三档：逐字节相同 →
PASS；不同但分叉点在后半段 → PASS + 记录分叉点；分叉在前半段 → FAIL（长度不参与判定）。实测不投机 8/16/32 token
全部逐字节相同，33~40 token 之间才翻。其二：`verify/verify_seq_rm_exact.py` 的 B 组（RS 形态）从绿变红，因为 `ab3768c` 之后
`rs_max = n_seq_tokens − 1`，而 probe 是逐 token decode （`llama_batch_get_one(&t, 1)`），所以 `rs_max =
0`、任何回退都必须被拒。不改 probe：用多 token ubatch 能让 B 组变绿，但「重放 logits 逐位相同」会因 1 行 vs k+1 行的
kernel 差异（0.27 量级）失效；所以更新 known-fail 集合（A1 A2 B1 B2 B3 已知红、C1 绿）并把原因写进文件头。

**副本验收的一处静默错**：做 worktree 副本验收时 `verify/verify.sh` 的 `ROOT` 改了，但 `verify/verify_multislot.py` 里的 `ROOT` / `BIN`
是硬编码的没改，组 5 实际跑的是主仓二进制，表现却是「通过」。判据是机械的：做副本验收时
`grep -n 'ROOT\|BIN\|build-split\|H:'` 每一个脚本，别只改主入口。
2026-11-10 脚本入库 `verify/` 后这处缺陷从结构上消掉了：所有脚本统一从 `QLH_BUILD_DIR` 取二进制目录，
缺二进制点名退出 3 而不是静默跑别处的二进制（`../verify/README.md`）。

**2B（`0:20` + `20:24`）上「代码没退化」的判据**：N=1 直通时默认 / `--split-layer-range 0:24` / `0:20` 三种配置输出与改动前逐字节相同；N=2 本地双段 logits 逐位相同、`seq_rm` 广播后两段 `pos_max` 同步下降；N=2 经 llama-server 时两个端口 `reasoning_content` 一致；跨进程 logits 逐位相同；投机判据分两段（无回退时与**整模投机**逐字节相同；「与不投机相同」降级为 INFO——它测的是 batch invariance）；链死 36 项。

**验收（主仓，`--clean-first` 全量构建之后）**：`verify/verify.sh` 52/0/0、`verify/verify_chain_death.py`
36/0、`verify/verify_grammar_spec.sh` 23/0、`verify/verify_rbudget_spec.sh` 36/0、`verify/verify_seq_id.py` 9/0、 `verify/verify_mtp.sh`
6/0、`verify/verify_mtp_e2e.sh` 6/0、`verify/verify_seq_rm_exact.py --expect-known-fail` 1 pass / 5 known-fail / 0 unexpected。grammar
/ rbudget / seq_id / mtp 四套最近一次跑在 `0c5c3bdd` 上——七文件改动（段级 offload、MoE
缓存默认、显存预算）不在这四条路径上，没有重跑。

## 5. 与上游版本的两次对齐（已整体吸收）

逐行对比 `src/models/dflash.cpp` 与上游 `f477f795c` 时发现：两边只差 118 行，而且全部是「上游改了
API，我们没跟上」。两处都是上游提交：

| 上游 | 内容 | 在我们这里的形态 |
| --- | --- | --- |
| `fb5997274`（#31073） | **per-layer rotation**：rotary 常量从「每次 `set_input` 从 host memcpy」改成「kv-cache 构造时按层建张量、放进独立的 `attn_rot_bufs`、直接上传」，`self_k_rot` → `self_k_rot[il]`；按层是因为每层 cache 的 buffer / 放置可以不同 | `llm_graph_input_*` 的 `self_k_rot` / `self_v_rot` 改成按层数组，`build_input_k_rot(ctx0)` 换成 `get_k_rot()`，graph 层那批 `set_input_k_rot` / `set_input_v_rot` 的 host 拷贝整块删掉；kv-cache 侧新增 `attn_rot_bufs` 的建立与上传 |
| `ca8daad19`（#31558） | **`block_uniform`**：给 `llm_attn_mask` 加 `block_uniform`，`iq = block_uniform ? seq_first[seq_id] : i`——同 seq 的后续 token 复制首行 | `llm_graph_input_attn_k_iswa` 加 `self_kq_mask_swa_block_uniform`（默认 false）；`set_input_kq_mask` 加同名参数；`args_set_input_kq_mask` 加 `seq_first` + `iq`；`dflash.cpp` 删掉自造的 `build_dspark_block_mask()`（15 行 reshape/repeat），换成一行 `self_kq_mask_swa_block_uniform = true` |

移植方式是等价适配，不是合并上游：不引入 `llm_attn_mask` 结构、不改 graph 构造签名（上游那两个提交顺带重构了
`graph(const llm_graph_params&)`，那属于完整 rebase）。数值等价由 tap 逐位证明：组 9/10 的 8 路 / 5 路 tap
与整模逐位相同。改动面（`git diff c5bd787b ec25cedda`）：13 个被 API 变更牵连的文件合计净 **+3 行**，其中
`src/llama-graph.cpp` +32 −103、`src/models/dflash.cpp` +16 −29；`graph.cpp` 删掉的一百多行主要是上游 `fb5997274`
一并删掉的 host 拷贝调用，我们自己的胶水只有 `build_dspark_block_mask()` 那 15
行，整块消失、换成一个布尔标志。**这一节已退场**：这 13 个文件现在与上游逐字节相同，`docs/llamacpp-split.patch`
里不再出现它们；留档是为了记录「先等价适配、后整体
rebase」这条路径怎么判断——判断标准是「改动能否独立成立」，不依赖上游新 API 签名的才移动，依赖的记下来放进完整 rebase。

## 6. 工具层参数校验（非分布式必需）

`common/arg.cpp`，14 行，为 `--spec-type ngram-simple` 加一条交叉校验；它出现在 patch 里是因为 patch
按工作区快照整体生成。`ngram-map.cpp:95-101` 的实现是：`copy_max = min(size_m, 剩余历史)`，只要 `copy_max <
n_draft_min`（即 `size_ngram`）就返回空草稿，所以 **`size_m < size_n` 是恒空
配置**——一个草稿都产不出来，且没有任何警告。上游默认值 12/48（`m >> n`）走不到这个分支，用户
单独调这两个参数时会。校验放在 `common_params_parse` 解析完成之后而不是各选项回调里，因为两个
参数在命令行上可以任意顺序出现，放在回调里会让 `-size-n 60 -size-m 100` 误报。验证（不需要真模型）： `-size-n 4 -size-m
3` 报 `ngram size M (3) must be >= ngram size N (4): smaller values produce no drafts at all
(ngram-map.cpp:95-101)`，`-size-n 2 -size-m 7` 通过。这条与上游无关：本项目有效组合是 `m > n`，
校验只在本仓启动参数上生效。
