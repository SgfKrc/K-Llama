# 已确认但有意未修的问题

> 状态：**待办（未实现）**
> 更新日期：2026-11-11
> 适用范围：**管**「已确认存在、判据成立、有意推迟处理」的代码与验收问题——每条带文件与符号，
> 写清怎么改、为什么现在不动。**不管**：已修的缺陷（看 git 与 [`PATCHES.md`](deploy/PATCHES.md)、[`FIXES.md`](issues/FIXES.md)）、
> 产品需求与排期、单缺陷的定位过程（那类进 `docs/issues/ISSUE-*.md`）。读者是下一个动这块代码的人：
> 这份清单只服务一个目的：下一次改动之前先读到这里，避免同一处问题被重新发现、被当成已经修完。

代码引用口径：`src/`、`tools/`、`common/` 指 `H:\llamacpp-c` 的 fork 树（HEAD `4acf03c8b`，基底
`09338b85e`），其余路径相对本仓库。行数按 `4acf03c8b` 的函数边界统计，只作量级参考；符号名是稳定引用。

## A. 结构性重构（改动面跨文件，单独一轮）

### A1. 协议 body 布局两端各写一份 —— **已修（2026-11-11）**
- **位置**：`tools/split-stage/main.cpp` 的 `reply_parts()` 七处调用点正向加总 `n_bytes`（relay+tap、verify+tap、verify+nextn、cand+tap、cand+nextn、plain+tap、plain+nextn）；`src/llama-split-remote.cpp` 的 `llama_split_remote::decode()` 反向推 `nextn_off` / `nextn_bytes` / `tap_bytes`，再用 `cand_bytes = rsp.n_bytes − nextn_bytes − tap_bytes` 反解；`src/llama-split-proto.h` 里只有 wire 结构体与 `static_assert`，零个 encode/decode 函数。
- **判据**：单一事实源——同一条「body 怎么排、各段多宽」的规则有两份实现，一份加法一份减法。
- **建议**：在 `llama-split-proto.h` 加 `layout_of(flags, n_tokens, n_embd, n_vocab, n_taps)`，返回各段 offset/宽度；`reply_parts()` 用各段宽度之和算 `n_bytes`，`decode()` 用同一张表推 offset。配一个只测 `layout_of` 的单测，回环测试继续兜底。
- **怎么修**：`src/llama-split-proto.h` 新增 `body_layout` 与 `layout_of(flags, n_tokens, n_embd, n_vocab, n_taps)`（`total` / `offset_of(seg)` / `with_cand(n_cand)`），七处加总与四处反解都改读它；`tools/split-stage/main.cpp` 的 `answer()` 与四个 `answer_*()`、`src/llama-split-remote.cpp` 的 `decode()` 都用这一张表。自检两端都是显式 `if`（`build-split` 是 Release，`/DNDEBUG` 下 `assert` 是空的），另有四条 `static_assert(DECODE_FLAG_* == RSP_FLAG_*)`；候选段是唯一运行时宽度，由 `with_cand()` 补、读侧取余数。wire 结构体一个字节未动，不需要 clean-first。
- **为什么留**：改动面跨两个构建目标（`tools/split-stage` 与 `libllama`），且要先把六种回复形状归一成一张表。v5→v9 五次协议升级每次都付了双份改动 + 双份测试，这一条是那个成本的源头。

### A2. `decode()` 385 行 / 20 参数 / 6 种回复形状
- **位置**：`src/llama-split-remote.cpp` 的 `llama_split_remote::decode()`（约 385 行，20 个参数，其中 8 个是 `out_*` 出参）。
- **判据**：屎山——一个函数里叠了 hidden、hidden+tap、verify(+nextn/tap)、candidates(+nextn/tap)、plain logits(+nextn/tap) 六种回复的读取。
- **建议**：拆四个 body reader（hidden / verify+nextn / candidates / plain+tap），出参收进 `decode_reply_out` 结构体；入参里的 `out_hidden`/`out_logits`/`out_n_vocab`/`out_cands`/`out_verify`/`out_nextn`/`out_nextn_embd`/`out_tap` 随之消失。
- **为什么留**：要定 `decode_reply_out` 的字段语义（哪个通道与哪个互斥、哪个只在一帧内有效），与 A1 的布局表同一轮做。

### A3. `decode_run()` 404 行 / 4 条回答路径 —— **已修（2026-11-11）**
- **位置**：`tools/split-stage/main.cpp` 的 `decode_run()`（约 404 行）。12 个 return 点已由 `batch_guard` 收掉，剩下的是 relay / verify / cand / plain 四条路径各自组装 tap 与 nextn 的 `reply_parts`（三份重复）。
- **判据**：重复造轮子。
- **建议**：先抽一个 `answer()`，把 `n_bytes` / `flags` / `n_taps` 的组装收进去（内部调 A1 的 `layout_of`），再按通道拆四个 builder。
- **怎么修**：`tools/split-stage/main.cpp` 抽了 `answer()`（`n_bytes` / `flags` / `n_taps` 由 `layout_of(...).with_cand(...).total` 推出，并显式校验「各段长度之和 == 表的 total」）与四个 builder `answer_relay` / `answer_verify` / `answer_cand` / `answer_plain`；tap 与 nextn 的散装参数收成 `tap_parts` / `nextn_parts`。`decode_run()` 403 → 365 行，剩下的是四条路径的 decode 主体，不再拆。
- **为什么留**：与 A1 同一处布局，分两轮做等于改两遍。

### A4. 七个 op 方法各抄一遍「valid → 填 header → send → recv → check → status」
- **位置**：`src/llama-split-remote.cpp` 的 `state_size()` / `state_take()` / `state_give()` / `state_load()` / `hello()` / `seq_rm()` / `clear()`。
- **判据**：重复造轮子。
- **建议**：抽 `roundtrip(op, req, send_body, recv_body, &rsp)`，七处共用；`hello()` 的身份比较与 `greeted` 置位留在调用点。
- **为什么留**：七处的失败语义并不一样（`hello()` 要写 `err` 文案并决定连接是否还算「已握手」），要先定返回码约定再抽。

### A5. `split_attach` / `split_attach_remote` / `split_drive_remote` 三段结构重复
- **位置**：`src/llama-context.cpp` 的 `split_attach()`（约 227 行）、`split_attach_remote()`（约 175 行）、`split_drive_remote()`（约 198 行）。前两者各含「head 校验 / 逐段 tiling + 握手 / 装 union view」三段。
- **判据**：重复造轮子 + 正确性——两份校验已经漂移（见 B5）。
- **建议**：抽 `validate_head()` 与 `tile_check()`，两条 attach 路径共用；`split_drive_remote()` 与 `split_drive_local()` 的 hop 循环共用已有的 `split_build_hop()`。
- **为什么留**：抽完要重跑组 1/4/6/7（本地双段、跨进程、三段、远程链各一组），一轮只够改一处并验证。

### A6. `issue_expert_willneed()` 326 行
- **位置**：`src/slim-arc-prefetch.cpp` 的 `prefetch_scheduler::issue_expert_willneed()`（约 326 行）。
- **判据**：屎山——目标集、驻留策略、预算与计数混在一个函数里。
- **建议**：拆 `build_target_set()` / `apply_residency_policy()` / `issue_and_account()`。
- **为什么留**：它在 NOT WIRED UP 区（B1）。接线前拆一次、接线后按真实压力源再调一次，现在拆完还要改。

### A7. 进程清理六份实现，判据互不相同
- **位置**：`verify/verify.sh` 的 `kill_ours_tree()` / `cleanup_ours()`、`verify/ours_procs.ps1`、`verify/verify_dflash_dev_path.sh` 的第二份 `kill_ours_tree()` / `cleanup_ours()`、`verify/mtp_remeasure.sh` 的 `kill_ours()`、`verify/verify_grammar_spec.sh` 与 `verify/verify_rbudget_spec.sh` 的 `kill_ours()`、`verify/verify_chain_death.py` 的 `kill_93xx()`、`control/qlhctl.py` 的 `stop_pid()` / `_stop_entry()`。
- **判据**：重复造轮子——六份判据分别是「按端口」「按映像名」「按 `$BIN` 路径」「按 93xx 号段」「按命令行 + 层范围」。
- **建议**：统一到 `verify/_qlh_env.sh` 的 `qlh_kill_ours`，判据用「可执行文件在 `$BIN` 下」这一条（`verify.sh` 头部注释已论证过它覆盖最常见的三种残留），其余按需传参。`qlhctl.py` 保留自己的 `stop_pid()`：它是产品侧，要带端口与层范围做身份校验。
- **为什么留**：`verify/` 正在并行改动，统一要等它稳定；且「杀谁」的判据本身要定——远程段的进程本机看不见（`docs/deploy/ANDROID-SURFACE-段接入评估.md` §1 记了安卓上 `ps`/`pgrep` 的失效）。

### A8. `wait_health` 六份 / 就绪等待八份 / 启动命令行手写三十余处 / 模型默认路径八处
- **位置**：`wait_health()` 在 `verify/verify.sh`、`verify/mtp_remeasure.sh`、`verify/verify_dflash_dev_path.sh`、`verify/verify_grammar_spec.sh`、`verify/verify_mtp_e2e.sh`、`verify/verify_rbudget_spec.sh` 各一份，`verify/measure_state_bytes.py` 与 `verify/verify_chain_death.py` 再各一份；`llama-server` / `split-stage` 的命令行在 `verify/*.sh` 里手写三十余处；模型默认路径散在八处。
- **判据**：重复造轮子。
- **建议**：四样都收进 `verify/_qlh_env.sh`（已有 `qlh_require_bins` / `qlh_available_gb` 两个先例），对外暴露 `qlh_wait_health` / `qlh_start_stage` / `QLH_MODEL_*`。
- **为什么留**：与 A7 同一批文件，同一轮做；命令行手写处要先逐条核对 `--no-warmup`、`-c`、`-ngl 0`、`-b/-ub` 的差异，机械替换会改掉各组的测试条件。

## B. 设计层（要定接口语义）

### B1. SLIM-ARC 的约一千行未接线代码
- **位置**：`src/slim-arc-prefetch.cpp` 从 `register_expert_tensor()` 起的一段（已标 `[QLH] NOT WIRED UP`，为后续研究保留）：`cache_router_experts()`、`prefetch_experts()`、`cancel_expert_prefetch()`、`set_expert_residency_pressure()`、`set_expert_budget()`、`reset_expert_budget_usage()`。
- **判据**：正确性——代码在树上但不在任何调用路径上，指标恒 0。
- **建议**：接线要补三样：`src/llama-model-loader.cpp` 的 `init_mappings` 之后按 `ne[2]` 注册专家张量；`set_expert_residency_pressure()` 接真实压力源（`mincore` / PSAPI 采专家页驻留率）；`issue_expert_willneed()` 之前调 `expert_pressure_controller::update()`——该 controller 目前挂在 `prefetch_scheduler` 之外，要先加进来。
- **为什么留**：要定接口（压力源从哪来、controller 归谁），且没有真实压力源时接上只会让 `dump_metrics()` 输出一排 0。

### B2. `chain_kind` 三态枚举 —— **已修（2026-11-11）**
- **位置**：`common/speculative.cpp`——DFlash 用局部 `tail_remote`（`~1097`）加三重条件判「这一帧往哪走」，MTP 用成员 `is_chain` / `is_remote` / `is_mem_shared`（`~1821` 起）；「链上 / 远程 / 本地」的判据散在十处。
- **判据**：单一事实源——两套命名、两套拼法，历史上已经出过行为分叉。
- **建议**：`enum class chain_kind { WHOLE, LOCAL_TAIL, REMOTE_TAIL }`，构造函数里定一次，各处读它。`is_mem_shared` 与 `chain_heads` 保留：它们描述 MTP 的形态，不描述链形态。
- **为什么留**：要定语义——「远程 relay 链 + 本地 tail」这个组合现在没有名字，定枚举之前得先说清它归哪一类（`tools/server/server-context.cpp` 的 `ctx_tail_remote` 置位条件正是它的边界）。
- **怎么修**：`common/common.h` 的 `common_params_speculative_draft` 加 `enum class chain_kind { WHOLE, LOCAL_TAIL, REMOTE_TAIL }` + `kind()` + 三个 inline 谓词（无数据成员、布局不变）；MTP 的 `is_chain`/`is_remote` 成员删除，DFlash 的局部 `tail_remote` 删除，**21 个代码点收敛**。优先级（两字段都置时 remote 优先）写进注释，MTP 的互斥断言仍读原始字段（`kind()` 会掩盖冲突）。

### B3. `split_frame_answer` 结构体
- **位置**：`src/llama-context.h`——五个答案槽（`split_remote_logits` / `split_remote_cands` / `split_remote_verify` / `split_remote_nextn` / `split_remote_tap`）+ 五个标量（`split_remote_n_vocab` / `split_remote_logits_row` / `split_remote_cands_row` / `split_remote_nextn_embd` / `split_remote_nextn_row0`）+ 三个 armed（`split_cand_wanted` / `split_verify_wanted` / `split_nextn_wanted`）+ 三个 warned（`split_verify_row_warned` / `split_nextn_warned` / `split_tap_warned`）散落。
- **判据**：屎山 + 正确性——store 顺序有隐性依赖：`split_remote_store_cands()` 与 `split_remote_store_logits()` 都先调 `split_remote_reset()`，先存 nextn 会被 reset 吃掉（`llama-context.h` 的注释自己记了「实测丢过」）。
- **建议**：一个 `split_frame_answer` 结构体装槽与标量，armed / warned 各一个位域；reset 从逐字段清改成整结构体赋值。
- **为什么留**：要定「一帧只有一个答案」的接口语义，`get_logits_ith()` / `get_embeddings_nextn()` / `get_embeddings_layer_inp()` 等公开访问器要跟着改。

### B4. `split_active()` 谓词
- **位置**：`src/llama-context.cpp` 里 `!split_stages.empty() || !split_remotes.empty()` 出现 3 次、`!split_remotes.empty()` 出现 7 次；`src/llama-context.h` 已写明 `split_memory` 是「Non-null iff the chain has stages」。
- **判据**：单一事实源——三个判据表达同一件事。
- **建议**：加 `bool split_active() const`，调用点全换。
- **为什么留**：两条判据当前等价但语义不同（一个问容器、一个问 union view）。选哪条要先确认 detach 路径上两者的更新顺序（`split_attach(…, 0)` 里 `split_memory.reset()` 在 `split_stages.clear()` 之前）。

### B5. 本地 / 远程 head 校验抽公共函数
- **位置**：`src/llama-context.cpp` 的 `split_attach()` 与 `split_attach_remote()`。远程版经 `identity_mismatch()`（`src/llama-split-remote.cpp`）比 `n_vocab` / `n_embd_inp` / `n_embd_out` / `n_layer_src` / `layer_lo` / `layer_hi` / `role`；本地版只比 `n_layer_src`、层连续性、arch、宽度与 embeddings，不验 `n_vocab`。
- **判据**：正确性——两份校验已经漂移；`n_ctx` / `n_seq_max` / `n_rs_seq` 两端都没有，`stage_id`（`src/llama-split-proto.h`）里根本没有这三个字段。
- **建议**：抽 `split_check_head(const llama_context &, const head_expect &)` 供两条路径共用；`stage_id` 若要加 `n_ctx` / `n_seq_max` / `n_rs_seq`，同一轮加。
- **为什么留**：给 `stage_id` 加字段就是协议变更，要连 `static_assert(sizeof(stage_id) == 28)`、两端的版本检查与 [`PROTOCOL-V9.md`](design/PROTOCOL-V9.md) 一起改。

### B6. M-RoPE 位置展开是逐字复制
- **位置**：`src/llama-batch.cpp` 的 pos 展开循环（`pos[j*n_tok + i]`，`p = j < 3 ? tok.pos[0] : 0`）与 `src/llama-context.cpp` 的 `split_expand_pos()`（由 `split_remote_positions()` 调用）；`llama-batch.cpp` 的注释已经写明「两处必须一起改」。
- **判据**：单一事实源——同步机制只有一条注释；`GGML_ASSERT(n_pos_per_embd <= GGML_MROPE_SECTIONS)` 只防越界，不防规则漂移。
- **建议**：把展开提成 `llama_batch_ext` 的公开辅助函数（要动 `llama-batch.{h,cpp}`），两处都调。
- **为什么留**：动的是上游文件，要在补丁里单列一处 diff 并进 [`PATCHES.md`](deploy/PATCHES.md)，与下一次重导一起做。

### B7. `env_bool` / `env_int` 统一
- **位置**：`src/slim-arc-prefetch.cpp`——六处各自写 `from_chars` + 范围检查 + `fprintf`（`parse_expert_hot_budget_bytes()`、`parse_expert_hot_admission_threshold()`、`parse_popularity_k()` 等）；另有六处运行期 `getenv`，其中 `SLIM_ARC_DYNAMIC_MADV` 在 `set_phase()` 与 `on_layer_boundary()` 各读一次，后者在每层边界热路径上。
- **判据**：重复造轮子。
- **建议**：一个 `env_bool(name, def)` / `env_int(name, lo, hi, def)`，构造时缓存；热路径那两处先提到构造期。
- **为什么留**：热路径那两处的现有语义是「运行期可改」，提成缓存会去掉这个能力。要先定「允许运行期改的开关」清单，再动手。

### B8. `common_speculative_init_from_params` 的公开接口语义
- **位置**：`common/speculative.cpp` 的 `common_speculative_init_from_params()`——`cparams.n_rs_seq = 0`（`~3754`），draft ctx 的 `n_rs_seq` 恒为 0；两处判据在读它：`tools/server/server-context.cpp` 的 `use_ckpt_dft`（`~4524`）与 `n_draft > llama_n_rs_seq(ctx_tgt)`（`~6109`）。现在靠「`ctx_tail_remote` 只在 `ctx_tail == nullptr` 时置 true」这条库外不变量兜着。
- **判据**：正确性——读一个恒 0 的值做分支，`draft.size() > 0` 恒真，draft checkpoint 恒被取用。
- **建议**：`common_speculative_init_result` 直接暴露 `draft_rs_seq` 与 `draft_is_remote`，server 不再读 `llama_n_rs_seq(ctx_dft)`；或给 remote draft 一个真实值。
- **为什么留**：要定公开接口（谁拥有 draft 的 KV 能力），而 `common/` 是三个入口共用的头文件。

### B9. `scripts/termux-install.sh` 整份面向已退场的 v2 —— **已标注（2026-11-11）**
- **位置**：`scripts/termux-install.sh`（装 `stage_worker.py` / `stage_supervisor.py` / 生成的 `topology.toml` / shim），仍被 [`PACKAGING.md`](deploy/PACKAGING.md) 与 [`DELIVERY.md`](deploy/DELIVERY.md) 引用。
- **判据**：单一事实源——文档指向已退场的实现。
- **建议**：归档到 `_archive/`（本机工具通道），两篇文档改成「已退场」并指向 `slimarc/prototype`；若保留，改成装 v2.1 的 `split-stage` + `qlhctl`。
- **为什么留**：要先定安卓交付形态（Termux 直接跑 `split-stage`，还是保留 shim 层），见 D3。
- **怎么处理**：不删脚本（保留 v2 部署形态的记录），在脚本头加状态块写明「已退场、照它装会得到不存在的系统、v2.1 见哪两篇」；`PACKAGING.md` 的引用处加同样的提示并把命令标成「v2 口径」；`DELIVERY.md` 的引用处改成「v2 用 …（已退场），v2.1 见 …」。

### B10. head 偶发发出无效 `cand_req`，把整条链标成不可信 —— **未修（2026-11-14 组 15 首跑发现）**
- **现象**：多序列混合批 + 候选通道时，stage 拒帧，两种消息交替出现：`OP_DECODE candidate request is out of range (n_steps=0, 40 bytes vs 40)` 与 `candidate request named a sampler step this build does not have`。随后 head 把整条链标成 `no longer trustworthy`，**正在 decode 的另一个序列被打断**（实测 B 只输出 25 字符，基线 361）。
- **判据**：正确性——不是「报错」而是「静默打断另一个序列」，形态与组 15 判据 5 要防的「输出像正常但错」同类。
- **复现**：单独复现 4 轮命中 2 轮，两种拒绝交替；判据 2 走候选通道的自然路径两次都过，但理论上同样可能撞上。
- **位置**：**待定位**。指向 head 侧构造 `cand_req` 的地方（`n_steps=0` 或 `steps[]` 含 0），`src/llama-context.cpp` 与 `common/speculative.cpp` 都有候选帧构造路径，尚未二分到具体行。
- **为什么留**：组 15 落地时只做到「把它从判据 3 的干扰请求里排掉」（改用 `logprobs` 整行路径），没有动实现——它不属于 `seq_rm` 隔离语义，混在判据 3 里会让失败原因失真。
- **怎么修**：先二分定位 `cand_req` 构造点，确认 `n_steps=0` 是「合法状态下不该出现」还是「某个边界条件下允许」；若是前者，head 侧应拒绝发出并在本地报错，而不是让 stage 拒帧后把链打死。

## C. 低危 / 风格（攒着一起做）

### C1. 组 11 用 `sed` 解析子脚本输出计分 —— **已修（2026-11-11）**
- **位置**：`verify/verify.sh` 组 11 的 `sed -n 's/^  \[PASS\] //p'` 与 `[FAIL]` 两段。
- **判据**：重复造轮子——父子两套 PASS 计数耦合。
- **建议**：子脚本输出一行机器可读汇总（`PASS=n FAIL=m`），父脚本读它；现有的 `EXPECT_PASS` 项数断言继续兜底。
- **为什么留**：低危（项数断言已能抓「静默少跑」），且 `verify/` 在途改动。
- **怎么修**：子脚本的 `ok()`/`bad()` 额外写 `PASS\t消息` / `FAIL\t消息` 到 `$QLH_GROUP_RESULT`，`verify.sh` 组 11 读该文件计分，不再 `sed` 刮输出；独立跑子脚本不传变量则无文件、行为不变。

### C2. ~~`docs/c-scheme-full.patch` 是死文件~~（已删，2026-11-11）
- **位置**：原 `docs/c-scheme-full.patch`（14 文件，第一代 C-scheme），全仓零引用，已被现行主补丁取代，清理时删除。保留此条只为记录它存在过。
- **判据**：单一事实源——第二份补丁快照，与 `llamacpp-split.patch` 的现行口径并存。
- **建议**：删除，或移入 `docs/legacy/` 并在归档索引登记一行。
- **为什么留**：删除前要确认没有验证脚本把它当输入（当前 `grep` 零命中），归档通道一次做掉。

### C3. 路径规范化两份实现 —— **已修（2026-11-11）**
- **位置**：`verify/_qlh_env.sh` 的 `_qlh_norm()`（`case`）与 `verify/_qlh_env.py` 的 `_norm()`（正则）。
- **判据**：重复造轮子。
- **建议**：保留 sh 版为事实源，py 版改成同一张规则表，或在注释里把「三种写法」指到同一处。
- **为什么留**：低危（两份当前行为一致），跨语言共用要付一次子进程开销。
- **怎么修**：`_qlh_env.sh` 的 `_qlh_norm` 标为唯一事实源，`_qlh_env.py` 的 `_norm` 加同一张四条规则表并互相指名。

### C4. `--rs-seq 8` 五处字面量 —— **已修（2026-11-11）**
- **位置**：`verify/verify.sh` 的 `RS_SEQ=8` 之外，`verify/verify_dflash_dev_path.sh`、`verify/verify_grammar_spec.sh`、`verify/verify_mtp.sh`、`verify/verify_mtp_e2e.sh`、`verify/verify_rbudget_spec.sh` 各自手写 `--rs-seq 8`。
- **判据**：单一事实源。
- **建议**：进 `verify/_qlh_env.sh` 的 `QLH_RS_SEQ`，各脚本改成 `--rs-seq "$QLH_RS_SEQ"`。
- **为什么留**：与 A8 同一批文件，同一轮做。
- **怎么修**：`QLH_RS_SEQ=${QLH_RS_SEQ:-8}` 进 `_qlh_env.sh` 并 export，五个 sh 与 `verify.sh` 全部改用它。

### C5. 死变量与不可达分支 —— **已修（2026-11-11）**
- **位置**：`verify/verify.sh` 的 `n1=""`（组 2 里先赋值后使用，这行初始化没有读者）与 `SERVER_PID=$!`（只在赋值处出现一次，注释写明「仅供诊断」）；`control/qlhctl.py` 的 `preflight_ports()` 里 `if pid is None:`——`conflicts` 只在 `pid` 非 None 时 append，该分支不可达。
- **判据**：屎山。
- **建议**：删掉 `n1=""` 与 `SERVER_PID`；`preflight_ports()` 的 `pid is None` 处理要么删，要么把 `listening.get()` 的 None 也放进 `conflicts`。
- **为什么留**：低危，且 `verify.sh` 与 `qlhctl.py` 都在途改动。
- **怎么修**：删 `verify.sh` 的 `n1=""` 与 `SERVER_PID`、删 `qlhctl.py` 的 `preflight_ports()` 不可达分支。

### C6. `extern "C"` 重复声明 —— **已修（2026-11-11）**
- **位置**：`src/llama-context.cpp` 的 `llama_set_embeddings_nextn()` 与 `llama_get_embeddings_nextn()` 定义处各带一次 `extern "C"`，`src/llama-ext.h` 已经声明过。
- **判据**：屎山（rebase 噪声）。
- **建议**：删掉定义处的 `extern "C"`；C++ 里定义处带它不改变链接名。
- **为什么留**：改 fork 源码要重导补丁，攒到下一次重导一起做。
- **怎么修**：删 `src/llama-context.cpp` 定义处两处冗余 `extern "C"`（`llama-ext.h` 已声明 C 链接）。

### C7. `speculative.cpp` 的 `#7` 条件 —— **原建议有误，已撤回（2026-11-11）**
- **位置**：`common/speculative.cpp` 的 `if (is_chain && !is_mem_shared && !keep_kv)`（`~2654`）。
- **原判据**：屎山——看起来 `is_chain` 已蕴含 `!is_mem_shared`，三条件可简化。
- **原建议（错）**：收成 `!keep_kv`，或保留 `is_chain` 去掉 `!is_mem_shared`。
- **实测为什么不等价**：`!keep_kv` 对**整模 MTP**（非链、非远程）恒真，于是 `draft()` 会每轮执行
  `llama_memory_clear(ctx_dft, true)`；原判据在该路径为 false、不清。清了会让 draft 退化——非
  keep_kv 路径依赖 `process()` 填入的 prompt draft KV 与 `chain_heads` 的逐头 rebuild。另一个选项
  「去掉 `!is_mem_shared`」也不成立：`is_chain` 在库调用者手上不蕴含 `!is_mem_shared`，会清掉
  目标 KV。
- **结论**：三条件**都是必需的**，不是冗余。只做了 B2 的判据收敛（改读 `params.is_local_tail()`），
  条件本身保留，并在代码里写明保留 `!is_mem_shared` 的理由。**不要再照着本条原建议去简化。**

### C8. `common_sampler_copy` 注释与实现不符 —— **已修（2026-11-11）**
- **位置**：`common/sampling.cpp` 的 `common_sampler_copy()` 与 `sampler_plan_copy()`。
- **判据**：屎山——注释声称两路同规则，实际 `common_sampler_clone()` 不复制 `t_total_us`、`common_sampler_copy()` 复制；`dst->cur_p.data` 的重指依赖「`src->cur_p.data` 指向 `src->cur`」这条没写下的前提。
- **建议**：注释补上这两条差异，或把 `t_total_us` 也挪进 `sampler_plan_copy()`；`cur_p.data` 重指加一条断言。
- **为什么留**：低危（`cur_p` 只在采样中间态被读），且 `common/` 与上游的 diff 越少越好。
- **怎么修**：`sampler_plan_copy()` 注释写明 `t_total_us` 有意不在此处（clone 从零、copy 恢复检查点）；`common_sampler_copy()` 在 `cur_p.data` 重指前加断言。

## D. 需要外部条件

### D1. GPU 路径的完整验收
- **位置**：`verify/verify.sh` 的段级 offload 组（测量装置 `verify/probes/seg_probe.cpp`）与 [`SEGMENT-OFFLOAD.md`](measure/SEGMENT-OFFLOAD.md) §3/§4/§5。
- **判据**：正确性（验收覆盖）——自动组只覆盖一个小模型 + 少量 `-ngl` 层的段级 offload 正确性。
- **建议**：自动验收覆盖 35B 的 head 段与 tail 段、`op_offload` 开关两侧、MoE 缓存命中率、显存预算的预检与后检。
- **为什么留**：等外部条件——35B 整模装不下 8 GB，`op_offload` 的 +28%、MoE 缓存 +95% 与显存预算的数字只有手工 probe；一台机器的一张卡不构成结论。

### D2. SLIM-ARC 的 iGPU 形态
- **位置**：[`SLIM-ARC-接入计划.md`](design/SLIM-ARC-接入计划.md) §6.3（iGPU 上装不上），§6.1/§6.2 的预取通路已在 2026-11-11 复核为已修。
- **判据**：正确性（形态未验证）。
- **建议**：核显机器上重跑预取通路，记录 `issued` 与命中率；装不上就把结论并入 [`IGPU-SLOWDOWN.md`](measure/IGPU-SLOWDOWN.md)。
- **为什么留**：等外部条件（需要一台核显机器 + 一次完整实测）。

### D4. 三段链的切分参数没有落盘（复测已放弃，见下）
- **位置**：三段链那次运行的日志里只有 `[split] attached 2 remote stage(s): 100.100.52.106:9331,
  100.99.211.13:9332`，**没有 `--split-layer-range` 的实际值**。本机 head `0:32`、Surface relay
  `32:36`、Y700 tail `36:40` 是从启动命令推的，不在日志里。
- **判据**：可复现性——`MODEL-SWAP.md` §11.8 的三段链 **9.09 t/s** 无法从日志复算。
- **建议**：attach 成功时把 `split_layer_lo/hi` 与每个 `--split-remote` 打进一行无条件日志（像
  `[split] attached N remote stage(s)` 那样）。
- **处置（2026-11-13）：复测放弃**。2026-11-12 用同拓扑（`0:24/24:32/32:40`、`-t 8`、941 token、客户端
  本机）复测只有 **5.85 t/s**；而同 head `0:32`、同 `-t 14` 的两段链是 5.98–6.29——**三段比两段快 52%
  在物理上说不通**。用户的判断是「怀疑是旧版本来源不明数据填进来了」，所以 **9.09 只作追溯线索、
  不作现行值**（`INDEX.md` §1 已这么标），**不再花机器去复现**。现行三段链数字用 2026-11-12 的
  **5.85**（`measurements.csv` `20261112-D1`）。
- **仍然建议改 attach 日志**：不管 9.09 要不要复现，「切分参数不落盘」本身是可复现性缺陷——下次
  谁再跑三段链，日志里还是看不到切点。改动很小（`src/llama-context.cpp` 的 attach 处打一行）。

### D5. 跨机投机还能不能调（开放问题，留给读者）
- **现状**：同口径实测（941 token、`-t 8`、direct、客户端本机）两段 → Y700 的 DSpark k=2 是 **+51%**
  （7.46 → 11.35），两段 → Surface 是 **+15%**（10.77 → 12.36），三段是 **−0.4%**（基本不划算）。
- **为什么三段不行**：DSpark 的 5 个 target `[2,10,20,30,37]` 在两段上只跨线 1 个、三段上跨线 **2** 个；
  多一次跨线往返就把投机的收益吃掉。DFlash 更极端（6 个 target，三段跨线 3 个，**−31%**）。
- **还能调什么**（未做，没有结论）：
  - **切点**：把 head 的层数选在「tap 尽量落在本地」的位置——DSpark 的 targets 是 `[2,10,20,30,37]`，
    如果 head 切到 `0:31` 而不是 `0:32`，第 5 个 tap（`30`）仍在 head、`37` 在 tail，跨线仍是 1 个；
    但切到 `0:21` 就只跨 `30,37` 两个。**切点与 targets 的对齐关系没系统测过。**
  - **k**：全程只测了 k=2。k=3/4 在跨机上是否更划算（更多行摊一次往返）没测。
  - **catch-up**：`MODEL-SWAP.md` §11.4 的自适应 catch-up 让链上接受率回到整模（0.688 → 0.91），
    但它对**跨机端到端 t/s** 的净效果只在 941-token 口径下测过一次。
- **为什么留作开放问题**：调优要占两台机器连续跑几小时，而收益上限受「跨线 tap 数」这个结构约束——
  段数越多越难赚。**交给后续读者**：如果他们换更强的尾段机器、或能改 tap 的传输方式（比如把 tap
  与 hidden 合并成一帧），这条线上还有空间。
### D3. 安卓 / ARM 的自动验收
- **位置**：[`ANDROID-SURFACE-段接入评估.md`](deploy/ANDROID-SURFACE-段接入评估.md) §6「证据等级」与 §7「已完成 / 下一步」；相关 `scripts/termux-install.sh`（B9）。
- **判据**：正确性（验收覆盖）——安卓结论全部来自手工实测，没有脚本支撑。
- **建议**：把 §1「安卓侧怎么跑」做成脚本（起 `split-stage`、一条链、三问），进 `verify/` 的外部设备组；进程判据要换（§1 已记：安卓上 `ps`/`pgrep` 看不见别的进程）。
- **为什么留**：等外部条件（需要一台安卓设备常驻）。

## 优先级

下一轮最该先做的三条，按「做完能解锁多少、不做会重复付多少」排。**2026-11-11 更新：三条已全部完成**
（A1/A3 见上、B2 见下），下表保留原排序作记录——它是「为什么这三条先做」的判据，不是待办。

| 序 | 条目 | 理由 |
| --- | --- | --- |
| 1 | **A1 协议布局** ✅ 已修 | 它是 A2/A3/A4 的前置：那三处都要靠同一张布局表收口，先拆函数等于把两份实现拆成四份。v5→v9 五次协议升级每次都付了双份改动 + 双份测试，这是唯一能终止那笔重复的改动。 |
| 2 | **B2 `chain_kind`** ✅ 已修 | 唯一一条已经造成过行为分叉的问题，且它挡着 B5/B8 的语义整理：链形态没有名字，「远程 relay + 本地 tail」该走哪套校验就说不清。改动集中在一个文件，验证成本低。 |
| 3 | **A3 `decode_run`** ✅ 已修 | 改动面在 `tools/split-stage` 一个构建目标内，回环组 1/6/7 就能验证，风险最小；但要排在 A1 之后，否则拆完仍是三份 `reply_parts`。 |

其余的顺序建议：A4 / A7 / A8 是同一类文件级重复（`verify/` + 远端客户端），等 `verify/` 的在途改动落定后一轮做完；A5 与 B5 同一处代码，一起做；C 类攒到下一次补丁重导；D 类等设备与显存条件，不占轮次。
